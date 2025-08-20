import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Import local modules
from lib.data.IncrementalDataLoader import LogsDataLoader
from lib.model.incremental_model import (IncrementalLSTMClassifier, train_model, 
                                       predict_model, update_model, compute_embeddings, 
                                       finetune_classifier)
from lib.utils.augmentation import (intra_class_mixup, distribution_aug)
from lib.utils.drift_detection import detect_drifts
from lib.utils.helper import (compute_embeddings_with_raw, create_dataloader_from_buffer,
                            combine_dataloaders)
from lib.utils.replay_buffer import (update_replay_buffer, init_global_prototypes)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CIL2D for Next Activity Prediction")
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="BPIC15_1", 
                        help="Name of the dataset")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to the data directory")
    parser.add_argument("--window_type", type=str, default="month", choices=["day", "week", "month", None],
                        help="Type of time window for test batches")
    parser.add_argument("--train_test_ratio", type=float, default=0.10,
                        help="Ratio for splitting training and test data")
    
    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="Dimension of activity embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Dimension of LSTM hidden state")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs for training and updates")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Base learning rate for model training (lr/2 will be used for updates)")
    
    # Drift detection parameters
    parser.add_argument("--drift_threshold", type=float, default=0.05,
                        help="default threshold for detecting shifted activities")
    parser.add_argument("--novelty_threshold_factor", type=float, default=2.0,
                        help="Factor for novelty threshold")
    parser.add_argument("--use_cosine", action="store_true",
                        help="Use cosine distance instead of Euclidean")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="Minimum number of samples required to compute reliable statistics")
    
    # Augmentation parameters
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Mixing parameter for intra-class mixup with lambda ~ beta(alpha, alpha)")
    parser.add_argument("--buffer_size_per_class", type=int, default=20,
                        help="Number of representative samples to keep per class in the replay buffer")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training/evaluation")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    return parser.parse_args()

def create_output_dir(args):
    """Create output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.window_type:
        output_dir = os.path.join(args.output_dir, args.dataset, args.window_type, f"run_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, args.dataset, "no_window", f"run_{timestamp}")
        
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_results(metrics, output_dir):
    """Save metrics to CSV file."""
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(output_dir, f"results.csv"), index=False)
    return df

def save_parameters(args, output_dir):
    """Save all parameter settings to a file."""
    params_dict = vars(args)
    
    with open(os.path.join(output_dir, "parameters.json"), 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    params_df = pd.DataFrame([params_dict])
    params_df.to_csv(os.path.join(output_dir, "parameters.csv"), index=False)
    
    return params_dict

def plot_accuracy_over_time(batch_accuracies, batch_timestamps, output_dir):
    """Plot accuracy over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(batch_timestamps, batch_accuracies, marker='o')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_time.png'))
    plt.close()

def main():
    """Main function for the CIL2D pipeline."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = create_output_dir(args)
    
    # Save all parameter settings
    save_parameters(args, output_dir)
    
    print(f"=== CIL2D for {args.dataset} ===")
    print(f"Results will be saved to: {output_dir}")
    
    # === LOAD DATA ===
    print("\n=== Loading Data ===")
    loader = LogsDataLoader(
        dataset_name=args.dataset,
        dir_path=args.data_dir,
        window_type=args.window_type
    )
    loader.load_data()
    train_df, test_df = loader.split_train_test(args.train_test_ratio)
    
    # === MODEL TRAINING ===
    print("\n=== Training Model ===")
    train_dataloader = loader.encode_and_prepare(train_df, args.batch_size, shuffle=True)
    vocab_size = len(loader.vocab_mapper.token_vocab)
    num_classes = len(loader.vocab_mapper.label_vocab)
    
    # Determine device
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize and train the model
    model = IncrementalLSTMClassifier(
        vocab=loader.vocab_mapper.token_vocab,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        padding_idx=loader.vocab_mapper.pad_idx
    ).to(device)
    
    model, training_stats = train_model(
        model=model,
        dataloader=train_dataloader,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
        device=device
    )

    # Extract features and raw inputs for training data
    train_data_by_label = compute_embeddings_with_raw(model, train_dataloader, device=device)
    
    # Update prototypes
    model.update_prototypes({k: [e for e, _, _ in v] for k, v in train_data_by_label.items()})
    
    # Initialize global prototypes from training data
    global_prototypes = init_global_prototypes(train_data_by_label, model, device)
    
    # Initialize replay buffer
    replay_buffer = {'X': [], 'y': [], 'samples_by_label': {}}
    buffer_size_per_class = args.buffer_size_per_class
    replay_buffer, global_prototypes = update_replay_buffer(
        replay_buffer, 
        train_data_by_label, 
        global_prototypes,
        buffer_size_per_class=buffer_size_per_class,
        device=device
    )

    # === PREDICTION ===
    test_batches = loader.create_batches(test_df)
    
    # Tracking variables
    batch_predictions = []
    batch_ground_truth = []
    batch_accuracies = []
    batch_timestamps = []
    batch_update_times = []
    total_update_time = 0
    
    # Process each batch
    for i, (batch_time, batch_df) in enumerate(test_batches.items()):
        print(f"\n=== Predicting batch {i+1}/{len(test_batches)} - {batch_time} ===")
        
        model = model.to(device)
        test_dataloader = loader.encode_and_prepare(batch_df, args.batch_size, shuffle=False)
        
        # Check for vocabulary expansion
        try:
            if len(loader.vocab_mapper.token_vocab) > len(model.embed.vocab):
                tokens_to_add = [t for t in loader.vocab_mapper.token_vocab.keys() 
                                if t not in model.embed.vocab]
                if tokens_to_add:
                    model.embed.expand_vocab(tokens_to_add)
                
            if len(loader.vocab_mapper.label_vocab) > model.classifier.out_features:
                model.incremental_learning([], len(loader.vocab_mapper.label_vocab))
                            
        except Exception as e:
            print(f"Error during vocabulary expansion: {e}")
            raise
            
        # Make predictions
        accuracy, predictions, ground_truth = predict_model(model, test_dataloader, device=device)
        curr_data_by_label = compute_embeddings_with_raw(model, test_dataloader, device=device)
        curr_embeddings = {k: [e for e, _, _ in v] for k, v in curr_data_by_label.items()}
        
        # Store results
        batch_predictions.append(predictions)
        batch_ground_truth.append(ground_truth)
        batch_accuracies.append(accuracy)
        batch_timestamps.append(batch_time)
        print(f"Batch accuracy: {accuracy*100:.2f}")
        
        # === DRIFT DETECTION ===
        update_start_time = time.time()
        print("-> Drift Detection <-")
    
        unseen_activities, drifting_activities, novel_activities, drift_scores = detect_drifts(
            curr_embeddings, 
            global_prototypes,
            replay_buffer,
            default_threshold=args.drift_threshold, 
            novelty_threshold_factor=args.novelty_threshold_factor, 
            use_cosine=args.use_cosine,
            min_samples=args.min_samples
        )
        
        need_update = bool(unseen_activities or drifting_activities or novel_activities)
        if need_update:
            if (unseen_activities or novel_activities) and len(test_dataloader.dataset) > 10:
                model = model.to(device)
                replay_loader = create_dataloader_from_buffer(
                    replay_buffer, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    max_case_length=loader.max_case_length
                )
                combined_loader = combine_dataloaders(test_dataloader, replay_loader, batch_size=args.batch_size)
                model = update_model(
                    model, 
                    dataloader=combined_loader,
                    lr=args.learning_rate/2,  # Half of base learning rate for updates
                    epochs=args.epochs,
                    patience=5,
                    device=device,
                )

                # Recompute embeddings on combined loader using updated model
                curr_embeddings = compute_embeddings(model, combined_loader, device=device)
                real_features = []
                real_labels = []
                for act, embeddings in curr_embeddings.items():
                    real_features.extend(embeddings)
                    real_labels.extend([act] * len(embeddings))
                
                # Convert real features/labels to tensors
                real_features = torch.stack(real_features)
                real_labels = torch.tensor(real_labels, device=device)
                
                # Data augmentation for all shifted activities
                # Within-activity augmentation for drifting activities
                aug_X1 = []
                aug_Y1 = []
                if drifting_activities:
                    aug_emb_dict = intra_class_mixup(
                        drifting_activities, 
                        curr_embeddings, 
                        replay_buffer,
                        alpha=args.alpha, 
                        mix_times=4
                    )
                    if aug_emb_dict:
                        for act, embeddings in aug_emb_dict.items():
                            for e in embeddings:
                                if e.dim() != 1:
                                    e = e.view(-1)
                                aug_X1.append(e)
                                aug_Y1.append(int(act))
                if aug_X1:
                    aug_X1 = torch.stack(aug_X1, dim=0)
                    aug_Y1 = torch.tensor(aug_Y1, dtype=torch.long, device=device)
                else:
                    aug_X1 = torch.tensor([])
                    aug_Y1 = torch.tensor([], dtype=torch.long, device=device)

                # Prototype jittering for novel/unseen activities
                aug_X2 = torch.tensor([])
                aug_Y2 = torch.tensor([], dtype=torch.long)
                if novel_activities or unseen_activities:
                    F_aug, Y_aug = distribution_aug(
                        novel_activities.union(unseen_activities), 
                        curr_embeddings, 
                        gamma=0.3, 
                        eps=1e-3
                    )
                    if F_aug is not None and Y_aug is not None:
                        aug_X2 = F_aug
                        aug_Y2 = Y_aug
                
                print(f"Number of augmented samples: {len(aug_X1) + len(aug_X2)}")
                
                # Combine augmented samples if we have any
                if aug_X1.numel() > 0 and aug_X2.numel() > 0:
                    aug_X = torch.cat([aug_X1, aug_X2])
                    aug_Y = torch.cat([aug_Y1, aug_Y2])
                elif aug_X1.numel() > 0:
                    aug_X = aug_X1
                    aug_Y = aug_Y1
                elif aug_X2.numel() > 0:
                    aug_X = aug_X2
                    aug_Y = aug_Y2
                else:
                    aug_X = torch.tensor([])
                    aug_Y = torch.tensor([], dtype=torch.long)
                
                # Finetune classifier on augmented samples and combined loader
                if aug_X.numel() > 0 and aug_Y.numel() > 0:
                    model = finetune_classifier(
                        model, 
                        real_features, 
                        real_labels, 
                        aug_X, 
                        aug_Y, 
                        lr=args.learning_rate/2,  # Half of base learning rate for finetuning
                        epochs=args.epochs, 
                        patience=5,
                        device=device
                    )

            elif drifting_activities:
                # Finetune classifier on drifting activities only
                aug_emb_dict = intra_class_mixup(
                    drifting_activities, 
                    curr_embeddings, 
                    replay_buffer,
                    alpha=args.alpha, 
                    mix_times=4
                )
                aug_X1 = []
                aug_Y1 = []
                if aug_emb_dict:
                    for act, embeddings in aug_emb_dict.items():
                        for e in embeddings:
                            if e.dim() != 1:
                                e = e.view(-1)
                            aug_X1.append(e)
                            aug_Y1.append(int(act))
                if aug_X1:
                    aug_X1 = torch.stack(aug_X1, dim=0)
                    aug_Y1 = torch.tensor(aug_Y1, dtype=torch.long, device=aug_X1.device)
                else:
                    aug_X1 = torch.tensor([])
                    aug_Y1 = torch.tensor([], dtype=torch.long, device=device)

                # Calculate real features and labels of combined loader
                replay_loader = create_dataloader_from_buffer(
                    replay_buffer, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    max_case_length=loader.max_case_length
                )
                combined_loader = combine_dataloaders(test_dataloader, replay_loader, batch_size=args.batch_size)
                curr_embeddings = compute_embeddings(model, combined_loader, device=device)
                real_features = []
                real_labels = []
                for act, embeddings in curr_embeddings.items():
                    real_features.extend(embeddings)
                    real_labels.extend([act] * len(embeddings))
                real_features = torch.stack(real_features)
                real_labels = torch.tensor(real_labels, device=device)
                
                # Finetune classifier
                model = finetune_classifier(
                    model, 
                    real_features, 
                    real_labels, 
                    aug_X1, 
                    aug_Y1, 
                    lr=args.learning_rate/2,  # Half of base learning rate for finetuning
                    epochs=args.epochs, 
                    patience=5,
                    device=device
                )

        # Update model prototypes
        model.update_prototypes(curr_embeddings)
        
        # Update replay buffer and global prototypes
        buffer_size = args.buffer_size_per_class
        replay_buffer, global_prototypes = update_replay_buffer(
            replay_buffer,
            curr_data_by_label,
            global_prototypes,
            buffer_size_per_class=buffer_size,
            drift_scores=drift_scores,
            device=device
        )

        update_time = time.time() - update_start_time
        batch_update_times.append(update_time)
        total_update_time += update_time
    
    # === PERFORMANCE EVALUATION ===
    print("\n=== Performance Evaluation ===")
    
    all_predictions = torch.cat([pred.cpu() for pred in batch_predictions])
    all_ground_truth = torch.cat([gt.cpu() for gt in batch_ground_truth])
    
    all_predictions = all_predictions.cpu().numpy()
    all_ground_truth = all_ground_truth.cpu().numpy()
    
    metrics = {
        "accuracy": f"{accuracy_score(all_ground_truth, all_predictions)*100:.2f}",
        "total_update_time": f"{total_update_time:.2f}"
    }
    
    print("\nPerformance Summary:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Total update time: {metrics['total_update_time']}")
    
    save_results(metrics, output_dir)
    
    batch_results = pd.DataFrame({
        "batch_time": batch_timestamps,
        "batch_accuracy": [f"{acc*100:.2f}" for acc in batch_accuracies]
    })
    batch_results.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
    
    plot_accuracy_over_time(batch_accuracies, batch_timestamps, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    
    return output_dir, metrics

if __name__ == "__main__":
    main() 