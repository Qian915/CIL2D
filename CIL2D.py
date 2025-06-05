import os
import sys
import time
import torch
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from scipy.spatial import distance
import torch.nn.functional as F
import json


# Import local modules
from lib.data.IncrementalDataLoader import LogsDataLoader
from lib.model.incremental_model import (DynamicEmbedding, IncrementalLSTMClassifier, train_model, evaluate_model, model_predict, update_model, compute_embeddings, finetune_classifier)

def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Class Incremental Learning with Drift Detection and Data Augmentation for Dynamic Processes")
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="Sepsis", 
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
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Learning rate for the optimizer")
    parser.add_argument("--lr_model_update", type=float, default=0.001,
                        help="Learning rate for the model update")  # training lr / 2 for full update
    parser.add_argument("--epochs_finetune", type=int, default=10,
                        help="Number of epochs for the model update")
    parser.add_argument("--lr_finetune", type=float, default=0.0002,
                        help="Learning rate for the finetune")  # training lr / 10 for finetuning classifier!

    
    # Drift detection parameters
    parser.add_argument("--drift_threshold", type=float, default=0.05,
                        help="Threshold for detecting concept drift")
    parser.add_argument("--novelty_threshold_factor", type=float, default=2.0,
                        help="Factor for novelty threshold")
    parser.add_argument("--use_cosine", action="store_true",
                        help="Use cosine distance instead of Euclidean")
    parser.add_argument("--min_samples", type=int, default=3,
                        help="Minimum number of samples required to compute reliable statistics")
    
    # Data Augmentation parameters
    parser.add_argument("--alpha", type=float, default=20.0,
                        help="Mixing parameter for feature augmentation")
    parser.add_argument("--max_samples_per_activity", type=int, default=500,
                        help="Maximum number of augmented samples per activity")
    parser.add_argument("--num_closest", type=int, default=5,
                        help="Number of closest activities to use for cross-activity augmentation")

    # Replay Buffer parameters
    parser.add_argument("--buffer_size_per_class", type=int, default=20,
                        help="Number of representative samples to keep per class in the replay buffer")
    
    # Misc parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training/evaluation")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    return parser.parse_args()


def create_output_dir(args):
    """Create output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Include window_type in the path if it's specified
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
    # Convert args to dictionary
    params_dict = vars(args)
    
    # Save as JSON for easy reading
    with open(os.path.join(output_dir, "parameters.json"), 'w') as f:
        json.dump(params_dict, f, indent=4)
    
    # Also save as CSV for potential analysis
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

def detect_drifts_with_global_prototypes(curr_embeddings, global_prototypes, replay_buffer,
                               default_threshold=0.2, novelty_threshold_factor=2.0, use_cosine=True, min_samples=3):
    """Detect drifts, novelty, and unseen activities using global prototypes.
    
    Args:
        curr_embeddings: Dictionary mapping activity indices to lists of embeddings (current batch)
        global_prototypes: Dictionary mapping activity indices to global prototypes
        replay_buffer: Dictionary containing replay buffer data for threshold computation
        default_threshold: Default threshold for drift detection
        novelty_threshold_factor: Factor for computing novelty threshold
        use_cosine: Whether to use cosine distance instead of Euclidean
        min_samples: Minimum number of samples required to compute reliable statistics
        
    Returns:
        Tuple of (unseen_activities, drifting_activities, novel_activities) sets
    """
    
    # Get device
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
    
    # Set default thresholds based on distance metric
    if use_cosine:
        base_drift_threshold = 0.1  # Cosine distance threshold
    else:
        base_drift_threshold = default_threshold  # Euclidean distance threshold
    base_novelty_threshold = base_drift_threshold * novelty_threshold_factor
    
    # Get all known activity indices 
    known_activity_indices = set(global_prototypes.keys())
    
    # Convert activity indices to integers to ensure consistent comparison
    known_activity_indices = {int(idx) for idx in known_activity_indices}
    curr_activity_indices = {int(idx) for idx in curr_embeddings.keys()}
    
    # Identify unseen activities (in current data but not in global prototypes)
    unseen_activities = curr_activity_indices - known_activity_indices
    
    # Get drift thresholds based on the replay buffer of past data
    try:
        drift_thresholds, novelty_thresholds = compute_drift_thresholds(
            replay_buffer=replay_buffer,
            threshold=default_threshold,
            novelty_threshold_factor=novelty_threshold_factor,
            use_cosine=use_cosine,
            min_samples=min_samples
        )
    except Exception as e:
        print(f"Warning: Error computing drift thresholds: {e}")
        drift_thresholds = {}
        novelty_thresholds = {}

    # Track activities with drift and novel patterns
    drifting_activities = set()
    novel_activities = set()
    
    # Process each activity in the current batch
    for activity_idx in curr_activity_indices:
        # Skip unseen activities (they will be handled separately)
        if activity_idx not in known_activity_indices:
            continue
            
        # Get embeddings for current activity
        activity_embeddings = curr_embeddings[activity_idx]
        if not activity_embeddings:
            continue
            
        try:
            # Get global prototype for this activity
            global_prototype = global_prototypes[activity_idx].to(device)
            
            # Calculate distances from each embedding to the global prototype
            distances = []
            for embed in activity_embeddings:
                embed = embed.to(device)
                if use_cosine:
                    # Cosine similarity
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        embed.unsqueeze(0), global_prototype.unsqueeze(0)).item()
                    dist = 1.0 - cosine_sim
                else:
                    # Euclidean distance
                    dist = torch.norm(embed - global_prototype).item()
                distances.append(dist)
            
            # Calculate mean distance to determine drift
            if distances:
                mean_distance = sum(distances) / len(distances)
                
                # Get thresholds for this activity (or use base thresholds as fallback)
                drift_threshold = drift_thresholds.get(activity_idx, base_drift_threshold)
                novelty_threshold = novelty_thresholds.get(activity_idx, base_novelty_threshold)
                
                # Check against thresholds
                if mean_distance > novelty_threshold:
                    # Novelty detected - a significant new pattern of existing activity
                    novel_activities.add(activity_idx)
                    #print(f"Novel pattern detected for activity {activity_idx}: distance = {mean_distance:.4f}, threshold = {novelty_threshold:.4f}")
                elif mean_distance > drift_threshold:
                    # Drift detected - existing activity pattern has changed
                    drifting_activities.add(activity_idx)
                    #print(f"Drift detected for activity {activity_idx}: distance = {mean_distance:.4f}, threshold = {drift_threshold:.4f}")
                else:
                    # No drift or novelty detected
                    #print(f"No drift for activity {activity_idx}: distance = {mean_distance:.4f}, threshold = {drift_threshold:.4f}")
                    pass
        
        except Exception as e:
            print(f"Warning: Error processing activity {activity_idx}: {e}")
            continue
    
    print(f"Found {len(unseen_activities)} unseen activities, {len(drifting_activities)} drifting activities, "
          f"and {len(novel_activities)} novel activities")
    
    return unseen_activities, drifting_activities, novel_activities

def compute_drift_thresholds(replay_buffer, threshold=0.2, novelty_threshold_factor=2.0, use_cosine=True, min_samples=3):
    """Compute thresholds for drift and novelty detection based on intra-cluster variance.
    
    Args:
        replay_buffer: Dictionary containing replay buffer data
        threshold: Default threshold to use when adaptive threshold can't be computed
        novelty_threshold_factor: Multiplier to get novelty threshold from drift threshold
        use_cosine: Whether to use cosine distance instead of Euclidean
        min_samples: Minimum samples needed to compute reliable statistics
        
    Returns:
        Tuple of (drift_thresholds, novelty_thresholds) dictionaries
    """
    # Get the global device if exists
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
    
    # Set default thresholds based on distance metric
    if use_cosine:
        default_drift_threshold = 0.1  # Cosine distances are in [0,2] range
        default_novelty_threshold = default_drift_threshold * novelty_threshold_factor
    else:
        default_drift_threshold = threshold  # User-specified or default Euclidean threshold
        default_novelty_threshold = default_drift_threshold * novelty_threshold_factor
    
    # Initialize threshold dictionaries with defaults (in case computation fails)
    drift_thresholds = {}
    novelty_thresholds = {}
    
    # Validate replay buffer structure
    if not replay_buffer or not isinstance(replay_buffer, dict) or 'samples_by_label' not in replay_buffer:
        print("Warning: Invalid replay buffer structure, using default thresholds")
        return drift_thresholds, novelty_thresholds
    
    try:
        # Get embeddings_by_label from replay buffer
        samples_by_label = replay_buffer['samples_by_label']
        
        # Check that samples_by_label is properly formatted
        if not isinstance(samples_by_label, dict):
            print(f"Warning: samples_by_label is not a dictionary, using default thresholds")
            return drift_thresholds, novelty_thresholds
            
        # Extract embeddings from samples, safely handling potential errors
        embeddings_by_label = {}
        for label, samples in samples_by_label.items():
            try:
                # Ensure label is properly formatted and samples is a non-empty list
                if not samples or not isinstance(samples, list):
                    continue
                    
                # Get embeddings (first element of each sample)
                embeddings = []
                for sample in samples:
                    if isinstance(sample, tuple) and len(sample) >= 1:
                        embedding = sample[0]
                        if isinstance(embedding, torch.Tensor):
                            embeddings.append(embedding)
                            
                if len(embeddings) >= min_samples:
                    embeddings_by_label[label] = embeddings
            except Exception as e:
                print(f"Warning: Error extracting embeddings for label {label}: {e}")
                continue
        
        # Calculate adaptive thresholds based on intra-cluster variance
        for activity_idx, embeddings in embeddings_by_label.items():
            # Skip if too few samples for reliable statistics
            if len(embeddings) < min_samples:
                # Set default thresholds for activities with insufficient samples
                drift_thresholds[activity_idx] = default_drift_threshold
                novelty_thresholds[activity_idx] = default_novelty_threshold
                continue
                
            try:
                # Calculate distances between each embedding and the mean of this window
                window_embeddings = torch.stack([emb.to(device) for emb in embeddings])
                window_mean = window_embeddings.mean(dim=0)
                
                # Calculate distances based on specified metric
                distances = []
                if use_cosine:
                    # Cosine distance: 1 - cos(θ) ranges from 0 (identical) to 2 (opposite)
                    for emb in embeddings:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            emb.to(device).unsqueeze(0), window_mean.unsqueeze(0)).item()
                        distances.append(1.0 - cos_sim)
                else:
                    # Euclidean distance
                    distances = [torch.norm(emb.to(device) - window_mean).item() for emb in embeddings]
                
                # Use robust statistics: median and median absolute deviation (MAD)
                # for better resilience against outliers
                median_distance = np.median(distances)
                mad = np.median([abs(d - median_distance) for d in distances])
                
                # Avoid division by zero or very small MAD
                if mad < 1e-6:
                    # Fall back to standard deviation if MAD is too small
                    std_distance = np.std(distances)
                    
                    # If both MAD and std are too small, use a fixed minimum value
                    if std_distance < 1e-6:
                        if use_cosine:
                            mad = 0.01  # Minimum value for cosine distance
                        else:
                            mad = 0.05  # Minimum value for Euclidean distance
                    else:
                        mad = std_distance / 1.4826  # Convert std to approximate MAD
                
                # Set adaptive thresholds using median + k*MAD formula
                # k=3 covers ~99% of normal distribution
                k = 3.0
                drift_thresholds[activity_idx] = median_distance + k * mad
                
                # Set novelty threshold higher than drift threshold
                novelty_thresholds[activity_idx] = drift_thresholds[activity_idx] * novelty_threshold_factor
                
                # Default minimum thresholds as a safety measure
                if use_cosine:
                    drift_thresholds[activity_idx] = max(drift_thresholds[activity_idx], 0.05)
                    novelty_thresholds[activity_idx] = max(novelty_thresholds[activity_idx], 0.1)
                else:
                    # For Euclidean, ensure thresholds aren't too small
                    drift_thresholds[activity_idx] = max(drift_thresholds[activity_idx], threshold * 0.5)
                    novelty_thresholds[activity_idx] = max(novelty_thresholds[activity_idx], threshold)
                
                # Log detailed statistics for debugging (optional)
                # print(f"Activity {activity_idx} (n={len(distances)}): median={median_distance:.4f}, "
                #       f"MAD={mad:.4f}, drift_threshold={drift_thresholds[activity_idx]:.4f}, "
                #       f"novelty_threshold={novelty_thresholds[activity_idx]:.4f}")
                      
            except Exception as e:
                print(f"Warning: Error computing thresholds for activity {activity_idx}: {e}")
                # Use default thresholds for failed activities
                drift_thresholds[activity_idx] = default_drift_threshold
                novelty_thresholds[activity_idx] = default_novelty_threshold
    
    except Exception as e:
        print(f"Warning: Error in threshold computation: {e}")
        # Return empty thresholds on error, which will fall back to defaults
    
    # For any activity not covered, ensure we have at least default thresholds
    global_prototypes = replay_buffer.get('global_prototypes', {})
    for activity_idx in global_prototypes.keys():
        if activity_idx not in drift_thresholds:
            drift_thresholds[activity_idx] = default_drift_threshold
            novelty_thresholds[activity_idx] = default_novelty_threshold
    
    return drift_thresholds, novelty_thresholds

def safe_one_hot(idx, num_classes=None, device=None):
    """Create a one-hot encoded tensor with bounds checking.
    
    Args:
        idx: The index to encode as one-hot
        num_classes: Number of classes for one-hot encoding
        device: Device to create the tensor on
        
    Returns:
        A one-hot encoded tensor with the specified index set to 1.0
    """
    # Use global one_hot_size if available and no specific num_classes provided
    if num_classes is None and 'one_hot_size' in globals():
        num_classes = one_hot_size
    
    # Default num_classes if still None
    if num_classes is None:
        num_classes = 1000  # Fallback, but should never hit this with global one_hot_size
        
    # If idx is too large, print warning and use the maximum valid index
    if idx >= num_classes:
        print(f"Warning: Activity index {idx} is out of bounds for one-hot encoding with size {num_classes}")
        # Use the maximum valid index instead
        idx = num_classes - 1
        
    # Ensure the tensor is created on the specified device
    idx_tensor = torch.tensor(idx, dtype=torch.long)
    if device is not None:
        idx_tensor = idx_tensor.to(device)
        
    # Create one-hot encoding and convert to float
    one_hot = F.one_hot(idx_tensor, num_classes=num_classes).float()
    
    # Ensure the tensor is on the specified device
    if device is not None:
        one_hot = one_hot.to(device)
        
    return one_hot

def aug_within_activity(activities, curr_embeddings, global_prototypes, alpha=20.0, max_samples_per_activity=100):
    """Enhanced feature augmentation within the same activity using global prototypes.
    
    Args:
        activities: List of activities to augment
        curr_embeddings: Dictionary of current embeddings
        global_prototypes: Dictionary of global prototypes by activity
        alpha: Beta distribution parameter for mixing
        max_samples_per_activity: Maximum number of augmented samples per activity
        
    Returns:
        Dictionary of augmented embeddings {act_idx: [emb1, emb2, ...]}
    """
    global device
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mixed_embeddings = {}
    
    for act in activities:
        # Skip if activity doesn't exist in both current embeddings and global prototypes
        if act not in curr_embeddings or act not in global_prototypes:
            continue
            
        if len(curr_embeddings[act]) == 0:
            continue
            
        mixed_embeddings[act] = []
        
        # Move all tensors to the same device
        curr_embs = [emb.to(device) for emb in curr_embeddings[act]]
        # Get the global prototype for this activity
        global_proto = global_prototypes[act].to(device)
        
        # For each current embedding, mix with the global prototype
        for curr_emb in curr_embs:
            # Generate a random mixing factor
            lam = np.random.beta(alpha, alpha)
            if lam < 0.4 or lam > 0.6:  # Ensure meaningful mixing
                lam = 0.5
                
            # Mix current embedding with global prototype
            mixed = lam * curr_emb + (1 - lam) * global_proto
            mixed_embeddings[act].append(mixed)
            
            # Stop if we've reached the maximum sample count
            if len(mixed_embeddings[act]) >= max_samples_per_activity:
                break
    
    return mixed_embeddings

def find_n_closest_activities_global(global_prototypes, act, n=1, use_cosine=True):
    """Find the N closest activities to the given activity using global prototypes.
    
    Args:
        global_prototypes: Dictionary mapping activity indices to global prototypes
        act: The activity to find closest matches for
        n: Number of closest activities to return
        use_cosine: Whether to use cosine distance
        
    Returns:
        List of N closest activities
    """
    if act not in global_prototypes:
        return []
        
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
    
    # Get the prototype for the target activity
    target_proto = global_prototypes[act].to(device)
    
    # Calculate distances to all other prototypes
    distances = {}
    for other_act, other_proto in global_prototypes.items():
        if other_act == act:
            continue
            
        other_proto = other_proto.to(device)
        
        if use_cosine:
            # Calculate cosine similarity: 1 - similarity gives distance
            cosine_sim = torch.nn.functional.cosine_similarity(
                target_proto.unsqueeze(0), other_proto.unsqueeze(0)).item()
            distances[other_act] = 1.0 - cosine_sim
        else:
            # Calculate Euclidean distance
            distances[other_act] = torch.norm(target_proto - other_proto).item()
    
    # Sort activities by distance
    sorted_acts = sorted(distances.keys(), key=lambda x: distances[x])
    
    # Return the N closest activities
    return sorted_acts[:min(n, len(sorted_acts))]

def aug_across_activity(novel_activities, curr_embeddings, global_prototypes, 
                        alpha=20.0, max_samples_per_activity=100, num_closest=1):
    """Enhanced feature augmentation between different activities using global prototypes.
    
    Args:
        novel_activities: List of activities to augment
        curr_embeddings: Dictionary of current embeddings
        global_prototypes: Dictionary of global prototypes by activity
        alpha: Beta distribution parameter
        max_samples_per_activity: Maximum augmented samples per activity pair
        num_closest: Number of closest activities to consider for each novel activity
        
    Returns:
        List of mixed embeddings, list of mixed labels
    """
    global one_hot_size, device
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mixed_embeddings_list = []
    mixed_labels_list = []
    
    # For each novel activity
    for act in novel_activities:
        if act not in curr_embeddings or not curr_embeddings[act]:
            continue
            
        if act not in global_prototypes:
            continue
            
        # Find N closest activities using global prototypes
        closest_acts = find_n_closest_activities_global(global_prototypes, act, num_closest)
        
        if not closest_acts:
            continue
            
        # Get embeddings for the current activity
        curr_embs = [emb.to(device) for emb in curr_embeddings[act]]
        # Create one-hot encoding for current activity
        one_hot_act = safe_one_hot(act, one_hot_size, device=device)
        
        # Calculate max samples per closest activity to stay within overall limit
        max_samples_per_closest = max_samples_per_activity // len(closest_acts)
        if max_samples_per_closest < 1:
            max_samples_per_closest = 1
        
        # Mix with each of the closest activities
        for closest_act in closest_acts:
            if closest_act not in global_prototypes:
                continue
                
            # Get global prototype for closest activity
            closest_proto = global_prototypes[closest_act].to(device)
            # Create one-hot encoding for closest activity
            one_hot_closest_act = safe_one_hot(closest_act, one_hot_size, device=device)
            
            # Mix each current embedding with the prototype of the closest activity
            samples_added = 0
            for curr_emb in curr_embs:
                lam = np.random.beta(alpha, alpha)
                
                # Mix embeddings
                mixed_emb = lam * curr_emb + (1 - lam) * closest_proto
                # Mix labels
                mixed_label = lam * one_hot_act + (1 - lam) * one_hot_closest_act
                
                mixed_embeddings_list.append(mixed_emb)
                mixed_labels_list.append(mixed_label)
                
                samples_added += 1
                if samples_added >= max_samples_per_closest:
                    break
    
    return mixed_embeddings_list, mixed_labels_list

def compute_embeddings_with_raw(model, dataloader, device=None):
    """
    Compute embeddings and keep raw inputs + labels.
    
    Args:
        model: The model to compute embeddings with
        dataloader: DataLoader with inputs, labels, and lengths
        device: Device to use for computation
        
    Returns:
        Dictionary mapping label indices to lists of (embedding, raw_input, label) tuples
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()

    data_by_label = {}  # {label: [(embedding, raw_input, label)]}

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model.extract_features(inputs)

            if labels.ndim == 2:  # One-hot
                labels = labels.argmax(dim=1)

            for i in range(len(inputs)):
                label = labels[i].item()
                embedding = embeddings[i].detach().cpu()
                raw_input = inputs[i].detach().cpu()

                if label not in data_by_label:
                    data_by_label[label] = []
                data_by_label[label].append((embedding, raw_input, label))

    return data_by_label  # Will be used for both prototypes and buffer

def init_global_prototypes(data_by_label, model, device=None):
    """
    Initialize global prototypes for each class.
    
    Args:
        data_by_label: Dictionary mapping label indices to lists of (embedding, raw_input, label) tuples
        model: The model containing prototypes
        device: Device for tensor operations
        
    Returns:
        Dictionary mapping label indices to global prototypes
    """
    if device is None:
        device = next(model.parameters()).device
    
    global_prototypes = {}
    
    for label, samples in data_by_label.items():
        if not samples:
            continue
        
        embeddings = torch.stack([e for e, _, _ in samples])
        # Initialize global prototype as mean of all embeddings for this class
        global_prototypes[label] = embeddings.mean(dim=0).to(device)
    
    return global_prototypes

def update_global_prototypes(global_prototypes, curr_data_by_label, beta=0.9, device=None):
    """
    Update global prototypes using exponential moving average.
    
    Args:
        global_prototypes: Dictionary mapping label indices to global prototypes
        curr_data_by_label: Dictionary mapping label indices to current batch embeddings
        beta: Momentum parameter (higher = more weight to old prototype)
        device: Device for tensor operations
        
    Returns:
        Updated global prototypes dictionary
    """
    if device is None:
        if global_prototypes and next(iter(global_prototypes.values())).device:
            device = next(iter(global_prototypes.values())).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for label, samples in curr_data_by_label.items():
        if not samples:
            continue
        
        # Compute current prototype for this class
        embeddings = torch.stack([e for e, _, _ in samples]).to(device)
        current_prototype = embeddings.mean(dim=0)
        
        if label in global_prototypes:
            # Update existing global prototype using EMA
            global_prototypes[label] = beta * global_prototypes[label] + (1 - beta) * current_prototype
        else:
            # Initialize new global prototype
            global_prototypes[label] = current_prototype
    
    return global_prototypes

def compute_blended_prototypes(global_prototypes, curr_data_by_label, alpha=0.7, device=None):
    """
    Compute blended prototypes combining global and current prototypes.
    
    Args:
        global_prototypes: Dictionary mapping label indices to global prototypes
        curr_data_by_label: Dictionary mapping label indices to current batch embeddings
        alpha: Weight for global prototype (higher = more weight to global)
        device: Device for tensor operations
        
    Returns:
        Dictionary mapping label indices to blended prototypes
    """
    if device is None:
        if global_prototypes and next(iter(global_prototypes.values())).device:
            device = next(iter(global_prototypes.values())).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    blended_prototypes = {}
    
    for label, samples in curr_data_by_label.items():
        if not samples:
            continue
        
        # Skip if we don't have a global prototype yet
        if label not in global_prototypes:
            continue
        
        # Compute current prototype for this class
        embeddings = torch.stack([e for e, _, _ in samples]).to(device)
        current_prototype = embeddings.mean(dim=0)
        
        # Blend global and current prototypes
        blended_prototypes[label] = alpha * global_prototypes[label] + (1 - alpha) * current_prototype
    
    return blended_prototypes

def select_representative_samples(data_by_label, prototypes, buffer_size_per_class=10, device=None):
    """
    Select the most representative samples based on distance to prototypes.
    
    Args:
        data_by_label: Dictionary mapping label indices to lists of (embedding, raw_input, label) tuples
        prototypes: Dictionary mapping label indices to prototypes (can be global, current or blended)
        buffer_size_per_class: Maximum number of samples to select per class
        device: Device for tensor operations
        
    Returns:
        Dictionary mapping label indices to lists of (embedding, raw_input, label) tuples
    """
    if device is None:
        if prototypes and next(iter(prototypes.values())).device:
            device = next(iter(prototypes.values())).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    representative_samples = {}
    
    for label, samples in data_by_label.items():
        if not samples or label not in prototypes:
            continue
        
        # Get prototype for this class
        prototype = prototypes[label].to(device)
        
        # Compute distances from each embedding to the prototype
        embeddings = torch.stack([e.to(device) for e, _, _ in samples])
        distances = torch.norm(embeddings - prototype.unsqueeze(0), dim=1)
        
        # Get indices of closest samples
        k = min(buffer_size_per_class, len(distances))
        closest_indices = torch.topk(distances, k=k, largest=False).indices.cpu().numpy()
        
        # Select the representative samples
        representative_samples[label] = [samples[idx] for idx in closest_indices]
    
    return representative_samples

def update_replay_buffer_with_blending(replay_buffer, curr_data_by_label, global_prototypes, 
                                      buffer_size_per_class=10, alpha=0.7, beta=0.9, device=None):
    """
    Update replay buffer using blended prototypes for sample selection.
    
    Args:
        replay_buffer: Dictionary with 'X', 'y', and 'samples_by_label' data
        curr_data_by_label: Dictionary mapping label indices to current batch embeddings
        global_prototypes: Dictionary mapping label indices to global prototypes
        buffer_size_per_class: Maximum number of samples to keep per class
        alpha: Weight for global prototype in blending
        beta: Momentum for global prototype update
        device: Device for tensor operations
        
    Returns:
        Updated replay buffer and global prototypes
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize global prototypes if not provided -> skip it as we init it after model training!
    #if not global_prototypes:
    #    global_prototypes = init_global_prototypes(curr_data_by_label, None, device)
    
    # Update global prototypes with current data
    global_prototypes = update_global_prototypes(global_prototypes, curr_data_by_label, beta, device)
    
    # Compute blended prototypes
    blended_prototypes = compute_blended_prototypes(global_prototypes, curr_data_by_label, alpha, device)
    
    # Initialize or get existing samples by label from replay buffer
    if 'samples_by_label' not in replay_buffer:
        replay_buffer['samples_by_label'] = {}
    
    samples_by_label = replay_buffer['samples_by_label']
    
    # For each class in current batch, select representative samples
    curr_representatives = select_representative_samples(curr_data_by_label, blended_prototypes, 
                                                       buffer_size_per_class, device)
    
    # Merge with existing samples in replay buffer
    for label, new_samples in curr_representatives.items():
        if label in samples_by_label:
            # Combine existing and new samples
            combined_samples = samples_by_label[label] + new_samples
            
            # Re-select most representative samples using global prototype
            if label in global_prototypes:
                proto_dict = {label: global_prototypes[label]}
                data_dict = {label: combined_samples}
                selected = select_representative_samples(data_dict, proto_dict, 
                                                       buffer_size_per_class, device)
                if label in selected:
                    samples_by_label[label] = selected[label]
        else:
            # Add new class to replay buffer
            samples_by_label[label] = new_samples
    
    # Reconstruct X and y from samples_by_label
    X, y = [], []
    for label, samples in samples_by_label.items():
        for _, raw_input, label_val in samples:
            X.append(raw_input)
            y.append(label_val)
    
    # Update replay buffer
    if X:
        replay_buffer['X'] = torch.stack(X)
        replay_buffer['y'] = torch.tensor(y)
        print(f"Updated replay buffer with {len(X)} samples across {len(samples_by_label)} classes")
    else:
        replay_buffer['X'] = torch.tensor([])
        replay_buffer['y'] = torch.tensor([])
        print("Warning: Empty replay buffer created")
    
    return replay_buffer, global_prototypes

def create_dataloader_from_buffer(replay_buffer, batch_size=32, shuffle=True, max_case_length=None):
    """
    Create a DataLoader from a replay buffer.
    
    Args:
        replay_buffer: Dictionary with 'X' and 'y' tensors
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        max_case_length: Maximum sequence length (if None, uses X's second dimension)
        
    Returns:
        DataLoader with replay buffer data
    """
    if not replay_buffer['X'].size(0):
        return None
    
    # Create sequence lengths (using max_case_length to match loader settings)
    if max_case_length is None:
        max_case_length = replay_buffer['X'].size(1)
    
    # The sequences are already padded to max_case_length in replay buffer
    # So we set all lengths to max_case_length to be consistent
    seq_lengths = torch.ones(replay_buffer['X'].size(0), dtype=torch.long) * max_case_length
    
    # Convert labels to one-hot if needed (consistent with original dataloader)
    if replay_buffer['y'].ndim == 1:
        num_classes = replay_buffer['y'].max().item() + 1
        y_onehot = torch.zeros(replay_buffer['y'].size(0), num_classes)
        y_onehot.scatter_(1, replay_buffer['y'].unsqueeze(1), 1)
        labels = y_onehot
    else:
        labels = replay_buffer['y']
        
    # Create TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(replay_buffer['X'], labels, seq_lengths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def combine_dataloaders(current_loader, replay_loader, batch_size=32, shuffle=True):
    """
    Combine current batch DataLoader with replay buffer DataLoader.
    
    Args:
        current_loader: DataLoader for current batch
        replay_loader: DataLoader from replay buffer
        batch_size: Batch size for combined DataLoader
        shuffle: Whether to shuffle the combined data
        
    Returns:
        Combined DataLoader or original current_loader if replay_loader is None
    """
    if replay_loader is None:
        return current_loader
        
    # Extract data from both loaders
    X_current, y_current, len_current = [], [], []
    X_replay, y_replay, len_replay = [], [], []
    
    # Extract data from current loader
    for inputs, labels, lengths in current_loader:
        X_current.append(inputs)
        y_current.append(labels)
        len_current.append(lengths)
    
    # Extract data from replay loader
    for inputs, labels, lengths in replay_loader:
        X_replay.append(inputs)
        y_replay.append(labels)
        len_replay.append(lengths)
    
    # Get dimensions
    current_classes = y_current[0].shape[1]
    replay_classes = y_replay[0].shape[1]
    
    # Handle different class dimensions (happens during incremental learning)
    if current_classes != replay_classes:
        print(f"Handling class dimension mismatch: current={current_classes}, replay={replay_classes}")
        # Determine the larger dimensionality
        max_classes = max(current_classes, replay_classes)
        
        # Resize current data labels if needed
        if current_classes < max_classes:
            resized_y_current = []
            for y in y_current:
                # Create new tensor with larger dimension
                new_y = torch.zeros(y.shape[0], max_classes, device=y.device)
                # Copy existing data
                new_y[:, :current_classes] = y
                resized_y_current.append(new_y)
            y_current = resized_y_current
        
        # Resize replay data labels if needed
        if replay_classes < max_classes:
            resized_y_replay = []
            for y in y_replay:
                # Create new tensor with larger dimension
                new_y = torch.zeros(y.shape[0], max_classes, device=y.device)
                # Copy existing data
                new_y[:, :replay_classes] = y
                resized_y_replay.append(new_y)
            y_replay = resized_y_replay
    
    # Combine data
    X_combined = torch.cat(X_current + X_replay, dim=0)
    y_combined = torch.cat(y_current + y_replay, dim=0)
    len_combined = torch.cat(len_current + len_replay, dim=0)
    
    # Create combined dataset and loader
    combined_dataset = torch.utils.data.TensorDataset(X_combined, y_combined, len_combined)
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return combined_loader

def main(args=None):
    # Create output directory
    output_dir = create_output_dir(args)
    
    # Save all parameter settings
    save_parameters(args, output_dir)
    
    print(f"=== CIL2D for {args.dataset} ===")
    #print(f"Results will be saved to: {output_dir}")
    
    # === LOAD DATA ===
    print("\n=== Loading Data ===")
    loader = LogsDataLoader(
        dataset_name=args.dataset,
        dir_path=args.data_dir,
        window_type=args.window_type
    )
    loader.load_data()
    # Split into training and test sets
    train_df, test_df = loader.split_train_test(args.train_test_ratio)
    
    # === MODEL TRAINING ===
    print("\n=== Training Model ===")
    
    # Encode training data
    train_dataloader = loader.encode_and_prepare(train_df, args.batch_size, shuffle=True)
    # Get vocabulary size and number of classes
    vocab_size = len(loader.vocab_mapper.token_vocab)
    num_classes = len(loader.vocab_mapper.label_vocab)
    
    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize and train the model
    train_start_time = time.time()
    model = IncrementalLSTMClassifier(
        vocab=loader.vocab_mapper.token_vocab,
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        padding_idx=loader.vocab_mapper.pad_idx
    ).to(device)  
    # Set the global one_hot_size after initializing model
    global one_hot_size
    one_hot_size = max(vocab_size, num_classes)
    
    model, training_stats = train_model(
        model=model,
        dataloader=train_dataloader,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.learning_rate,
        device=device
    )
    train_end_time = time.time()

    # Extract features and raw inputs from training data
    train_data_by_label = compute_embeddings_with_raw(model, train_dataloader, device=device)
    
    # Update prototypes
    model.update_prototypes({k: [e for e, _, _ in v] for k, v in train_data_by_label.items()})
    
    # Initialize global prototypes from training data
    global_prototypes = init_global_prototypes(train_data_by_label, model, device)
    
    # Initialize replay buffer with representative samples from training data
    replay_buffer = {'X': [], 'y': [], 'samples_by_label': {}}
    buffer_size_per_class = args.buffer_size_per_class
    #print(f"Using replay buffer size of {buffer_size_per_class} samples per class")
    replay_buffer, global_prototypes = update_replay_buffer_with_blending(
        replay_buffer, 
        train_data_by_label, 
        global_prototypes,
        buffer_size_per_class=buffer_size_per_class,
        alpha=0.7,  # weight for global vs. current prototype for sample selection
        beta=0.9,   # more weight to past prototypes
        device=device
    )

    # Save the trained model
    model_path = os.path.join(output_dir, "model.pt")
    model.save_model(model_path)
    #print(f"Model saved to {model_path}")
    
    # === PREDICTION ===

    # Create test batches by time window
    test_batches = loader.create_batches(test_df)
    
    # Tracking variables
    batch_predictions = []
    batch_ground_truth = []
    batch_accuracies = []
    batch_timestamps = []
    batch_update_times = []
    batch_drift_scores = []
    num_updates = 0
    num_model_updates = 0
    total_update_time = 0
    
    # Process each batch
    for i, (batch_time, batch_df) in enumerate(test_batches.items()):
        print(f"\n=== Predicting batch {i+1}/{len(test_batches)} - {batch_time} ===")
        
        model = model.to(device)
        
        # Encode test data
        test_dataloader = loader.encode_and_prepare(batch_df, args.batch_size, shuffle=False)
        
        # Check for vocabulary and expand model architecture if needed
        try:
            if len(loader.vocab_mapper.token_vocab) > len(model.embed.vocab):
                # Get only new tokens that are not in the current vocabulary
                tokens_to_add = [t for t in loader.vocab_mapper.token_vocab.keys() 
                                if t not in model.embed.vocab]
                if tokens_to_add:
                    model.embed.expand_vocab(tokens_to_add)
                
            if len(loader.vocab_mapper.label_vocab) > model.classifier.out_features:
                model.incremental_learning([], len(loader.vocab_mapper.label_vocab))
                
            # Update one_hot_size whenever the vocabulary expands
            one_hot_size = max(len(model.embed.vocab), len(loader.vocab_mapper.label_vocab))
            
        except Exception as e:
            print(f"Error during vocabulary expansion: {e}")
            print(f"Current token vocab size: {len(loader.vocab_mapper.token_vocab)}")
            print(f"Current model vocab size: {len(model.embed.vocab)}")
            print(f"Current label vocab size: {len(loader.vocab_mapper.label_vocab)}")
            print(f"Current output features: {model.classifier.out_features}")
            raise
            
        # Make predictions
        accuracy, predictions, ground_truth = model_predict(model, test_dataloader, device=device)
        # Extract features and raw inputs for test data
        curr_data_by_label = compute_embeddings_with_raw(model, test_dataloader, device=device)
        
        # Get current embeddings
        curr_embeddings = {k: [e for e, _, _ in v] for k, v in curr_data_by_label.items()}
        
        # Store and calculate batch accuracy
        batch_predictions.append(predictions)
        batch_ground_truth.append(ground_truth)
        batch_accuracies.append(accuracy)
        batch_timestamps.append(batch_time)
        print(f"Batch accuracy: {accuracy*100:.2f}")
        
        # === DRIFT DETECTION ===
        update_start_time = time.time()
        print("-> Drift Detection <-")
    
        # Use global prototypes for drift detection
        unseen_activities, drifting_activities, novel_activities = detect_drifts_with_global_prototypes(
            curr_embeddings, 
            global_prototypes,
            replay_buffer,
            default_threshold=args.drift_threshold, 
            novelty_threshold_factor=args.novelty_threshold_factor, 
            use_cosine=args.use_cosine,
            min_samples=args.min_samples
        )
        
        # === DATA AUGMENTATION ===
        need_update = bool(unseen_activities or drifting_activities or novel_activities)
        if need_update:
            print("-> Data Augmentation <-")
            ### feature augmentation
            aug_features = []
            aug_labels = []     # one-hot encoded labels!
            
            if drifting_activities:
                aug_emb_dict = aug_within_activity(drifting_activities, curr_embeddings, global_prototypes, alpha=args.alpha, max_samples_per_activity=args.max_samples_per_activity)
                for act, embeddings in aug_emb_dict.items():
                    aug_features.extend(embeddings)
                    aug_labels.extend([safe_one_hot(act, one_hot_size, device=device) for _ in embeddings])
                    
            if novel_activities:
                aug_emb_dict = aug_within_activity(novel_activities, curr_embeddings, global_prototypes, alpha=args.alpha, max_samples_per_activity=args.max_samples_per_activity)                    
                for act, embeddings in aug_emb_dict.items():
                    aug_features.extend(embeddings)
                    aug_labels.extend([safe_one_hot(act, one_hot_size, device=device) for _ in embeddings])
                
                aug_emb_list, aug_lab_list = aug_across_activity(novel_activities, curr_embeddings, global_prototypes, alpha=args.alpha, max_samples_per_activity=args.max_samples_per_activity, num_closest=args.num_closest)                    
                aug_features.extend(aug_emb_list)
                aug_labels.extend(aug_lab_list)
                
            if unseen_activities:                
                aug_emb_dict = aug_within_activity(unseen_activities, curr_embeddings, global_prototypes, alpha=args.alpha, max_samples_per_activity=args.max_samples_per_activity)
                for act, embeddings in aug_emb_dict.items():
                    aug_features.extend(embeddings)
                    aug_labels.extend([safe_one_hot(act, one_hot_size, device=device) for _ in embeddings])
                
                aug_emb_list, aug_lab_list = aug_across_activity(unseen_activities, curr_embeddings, global_prototypes, alpha=args.alpha, max_samples_per_activity=args.max_samples_per_activity, num_closest=args.num_closest)                    
                aug_features.extend(aug_emb_list)
                aug_labels.extend(aug_lab_list)
                
            #print(f"Number of augmented samples: {len(aug_features)}")
                
            # === UPDATE MODEL ===
            model = model.to(device)
            # fine-tune classifier first on augmented samples
            if len(aug_features) > 10:
                print("-> Finetuning Classifier <-")
                model = finetune_classifier(model, aug_features, aug_labels, 
                                lr=args.lr_finetune, 
                                epochs=args.epochs_finetune, 
                                device=device)
                
            # incremental learning if unseen or novel activities: update on current data and a buffer of past samples!
            if (unseen_activities or novel_activities) and len(test_dataloader.dataset) > 10:
                model = model.to(device)
                
                # Create DataLoader from replay buffer
                replay_loader = create_dataloader_from_buffer(
                    replay_buffer, 
                    batch_size=args.batch_size, 
                    shuffle=True,
                    max_case_length=loader.max_case_length
                )
                
                # Use combined data (current + replay) for model update
                if replay_loader:
                    combined_loader = combine_dataloaders(test_dataloader, replay_loader, batch_size=args.batch_size)
                    print("-> Updating Full Model <-")
                    model = update_model(
                        model, 
                        dataloader=combined_loader,
                        lr=args.lr_model_update,
                        epochs=args.epochs,
                        patience=args.patience,
                        device=device
                    )
                else:
                    print("No replay buffer available, using only current batch for update")
                    model = update_model(
                        model, 
                        dataloader=test_dataloader,
                            lr=args.lr_model_update,
                        epochs=args.epochs,
                        patience=args.patience,
                        device=device
                    )
                num_model_updates += 1

        # update model prototypes after drift detection
        model.update_prototypes(curr_embeddings)
        
        # === Post-prediction Update ===
        print("-> Updating Replay Buffer and Global Prototypes <-")
        # Update the replay buffer and global prototypes            
        replay_buffer, global_prototypes = update_replay_buffer_with_blending(
            replay_buffer,
            curr_data_by_label,
            global_prototypes,
            buffer_size_per_class=args.buffer_size_per_class,
            alpha=0.7,  # Balance between stability and adaptivity
            beta=0.9,   # Momentum for global prototype update
            device=device
        )

        update_time = time.time() - update_start_time
        batch_update_times.append(update_time)
        total_update_time += update_time
        num_updates += 1
    
    # === PERFORMANCE EVALUATION ===
    print("\n=== Performance Evaluation ===")
    
    # Combine results from all batches
    all_predictions = torch.cat([pred.cpu() for pred in batch_predictions])
    all_ground_truth = torch.cat([gt.cpu() for gt in batch_ground_truth])
    
    all_predictions = all_predictions.cpu().numpy()
    all_ground_truth = all_ground_truth.cpu().numpy()
    
    # Calculate overall metrics
    metrics = {
        "accuracy": accuracy_score(all_ground_truth, all_predictions),
        "f1_weighted": f1_score(all_ground_truth, all_predictions, average="weighted"),
        "precision_weighted": precision_score(all_ground_truth, all_predictions, average="weighted"),
        "recall_weighted": recall_score(all_ground_truth, all_predictions, average="weighted"),
        "training_time": train_end_time - train_start_time,
        "num_updates": num_updates,
        "num_model_updates": num_model_updates,
        "total_update_time": total_update_time,
        "avg_update_time": total_update_time / max(1, num_updates)
    }
    
    # Print summary of key metrics
    print("\nPerformance Summary:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score (weighted): {metrics['f1_weighted']}")
    print(f"Training time: {metrics['training_time']}")
    print(f"Total update time: {metrics['total_update_time']}")
    
    # Save results
    save_results(metrics, output_dir)
    
    # Save batch results with formatted accuracies
    batch_results = pd.DataFrame({
        "batch_time": batch_timestamps,
        "batch_accuracy": [f"{acc*100:.2f}" for acc in batch_accuracies],
        "batch_update_time": [t for t in batch_update_times]
    })
    batch_results.to_csv(os.path.join(output_dir, "batch_results.csv"), index=False)
    
    # Plot accuracy over time
    plot_accuracy_over_time(batch_accuracies, batch_timestamps, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    
    # Return metrics for potential aggregation
    return output_dir, metrics

if __name__ == "__main__":
    args = parse_arguments()
    output_dir, _ = main(args) 