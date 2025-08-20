import torch
import torch.nn as nn
from torch.utils.data import random_split
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F


def soft_cross_entropy(preds, targets):
    """
    Compute cross entropy loss with soft labels.
    Works with both one-hot encoded and soft (mixup) labels if data augmentation is used.
    
    Args:
        preds: Raw logits from the model
        targets: Target probabilities (one-hot or soft labels)
    
    Returns:
        loss: Mean cross entropy loss
    """
    log_probs = F.log_softmax(preds, dim=1)
    loss = -torch.sum(targets * log_probs, dim=1).mean()
    return loss


class DynamicEmbedding(nn.Module):
    def __init__(self, vocab, embed_dim, padding_idx=0):
        """
        Initialize the DynamicEmbedding module.

        Args:
            vocab: Vocabulary dictionary mapping tokens to their indices.
            embed_dim: Dimension of the embedding vectors.
            padding_idx: Index of the padding token.
        """
        super().__init__()
        self.vocab = vocab.copy()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(len(self.vocab), embed_dim, padding_idx=padding_idx)
    
    def expand_vocab(self, new_tokens):
        """
        Expands the embedding matrix to accommodate new tokens.

        Args:
            new_tokens (list): List of new tokens to add to the vocabulary
        """
        old_vocab_size = len(self.vocab)
        tokens_to_add = [t for t in new_tokens if t not in self.vocab]
        
        if not tokens_to_add:
            return  # No new tokens to add
        
        # Create a new embedding matrix and copy the old weights
        new_vocab_size = old_vocab_size + len(tokens_to_add)
        new_embedding = nn.Embedding(new_vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        with torch.no_grad():
            new_embedding.weight[:old_vocab_size] = self.embedding.weight
            # Initialize new embeddings with small random values
            new_embedding.weight[old_vocab_size:] = torch.randn(
                len(tokens_to_add), self.embed_dim) * 0.02
        self.embedding = new_embedding

        # Update vocabulary
        for token in tokens_to_add:
            self.vocab[token] = len(self.vocab)
    
    def forward(self, token_idxs):
        return self.embedding(token_idxs)


class IncrementalLSTMClassifier(nn.Module):
    def __init__(self, vocab, embed_dim=64, hidden_dim=128, num_classes=10, padding_idx=0):
        super().__init__()
        self.embed = DynamicEmbedding(vocab, embed_dim, padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.prototypes = {}  # Dict[int, List[torch.Tensor]]

    def forward(self, input_tensor):
        """
        Forward pass using pre-encoded inputs (already encoded by DataLoader)
        Args:
            input_tensor: Tensor of token indices, shape (B, T)
        Returns:
            logits: Tensor of logits, shape (B, num_classes)
        """
        embeds = self.embed(input_tensor)  # (B, T, E)
        lstm_out, _ = self.lstm(embeds)  # (B, T, H)
        features = lstm_out[:, -1, :]    # use final hidden state as features
        logits = self.classifier(features)
        return logits

    def extract_features(self, input_tensor):
        embeds = self.embed(input_tensor)  # (B, T, E)
        lstm_out, _ = self.lstm(embeds)  # (B, T, H)
        return lstm_out[:, -1, :]  # Features from the last hidden state

    def incremental_learning(self, new_tokens, new_classes):
        """
        Incremental learning to add new tokens and classes.
        """
        # Update embeddings to accommodate new tokens
        if new_tokens:
            self.embed.expand_vocab(new_tokens)
        
        # Update the classification head to accommodate new classes
        if new_classes > self.classifier.out_features:
            old_weight = self.classifier.weight.data
            old_bias = self.classifier.bias.data
            device = self.classifier.weight.device
            
            new_classifier = nn.Linear(self.classifier.in_features, new_classes).to(device)
            with torch.no_grad():
                new_classifier.weight[:self.classifier.out_features] = old_weight
                new_classifier.bias[:self.classifier.out_features] = old_bias
                
                # Initialize new weights with small random values
                new_classifier.weight[self.classifier.out_features:] = torch.randn(
                    new_classes - self.classifier.out_features, 
                    self.classifier.in_features, device=device) * 0.02
                new_classifier.bias[self.classifier.out_features:] = 0
                
            self.classifier = new_classifier


    def update_prototypes(self, embeddings_by_label):
        """
        Update the prototypes for the new classes.
        
        Args:
            embeddings_by_label: Dictionary mapping activity indices to lists of embeddings
        """
        for label, embeddings in embeddings_by_label.items():
            # Convert label to int for consistent indexing
            label = int(label)
            
            if label not in self.prototypes:
                self.prototypes[label] = []
            
            # Ensure embeddings is not empty
            if not embeddings:
                continue
            
            new_prototype = torch.stack(embeddings).mean(dim=0) if len(embeddings) > 1 else embeddings[0]
            self.prototypes[label].append(new_prototype)

    def save_model(self, path):
        """Save the model.
        
        Args:
            path: Path to save the model
        """ 
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': len(self.embed.vocab),
            'num_classes': self.classifier.out_features,
            'embedding_dim': self.embed.embed_dim,
            'hidden_dim': self.lstm.hidden_size
        }, path)


def train_model(model, dataloader, lr=0.002, epochs=100, patience=10, device=None):
    """
    Train the model on the given dataset.

    Args:
        dataloader: DataLoader with training and validation batches.
        lr: Learning rate for the optimizer.
        epochs: Number of training epochs.
        device: 'cpu' or 'cuda' if GPU available.
    """

    # Split dataset into training and validation sets
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create training and validation data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    )
    
    print(f"Split dataset: {train_size} training samples, {val_size} validation samples")

    # Move model to device
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.NAdam(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,       
    momentum_decay=0.004
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle one-hot encoded labels
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_stats['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn)
        training_stats['val_loss'].append(val_loss)
        training_stats['val_acc'].append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
                
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, training_stats

def evaluate_model(model, val_loader, loss_fn=None):
    """Evaluate the model on a dataset.
    
    Args:
        model: The model to be evaluated.
        dataloader: DataLoader with validation batches.
        criterion: Loss function.
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)
            loss = loss_fn(outputs, labels) if loss_fn else 0
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def model_predict(model, test_dataloader, device=None):
    """Predict the model on the given dataset.
    
    Args:
        model: The model to be predicted.
        test_dataloader: DataLoader with test batches.
        device: Device to use for computations.
    
    Returns:
        Tuple of (accuracy, predictions, ground_truth)
    """ 
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    predictions = []
    ground_truth = []
    
    try:
        with torch.no_grad():
            for inputs, labels, _ in test_dataloader:
                if labels.ndim == 2:
                    labels = labels.argmax(dim=1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
                ground_truth.append(labels)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Model vocabulary size: {len(model.embed.vocab)}")
        print(f"Model output classes: {model.classifier.out_features}")
        raise
    
    # Compute accuracy
    if not predictions or not ground_truth:
        return 0.0, torch.tensor([]), torch.tensor([])
        
    try:
        all_predictions = torch.cat(predictions)
        all_ground_truth = torch.cat(ground_truth)
        correct = (all_predictions == all_ground_truth).sum().item()
        total = all_ground_truth.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        return accuracy, all_predictions, all_ground_truth
    except Exception as e:
        print(f"Error computing accuracy: {e}")
        return 0.0, torch.tensor([]), torch.tensor([])

def update_model(model, dataloader, lr=0.001, epochs=100, patience=10, device=None):
    """Update the model on the given dataset.
    
    Args:
        model: The model to be updated.
        dataloader: DataLoader with training and validation batches.
    """
    # Split dataset into training and validation sets
    dataset = dataloader.dataset
    dataset_size = len(dataset)
    if dataset_size > 10:
        val_size = int(dataset_size * 0.2)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None  # No validation set
    
    # Create training and validation data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True,           
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
    )
    # Create validation data loader only if val_dataset is not None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0
        )
    else:
        val_loader = None
    # Move model to device
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.NAdam(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,       
    momentum_decay=0.004
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_stats = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle one-hot encoded labels
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        training_stats['train_loss'].append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, loss_fn)
            training_stats['val_loss'].append(val_loss)
            training_stats['val_acc'].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, Val Acc: {val_acc:.4f}")
                break

            #print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
            #      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

def finetune_classifier(model, real_features, real_labels, aug_features, aug_labels, lr=0.0002, epochs=100, patience=10, device=None):
    """Finetune the classifier with train/val split and early stopping.
    
    Args:
        model: The model to finetune
        real_features: Real feature tensors [N, hidden_dim]
        real_labels: Real labels [N]
        aug_features: Augmented feature tensors [M, hidden_dim]
        aug_labels: Augmented labels [M]
        lr: Learning rate
        epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to use
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check inputs
    if aug_features.numel() == 0 or aug_labels.numel() == 0:
        print("No augmented samples provided for fine-tuning")
        return
        
    # Ensure all tensors are on the correct device
    real_features = real_features.to(device)
    real_labels = real_labels.to(device)
    aug_features = aug_features.to(device)
    aug_labels = aug_labels.to(device)
    
    # Combine real and augmented samples
    all_features = torch.cat([real_features, aug_features])
    all_labels = torch.cat([real_labels, aug_labels])
    
    # Create dataset and split into train/val
    combined_features = torch.cat([aug_features, real_features], dim=0)
    combined_labels = torch.cat([aug_labels, real_labels], dim=0)
    dataset = torch.utils.data.TensorDataset(combined_features, combined_labels)
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    
    # Adjust batch size based on dataset size
    batch_size = 16  # Fixed batch size
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
    
    # Choose loss function based on label format
    if combined_labels.ndim == 2:  # One-hot encoded soft labels
        loss_fn = soft_cross_entropy
    else:  # Standard cross entropy for class indices
        loss_fn = nn.CrossEntropyLoss()
    
    # Only optimize classifier weights
    model.classifier.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=0.0001)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    
    for epoch in range(epochs):
        # Training phase
        model.classifier.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model.classifier(features)
            
            try:
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                print(f"Error during fine-tuning: {e}")
                print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
                print(f"Outputs shape: {outputs.shape}")
                print(f"Device check - Features: {features.device}, Labels: {labels.device}")
                return
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        if val_loader is not None:
            model.classifier.eval()
            val_loss = 0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model.classifier(features)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Fine-tuning stopped at epoch {epoch+1}")
                early_stopped = True
                break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # Print completion message if we didn't early stop
    if not early_stopped:
        if val_loader is not None:
            print(f"Fine-tuning completed all {epochs} epochs (no early stopping)")
        else:
            print(f"Fine-tuning completed all {epochs} epochs (no validation)")
    
    model.eval()
    return model

def compute_embeddings(model, dataloader, device=None):
    """Compute the embeddings of the model on the given dataset.
    
    Args:
        model: The model to be computed.
        dataloader: DataLoader with validation batches.
        
    Returns:
        Dictionary mapping activity indices to lists of embeddings
    """
    device = next(model.parameters()).device
    model.to(device)
    embeddings_by_label = {}
    model.eval()
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model.extract_features(inputs)
            
            # Convert one-hot to class indices
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)
                
            # Get embeddings for each activity
            for i, label_idx in enumerate(labels):
                label_idx = label_idx.item()  # Convert tensor to integer
                if label_idx not in embeddings_by_label:
                    embeddings_by_label[label_idx] = []
                embeddings_by_label[label_idx].append(embeddings[i])
                
    return embeddings_by_label
