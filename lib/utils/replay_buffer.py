import torch

def init_global_prototypes(data_by_label, model, device=None):
    """Initialize global prototypes for each class."""
    if device is None:
        device = next(model.parameters()).device
    
    global_prototypes = {}
    
    for label, samples in data_by_label.items():
        if not samples:
            continue
        embeddings = torch.stack([e for e, _, _ in samples])
        global_prototypes[label] = embeddings.mean(dim=0).to(device)
    
    return global_prototypes

def update_global_prototypes(global_prototypes, curr_data_by_label, drift_scores=None, device=None):
    """Update global prototypes using exponential moving average."""
    if device is None:
        if global_prototypes and next(iter(global_prototypes.values())).device:
            device = next(iter(global_prototypes.values())).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_beta = 0.9
    min_beta = 0.5
    max_beta = 0.95

    for label, samples in curr_data_by_label.items():
        if not samples:
            continue
        
        embeddings = torch.stack([e for e, _, _ in samples]).to(device)
        current_prototype = embeddings.mean(dim=0)
        
        if label in global_prototypes:
            if drift_scores is not None and label in drift_scores and drift_scores[label] is not None:
                beta = max(min_beta, min(max_beta, base_beta - 0.1 * drift_scores[label]))
            else:
                beta = base_beta
            global_prototypes[label] = beta * global_prototypes[label] + (1 - beta) * current_prototype
        else:
            global_prototypes[label] = current_prototype
    
    return global_prototypes

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

def update_replay_buffer(replay_buffer, curr_data_by_label, global_prototypes, 
                         buffer_size_per_class=10, drift_scores=None, device=None):
    """
    Update replay buffer using blended prototypes for sample selection.
    
    Args:
        replay_buffer: Dictionary with 'X', 'y', and 'samples_by_label' data
        curr_data_by_label: Dictionary mapping label indices to current batch embeddings
        global_prototypes: Dictionary mapping label indices to global prototypes
        buffer_size_per_class: Maximum number of samples to keep per class
        drift_scores: Dictionary mapping activity indices to drift scores
        device: Device for tensor operations
        
    Returns:
        Updated replay buffer and global prototypes
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update global prototypes with current data
    global_prototypes = update_global_prototypes(global_prototypes, curr_data_by_label, drift_scores, device)
    
    # Initialize or get existing samples by label from replay buffer
    if 'samples_by_label' not in replay_buffer:
        replay_buffer['samples_by_label'] = {}
    
    samples_by_label = replay_buffer['samples_by_label']
    
    # Select representative samples based on the global prototypes
    curr_representatives = select_representative_samples(curr_data_by_label, global_prototypes, 
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