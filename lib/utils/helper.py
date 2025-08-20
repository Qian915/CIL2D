import torch

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

    return data_by_label

def create_dataloader_from_buffer(replay_buffer, batch_size=32, shuffle=True, max_case_length=None):
    """Create a DataLoader from a replay buffer."""
    if not replay_buffer['X'].size(0):
        return None
    
    if max_case_length is None:
        max_case_length = replay_buffer['X'].size(1)
    
    seq_lengths = torch.ones(replay_buffer['X'].size(0), dtype=torch.long) * max_case_length
    
    if replay_buffer['y'].ndim == 1:
        num_classes = replay_buffer['y'].max().item() + 1
        y_onehot = torch.zeros(replay_buffer['y'].size(0), num_classes)
        y_onehot.scatter_(1, replay_buffer['y'].unsqueeze(1), 1)
        labels = y_onehot
    else:
        labels = replay_buffer['y']
        
    dataset = torch.utils.data.TensorDataset(replay_buffer['X'], labels, seq_lengths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def combine_dataloaders(current_loader, replay_loader, batch_size=32):
    """Combine current batch DataLoader with replay buffer DataLoader."""
    if replay_loader is None:
        return current_loader
        
    X_current, y_current, len_current = [], [], []
    X_replay, y_replay, len_replay = [], [], []
    
    for inputs, labels, lengths in current_loader:
        X_current.append(inputs)
        y_current.append(labels)
        len_current.append(lengths)
    
    for inputs, labels, lengths in replay_loader:
        X_replay.append(inputs)
        y_replay.append(labels)
        len_replay.append(lengths)
    
    current_classes = y_current[0].shape[1]
    replay_classes = y_replay[0].shape[1]
    
    if current_classes != replay_classes:
        max_classes = max(current_classes, replay_classes)
        
        if current_classes < max_classes:
            resized_y_current = []
            for y in y_current:
                new_y = torch.zeros(y.shape[0], max_classes, device=y.device)
                new_y[:, :current_classes] = y
                resized_y_current.append(new_y)
            y_current = resized_y_current
        
        if replay_classes < max_classes:
            resized_y_replay = []
            for y in y_replay:
                new_y = torch.zeros(y.shape[0], max_classes, device=y.device)
                new_y[:, :replay_classes] = y
                resized_y_replay.append(new_y)
            y_replay = resized_y_replay
    
    X_combined = torch.cat(X_current + X_replay, dim=0)
    y_combined = torch.cat(y_current + y_replay, dim=0)
    len_combined = torch.cat(len_current + len_replay, dim=0)
    
    combined_dataset = torch.utils.data.TensorDataset(X_combined, y_combined, len_combined)
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    return combined_loader 