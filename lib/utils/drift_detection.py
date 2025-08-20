import torch
import numpy as np
import torch.nn.functional as F

def detect_drifts(curr_embeddings, global_prototypes, replay_buffer,
                  default_threshold=0.2, novelty_threshold_factor=2.0, use_cosine=True, min_samples=3):
    """Detect drifts, novelty, and unseen activities using global prototypes."""
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
    
    drift_scores = {}

    if not isinstance(curr_embeddings, dict) or not isinstance(global_prototypes, dict):
        print("Warning: Invalid input parameters for drift detection")
        return set(), set(), set()
    
    if use_cosine:
        base_drift_threshold = 0.1
    else:
        base_drift_threshold = default_threshold
    base_novelty_threshold = base_drift_threshold * novelty_threshold_factor
    
    known_activity_indices = {int(idx) for idx in global_prototypes.keys()}
    curr_activity_indices = {int(idx) for idx in curr_embeddings.keys()}
    
    unseen_activities = curr_activity_indices - known_activity_indices
    
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

    drifting_activities = set()
    novel_activities = set()
    
    for activity_idx in curr_activity_indices:
        if activity_idx not in known_activity_indices:
            continue
            
        activity_embeddings = curr_embeddings[activity_idx]
        if not activity_embeddings:
            continue
            
        try:
            global_prototype = global_prototypes[activity_idx].to(device)
            
            distances = []
            for embed in activity_embeddings:
                embed = embed.to(device)
                if use_cosine:
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        embed.unsqueeze(0), global_prototype.unsqueeze(0)).item()
                    dist = 1.0 - cosine_sim
                else:
                    dist = torch.norm(embed - global_prototype).item()
                distances.append(dist)
            
            if distances:
                mean_distance = sum(distances) / len(distances)
                drift_scores[activity_idx] = mean_distance
                drift_threshold = drift_thresholds.get(activity_idx, base_drift_threshold)
                novelty_threshold = novelty_thresholds.get(activity_idx, base_novelty_threshold)
                
                if mean_distance > novelty_threshold:
                    novel_activities.add(activity_idx)
                elif mean_distance > drift_threshold:
                    drifting_activities.add(activity_idx)
        
        except Exception as e:
            print(f"Warning: Error processing activity {activity_idx}: {e}")
            continue
    
    print(f"Found {len(unseen_activities)} unseen activities, {len(drifting_activities)} drifting activities, "
          f"and {len(novel_activities)} novel activities")
    
    return unseen_activities, drifting_activities, novel_activities, drift_scores

def compute_drift_thresholds(replay_buffer, threshold=0.2, novelty_threshold_factor=2.0, use_cosine=True, min_samples=3):
    """Compute thresholds for drift and novelty detection based on intra-cluster variance."""
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
    
    if use_cosine:
        default_drift_threshold = 0.1
        default_novelty_threshold = default_drift_threshold * novelty_threshold_factor
    else:
        default_drift_threshold = threshold
        default_novelty_threshold = default_drift_threshold * novelty_threshold_factor
    
    drift_thresholds = {}
    novelty_thresholds = {}
    
    if not replay_buffer or not isinstance(replay_buffer, dict) or 'samples_by_label' not in replay_buffer:
        print("Warning: Invalid replay buffer structure, using default thresholds")
        return drift_thresholds, novelty_thresholds
    
    try:
        samples_by_label = replay_buffer['samples_by_label']
        
        if not isinstance(samples_by_label, dict):
            print(f"Warning: samples_by_label is not a dictionary, using default thresholds")
            return drift_thresholds, novelty_thresholds
            
        embeddings_by_label = {}
        for label, samples in samples_by_label.items():
            try:
                if not samples or not isinstance(samples, list):
                    continue
                    
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
        
        for activity_idx, embeddings in embeddings_by_label.items():
            if len(embeddings) < min_samples:
                drift_thresholds[activity_idx] = default_drift_threshold
                novelty_thresholds[activity_idx] = default_novelty_threshold
                continue
                
            try:
                window_embeddings = torch.stack([emb.to(device) for emb in embeddings])
                window_mean = window_embeddings.mean(dim=0)
                
                distances = []
                if use_cosine:
                    for emb in embeddings:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            emb.to(device).unsqueeze(0), window_mean.unsqueeze(0)).item()
                        distances.append(1.0 - cos_sim)
                else:
                    distances = [torch.norm(emb.to(device) - window_mean).item() for emb in embeddings]
                
                median_distance = np.median(distances)
                mad = np.median([abs(d - median_distance) for d in distances])
                
                if mad < 1e-6:
                    std_distance = np.std(distances)
                    
                    if std_distance < 1e-6:
                        if use_cosine:
                            mad = 0.01
                        else:
                            mad = 0.05
                    else:
                        mad = std_distance / 1.4826
                
                k = 3
                drift_thresholds[activity_idx] = median_distance + k * mad
                novelty_thresholds[activity_idx] = drift_thresholds[activity_idx] * novelty_threshold_factor
                
                if use_cosine:
                    drift_thresholds[activity_idx] = max(drift_thresholds[activity_idx], 0.1)
                    novelty_thresholds[activity_idx] = max(novelty_thresholds[activity_idx], 0.2)
                else:
                    drift_thresholds[activity_idx] = max(drift_thresholds[activity_idx], threshold * 0.5)
                    novelty_thresholds[activity_idx] = max(novelty_thresholds[activity_idx], threshold)
                      
            except Exception as e:
                print(f"Warning: Error computing thresholds for activity {activity_idx}: {e}")
                drift_thresholds[activity_idx] = default_drift_threshold
                novelty_thresholds[activity_idx] = default_novelty_threshold
    
    except Exception as e:
        print(f"Warning: Error in threshold computation: {e}")
    
    global_prototypes = replay_buffer.get('global_prototypes', {})
    for activity_idx in global_prototypes.keys():
        if activity_idx not in drift_thresholds:
            drift_thresholds[activity_idx] = default_drift_threshold
            novelty_thresholds[activity_idx] = default_novelty_threshold
    
    return drift_thresholds, novelty_thresholds 