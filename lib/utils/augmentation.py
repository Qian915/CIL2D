import torch
import numpy as np
import torch.nn.functional as F


def intra_class_mixup(activities, curr_embeddings, replay_buffer, alpha=0.4, mix_times=4):
    """Enhanced feature augmentation within the same activity using replay buffer samples."""
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']

    mixed_embeddings = {}
    
    for act in activities:
        if act not in curr_embeddings:
            continue
            
        curr_samples = curr_embeddings[act]
        if not curr_samples:
            continue
        real_count = len(curr_samples)
        aug_count = int(0.5 * real_count)

        buffer_samples = []
        if act in replay_buffer['samples_by_label']:
            buffer_samples = [e for e, _, _ in replay_buffer['samples_by_label'][act]]
        
        if not buffer_samples:
            continue
            
        mixed_embeddings[act] = []
        
        curr_samples = torch.stack([emb.to(device) for emb in curr_samples])
        buffer_samples = torch.stack([emb.to(device) for emb in buffer_samples])
        
        for _ in range(mix_times):
            curr_idx = torch.randperm(len(curr_samples))
            buffer_idx = torch.randperm(len(buffer_samples))
            
            samples_added = 0
            for i in range(min(aug_count, len(buffer_samples))):
                lam = np.random.beta(alpha, alpha)  # lambda is the interpolation parameter generated from beta distribution
                lam = max(lam, 1-lam)
                lam = max(0.7, min(lam, 0.9))
                mixed = lam * curr_samples[curr_idx[i]] + (1 - lam) * buffer_samples[buffer_idx[i]]
                mixed_embeddings[act].append(mixed)
                samples_added += 1
                if samples_added >= aug_count:
                    break
    
    return mixed_embeddings

def distribution_aug(activities, curr_embeddings, gamma: float = 0.3, eps: float = 1e-3):
    """Generate augmented samples using prototype jittering."""
    if 'device' not in globals():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = globals()['device']
        
    F_aug = []
    Y_aug = []
    
    activities = list(activities)
    if not activities:
        return None, None
            
    for act in activities:
        if act not in curr_embeddings:
            continue
            
        feats = curr_embeddings[act]
        if not feats:
            continue
            
        feats = torch.stack([f.to(device) for f in feats])
        real_count = len(feats)
        
        if real_count < 5:
            continue
            
        aug_count = int(0.5 * real_count)
        
        mu = feats.mean(0)
        X = feats - mu
        
        cov = (X.t() @ X) / max(1, feats.size(0)-1)
        cov_s = (1-gamma)*cov + gamma*torch.diag(torch.diag(cov)) + eps*torch.eye(cov.size(0), device=device)
        L = torch.linalg.cholesky(cov_s)
        z = torch.randn(aug_count, feats.size(1), device=device) @ L.T + mu
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
        y = torch.full((aug_count,), int(act), device=device)
        F_aug.append(z)
        Y_aug.append(y)
    
    if not F_aug:
        return None, None
        
    F_aug = torch.cat(F_aug, dim=0)
    Y_aug = torch.cat(Y_aug, dim=0)
    return F_aug, Y_aug 