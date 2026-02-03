import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_masks(data, train_ratio=0.6, val_ratio=0.2):
    if hasattr(data, "train_mask"):
        return data

    n = data.num_nodes
    perm = torch.randperm(n)

    t_end = int(train_ratio * n)
    v_end = int((train_ratio + val_ratio) * n)

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)

    data.train_mask[perm[:t_end]] = True
    data.val_mask[perm[t_end:v_end]] = True
    data.test_mask[perm[v_end:]] = True

    return data
