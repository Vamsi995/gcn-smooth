from collections import defaultdict
import numpy as np
import torch

from torch_geometric.datasets import Planetoid, Coauthor

from src.utils import set_seed, ensure_masks
from src.laplacian import compute_laplacian_eig
from src.models.gcn import GCN
from src.models.laplacian_lora import LaplacianLoRAGCN
from src.train import train_and_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEPTHS = [2, 4, 8, 16, 32]
SEEDS = [0, 1, 2]

HIDDEN_DIM = 64
EPOCHS_GCN = 200
EPOCHS_LORA = 400
K_EIGS = 64
ALPHA = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4

datasets = {
    "Cora": Planetoid("/tmp/Cora", "Cora"),
    "Citeseer": Planetoid("/tmp/Citeseer", "Citeseer"),
    "Pubmed": Planetoid("/tmp/Pubmed", "Pubmed"),
    "CoauthorCS": Coauthor("/tmp/CoauthorCS", "CS"),
    "CoauthorPhysics": Coauthor("/tmp/CoauthorPhysics", "Physics"),
}

results = defaultdict(lambda: defaultdict(list))

for name, dataset in datasets.items():
    data = ensure_masks(dataset[0]).to(device)
    U, lam = compute_laplacian_eig(data, K_EIGS, device)
    base_filter = 1.0 - lam

    for depth in DEPTHS:
        gcn_accs, lora_accs = [], []

        for seed in SEEDS:
            set_seed(seed)

            gcn = GCN(data.x.size(1), HIDDEN_DIM, dataset.num_classes, depth).to(device)
            gcn_accs.append(train_and_eval(gcn, data, EPOCHS_GCN, LR, WEIGHT_DECAY))

            lora = LaplacianLoRAGCN(
                data.x.size(1), HIDDEN_DIM, dataset.num_classes,
                depth, U, lam, base_filter, ALPHA
            ).to(device)

            lora_accs.append(train_and_eval(lora, data, EPOCHS_LORA, LR, WEIGHT_DECAY))

        results[name]["GCN"].append((np.mean(gcn_accs), np.std(gcn_accs)))
        results[name]["LaplacianLoRA"].append((np.mean(lora_accs), np.std(lora_accs)))

import json
with open("results/accuracy_vs_depth.json", "w") as f:
    json.dump(results, f, indent=2)
