from collections import defaultdict
import numpy as np
import torch

from torch_geometric.datasets import Planetoid, Coauthor

from src.utils import set_seed, ensure_masks
from src.laplacian import compute_laplacian_eig
from src.models.gcn import GCN
from src.models.laplacian_lora import LaplacianLoRAGCN
from src.train import train_and_eval
from src.metrics import embedding_variance, get_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEPTHS = [2, 4, 8, 16, 32]
SEEDS = [0, 1, 2]
HIDDEN_DIM = 64
EPOCHS_GCN = 200
EPOCHS_LORA = 400
K_EIGS = 64
ALPHA = 0.5

datasets = {
    "Cora": Planetoid("/tmp/Cora", "Cora"),
    "CoauthorCS": Coauthor("/tmp/CoauthorCS", "CS"),
}

results = defaultdict(lambda: defaultdict(list))

for name, dataset in datasets.items():
    data = ensure_masks(dataset[0]).to(device)
    U, lam = compute_laplacian_eig(data, K_EIGS, device)
    base_filter = 1.0 - lam

    for depth in DEPTHS:
        gcn_vars, lora_vars = [], []

        for seed in SEEDS:
            set_seed(seed)

            gcn = GCN(data.x.size(1), HIDDEN_DIM, dataset.num_classes, depth).to(device)
            train_and_eval(gcn, data, EPOCHS_GCN, 0.01, 5e-4)
            gcn_vars.append(embedding_variance(get_embeddings(gcn, data)))

            lora = LaplacianLoRAGCN(
                data.x.size(1), HIDDEN_DIM, dataset.num_classes,
                depth, U, lam, base_filter, ALPHA
            ).to(device)
            train_and_eval(lora, data, EPOCHS_LORA, 0.01, 5e-4)
            lora_vars.append(embedding_variance(get_embeddings(lora, data)))

        results[name]["GCN"].append((np.mean(gcn_vars), np.std(gcn_vars)))
        results[name]["LaplacianLoRA"].append((np.mean(lora_vars), np.std(lora_vars)))
