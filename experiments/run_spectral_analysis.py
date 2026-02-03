import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid, Coauthor

from src.utils import set_seed, ensure_masks
from src.laplacian import compute_laplacian_eig
from src.models.laplacian_lora import LaplacianLoRAGCN
from src.train import train_and_eval


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "Cora": Planetoid("/tmp/Cora", "Cora"),
    "CoauthorCS": Coauthor("/tmp/CoauthorCS", "CS"),
}

DEPTH_FOR_ANALYSIS = 16
DEPTHS_FOR_CONTRACTION = [2, 4, 8, 16, 32]

HIDDEN_DIM = 64
EPOCHS_LORA = 400
K_EIGS = 64
ALPHA = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4


@torch.no_grad()
def extract_theta(lora_model):
    """
    Extract θ(λ) exactly as used in forward().
    """
    lora_block = lora_model.lora
    lam = lora_block.eigvals.unsqueeze(-1)
    theta = torch.sigmoid(lora_block.theta_mlp(lam)).squeeze(-1)
    return theta.cpu().numpy()


def compute_propagation_eigenvalues(eigvals, theta, alpha):
    lam = eigvals.cpu().numpy()
    mu_gcn = 1.0 - lam
    mu_lora = (1.0 - lam) * (1.0 - alpha * theta)
    return lam, mu_gcn, mu_lora


def plot_propagation_spectrum(lam, mu_gcn, mu_lora, title):
    idx = np.argsort(lam)

    plt.figure(figsize=(6, 4))
    plt.plot(lam[idx], mu_gcn[idx], "--", label="GCN", linewidth=2)
    plt.plot(lam[idx], mu_lora[idx], label="Laplacian-LoRA", linewidth=2)

    plt.axhline(1.0, linestyle=":", color="gray")
    plt.axhline(-1.0, linestyle=":", color="gray")

    plt.xlabel("Laplacian Eigenvalue λ")
    plt.ylabel("Propagation Eigenvalue μ(λ)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # zoom near stability boundary
    plt.ylim(0.85, 1.02)
    plt.tight_layout()
    plt.show()


def plot_contraction_ratio(mu_gcn, mu_lora, depths, title):
    mu_gcn_sorted = np.sort(np.abs(mu_gcn))[::-1]
    mu_lora_sorted = np.sort(np.abs(mu_lora))[::-1]

    ratio_gcn = [(mu_gcn_sorted[1] / mu_gcn_sorted[0]) ** L for L in depths]
    ratio_lora = [(mu_lora_sorted[1] / mu_lora_sorted[0]) ** L for L in depths]

    plt.figure(figsize=(6, 4))
    plt.plot(depths, ratio_gcn, "o--", label="GCN")
    plt.plot(depths, ratio_lora, "o-", label="Laplacian-LoRA")

    plt.xlabel("Depth L")
    plt.ylabel("Spectral Contraction Ratio")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_retention(lam, mu_gcn, mu_lora, L_plot, title):
    idx = np.argsort(lam)

    E_gcn = np.abs(mu_gcn) ** L_plot
    E_lora = np.abs(mu_lora) ** L_plot

    plt.figure(figsize=(6, 4))
    plt.plot(lam[idx], E_gcn[idx], "--", label="GCN")
    plt.plot(lam[idx], E_lora[idx], label="Laplacian-LoRA")

    plt.xlabel("Laplacian Eigenvalue λ")
    plt.ylabel(f"Energy After {L_plot} Layers")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    set_seed(0)

    for name, dataset in DATASETS.items():
        print(f"\n=== Spectral Analysis: {name} ===")

        data = ensure_masks(dataset[0]).to(DEVICE)

        # Laplacian eigendecomposition
        U, eigvals = compute_laplacian_eig(data, K_EIGS, DEVICE)
        base_filter = 1.0 - eigvals

        # Train depth-16 Laplacian-LoRA
        model = LaplacianLoRAGCN(
            data.x.size(1),
            HIDDEN_DIM,
            dataset.num_classes,
            DEPTH_FOR_ANALYSIS,
            U,
            eigvals,
            base_filter,
            ALPHA,
        ).to(DEVICE)

        train_and_eval(
            model,
            data,
            epochs=EPOCHS_LORA,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        # Extract θ(λ)
        theta = extract_theta(model)

        # Compute propagation eigenvalues
        lam, mu_gcn, mu_lora = compute_propagation_eigenvalues(
            eigvals, theta, ALPHA
        )

        # ---- Plots ----
        plot_propagation_spectrum(
            lam,
            mu_gcn,
            mu_lora,
            title=f"Propagation Spectrum ({name})",
        )

        plot_contraction_ratio(
            mu_gcn,
            mu_lora,
            DEPTHS_FOR_CONTRACTION,
            title=f"Spectral Contraction vs Depth ({name})",
        )

        plot_energy_retention(
            lam,
            mu_gcn,
            mu_lora,
            L_plot=DEPTH_FOR_ANALYSIS,
            title=f"Energy Retention ({name}, L={DEPTH_FOR_ANALYSIS})",
        )
