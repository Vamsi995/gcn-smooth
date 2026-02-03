import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AnchoredLaplacianLoRA(nn.Module):
    def __init__(self, U, eigvals, base_filter):
        super().__init__()
        self.register_buffer("U", U)
        self.register_buffer("eigvals", eigvals)
        self.register_buffer("base_filter", base_filter)

        self.theta_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.theta_mlp[-1].weight)
        nn.init.zeros_(self.theta_mlp[-1].bias)

    def forward(self, H, alpha_l):
        proj = self.U.T @ H
        lam = self.eigvals.unsqueeze(-1)
        theta = torch.sigmoid(self.theta_mlp(lam)).squeeze(-1)
        g = self.base_filter * (1.0 - alpha_l * theta)
        return self.U @ (g.unsqueeze(1) * proj)


class LaplacianLoRAGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth, U, eigvals, base_filter, alpha):
        super().__init__()
        self.depth = depth
        self.alpha = alpha

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(depth - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

        self.lora = AnchoredLaplacianLoRA(U, eigvals, base_filter)

    def forward(self, x, edge_index):
        h = F.relu(self.convs[0](x, edge_index))

        for i, conv in enumerate(self.convs[1:-1], start=1):
            alpha_l = self.alpha * i / self.depth
            h = F.relu(conv(h, edge_index) + self.lora(h, alpha_l))

        return self.convs[-1](h, edge_index)
