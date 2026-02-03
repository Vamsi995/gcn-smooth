import torch
from torch_geometric.utils import get_laplacian
from scipy.sparse.linalg import eigsh

def compute_laplacian_eig(data, k, device):
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        normalization="sym",
        num_nodes=data.num_nodes,
    )

    L = torch.sparse_coo_tensor(
        edge_index, edge_weight,
        (data.num_nodes, data.num_nodes)
    ).to_dense().cpu().numpy()

    eigvals, eigvecs = eigsh(L, k=k, which="SM")

    return (
        torch.tensor(eigvecs, dtype=torch.float32, device=device),
        torch.tensor(eigvals, dtype=torch.float32, device=device),
    )
