import torch

def embedding_variance(H):
    mean = H.mean(dim=0, keepdim=True)
    return ((H - mean) ** 2).sum(dim=1).mean().item()


@torch.no_grad()
def get_embeddings(model, data):
    model.eval()
    return model(data.x, data.edge_index)
