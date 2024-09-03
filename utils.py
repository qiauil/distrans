import matplotlib.pyplot as plt
import numpy as np
import torch, math
import torch.nn as nn
from typing import *
from sklearn.datasets import make_moons,make_swiss_roll
from sklearn.preprocessing import StandardScaler

def get_data(dataset: str, n_points: int) -> np.ndarray:
    if dataset == "moons":
        data, _ = make_moons(n_points, noise=0.05)
    elif dataset == "swiss":
        data, _ = make_swiss_roll(n_points, noise=0.5)
        data = data[:, [0, 2]] / 10.0
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return StandardScaler().fit_transform(data)

def plot_dist(data:Union[np.ndarray, torch.Tensor], title:Optional[str]=None, ax=None):
    if ax is None:
        fig,ax_i=plt.subplots(figsize=(5,5))
    else:
        ax_i=ax
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    im=ax_i.hist2d(data[:, 0], data[:, 1], bins=128,range=[[-3, 3], [-3, 3]],cmap="Blues")
    if title is not None:
        ax_i.set_title(title)
    ax_i.axis("off")
    if ax is None:
        plt.show()
    else:
        return im
        
def wasserstein_distance(x, y, p=1):
    """
    Computes the Wasserstein distance between two distributions x and y.
    
    Args:
        x (torch.Tensor): A tensor of shape (n_samples, n_features).
        y (torch.Tensor): A tensor of shape (m_samples, n_features).
        p (int): The order of the Wasserstein distance. Defaults to 1 (W1 distance).
    
    Returns:
        float: The Wasserstein distance between x and y.
    """
    n = x.shape[0]
    m = y.shape[0]
    x_broadcast = x.unsqueeze(1).repeat(1, m, 1)  # shape (n, m, n_features)
    y_broadcast = y.unsqueeze(0).repeat(n, 1, 1)  # shape (n, m, n_features)
    pairwise_distances = torch.norm(x_broadcast - y_broadcast, dim=2, p=p)  # shape (n, m)
    min_row, _ = torch.min(pairwise_distances, dim=1)
    min_col, _ = torch.min(pairwise_distances, dim=0)
    distance = torch.mean(min_row) + torch.mean(min_col)
    return distance.item()

class Net(nn.Module):
    def __init__(self, in_dim: int=2, out_dim: int=2, h_dims: List[int]=[512]*4,dim_time=32) -> None:
        super().__init__()
        half_dim = dim_time // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.t_embeddings_base = torch.exp(torch.arange(half_dim) * -embeddings)
        
        ins = [in_dim + half_dim*2] + h_dims
        outs = h_dims + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)]
            )
        self.out = nn.Sequential(nn.Linear(out_dim, out_dim))
        
    def time_encoder(self, t: torch.Tensor) -> torch.Tensor:
        embeddings = t[:, None] * self.t_embeddings_base.to(t.device)[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self.time_encoder(t)), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)