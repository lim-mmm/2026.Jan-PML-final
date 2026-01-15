import math
import torch
from torch.utils.data import Dataset

def _sample_rings(n, K=4, r0=0.25, gap=0.18, sigma=0.03,
                 imbalance=0.0, broken=False, device="cpu"):
    """
    Sample 2D points from K concentric rings.
    Returns: [n,2] points roughly in [-1,1].
    """
    radii = torch.linspace(r0, r0 + gap * (K - 1), K, device=device)

    if imbalance <= 1e-8:
        probs = torch.ones(K, device=device) / K
    else:
        weights = torch.arange(1, K + 1, device=device).float() ** (1.0 + 4.0 * imbalance)
        probs = weights / weights.sum()

    idx = torch.multinomial(probs, num_samples=n, replacement=True)
    r = radii[idx] + sigma * torch.randn(n, device=device)

    theta = 2 * math.pi * torch.rand(n, device=device)

    if broken:
        a = math.pi / 3
        delta = math.pi / 5
        mask = (theta > a) & (theta < a + delta)
        while mask.any():
            theta[mask] = 2 * math.pi * torch.rand(int(mask.sum().item()), device=device)
            mask = (theta > a) & (theta < a + delta)

    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return x

def _render_point_to_28x28(xy, H=28, W=28, blob_sigma=1.2):
    """
    Render a 2D point (x,y) into a 28x28 image with a Gaussian blob.
    xy expected roughly in [-1,1].
    Output: [1,H,W] in [0,1]
    """
    device = xy.device
    px = (xy[0] + 1) * 0.5 * (W - 1)
    py = (xy[1] + 1) * 0.5 * (H - 1)

    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    g = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * blob_sigma ** 2))
    g = g / (g.max() + 1e-8)
    return g.unsqueeze(0)

class RingsImageDataset(Dataset):
    """
    Returns (x, y_dummy) where x is flattened 784 tensor in [-1,1].
    """
    def __init__(self, N=60000, K=4, r0=0.25, gap=0.18, sigma=0.03,
                 imbalance=0.0, broken=False, blob_sigma=1.2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        torch.manual_seed(seed)
        xy = _sample_rings(N, K=K, r0=r0, gap=gap, sigma=sigma,
                           imbalance=imbalance, broken=broken, device="cpu")

        imgs = torch.stack([_render_point_to_28x28(p, blob_sigma=blob_sigma) for p in xy], dim=0)  # [N,1,28,28]
        imgs = imgs * 2.0 - 1.0 
        self.x = imgs.view(N, -1).contiguous() 
        self.y = torch.zeros(N, dtype=torch.long) 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
