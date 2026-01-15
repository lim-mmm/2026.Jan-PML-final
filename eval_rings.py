import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import utils, transforms
from toy_rings_dataset import RingsImageDataset


class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoding path
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding path
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        h = self.tconv4(h4)
        h = h + self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.act(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class DDPM(nn.Module):
    def __init__(self, network, T=1000, beta_1=1e-4, beta_T=2e-2):
        super().__init__()
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        self.T = T
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])
        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        mean =  1./torch.sqrt(self.alpha[t]) * (xt - (self.beta[t])/torch.sqrt(1-self.alpha_bar[t])*self.network(xt, t)) 
        std = torch.where(t>0, torch.sqrt(((1-self.alpha_bar[t-1]) / (1-self.alpha_bar[t]))*self.beta[t]), 0)
        
        return mean + std*epsilon

    @torch.no_grad()
    def sample(self, shape, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        xT = torch.randn(shape, device=self.beta.device)
        xt = xT
        for tt in range(self.T, 0, -1):
            noise = torch.randn_like(xt) if tt > 1 else 0
            t_tensor = torch.tensor(tt, device=self.beta.device).expand(xt.shape[0], 1)
            xt = self.reverse_diffusion(xt, t_tensor, noise)
        return xt



def images_to_xy_com(imgs_01: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Center-of-mass decode to continuous (x,y) in [-1,1].
    imgs_01: [N,1,H,W] in [0,1]
    """
    N, _, H, W = imgs_01.shape
    device = imgs_01.device

    mass = imgs_01[:, 0, :, :].clamp(min=0.0)          # [N,H,W]
    Z = mass.sum(dim=(1, 2), keepdim=True) + eps       # [N,1,1]
    p = mass / Z                                       # normalize to prob

    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")     # [H,W]

    px = (p * xx).sum(dim=(1, 2))                      # [N]
    py = (p * yy).sum(dim=(1, 2))                      # [N]

    x = (px / (W - 1)) * 2.0 - 1.0
    y = (py / (H - 1)) * 2.0 - 1.0
    return torch.stack([x, y], dim=1)


@torch.no_grad()
def sample_images_01(model: DDPM, nsamples: int, seed: int) -> torch.Tensor:
    xt = model.sample((nsamples, 28 * 28), seed=seed)
    imgs = xt.view(-1, 1, 28, 28)
    imgs = (imgs + 1.0) / 2.0
    return imgs.clamp(0.0, 1.0)


def save_sample_grid(imgs_01: torch.Tensor, outpath: str, nrow: int = None):
    N = imgs_01.shape[0]
    if nrow is None:
        nrow = max(1, int(math.sqrt(N)))
    grid = utils.make_grid(imgs_01, nrow=nrow)
    pil = transforms.functional.to_pil_image(grid)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pil.save(outpath)


def plot_scatter_xy(xy: np.ndarray, outpath: str, title: str):
    plt.figure(figsize=(5, 5))
    plt.scatter(xy[:, 0], xy[:, 1], s=3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_radius_hist(r: np.ndarray, outpath: str, title: str, bins: int = 60):
    plt.figure(figsize=(6, 4))
    plt.hist(r, bins=bins)
    plt.xlabel("radius r")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def load_cfg(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def build_model(cfg: dict, device: torch.device) -> DDPM:
    T = int(cfg.get("T", 1000))
    unet = ScoreNet(lambda t: torch.ones(1).to(device))
    model = DDPM(unet, T=T).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/ddpm_toy_rings_difficult_fix", help="Training run directory containing model_last.pt and config.json")
    parser.add_argument("--ckpt", type=str, default="model_last.pt", help="Checkpoint filename inside run_dir")
    parser.add_argument("--nsamples", type=int, default=4096, help="Number of samples for geometry eval")
    parser.add_argument("--grid_samples", type=int, default=64, help="Number of samples for a sample grid")
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed")
    parser.add_argument("--bins", type=int, default=60, help="Bins for radius histogram")
    parser.add_argument("--also_plot_real", action="store_true", help="Also plot real dataset scatter/hist for comparison")
    args = parser.parse_args()

    run_dir = args.run_dir
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg(run_dir)
    model = build_model(cfg, device)

    ckpt_path = os.path.join(run_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    imgs_grid = sample_images_01(model, nsamples=args.grid_samples, seed=args.seed)
    save_sample_grid(imgs_grid, os.path.join(eval_dir, "samples_grid.png"))

    imgs = sample_images_01(model, nsamples=args.nsamples, seed=args.seed)
    xy = images_to_xy_com(imgs).cpu().numpy()
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)

    plot_scatter_xy(xy, os.path.join(eval_dir, "gen_scatter_xy.png"),
                    title="Generated points")
    plot_radius_hist(r, os.path.join(eval_dir, "gen_radius_hist.png"),
                     title="Generated radius histogram", bins=args.bins)

    if args.also_plot_real:
        keys = ["N", "K", "r0", "gap", "sigma", "imbalance", "broken", "blob_sigma", "seed"]
        toy_cfg = {k: cfg[k] for k in keys if k in cfg}
        toy_cfg["N"] = min(int(toy_cfg.get("N", 60000)), args.nsamples)
        ds = RingsImageDataset(**toy_cfg)

        x = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)  # [N,784] in [-1,1]
        imgs_real = ((x.view(-1, 1, 28, 28) + 1.0) / 2.0).clamp(0.0, 1.0)

        xy_real = images_to_xy_com(imgs_real).cpu().numpy()
        r_real = np.sqrt(xy_real[:, 0] ** 2 + xy_real[:, 1] ** 2)

        plot_scatter_xy(xy_real, os.path.join(eval_dir, "real_scatter_xy.png"),
                        title="Real data points")
        plot_radius_hist(r_real, os.path.join(eval_dir, "real_radius_hist.png"),
                         title="Real radius histogram (ground truth)", bins=args.bins)

    with open(os.path.join(eval_dir, "eval_summary.txt"), "w") as f:
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"nsamples: {args.nsamples}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"mean radius: {r.mean():.4f}\n")
        f.write(f"std radius: {r.std():.4f}\n")

    print(f"[OK] Saved evaluation outputs to: {eval_dir}")
    print(" - samples_grid.png")
    print(" - gen_scatter_xy.png")
    print(" - gen_radius_hist.png")
    if args.also_plot_real:
        print(" - real_scatter_xy.png")
        print(" - real_radius_hist.png")
    print(" - eval_summary.txt")


if __name__ == "__main__":
    main()
