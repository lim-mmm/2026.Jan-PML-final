import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import math


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[64, 128, 256, 512], embed_dim=256):
        super().__init__()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # encoder
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

        # decoder
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 = self.act(self.gnorm1(h1 + self.dense1(embed)))

        h2 = self.conv2(h1)
        h2 = self.act(self.gnorm2(h2 + self.dense2(embed)))

        h3 = self.conv3(h2)
        h3 = self.act(self.gnorm3(h3 + self.dense3(embed)))

        h4 = self.conv4(h3)
        h4 = self.act(self.gnorm4(h4 + self.dense4(embed)))

        h = self.tconv4(h4)
        h = self.act(self.tgnorm4(h + self.dense5(embed)))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = self.act(self.tgnorm3(h + self.dense6(embed)))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = self.act(self.tgnorm2(h + self.dense7(embed)))

        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class DDPM(nn.Module):
    def __init__(self, network, T=200, beta_1=1e-4, beta_T=2e-2):
        super().__init__()
        self._network = network
        self.network = lambda x, t: (
            self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))
        ).reshape(-1, 28 * 28)

        self.T = T
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def reverse_diffusion(self, xt, t, epsilon):
        mean = 1.0 / torch.sqrt(self.alpha[t]) * (
            xt - (self.beta[t]) / torch.sqrt(1 - self.alpha_bar[t]) * self.network(xt, t)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]),
            torch.zeros_like(t, dtype=xt.dtype),
        )
        return mean + std * epsilon

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


@torch.no_grad()
def compute_nn_d1_d2(gen_flat, train_flat, batch=256):

    N = train_flat.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 training samples for d1/d2 ratio (N>=2).")

    M = gen_flat.shape[0]
    train_norm2 = (train_flat ** 2).sum(dim=1)  
    d1 = torch.empty(M, device=gen_flat.device)
    d2 = torch.empty(M, device=gen_flat.device)
    nn = torch.empty(M, dtype=torch.long, device=gen_flat.device)

    for s in range(0, M, batch):
        e = min(s + batch, M)
        g = gen_flat[s:e]
        g_norm2 = (g ** 2).sum(dim=1, keepdim=True)
        dist2 = g_norm2 + train_norm2[None, :] - 2.0 * (g @ train_flat.t())
        dist2 = dist2.clamp_min(0.0)
        vals, idxs = torch.topk(dist2, k=2, largest=False, dim=1)
        d1[s:e] = torch.sqrt(vals[:, 0])
        d2[s:e] = torch.sqrt(vals[:, 1])
        nn[s:e] = idxs[:, 0]
    return d1, d2, nn


def summarize_nn(d1_cpu, d2_cpu, D):

    d1 = d1_cpu
    d2 = d2_cpu
    ratio = d1 / (d2 + 1e-12)
    mse_nn = (d1 ** 2) / D
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_p05": float(torch.quantile(ratio, 0.05).item()),
        "ratio_p50": float(torch.quantile(ratio, 0.50).item()),
        "ratio_p95": float(torch.quantile(ratio, 0.95).item()),
        "d1_min": float(d1.min().item()),
        "d1_p01": float(torch.quantile(d1, 0.01).item()),
        "d1_median": float(torch.quantile(d1, 0.50).item()),
        "mse_min": float(mse_nn.min().item()),
        "mse_p01": float(torch.quantile(mse_nn, 0.01).item()),
        "mse_median": float(torch.quantile(mse_nn, 0.50).item()),
    }


def x_to_uint8(x_m11: torch.Tensor) -> torch.Tensor:
    x01 = ((x_m11 + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.round(x01 * 255.0).to(torch.uint8)

def save_pairs_grid(gen_u8_list, train_u8, pairs, out_png, nshow=16):
 
    nshow = min(nshow, len(pairs))
    if nshow == 0:
        return

    imgs = []
    for gi_slot, ti in pairs[:nshow]:
        g = gen_u8_list[gi_slot].float() / 255.0
        t = train_u8[ti].float() / 255.0
        pair = torch.cat([g, t], dim=2)
        imgs.append(pair)

    imgs = torch.stack(imgs, dim=0)
    nrow = max(1, int(math.sqrt(nshow)))
    grid = utils.make_grid(imgs, nrow=nrow, padding=2)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(grid[0].cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="runs/mnist_ddpm_mem_exact_n2_seed0")
    ap.add_argument("--ckpt", type=str, default="model_last.pt")
    ap.add_argument("--subset_idx", type=str, default="subset_idx.npy")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--nsamples", type=int, default=50000)
    ap.add_argument("--eval_batch", type=int, default=128, help="sampling batch size to avoid OOM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=float, default=0.33, help="ratio threshold for f_mem (d1/d2 < k)")
    ap.add_argument("--save_pairs", type=int, default=32, help="max number of exact-match pairs to save")
    ap.add_argument("--save_topk_nn", type=int, default=32, help="save top-k nearest-neighbor pairs (by smallest d1_u8)")
    ap.add_argument("--out_json", type=str, default="mem_exact_results.json")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = args.run_dir

    ckpt_path = os.path.join(run_dir, args.ckpt)
    subset_idx_path = os.path.join(run_dir, args.subset_idx)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(subset_idx_path):
        raise FileNotFoundError(f"subset_idx not found: {subset_idx_path}")

    idx = np.load(subset_idx_path).astype(int)

    tf_discrete = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255.0).round() / 255.0),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
    ])
    ds = datasets.MNIST("./mnist_data", train=True, download=True, transform=tf_discrete)

    train = torch.stack([ds[i][0] for i in idx], dim=0).to(device) 
    train_u8 = x_to_uint8(train) 

    train_keys = {}
    for ti in range(train_u8.size(0)):
        key = train_u8[ti].view(-1).cpu().numpy().tobytes()
        train_keys.setdefault(key, []).append(ti)

    unet = ScoreNet(lambda t: torch.ones(1, device=device))
    model = DDPM(unet, T=args.T).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = {
        "run_dir": run_dir,
        "ckpt": args.ckpt,
        "T": int(args.T),
        "nsamples": int(args.nsamples),
        "eval_batch": int(args.eval_batch),
        "seed": int(args.seed),
        "k_ratio_threshold": float(args.k),
        "subset_n": int(train_u8.size(0)),
        "note": "Exact match is defined as identical uint8 (0..255) pixel grid after rounding.",
    }

    train_flat = train.view(train.size(0), -1) 
    train_flat_u8 = train_u8.view(train_u8.size(0), -1).float().to(device)
    D = train_flat.shape[1]

    d1_f_all, d2_f_all = [], []
    d1_u_all, d2_u_all = [], []

    exact_hits_train = np.zeros(train_u8.size(0), dtype=np.int64)
    exact_gen_indices = [] 

    exact_gen_u8_keep = [] 
    exact_pairs_for_png = [] 

    top_nn = []  

    M = args.nsamples
    bs = args.eval_batch

    for start in range(0, M, bs):
        b = min(bs, M - start)

        x = model.sample((b, 28 * 28), seed=args.seed + start).clamp(-1, 1)
        gen = x.view(-1, 1, 28, 28)
        gen_u8 = x_to_uint8(gen)

        gen_u8_cpu = gen_u8.cpu()
        for j in range(gen_u8_cpu.size(0)):
            key = gen_u8_cpu[j].view(-1).numpy().tobytes()
            if key in train_keys:
                gi_global = start + j
                exact_gen_indices.append(gi_global)
                for ti in train_keys[key]:
                    exact_hits_train[ti] += 1

                if len(exact_gen_u8_keep) < args.save_pairs:
                    slot = len(exact_gen_u8_keep)
                    exact_gen_u8_keep.append(gen_u8_cpu[j])
                    exact_pairs_for_png.append((slot, train_keys[key][0]))

        gen_flat = gen.view(gen.size(0), -1)
        d1_f, d2_f, _ = compute_nn_d1_d2(gen_flat, train_flat, batch=512)
        d1_f_all.append(d1_f.detach().cpu())
        d2_f_all.append(d2_f.detach().cpu())

        gen_flat_u8 = gen_u8.view(gen_u8.size(0), -1).float().to(device)
        d1_u, d2_u, nn_u = compute_nn_d1_d2(gen_flat_u8, train_flat_u8, batch=512)
        d1_u_cpu = d1_u.detach().cpu()
        d2_u_cpu = d2_u.detach().cpu()
        nn_u_cpu = nn_u.detach().cpu()

        d1_u_all.append(d1_u_cpu)
        d2_u_all.append(d2_u_cpu)

        if args.save_topk_nn > 0:
            for j in range(b):
                val = float(d1_u_cpu[j].item())
                ti = int(nn_u_cpu[j].item())
                top_nn.append((val, gen_u8_cpu[j], ti))
            top_nn.sort(key=lambda x: x[0])
            top_nn = top_nn[: args.save_topk_nn]

        del x, gen, gen_u8, gen_u8_cpu, gen_flat, gen_flat_u8, d1_f, d2_f, d1_u, d2_u, nn_u
        if device.type == "cuda":
            torch.cuda.empty_cache()

    d1_f_cpu = torch.cat(d1_f_all, dim=0)
    d2_f_cpu = torch.cat(d2_f_all, dim=0)
    d1_u_cpu = torch.cat(d1_u_all, dim=0)
    d2_u_cpu = torch.cat(d2_u_all, dim=0)

    results["nn_float"] = summarize_nn(d1_f_cpu, d2_f_cpu, D)
    results["f_mem_float"] = float(((d1_f_cpu / (d2_f_cpu + 1e-12)) < args.k).float().mean().item())

    results["nn_u8"] = summarize_nn(d1_u_cpu, d2_u_cpu, D)
    results["f_mem_u8"] = float(((d1_u_cpu / (d2_u_cpu + 1e-12)) < args.k).float().mean().item())

    exact_gen_unique = sorted(set(exact_gen_indices))
    exact_match_count = len(exact_gen_unique)
    exact_match_rate = exact_match_count / float(args.nsamples)

    results.update({
        "exact_match_count": int(exact_match_count),
        "exact_match_rate": float(exact_match_rate),
        "num_train_images_hit": int((exact_hits_train > 0).sum()),
        "max_hits_on_a_single_train_image": int(exact_hits_train.max()) if exact_hits_train.size > 0 else 0,
    })

    if len(exact_pairs_for_png) > 0:
        out_png = os.path.join(run_dir, "mem_exact_pairs.png")
        save_pairs_grid(exact_gen_u8_keep, train_u8.cpu(), exact_pairs_for_png, out_png, nshow=args.save_pairs)
        results["pairs_png"] = out_png
    else:
        results["pairs_png"] = None

    if args.save_topk_nn > 0 and len(top_nn) > 0:
        nn_png = os.path.join(run_dir, "top_nn_pairs.png")
        nn_gen_keep = [x[1] for x in top_nn]  
        nn_pairs = [(i, x[2]) for i, x in enumerate(top_nn)] 
        save_pairs_grid(nn_gen_keep, train_u8.cpu(), nn_pairs, nn_png, nshow=min(args.save_topk_nn, len(top_nn)))
        results["top_nn_png"] = nn_png
        results["top_nn_d1_u8"] = [float(x[0]) for x in top_nn]
    else:
        results["top_nn_png"] = None

    print(json.dumps(results, indent=2))
    with open(os.path.join(run_dir, args.out_json), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Wrote {args.out_json} (and optional pngs)")

if __name__ == "__main__":
    main()
