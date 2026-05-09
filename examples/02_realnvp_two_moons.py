"""
RealNVP coupling-layer flow on the two-moons dataset.

Trains an 8-layer affine RealNVP (zuko.flows.NICE with default affine coupling
plus alternating mask) by maximum likelihood. After training, snapshots the
marginal sample distribution after k = 0, 2, 4, 6, 8 coupling layers in five
panels — illustrating how the standard-Gaussian base distribution gets folded
into the two-moons shape one coupling layer at a time.

Output: figures/realnvp_two_moons.{pdf,png} (5-panel ladder)
        figures/realnvp_layer_{0,2,4,6,8}.{pdf,png} (individual square panels)
"""
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.optim.lr_scheduler import CosineAnnealingLR

import zuko

PRIMARY = "#1f3a93"
GRID_GRAY = "#727176"

SEED = 0
TRANSFORMS = 8
HIDDEN = (128, 128, 128)
N_STEPS = 8_000
BATCH = 256
LR = 1e-3
LR_MIN = 1e-5
NOISE = 0.05
N_DATA = 4096
N_PTS_PLOT = 1500
SNAP_LAYERS = (0, 2, 4, 6, 8)
XLIM = (-2.5, 2.5)
YLIM = (-1.5, 1.5)


def make_dataset(rng: np.random.Generator) -> torch.Tensor:
    x_np, _ = make_moons(N_DATA, noise=NOISE, random_state=int(rng.integers(0, 2**31 - 1)))
    x_np = x_np - x_np.mean(0)
    x_np = x_np / x_np.std(0)
    return torch.tensor(x_np, dtype=torch.float32)


def train(flow: zuko.flows.Flow, x: torch.Tensor, device: torch.device) -> None:
    flow.to(device)
    x = x.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)
    n = x.shape[0]
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=device)
        xb = x[idx]
        loss = -flow().log_prob(xb).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 1_000 == 0:
            print(f"  step {step + 1:>5d} / {N_STEPS}  NLL={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")


@torch.no_grad()
def snapshot_layers(flow: zuko.flows.Flow, n_pts: int, device: torch.device) -> List[torch.Tensor]:
    """Sample n_pts from base, apply 0, 2, 4, 6, 8 coupling layers, return tensors.

    The flow's composed transform applies T_0, T_1, ..., T_{K-1} forward (x -> z).
    Going z -> x requires applying the inverses in REVERSE order:
        x = T_0^{-1}(T_1^{-1}(... T_{K-1}^{-1}(z))).
    "After k coupling layers" = applied k inverse transforms starting from z.
    """
    flow.eval()
    base = flow.base()
    z = base.sample((n_pts,)).to(device)
    snapshots: List[torch.Tensor] = []
    if 0 in SNAP_LAYERS:
        snapshots.append(z.detach().cpu().clone())
    x = z.clone()
    transforms_reversed = list(reversed(list(flow.transform.transforms)))
    for k, lazy_t in enumerate(transforms_reversed, start=1):
        t = lazy_t(None)
        x = t.inv(x)
        if k in SNAP_LAYERS:
            snapshots.append(x.detach().cpu().clone())
    return snapshots


def render(snapshots: List[torch.Tensor], out_basename: str) -> None:
    fig, axes = plt.subplots(1, len(SNAP_LAYERS), figsize=(15, 3.0), sharex=True, sharey=True)
    for ax, pts, k in zip(axes, snapshots, SNAP_LAYERS):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=PRIMARY, alpha=0.6, edgecolors="none")
        ax.set_title(f"Layer {k}", fontsize=14, color=GRID_GRAY)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_GRAY)
            spine.set_linewidth(0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_basename) or ".", exist_ok=True)
    fig.savefig(out_basename + ".pdf", dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_basename + ".png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {out_basename}.pdf and {out_basename}.png")


def render_individual(snapshots: List[torch.Tensor], out_dir: str) -> None:
    """Save one square, untitled density PDF per snapshot."""
    os.makedirs(out_dir, exist_ok=True)
    for pts, k in zip(snapshots, SNAP_LAYERS):
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=PRIMARY, alpha=0.6, edgecolors="none")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_GRAY)
            spine.set_linewidth(0.6)
        path_pdf = os.path.join(out_dir, f"realnvp_layer_{k}.pdf")
        path_png = os.path.join(out_dir, f"realnvp_layer_{k}.png")
        fig.savefig(path_pdf, dpi=300, bbox_inches="tight", transparent=True)
        fig.savefig(path_png, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"  wrote {path_pdf}")


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    x = make_dataset(rng)
    flow = zuko.flows.NICE(
        features=2,
        transforms=TRANSFORMS,
        hidden_features=HIDDEN,
        activation=torch.nn.SiLU,
    )
    train(flow, x, device)
    snapshots = snapshot_layers(flow, N_PTS_PLOT, device)
    fig_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    render(snapshots, os.path.join(fig_dir, "realnvp_two_moons"))
    render_individual(snapshots, fig_dir)


if __name__ == "__main__":
    main()
