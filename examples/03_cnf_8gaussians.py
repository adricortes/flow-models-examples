"""
Continuous Normalizing Flow (exact-trace) on a mixture of 8 Gaussians.

Trains a 2D CNF (zuko.flows.CNF with exact=True, i.e., exact divergence via
autograd batch-Jacobian, suitable in low D) by maximum likelihood on data drawn
from a ring of 8 Gaussian modes (radius 2, std 0.15).

Renders two panels:
  (a) sample scatters at four times t in {0, 0.33, 0.67, 1}.
  (b) trajectories of 50 sample particles from t=0 to t=1.

Output: figures/cnf_8gaussians.{pdf,png}
        figures/cnf_8gaussians_model.pt  (checkpoint)
        figures/cnf_8gaussians_losses.npy
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from zuko.flows import CNF

PRIMARY = "#1f3a93"
GRID_GRAY = "#727176"

SEED = 0
N_DATA = 8192
N_STEPS = 8_000
BATCH = 1024
LR = 1e-3
LR_MIN = 1e-5
HIDDEN = (256, 256, 256, 256)
N_MODES = 8
RADIUS = 2.0
MODE_STD = 0.15

N_TRAJ = 50
N_SAMPLES_PER_T = 600
TRAJ_STEPS = 60
PLOT_RANGE = 3.6
TIME_SLICES = (0.0, 0.33, 0.67, 1.0)
SLICE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
BLOB_GRID_RES = 240

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")
OUT_BASENAME = "cnf_8gaussians"
CKPT_PATH = os.path.join(OUT_DIR, f"{OUT_BASENAME}_model.pt")


@dataclass
class TrainResult:
    flow: CNF
    losses: np.ndarray


def mode_centers() -> np.ndarray:
    angles = 2 * math.pi * np.arange(N_MODES) / N_MODES
    return RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def make_dataset(rng: np.random.Generator) -> torch.Tensor:
    centers = mode_centers()
    assignments = rng.integers(0, N_MODES, size=N_DATA)
    means = centers[assignments]
    noise = rng.normal(scale=MODE_STD, size=(N_DATA, 2))
    return torch.tensor(means + noise, dtype=torch.float32)


def build_flow(device: torch.device) -> CNF:
    return CNF(
        features=2,
        hidden_features=HIDDEN,
        exact=True,
        atol=1e-5,
        rtol=1e-5,
    ).to(device)


def train(flow: CNF, x: torch.Tensor, device: torch.device) -> TrainResult:
    x = x.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)

    losses = np.empty(N_STEPS, dtype=np.float64)
    n = x.shape[0]
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=device)
        xb = x[idx]
        loss = -flow().log_prob(xb).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses[step] = float(loss.detach().cpu())
        if (step + 1) % 200 == 0:
            print(f"  step {step + 1:5d} / {N_STEPS}   loss = {losses[step]:.4f}", flush=True)

    return TrainResult(flow=flow, losses=losses)


def velocity(flow: CNF, t: float, x: torch.Tensor) -> torch.Tensor:
    transform = flow.transform
    t_tensor = torch.full(x.shape[:-1], float(t), dtype=x.dtype, device=x.device)
    return transform.f(t_tensor, x)


def integrate_until(flow: CNF, z0: torch.Tensor, t_end: float, n_steps: int) -> torch.Tensor:
    if t_end == 0.0:
        return z0
    dt = t_end / n_steps
    z = z0
    for k in range(n_steps):
        t = k * dt
        k1 = velocity(flow, t,            z)
        k2 = velocity(flow, t + dt / 2,   z + 0.5 * dt * k1)
        k3 = velocity(flow, t + dt / 2,   z + 0.5 * dt * k2)
        k4 = velocity(flow, t + dt,       z +       dt * k3)
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z


def integrate_full_trajectory(flow: CNF, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
    dt = 1.0 / n_steps
    states = torch.empty(n_steps + 1, *z0.shape, dtype=z0.dtype, device=z0.device)
    states[0] = z0
    z = z0
    for k in range(n_steps):
        t = k * dt
        k1 = velocity(flow, t,            z)
        k2 = velocity(flow, t + dt / 2,   z + 0.5 * dt * k1)
        k3 = velocity(flow, t + dt / 2,   z + 0.5 * dt * k2)
        k4 = velocity(flow, t + dt,       z +       dt * k3)
        z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[k + 1] = z
    return states


def background_blobs(ax: plt.Axes) -> None:
    axis = np.linspace(-PLOT_RANGE, PLOT_RANGE, BLOB_GRID_RES)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid = np.stack([xx, yy], axis=-1)
    src_var = 1.0
    src = np.exp(-0.5 * (grid[..., 0] ** 2 + grid[..., 1] ** 2) / src_var)
    centers = mode_centers()
    tgt_var = MODE_STD ** 2
    tgt = np.zeros_like(src)
    for cx, cy in centers:
        tgt += np.exp(-0.5 * ((grid[..., 0] - cx) ** 2 + (grid[..., 1] - cy) ** 2) / tgt_var)
    tgt /= tgt.max() + 1e-12
    src /= src.max() + 1e-12
    ax.imshow(
        np.zeros_like(src),
        extent=(-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE),
        origin="lower", cmap="gray", vmin=0, vmax=1, alpha=0,
    )
    src_alpha = 0.18 * src
    ax.imshow(
        np.stack([src_alpha, np.zeros_like(src), np.zeros_like(src), src_alpha], axis=-1),
        extent=(-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE),
        origin="lower", aspect="equal",
    )
    tgt_alpha = 0.30 * tgt
    ax.imshow(
        np.stack([np.zeros_like(tgt), np.zeros_like(tgt) + 0.4 * tgt, 0.7 * tgt + 0.3, tgt_alpha], axis=-1),
        extent=(-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE),
        origin="lower", aspect="equal",
    )


def render(flow: CNF, device: torch.device, out_path_pdf: str, out_path_png: str) -> None:
    flow.eval()
    torch.manual_seed(SEED + 1)
    z0 = torch.randn(N_SAMPLES_PER_T, 2, device=device)
    z0_traj = torch.randn(N_TRAJ, 2, device=device)

    sample_slices: dict[float, np.ndarray] = {}
    with torch.no_grad():
        for t in TIME_SLICES:
            z_t = integrate_until(flow, z0, t, n_steps=max(int(t * TRAJ_STEPS), 1) if t > 0 else 1)
            sample_slices[t] = z_t.cpu().numpy()

    with torch.no_grad():
        traj = integrate_full_trajectory(flow, z0_traj, TRAJ_STEPS).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.04, wspace=0.10)

    ax_a = axes[0]
    background_blobs(ax_a)
    for t, color in zip(TIME_SLICES, SLICE_COLORS):
        pts = sample_slices[t]
        ax_a.scatter(
            pts[:, 0], pts[:, 1],
            s=8, color=color, alpha=0.55,
            edgecolors="none", label=f"$t = {t:.2f}$",
        )
    ax_a.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax_a.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax_a.set_aspect("equal")
    ax_a.set_title(r"Samples from the learned ODE", color=PRIMARY, fontweight="bold", fontsize=15)
    leg = ax_a.legend(loc="upper right", frameon=False, fontsize=10, markerscale=1.5,
                      handletextpad=0.3, labelspacing=0.2)
    for txt in leg.get_texts():
        txt.set_color(GRID_GRAY)
    _strip(ax_a)

    ax_b = axes[1]
    background_blobs(ax_b)
    for i in range(N_TRAJ):
        ax_b.plot(traj[:, i, 0], traj[:, i, 1], color="black", lw=0.7, alpha=0.55)
    ax_b.scatter(traj[0, :, 0], traj[0, :, 1], s=10, color="black", zorder=3)
    ax_b.scatter(traj[-1, :, 0], traj[-1, :, 1], s=10, color="black", zorder=3)
    ax_b.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax_b.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax_b.set_aspect("equal")
    ax_b.set_title(r"Trajectories of the learned ODE", color=PRIMARY, fontweight="bold", fontsize=15)
    _strip(ax_b)

    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _strip(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(GRID_GRAY)
        ax.spines[side].set_linewidth(0.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--render-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    flow = build_flow(device)

    if args.render_only:
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(CKPT_PATH)
        flow.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    else:
        print("[1/3] Building dataset...", flush=True)
        x = make_dataset(rng)
        print(f"[2/3] Training (exact, hidden={HIDDEN}, {N_STEPS} steps, batch={BATCH})...", flush=True)
        result = train(flow, x, device)
        print(f"  final loss (last 50 mean) = {float(result.losses[-50:].mean()):.4f}", flush=True)
        torch.save(flow.state_dict(), CKPT_PATH)
        print(f"  wrote checkpoint {CKPT_PATH}", flush=True)
        np.save(os.path.join(OUT_DIR, f"{OUT_BASENAME}_losses.npy"), result.losses)

    print("[render] Rendering...", flush=True)
    pdf = os.path.join(OUT_DIR, f"{OUT_BASENAME}.pdf")
    png = os.path.join(OUT_DIR, f"{OUT_BASENAME}.png")
    render(flow, device, pdf, png)
    print(f"  wrote {pdf}", flush=True)
    print(f"  wrote {png}", flush=True)


if __name__ == "__main__":
    main()
