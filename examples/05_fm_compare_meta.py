"""
Flow Matching: OT-CFM vs Gaussian-VP-FM, using Meta's `flow_matching` library.

Companion to `05_fm_compare.py` (from-scratch). Same target (8 Gaussians on a
circle), same hyperparameters, same figure layout. The two probability paths
are obtained from the library:

  1. OT-CFM:
        path = CondOTProbPath()
     which is shorthand for
        AffineProbPath(scheduler=CondOTScheduler())  -> alpha_t=t, sigma_t=1-t
     i.e., x_t = t * x_1 + (1 - t) * x_0  with target dx_t = x_1 - x_0.

  2. Gaussian-VP-FM:
        path = AffineProbPath(scheduler=VPScheduler(beta_min=0.1, beta_max=20.0))
     The scheduler computes alpha_t, sigma_t, and their derivatives; the path's
     `sample(x_0, x_1, t)` returns x_t and dx_t (target conditional velocity)
     in closed form.

Sampling: ODESolver(velocity_model=ModelWrapper(model)).sample(...)
          with method="dopri5" and an 8-point time_grid.

Outputs (in figures/):
    fm_compare_meta_ot_a.{pdf,png},     fm_compare_meta_ot_b.{pdf,png}
    fm_compare_meta_gauss_a.{pdf,png},  fm_compare_meta_gauss_b.{pdf,png}
    fm_compare_meta_ot_model.pt,        fm_compare_meta_gauss_model.pt
    fm_compare_meta_ot_losses.npy,      fm_compare_meta_gauss_losses.npy

Note: `flow_matching` is licensed CC-BY-NC. Importing the library is fine for
research / teaching use; commercial use is restricted.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from flow_matching.path import AffineProbPath, CondOTProbPath, ProbPath
from flow_matching.path.scheduler import VPScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

PRIMARY = "#1f3a93"
GRID_GRAY = "#727176"

SEED = 0
N_DATA = 8192
N_STEPS = 8_000
BATCH = 4096
LR = 1e-3
LR_MIN = 1e-5
HIDDEN = 256
DEPTH = 4

N_MODES = 8
RADIUS = 2.0
MODE_STD = 0.15

N_SAMPLES_PER_T = 2000
PLOT_RANGE = 3.6

TIMES_ALL = tuple(k / 7.0 for k in range(8))
TIMES_A = TIMES_ALL[:4]
TIMES_B = TIMES_ALL[4:]

PANEL_COLORS_A = ("#1f77b4", "#3a8fbe", "#5fa8c8", "#7fc1d2")
PANEL_COLORS_B = ("#d4a52a", "#e08820", "#dd5c1c", "#d62728")

BETA_MIN = 0.1
BETA_MAX = 20.0

ODE_RTOL = 1e-5
ODE_ATOL = 1e-5
ODE_METHOD = "dopri5"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")
OT_CKPT = os.path.join(OUT_DIR, "fm_compare_meta_ot_model.pt")
GAUSS_CKPT = os.path.join(OUT_DIR, "fm_compare_meta_gauss_model.pt")


class VelocityMLP(nn.Module):
    """MLP velocity field (x, t) -> R^2."""

    def __init__(self, hidden: int = HIDDEN, depth: int = DEPTH) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=-1))


class VelocityWrapper(ModelWrapper):
    """Adapter for ODESolver."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        return self.model(x=x, t=t)


@dataclass
class TrainResult:
    model: VelocityMLP
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


def build_model(device: torch.device) -> VelocityMLP:
    return VelocityMLP(hidden=HIDDEN, depth=DEPTH).to(device)


def train(model: VelocityMLP, path: ProbPath, x: torch.Tensor,
          device: torch.device, tag: str) -> TrainResult:
    """Single training loop; the choice of `path` selects OT-CFM vs VP-Gaussian-FM."""
    x = x.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)

    losses = np.empty(N_STEPS, dtype=np.float64)
    n = x.shape[0]
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=device)
        x1 = x[idx]
        z0 = torch.randn(BATCH, 2, device=device)
        t = torch.rand(BATCH, device=device)

        sample = path.sample(x_0=z0, x_1=x1, t=t)
        u_pred = model(sample.x_t, t)
        loss = ((u_pred - sample.dx_t) ** 2).sum(dim=-1).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses[step] = float(loss.detach().cpu())
        if (step + 1) % 200 == 0:
            print(f"  [{tag}] step {step + 1:5d} / {N_STEPS}   loss = {losses[step]:.4f}",
                  flush=True)

    return TrainResult(model=model, losses=losses)


def sample_at_times(
    model: VelocityMLP,
    n_samples: int,
    times: tuple[float, ...],
    device: torch.device,
    seed_offset: int = 1,
) -> dict[float, np.ndarray]:
    """For each tau in `times`, integrate from 0 to tau via ODESolver/dopri5
    starting from a SHARED z_0 ~ N(0, I_2)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(SEED + seed_offset)
    z0 = torch.randn(n_samples, 2, device=device, generator=g)

    wrapped = VelocityWrapper(model).to(device)
    solver = ODESolver(velocity_model=wrapped)

    out: dict[float, np.ndarray] = {}
    with torch.no_grad():
        # Use a single call with return_intermediates to fetch all 8 times at once.
        time_grid = torch.tensor(list(times), device=device, dtype=z0.dtype)
        # ODESolver requires monotonic time grid; ensure it.
        traj = solver.sample(
            x_init=z0,
            step_size=None,
            method=ODE_METHOD,
            atol=ODE_ATOL,
            rtol=ODE_RTOL,
            time_grid=time_grid,
            return_intermediates=True,
        )
    for i, tau in enumerate(times):
        out[tau] = traj[i].detach().cpu().numpy()
    return out


def _strip(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(GRID_GRAY)
        ax.spines[side].set_linewidth(0.6)


def _fmt_t(t: float) -> str:
    if t == 0.0:
        return r"$t = 0$"
    if abs(t - 1.0) < 1e-9:
        return r"$t = 1$"
    k = int(round(t * 7))
    return rf"$t = {k}/7$"


def render_strip(
    samples_per_t: dict[float, np.ndarray],
    times: tuple[float, ...],
    out_pdf: str,
    out_png: str,
    palette: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.06)

    for ax, tau, color in zip(axes, times, palette):
        pts = samples_per_t[tau]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, color=color, alpha=0.55, edgecolors="none")
        ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
        ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
        ax.set_aspect("equal")
        ax.set_title(_fmt_t(tau), color=PRIMARY, fontweight="bold", fontsize=13)
        _strip(ax)

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _train_or_load(
    name: str,
    ckpt: str,
    path: ProbPath,
    data: torch.Tensor,
    device: torch.device,
) -> VelocityMLP:
    model = build_model(device)
    if os.path.exists(ckpt):
        print(f"[{name}] checkpoint exists at {ckpt} -> loading", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return model
    print(f"[{name}] training (hidden={HIDDEN}x{DEPTH}, {N_STEPS} steps, batch={BATCH})",
          flush=True)
    result = train(model, path, data, device, tag=name)
    print(f"[{name}] final loss (last 50 mean) = {float(result.losses[-50:].mean()):.4f}",
          flush=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[{name}] wrote checkpoint {ckpt}", flush=True)
    losses_path = ckpt.replace("_model.pt", "_losses.npy")
    np.save(losses_path, result.losses)
    print(f"[{name}] wrote losses {losses_path}", flush=True)
    return model


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("[data] building dataset (8 Gaussians, r=2, std=0.15)...", flush=True)
    data = make_dataset(rng)

    ot_path = CondOTProbPath()
    gauss_path = AffineProbPath(scheduler=VPScheduler(beta_min=BETA_MIN, beta_max=BETA_MAX))

    torch.manual_seed(SEED)
    ot_model = _train_or_load("OT", OT_CKPT, ot_path, data, device)

    torch.manual_seed(SEED)
    gauss_model = _train_or_load("GAUSS", GAUSS_CKPT, gauss_path, data, device)

    print(f"[sample] OT @ {len(TIMES_ALL)} times, n={N_SAMPLES_PER_T}", flush=True)
    ot_samples = sample_at_times(ot_model, N_SAMPLES_PER_T, TIMES_ALL, device, seed_offset=1)
    print(f"[sample] GAUSS @ {len(TIMES_ALL)} times, n={N_SAMPLES_PER_T}", flush=True)
    gauss_samples = sample_at_times(gauss_model, N_SAMPLES_PER_T, TIMES_ALL, device, seed_offset=2)

    targets = [
        ("fm_compare_meta_ot_a", ot_samples, TIMES_A, PANEL_COLORS_A),
        ("fm_compare_meta_ot_b", ot_samples, TIMES_B, PANEL_COLORS_B),
        ("fm_compare_meta_gauss_a", gauss_samples, TIMES_A, PANEL_COLORS_A),
        ("fm_compare_meta_gauss_b", gauss_samples, TIMES_B, PANEL_COLORS_B),
    ]
    for stem, samples, times, palette in targets:
        pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
        png = os.path.join(OUT_DIR, f"{stem}.png")
        render_strip(samples, times, pdf, png, palette)
        print(f"Saved: {pdf}", flush=True)
        print(f"Saved: {png}", flush=True)


if __name__ == "__main__":
    main()
