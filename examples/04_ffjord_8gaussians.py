"""
FFJORD (Hutchinson stochastic-trace CNF) on a mixture of 8 Gaussians.

Same target as 03_cnf_8gaussians.py (ring of 8 modes), but the divergence in
the log-likelihood is estimated with the Hutchinson trace estimator
(exact=False) instead of computed exactly. This matches Grathwohl et al.'s
FFJORD (2019) and is the scalable approach in higher dimensions, at the cost
of a noisier gradient. We compensate with more steps (14k vs 8k) and a larger
batch (2048 vs 1024).

Renders sample scatters at 8 times t in {0, 1/7, ..., 6/7, 1}, in two strips:
  Row A: t = 0, 1/7, 2/7, 3/7
  Row B: t = 4/7, 5/7, 6/7, 1

Output: figures/ffjord_8gaussians_a.{pdf,png}
        figures/ffjord_8gaussians_b.{pdf,png}
        figures/ffjord_8gaussians_model.pt
        figures/ffjord_8gaussians_losses.npy
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
ACCENT = "#d68749"
GRID_GRAY = "#727176"

SEED = 0
N_DATA = 8192
N_STEPS = 14_000
BATCH = 2048
LR = 1e-3
LR_MIN = 1e-5
HIDDEN = (256, 256, 256, 256)
N_MODES = 8
RADIUS = 2.0
MODE_STD = 0.15

N_TIME_SAMPLES = 1500
PLOT_RANGE = 3.6
BLOB_GRID_RES = 240
TIME_SLICES_ROW1 = (0.0, 1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0)
LABELS_ROW1 = (r"$t = 0$", r"$t = \frac{1}{7}$", r"$t = \frac{2}{7}$", r"$t = \frac{3}{7}$")
TIME_SLICES_ROW2 = (4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0, 1.0)
LABELS_ROW2 = (r"$t = \frac{4}{7}$", r"$t = \frac{5}{7}$", r"$t = \frac{6}{7}$", r"$t = 1$")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")
OUT_BASENAME = "ffjord_8gaussians"
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
        exact=False,                # Hutchinson stochastic trace -> FFJORD
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


def target_blobs(ax: plt.Axes) -> None:
    """Soft light-gray density heatmap of the 8 target modes."""
    axis = np.linspace(-PLOT_RANGE, PLOT_RANGE, BLOB_GRID_RES)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    centers = mode_centers()
    tgt_var = MODE_STD ** 2
    tgt = np.zeros_like(xx)
    for cx, cy in centers:
        tgt += np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / tgt_var)
    tgt /= tgt.max() + 1e-12

    rgba = np.zeros((BLOB_GRID_RES, BLOB_GRID_RES, 4))
    rgba[..., 0:3] = 0.45
    rgba[..., 3] = 0.55 * tgt
    ax.imshow(
        rgba,
        extent=(-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE),
        origin="lower", aspect="equal",
    )


def _compute_slices(flow: CNF, device: torch.device,
                    times: tuple[float, ...]) -> dict[float, np.ndarray]:
    """For each time tau in [0, 1] (tau=0 noise, tau=1 data), integrate the
    same source batch via zuko's adaptive solver.

    zuko's stored f maps data -> base, so internally t=0 is data, t=1 is noise.
    Sampling = transform.inv integrates backward from zuko_t=1 down to zuko_t=0.
        tau = 0   <->  z0 ~ p_init  (noise)        <->  zuko_t = 1
        tau = 1   <->  data samples                <->  zuko_t = 0
        tau in (0,1) -> integrate inv from zuko_t=1 down to zuko_t=(1 - tau).
    """
    from zuko.transforms import FreeFormJacobianTransform

    flow.eval()
    torch.manual_seed(SEED + 1)
    z0 = torch.randn(N_TIME_SAMPLES, 2, device=device)

    built = flow()
    full_t = built.transform

    out: dict[float, np.ndarray] = {}
    with torch.no_grad():
        for tau in times:
            if tau == 0.0:
                out[tau] = z0.cpu().numpy()
                continue
            zuko_t1 = 1.0 - float(tau)
            partial = FreeFormJacobianTransform(
                f=full_t.f,
                t0=full_t.t1,
                t1=torch.tensor(zuko_t1, device=device),
                phi=full_t.phi,
                atol=full_t.atol,
                rtol=full_t.rtol,
                exact=full_t.exact,
            )
            x_tau = partial(z0)
            out[tau] = x_tau.cpu().numpy()
    return out


def _render_row(samples: dict[float, np.ndarray],
                slices: tuple[float, ...], labels: tuple[str, ...],
                out_path_pdf: str, out_path_png: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.04, wspace=0.06)

    for ax, t, label in zip(axes, slices, labels):
        pts = samples[t]
        target_blobs(ax)
        ax.scatter(pts[:, 0], pts[:, 1], s=5, color=ACCENT, alpha=0.55, edgecolors="none")
        ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
        ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
        ax.set_aspect("equal")
        ax.set_title(label, color=PRIMARY, fontweight="bold", fontsize=16)
        _strip(ax)

    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def render(flow: CNF, device: torch.device, out_path_pdf: str, out_path_png: str) -> None:
    """Two row figures (A: t in {0,1/7,2/7,3/7}; B: t in {4/7,5/7,6/7,1})."""
    all_times = TIME_SLICES_ROW1 + TIME_SLICES_ROW2
    samples = _compute_slices(flow, device, all_times)

    base_pdf = out_path_pdf[:-4]
    base_png = out_path_png[:-4]
    pdf_a = f"{base_pdf}_a.pdf"
    png_a = f"{base_png}_a.png"
    pdf_b = f"{base_pdf}_b.pdf"
    png_b = f"{base_png}_b.png"

    _render_row(samples, TIME_SLICES_ROW1, LABELS_ROW1, pdf_a, png_a)
    _render_row(samples, TIME_SLICES_ROW2, LABELS_ROW2, pdf_b, png_b)


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
        flow.load_state_dict(
            torch.load(CKPT_PATH, map_location=device, weights_only=True),
            strict=False,
        )
    else:
        print("[1/3] Building dataset...", flush=True)
        x = make_dataset(rng)
        print(f"[2/3] Training FFJORD (Hutchinson, hidden={HIDDEN}, "
              f"{N_STEPS} steps, batch={BATCH})...", flush=True)
        result = train(flow, x, device)
        print(f"  final loss (last 50 mean) = {float(result.losses[-50:].mean()):.4f}",
              flush=True)
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
