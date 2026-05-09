"""
Flow Matching on the two-moons dataset.

Trains a small MLP velocity field u_theta(t, x) with the OT-CFM objective on
the linear interpolant. To get visually clean intermediate marginals, we use
mini-batch optimal-transport pairing between source and target samples within
each mini-batch (Hungarian algorithm). Without this step, FM learns a folding
flow because E[x_1 - x_0 | x_t] points toward the data centroid first.

Architecture: MLP, hidden=128, depth=4, SiLU activations.
Objective:    L = E_{t, x_0, x_1} || u_theta(t, x_t) - (x_1 - x_0) ||^2
              with x_t = (1 - t) x_0 + t x_1, t ~ Uniform[0, 1],
              x_0 ~ N(0, I), x_1 ~ make_moons.
Sampling:     Heun 2nd-order ODE integrator, 100 sub-steps, t in [0, 1].

Output: figures/fm_two_moons.{pdf,png}
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_moons
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

PRIMARY = "#1f3a93"
GRID_GRAY = "#727176"

SEED = 0
HIDDEN = 128
DEPTH = 4
N_STEPS = 30_000
BATCH = 256
LR = 2e-3
LR_MIN = 1e-5
NOISE = 0.05
N_PTS_PLOT = 1500
N_INT = 100
SNAP_TS = (0.0, 0.25, 0.50, 0.75, 1.00)
XLIM = (-2.5, 2.5)
YLIM = (-1.5, 1.5)


class FlowMLP(nn.Module):
    """MLP velocity field u_theta(t, x) for 2D flow matching."""

    def __init__(self, dim: int = 2, hidden: int = HIDDEN, depth: int = DEPTH) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self.net(torch.cat((t, x), dim=-1))

    def heun_step(self, x: Tensor, t0: Tensor, t1: Tensor) -> Tensor:
        """Heun's 2nd-order ODE step: x_{t1} = x + dt * 0.5 * (k1 + k2)."""
        dt = (t1 - t0).view(1, 1)
        t0b = t0.view(1, 1).expand(x.shape[0], 1).to(x.device)
        t1b = t1.view(1, 1).expand(x.shape[0], 1).to(x.device)
        k1 = self(t=t0b, x=x)
        k2 = self(t=t1b, x=x + dt * k1)
        return x + 0.5 * dt * (k1 + k2)


def train(device: torch.device) -> FlowMLP:
    flow = FlowMLP().to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    for step in range(N_STEPS):
        x1_np, _ = make_moons(BATCH, noise=NOISE, random_state=rng.integers(0, 2**31 - 1))
        x1 = torch.tensor(x1_np, dtype=torch.float32, device=device)
        x0 = torch.randn_like(x1)
        # Mini-batch OT pairing via Hungarian algorithm.
        with torch.no_grad():
            cost = torch.cdist(x0, x1).cpu().numpy()
            _, col = linear_sum_assignment(cost)
        x1 = x1[torch.tensor(col, device=device)]
        t = torch.rand(BATCH, 1, device=device)
        x_t = (1 - t) * x0 + t * x1
        target = x1 - x0
        opt.zero_grad()
        loss = loss_fn(flow(t=t, x=x_t), target)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 5_000 == 0:
            print(f"  step {step + 1:>6d} / {N_STEPS}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
    return flow


def integrate(flow: FlowMLP, x0: Tensor, n_int: int = N_INT) -> dict[float, Tensor]:
    """Heun-integrate from t=0 to t=1; return snapshots at the requested times."""
    flow.eval()
    ts = torch.linspace(0, 1, n_int + 1, device=x0.device)
    snap_set = set(SNAP_TS)
    snapshots: dict[float, Tensor] = {}
    if 0.0 in snap_set:
        snapshots[0.0] = x0.detach().cpu().clone()
    x = x0.clone()
    with torch.no_grad():
        for i in range(n_int):
            x = flow.heun_step(x, ts[i], ts[i + 1])
            t_now = float(ts[i + 1].item())
            for t_snap in SNAP_TS:
                if abs(t_now - t_snap) < 1e-9 and t_snap not in snapshots:
                    snapshots[t_snap] = x.detach().cpu().clone()
    return snapshots


def render(snapshots: dict[float, Tensor], out_basename: str) -> None:
    fig, axes = plt.subplots(1, len(SNAP_TS), figsize=(15, 3.0), sharex=True, sharey=True)
    for ax, t_lab in zip(axes, SNAP_TS):
        pts = snapshots[t_lab]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, c=PRIMARY, alpha=0.7, edgecolors="none")
        ax.set_title(f"$t = {t_lab:.2f}$", fontsize=14, color=GRID_GRAY)
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


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    flow = train(device)
    x0 = torch.randn(N_PTS_PLOT, 2, device=device)
    snapshots = integrate(flow, x0)
    missing = [t for t in SNAP_TS if t not in snapshots]
    if missing:
        raise RuntimeError(f"missing snapshots for t in {missing}")
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "fm_two_moons")
    render(snapshots, os.path.normpath(out))


if __name__ == "__main__":
    main()
