"""
Flow Matching on the two-moons dataset, using Meta's `flow_matching` library.

Companion to `01_fm_two_moons.py` (from-scratch). Same training data,
hyperparameters, and figure layout — but the linear interpolant, target
velocity, and ODE sampling are delegated to the library:

    - flow_matching.path.CondOTProbPath
        Provides the OT/conditional probability path
            x_t = alpha_t * x_1 + sigma_t * x_0  with  alpha_t = t, sigma_t = 1-t.
        path.sample(x_0, x_1, t).x_t  is the interpolant.
        path.sample(x_0, x_1, t).dx_t is the conditional target velocity (= x_1 - x_0).

    - flow_matching.utils.ModelWrapper
        Adapter so the user's MLP plugs into ODESolver's expected (x, t) signature.

    - flow_matching.solver.ODESolver
        Adaptive ODE integration with torchdiffeq under the hood (here: dopri5).

The mini-batch OT pairing trick (Hungarian algorithm via scipy) is not built
into the library, so we keep that step manual — same as in the from-scratch
version.

Output: figures/fm_two_moons_meta.{pdf,png}

Note: `flow_matching` is licensed CC-BY-NC. Importing the library is fine for
research / teaching use; commercial use is restricted.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_moons
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

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
SNAP_TS = (0.0, 0.25, 0.50, 0.75, 1.00)
XLIM = (-2.5, 2.5)
YLIM = (-1.5, 1.5)


class MLP(nn.Module):
    """Plain MLP velocity field f(x, t): x in R^d, t in R -> R^d."""

    def __init__(self, dim: int = 2, hidden: int = HIDDEN, depth: int = DEPTH) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # t: (B,) or scalar; x: (B, d)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=-1))


class VelocityWrapper(ModelWrapper):
    """Adapter so ODESolver can call our MLP with the (x, t, **extras) signature."""

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x=x, t=t)


def train(device: torch.device) -> MLP:
    model = MLP().to(device)
    path = CondOTProbPath()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    sched = CosineAnnealingLR(opt, T_max=N_STEPS, eta_min=LR_MIN)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(SEED)
    for step in range(N_STEPS):
        x1_np, _ = make_moons(BATCH, noise=NOISE, random_state=rng.integers(0, 2**31 - 1))
        x1 = torch.tensor(x1_np, dtype=torch.float32, device=device)
        x0 = torch.randn_like(x1)
        # Mini-batch OT pairing (Hungarian) — kept manual; not in the library.
        with torch.no_grad():
            cost = torch.cdist(x0, x1).cpu().numpy()
            _, col = linear_sum_assignment(cost)
        x1 = x1[torch.tensor(col, device=device)]
        t = torch.rand(BATCH, device=device)

        # The library handles the interpolant + target velocity:
        sample = path.sample(x_0=x0, x_1=x1, t=t)
        u_pred = model(sample.x_t, t)

        opt.zero_grad()
        loss = loss_fn(u_pred, sample.dx_t)
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 5_000 == 0:
            print(f"  step {step + 1:>6d} / {N_STEPS}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
    return model


def integrate(model: MLP, x0: Tensor, device: torch.device) -> dict[float, Tensor]:
    """Sample at the SNAP_TS times via ODESolver (dopri5)."""
    model.eval()
    wrapped = VelocityWrapper(model).to(device)
    solver = ODESolver(velocity_model=wrapped)
    time_grid = torch.tensor(list(SNAP_TS), device=device, dtype=x0.dtype)
    with torch.no_grad():
        traj = solver.sample(
            x_init=x0,
            step_size=None,
            method="dopri5",
            atol=1e-5,
            rtol=1e-5,
            time_grid=time_grid,
            return_intermediates=True,
        )
    return {float(t): traj[i].detach().cpu().clone() for i, t in enumerate(SNAP_TS)}


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
    model = train(device)
    x0 = torch.randn(N_PTS_PLOT, 2, device=device)
    snapshots = integrate(model, x0, device)
    out = os.path.join(os.path.dirname(__file__), "..", "figures", "fm_two_moons_meta")
    render(snapshots, os.path.normpath(out))


if __name__ == "__main__":
    main()
