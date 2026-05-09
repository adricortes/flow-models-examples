# Flow-Based Generative Models — Example Codes

Companion code for the talk *From Normalizing Flows to Flow Matching* (DMA 2026).
Each script in `examples/` is a self-contained, single-file PyTorch implementation
of one figure from the talk. The goal is pedagogical clarity over performance:
small datasets (2D toy distributions), small models (MLPs), short training
runs, and one figure per script.

---

## What's here

| Script | Method | Dataset | Estimator / objective |
|---|---|---|---|
| `examples/01_fm_two_moons.py` | Flow Matching (OT-CFM) | `make_moons` | Mini-batch OT pairing on linear interpolant |
| `examples/02_realnvp_two_moons.py` | RealNVP (coupling-layer NF) | `make_moons` | Maximum likelihood (exact) |
| `examples/03_cnf_8gaussians.py` | Continuous Normalizing Flow | 8 Gaussians on a circle | MLE with **exact** trace divergence |
| `examples/04_ffjord_8gaussians.py` | FFJORD | 8 Gaussians on a circle | MLE with **Hutchinson** stochastic trace |
| `examples/05_fm_compare.py` | OT-CFM vs Gaussian-VP-FM | 8 Gaussians on a circle | Two FM objectives, side-by-side |

Each script saves figures (`.pdf` + `.png`), training losses (`.npy`), and a
model checkpoint (`.pt`) to `figures/`.

---

## Setup

These examples target Python 3.10+ and PyTorch 2.x. Two of the examples use
[`zuko`](https://github.com/probabilists/zuko) for normalizing-flow primitives,
and one uses [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) for
adaptive ODE integration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A GPU helps (especially for `04_ffjord_8gaussians.py` and `05_fm_compare.py`)
but is not required — every script auto-falls-back to CPU. On CPU, expect
each script to take a few minutes (FM two-moons, RealNVP) to ~30 minutes
(FFJORD).

---

## Running

From the repo root:

```bash
python examples/01_fm_two_moons.py
python examples/02_realnvp_two_moons.py
python examples/03_cnf_8gaussians.py
python examples/04_ffjord_8gaussians.py
python examples/05_fm_compare.py
```

The CNF and FFJORD scripts also accept `--render-only` to skip training and
load the saved checkpoint:

```bash
python examples/03_cnf_8gaussians.py --render-only
```

---

## Notation conventions

Across all scripts and the talk:

- $z_0 \sim p_{\text{init}} = \mathcal{N}(\mathbf{0}, I)$ — source / noise sample
- $x_1 \sim p_{\text{data}}$ — data sample
- $t \in [0, 1]$ — time / interpolation variable, with $t = 0$ noise and $t = 1$ data
- $u_\theta(t, x)$ — learned velocity field (the flow-matching network)
- $p_t$ — marginal probability path at time $t$

For RealNVP only (Part 2 of the talk), bold $\mathbf{t}$ denotes the
*translation vector* in the affine coupling, not time.

---

## Reduce training time for laptops

Each script's hyperparameters are at the top of the file (`SEED`, `N_STEPS`,
`BATCH`, `HIDDEN`, ...). For a CPU-only test run, halve `N_STEPS` and `BATCH`
— results will look noisier but the qualitative behavior is preserved.

---

## Credits

The implementations build on patterns from:

- Lipman, Chen, Ben-Hamu, Nickel, Le, *Flow Matching for Generative Modeling* (ICLR 2023) — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, *FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models* (ICLR 2019) — [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
- Dinh, Sohl-Dickstein, Bengio, *Density Estimation using Real NVP* (ICLR 2017) — [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations* (NeurIPS 2018) — [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- The [`flow_matching`](https://github.com/facebookresearch/flow_matching) reference implementation by Meta AI.
- The [`zuko`](https://github.com/probabilists/zuko) probabilistic-flow library.

---

## License

MIT. See `LICENSE`.
