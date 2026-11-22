"""Fitting and plotting for scaling-law experiments (pure functions; see hedy.scaling for the runner)."""
import json
import math

import numpy as np
import torch


def load_runs(path, tag_prefix=None):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if tag_prefix is None or r.get("tag", "").startswith(tag_prefix)]


# ----------------------------------------------------------------------------- approach 2: IsoFLOP minima
def isoflop_minimum(N, L, deg=2):
    """Fit loss ~ quadratic in log N at one compute budget; return (N*, L*, bracketed, curvature).
    `bracketed` requires BOTH that the parabola's vertex lies inside the range of sizes tried AND that the
    best run actually observed is not an endpoint. (Three points always fit a parabola exactly, so a
    vertex inside the range is not enough: loss that simply rises from the smallest model has its true
    optimum off the grid, and the parabola's vertex is then an extrapolation artifact.)"""
    order = np.argsort(N)
    x, y = np.log(np.asarray(N, float))[order], np.asarray(L, float)[order]
    c2, c1, c0 = np.polyfit(x, y, 2)
    xs = -c1 / (2 * c2) if c2 > 0 else float("nan")
    best = int(np.argmin(y))
    inside = bool(c2 > 0 and x.min() < xs < x.max() and 0 < best < len(y) - 1)
    return float(np.exp(xs)), float(c0 - c1 ** 2 / (4 * c2)) if c2 > 0 else float("nan"), inside, float(c2)


def isoflop_table(runs):
    """{C: (N*, L*, D*, inside, n_points)} for every compute level in `runs`."""
    out = {}
    groups = {}
    for r in runs:
        key = r.get("tag", "").split(":")[-1] if r.get("tag", "").startswith("grid:") else f"{r['C']:.0e}"
        groups.setdefault(float(key), []).append(r)
    for C, rs in sorted(groups.items()):
        if len(rs) < 3:
            continue
        n, l, inside, _ = isoflop_minimum([r["N"] for r in rs], [r["val_loss"] for r in rs])
        out[C] = (n, l, C / (6 * n), inside, len(rs))
    return out


def powerlaw(x, y):
    """y = k * x^a by least squares in log-log. Returns (a, k, stderr_of_a)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    n = len(lx)
    se = math.sqrt(float(np.sum((ly - A @ np.array([a, b])) ** 2)) / max(n - 2, 1) / float(np.sum((lx - lx.mean()) ** 2))) if n > 2 else float("nan")
    return float(a), float(math.exp(b)), se


# ----------------------------------------------------------------------------- approach 3: parametric fit
def fit_parametric(N, D, L, starts=None, steps=250):
    """Fit L = E + A/N^alpha + B/D^beta (Hoffmann et al. 2022) minimizing a Huber loss on log L, with
    L-BFGS from a grid of starting points; returns the best fit as a dict. Parameterized through
    logsumexp over logA - alpha*logN, logB - beta*logD and logE so everything stays positive."""
    logN, logD = torch.log(torch.tensor(N, dtype=torch.float64)), torch.log(torch.tensor(D, dtype=torch.float64))
    target = torch.log(torch.tensor(L, dtype=torch.float64))
    best = None
    inits = starts or [(a, b, -0.5, al, be) for a in (3.0, 8.0) for b in (3.0, 8.0)
                       for al in (0.3, 0.6) for be in (0.3, 0.6)]
    for init in inits:
        th = torch.tensor(init, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            pred = torch.logsumexp(torch.stack([th[0] - th[3] * logN, th[1] - th[4] * logD, th[2].expand_as(logN)]), dim=0)
            loss = torch.nn.functional.huber_loss(pred, target, delta=1e-3, reduction="sum")
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            continue
        loss = closure().item()
        if np.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, th.detach().clone())
    a, b, e, al, be = best[1].tolist()
    return {"A": math.exp(a), "B": math.exp(b), "E": math.exp(e), "alpha": al, "beta": be, "loss": best[0]}


def predict(p, N, D):
    N, D = np.asarray(N, float), np.asarray(D, float)
    return p["E"] + p["A"] / N ** p["alpha"] + p["B"] / D ** p["beta"]


def optimal_allocation(p, C):
    """Minimize E + A/N^a + B/D^b subject to 6ND = C (closed form). Returns (N*, D*, L*)."""
    al, be, A, B = p["alpha"], p["beta"], p["A"], p["B"]
    G = (al * A / (be * B)) ** (1 / (al + be))
    N = G * (C / 6) ** (be / (al + be))
    D = C / (6 * N)
    return N, D, float(predict(p, N, D))


def exponents(p):
    """N* ~ C^a and D* ~ C^b with a = beta/(alpha+beta), b = alpha/(alpha+beta), a + b = 1."""
    return p["beta"] / (p["alpha"] + p["beta"]), p["alpha"] / (p["alpha"] + p["beta"])
