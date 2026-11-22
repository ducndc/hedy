"""Scaling-law experiments: train a family of models on different token budgets and fit L(N, D).

    python -m hedy.scaling run  --d 128 --L 4 --D 5000000 --lr 3e-3 --tag grid --out runs.jsonl
    python -m hedy.scaling plan --kind lr   > plan_lr.txt        # one line of `run` args per experiment
    python -m hedy.scaling plan --kind grid --lrs 128:3e-3,... > plan_grid.txt
    python -m hedy.scaling fit  --runs runs.jsonl --figs docs/figures

Family: head_dim 32, RoPE + RMSNorm + SwiGLU, vocab 1024, context 128, no dropout, batch 32x128 = 4096
tokens/step for every run. Compute is nominal C = 6 * N * D (N = all parameters; tied embedding counted once).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import get_batch, load_bin
from .model import GPT, GPTConfig
from .train import lr_at, make_optimizer

FAMILY = [(32, 2), (64, 2), (96, 3), (128, 4), (192, 4), (256, 5)]   # (width d, layers L); 58k .. 4.2M params
BLOCK, BATCH, VOCAB = 128, 32, 1024
TOKENS_PER_STEP = BLOCK * BATCH


def make_config(d, L):
    return GPTConfig(VOCAB, BLOCK, L, max(1, d // 32), d, 0.0, "rope", "rmsnorm", "swiglu", 0)


def n_params(d, L):
    return sum(p.numel() for p in GPT(make_config(d, L)).parameters())


@torch.no_grad()
def fixed_val_loss(model, val, n_windows=2400, batch=200):
    """Mean next-token loss on n_windows evenly spaced windows of the validation tokens: the SAME windows
    for every run, so differences between runs are not sampling noise in the evaluation."""
    model.eval()
    starts = np.linspace(0, len(val) - BLOCK - 2, n_windows).astype(int)
    tot = 0.0
    for i in range(0, n_windows, batch):
        s = starts[i : i + batch]
        x = torch.from_numpy(np.stack([val[j : j + BLOCK] for j in s]).astype(np.int64))
        y = torch.from_numpy(np.stack([val[j + 1 : j + BLOCK + 1] for j in s]).astype(np.int64))
        tot += torch.nn.functional.cross_entropy(model(x)[0].transpose(1, 2), y, reduction="sum").item()
    return tot / (n_windows * BLOCK)


def train_one(d, L, D, lr, seed=0, data_dir="data/tinystories-big", threads=2):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    train, val = load_bin(os.path.join(data_dir, "train.bin")), np.asarray(load_bin(os.path.join(data_dir, "val.bin")))
    steps = max(1, round(D / TOKENS_PER_STEP))
    assert steps * TOKENS_PER_STEP <= len(train), "would need more than one epoch"
    model = GPT(make_config(d, L))
    N = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, lr, 0.1, "cpu")
    warmup = min(100, max(10, steps // 10))
    recent, t0 = [], time.perf_counter()
    model.train()
    for step in range(steps):
        torch.manual_seed(seed * 1_000_003 + step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)
        loss = model(*get_batch(train, BATCH, BLOCK))[1]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        recent = (recent + [loss.item()])[-20:]
    secs = time.perf_counter() - t0
    D_actual = steps * TOKENS_PER_STEP
    return {"d": d, "L": L, "N": N, "D": D_actual, "C": 6 * N * D_actual, "steps": steps, "lr": lr, "seed": seed,
            "train_loss": float(np.mean(recent)), "val_loss": fixed_val_loss(model, val),
            "seconds": secs, "tok_per_s": D_actual / secs, "threads": threads}


def report(a):
    """Everything the study note quotes: noise, IsoFLOP minima, power-law and parametric fits,
    the extrapolation test and (with --figs) the figures."""
    from . import scalefit as sf
    allr = sf.load_runs(a.runs)
    grid = [r for r in allr if r["tag"].startswith("grid:")]
    noise = [r for r in allr if r["tag"] == "noise"]
    print(f"{len(grid)} grid runs, {len(noise)} replicate runs\n")

    print("== run-to-run noise (same config, different seed) ==")
    for d in sorted({r["d"] for r in noise}):
        base = [r for r in grid if r["d"] == d and any(abs(r["D"] - n["D"]) < 1 for n in noise if n["d"] == d)]
        vals = [r["val_loss"] for r in base + [n for n in noise if n["d"] == d]]
        print(f"d={d}: val losses {[round(v, 4) for v in vals]}  std {np.std(vals, ddof=1):.4f}  (range {max(vals) - min(vals):.4f})")

    print("\n== grid runs ==")
    print(f"{'level':>7} {'d':>4} {'N':>10} {'D':>11} {'tok/param':>9} {'val':>7} {'train':>7} {'secs':>6}")
    for r in sorted(grid, key=lambda r: (float(r["tag"].split(":")[1]), r["N"])):
        print(f"{float(r['tag'].split(':')[1]):>7.0e} {r['d']:>4} {r['N']:>10,} {r['D']:>11,} {r['D'] / r['N']:>9.1f} {r['val_loss']:>7.4f} {r['train_loss']:>7.4f} {r['seconds']:>6.0f}")

    tab = sf.isoflop_table(grid)
    print("\n== IsoFLOP minima (parabola in log N) ==")
    print(f"{'C':>7} {'N*':>10} {'D*':>11} {'tok/param':>9} {'L*':>7} {'bracketed':>9} {'points':>6}")
    for C, (n, l, d, inside, k) in tab.items():
        print(f"{C:>7.0e} {n:>10,.0f} {d:>11,.0f} {d / n:>9.1f} {l:>7.4f} {str(inside):>9} {k:>6}")
    ok = [C for C, v in tab.items() if v[3]]
    pl_n = pl_d = None
    if len(ok) >= 2:
        pl_n = sf.powerlaw(ok, [tab[C][0] for C in ok]); pl_d = sf.powerlaw(ok, [tab[C][2] for C in ok])
        print(f"IsoFLOP fit on {len(ok)} bracketed levels: N* ~ C^{pl_n[0]:.2f} (se {pl_n[1 + 1]:.2f}), D* ~ C^{pl_d[0]:.2f} (se {pl_d[2]:.2f})")

    N, D, L = (np.array([r[k] for r in grid], float) for k in ("N", "D", "val_loss"))
    p = sf.fit_parametric(N, D, L)
    a_exp, b_exp = sf.exponents(p)
    print(f"\n== parametric fit L = E + A/N^alpha + B/D^beta on all {len(grid)} runs ==")
    print(f"E={p['E']:.3f} A={p['A']:.3g} alpha={p['alpha']:.3f} B={p['B']:.3g} beta={p['beta']:.3f}; N* ~ C^{a_exp:.2f}, D* ~ C^{b_exp:.2f}")
    res = L - sf.predict(p, N, D)
    print(f"residuals: rms {np.sqrt(np.mean(res ** 2)):.4f}, max |res| {np.max(np.abs(res)):.4f}")
    rng = np.random.RandomState(0); boots = []
    for _ in range(a.boot):
        idx = rng.randint(0, len(N), len(N))
        try:
            q = sf.fit_parametric(N[idx], D[idx], L[idx], starts=[(math.log(p["A"]), math.log(p["B"]), math.log(p["E"]), p["alpha"], p["beta"])])
            boots.append((q["alpha"], q["beta"], sf.exponents(q)[0], q["E"]))
        except Exception:
            pass
    if boots:
        b = np.array(boots)
        print("bootstrap over runs (%d fits), 95%% interval: alpha %.2f-%.2f, beta %.2f-%.2f, N* exponent %.2f-%.2f, E %.2f-%.2f" % (
            len(b), *np.percentile(b[:, 0], [2.5, 97.5]), *np.percentile(b[:, 1], [2.5, 97.5]), *np.percentile(b[:, 2], [2.5, 97.5]), *np.percentile(b[:, 3], [2.5, 97.5])))
    print("optimum from the parametric fit:")
    for C in sorted(tab):
        n, d, l = sf.optimal_allocation(p, C)
        print(f"  C={C:.0e}: N*={n:,.0f}  D*={d:,.0f}  tok/param={d / n:.1f}  predicted L*={l:.4f}")

    top = max(float(r["tag"].split(":")[1]) for r in grid)
    tr = np.array([float(r["tag"].split(":")[1]) < top for r in grid])
    print(f"\n== extrapolation: fit on levels < {top:.0e}, predict level {top:.0e} ==")
    pe = sf.fit_parametric(N[tr], D[tr], L[tr])
    pred = sf.predict(pe, N[~tr], D[~tr]); act = L[~tr]
    for r, pr in zip([g for g, t in zip(grid, tr) if not t], pred):
        print(f"  d={r['d']:>3} N={r['N']:>9,}: predicted {pr:.4f}  actual {r['val_loss']:.4f}  error {100 * (pr - r['val_loss']) / r['val_loss']:+.2f}%")
    print(f"  mean |error| {100 * np.mean(np.abs(pred - act) / act):.2f}%   (fit: alpha={pe['alpha']:.2f} beta={pe['beta']:.2f} E={pe['E']:.2f})")
    best_pred = int(np.argmin(pred)); best_act = int(np.argmin(act))
    print(f"  best size predicted: N={[g for g, t in zip(grid, tr) if not t][best_pred]['N']:,}   best size actual: N={[g for g, t in zip(grid, tr) if not t][best_act]['N']:,}")

    if a.figs:
        from . import scaleplots as sp
        os.makedirs(a.figs, exist_ok=True)
        sp.isoflop_figure(grid, tab, os.path.join(a.figs, "scaling_isoflop.png"))
        if pl_n:
            sp.optimum_figure(tab, None, pl_n, pl_d, (min(tab) * 0.7, max(tab) * 1.5), os.path.join(a.figs, "scaling_optimum.png"), para=p)
        held = [g for g, t in zip(grid, tr) if not t]
        sp.extrapolation_figure(list(pred), list(act), [f"{g['N'] / 1e6:.2f}M" for g in held], os.path.join(a.figs, "scaling_extrapolation.png"), top)
        if a.lr_runs:
            sp.lr_figure([r for r in sf.load_runs(a.lr_runs) if r["d"] in (64, 128, 192, 256)], os.path.join(a.figs, "scaling_lr.png"))
        print(f"\nfigures written to {a.figs}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--d", type=int, required=True); r.add_argument("--L", type=int, required=True)
    r.add_argument("--D", type=float, required=True, help="training tokens"); r.add_argument("--lr", type=float, required=True)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--threads", type=int, default=2)
    r.add_argument("--tag", default=""); r.add_argument("--out", required=True)
    r.add_argument("--data_dir", default="data/tinystories-big")
    pl = sub.add_parser("plan")
    pl.add_argument("--kind", choices=["lr", "grid"], required=True)
    pl.add_argument("--lrs", default="", help="grid only: comma list d:lr")
    pl.add_argument("--levels", default="1e12,3e12,1e13,3e13"); pl.add_argument("--min_tokens", type=float, default=4e5)
    pl.add_argument("--max_tokens", type=float, default=6e7); pl.add_argument("--lr_tokens", type=float, default=6e5)
    pl.add_argument("--lr_grid", default="1e-3,3e-3,1e-2,3e-2"); pl.add_argument("--lr_dims", default="64,128,256")
    f = sub.add_parser("fit")
    f.add_argument("--runs", required=True); f.add_argument("--lr_runs", default=None)
    f.add_argument("--figs", default=None); f.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()

    if a.cmd == "run":
        res = train_one(a.d, a.L, int(a.D), a.lr, a.seed, a.data_dir, a.threads)
        res["tag"] = a.tag
        with open(a.out, "a") as f_:
            f_.write(json.dumps(res) + "\n")
        print(f"d={a.d} L={a.L} N={res['N']:,} D={res['D']:,} lr={a.lr}: val {res['val_loss']:.4f} "
              f"train {res['train_loss']:.4f} ({res['seconds']:.0f}s, {res['tok_per_s']:,.0f} tok/s)")
    elif a.cmd == "fit":
        report(a)
    elif a.cmd == "plan":
        fam = dict(FAMILY)
        if a.kind == "lr":
            for d in map(int, a.lr_dims.split(",")):
                for lr in a.lr_grid.split(","):
                    print(f"--d {d} --L {fam[d]} --D {a.lr_tokens:.0f} --lr {lr} --tag lr")
        else:
            lrs = {int(k): v for k, v in (kv.split(":") for kv in a.lrs.split(","))}
            for C in map(float, a.levels.split(",")):
                for d, L in FAMILY:
                    D = C / (6 * n_params(d, L))
                    if a.min_tokens <= D <= a.max_tokens:
                        print(f"--d {d} --L {L} --D {D:.0f} --lr {lrs[d]} --tag grid:{C:.0e}")


if __name__ == "__main__":
    main()
