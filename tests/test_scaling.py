import json
import math

import numpy as np
import torch

from hedy.scalefit import (exponents, fit_parametric, isoflop_minimum, isoflop_table, optimal_allocation,
                           powerlaw, predict)
from hedy.scaling import FAMILY, TOKENS_PER_STEP, fixed_val_loss, make_config, n_params, train_one
from hedy.model import GPT

TRUE = {"E": 1.2, "A": 8.0, "B": 25.0, "alpha": 0.4, "beta": 0.3}


def synthetic_grid(noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    N, D, L, tag = [], [], [], []
    for C in (1e12, 3e12, 1e13, 3e13):
        for n in (1.5e4, 4e4, 1e5, 2.5e5, 6e5, 1.5e6):
            d = C / (6 * n)
            N.append(n); D.append(d); tag.append(f"grid:{C:.0e}")
            L.append(float(predict(TRUE, n, d)) * (1 + noise * rng.randn()))
    return np.array(N), np.array(D), np.array(L), tag


def test_parametric_fit_recovers_known_law():
    N, D, L, _ = synthetic_grid()
    p = fit_parametric(N, D, L)
    for k in ("E", "alpha", "beta"):
        assert abs(p[k] - TRUE[k]) / TRUE[k] < 0.03, (k, p[k])
    assert np.allclose(predict(p, N, D), L, rtol=1e-3)


def test_optimal_allocation_matches_brute_force_and_exponents_sum_to_one():
    C = 1e14
    N, D, Lstar = optimal_allocation(TRUE, C)
    grid = np.logspace(4, 9, 4000)
    brute = predict(TRUE, grid, C / (6 * grid))
    assert abs(Lstar - brute.min()) < 1e-4 and abs(np.log(N / grid[brute.argmin()])) < 0.01
    a, b = exponents(TRUE)
    assert abs(a + b - 1) < 1e-12 and abs(a - 0.3 / 0.7) < 1e-12
    assert abs(6 * N * D - C) / C < 1e-9


def test_isoflop_minimum_and_table_recover_optimum():
    N, D, L, tag = synthetic_grid()
    runs = [{"N": n, "D": d, "C": 6 * n * d, "val_loss": l, "tag": t} for n, d, l, t in zip(N, D, L, tag)]
    tab = isoflop_table(runs)
    assert sorted(tab) == [1e12, 3e12, 1e13, 3e13]
    for C, (n_star, l_star, d_star, inside, k) in tab.items():
        n_true, d_true, l_true = optimal_allocation(TRUE, C)
        assert k == 6 and abs(n_star * d_star * 6 - C) / C < 1e-9
        if inside:
            assert abs(math.log(n_star / n_true)) < 0.35                 # a parabola in log N approximates the true curve
    assert tab[1e13][3] and tab[3e13][3]                                # the larger budgets bracket their optimum
    ok = [C for C, v in tab.items() if v[3]]
    a, _, _ = powerlaw(ok, [tab[C][0] for C in ok])
    assert abs(a - exponents(TRUE)[0]) < 0.15


def test_monotone_data_with_an_inside_vertex_is_not_called_bracketed():
    """Loss rising from the smallest model: a 3-point parabola can still have its vertex just inside the
    range, but the best observed run is an endpoint, so the optimum is off the grid."""
    N = np.array([5.8e4, 1.66e5, 4.31e5]); L = np.array([3.4699, 3.7218, 4.6318])
    n_star, _, bracketed, c2 = isoflop_minimum(N, L)
    assert c2 > 0 and N.min() < n_star < N.max() and bracketed is False


def test_isoflop_minimum_flags_optimum_outside_the_grid():
    N = np.array([1e5, 2e5, 4e5, 8e5]); L = 3 + 0.5 * (np.log(N) - np.log(1e7)) ** 2   # true optimum at 1e7, beyond the grid
    assert isoflop_minimum(N, L)[2] is False


def test_powerlaw_exact():
    x = np.array([1., 10., 100., 1000.])
    a, k, se = powerlaw(x, 3 * x ** 0.7)
    assert abs(a - 0.7) < 1e-9 and abs(k - 3) < 1e-9


def test_extrapolation_from_small_compute_is_close_on_clean_data():
    N, D, L, tag = synthetic_grid(noise=0.002)
    train = np.array([t != "grid:3e+13" for t in tag])
    p = fit_parametric(N[train], D[train], L[train])
    err = np.abs(predict(p, N[~train], D[~train]) - L[~train]) / L[~train]
    assert err.max() < 0.03


def test_family_and_plan_arithmetic():
    sizes = [n_params(d, L) for d, L in FAMILY]
    assert sizes == sorted(sizes) and sizes[0] < 1e5 and sizes[-1] > 3e6
    for d, L in FAMILY:
        cfg = make_config(d, L)
        assert cfg.n_embd % cfg.n_head == 0 and (cfg.n_embd // cfg.n_head) % 2 == 0   # RoPE needs an even head dim


def test_fixed_val_loss_is_deterministic_and_independent_of_rng(tmp_path):
    torch.manual_seed(0)
    m = GPT(make_config(32, 2)).eval()
    val = np.random.RandomState(0).randint(0, 1024, 20_000).astype(np.uint16)
    a = fixed_val_loss(m, val, n_windows=100, batch=50)
    torch.manual_seed(123); torch.rand(10)
    b = fixed_val_loss(m, val, n_windows=100, batch=50)
    assert a == b and abs(a - math.log(1024)) < 0.5            # untrained model ~ ln V


def test_train_one_smoke_and_compute_accounting(tmp_path):
    rng = np.random.RandomState(0)
    rng.randint(0, 1024, 60_000).astype(np.uint16).tofile(tmp_path / "train.bin")
    rng.randint(0, 1024, 12_000).astype(np.uint16).tofile(tmp_path / "val.bin")
    r = train_one(32, 2, 2 * TOKENS_PER_STEP, 3e-3, data_dir=str(tmp_path), threads=1)
    assert r["steps"] == 2 and r["D"] == 2 * TOKENS_PER_STEP and r["N"] == n_params(32, 2)
    assert r["C"] == 6 * r["N"] * r["D"] and r["val_loss"] > 0


def test_report_runs_end_to_end_and_recovers_the_synthetic_law(tmp_path, capsys):
    """The `fit` command: noise table, IsoFLOP minima, parametric fit, extrapolation test, figures."""
    import argparse
    from hedy.scaling import report
    N, D, L, tag = synthetic_grid(noise=0.002)
    rows = [{"d": 32, "L": 2, "N": n, "D": d, "C": 6 * n * d, "val_loss": l, "train_loss": l, "seconds": 1.0,
             "tok_per_s": 1.0, "tag": t, "seed": 0, "lr": 1e-3, "steps": 1, "threads": 1} for n, d, l, t in zip(N, D, L, tag)]
    rows += [{**rows[7], "seed": 1, "tag": "noise", "val_loss": rows[7]["val_loss"] + 0.01}]       # a replicate of one config
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(argparse.Namespace(runs=str(path), lr_runs=None, figs=str(tmp_path / "figs"), boot=2))
    out = capsys.readouterr().out
    assert "IsoFLOP minima" in out and "parametric fit" in out and "extrapolation: fit on levels" in out
    assert "alpha=0.4" in out.replace("alpha=0.40", "alpha=0.4") or "alpha=0.39" in out or "alpha=0.41" in out
    for f in ("scaling_isoflop.png", "scaling_optimum.png", "scaling_extrapolation.png"):
        assert (tmp_path / "figs" / f).stat().st_size > 5_000


def test_loss_command_reports_error_against_a_stored_prediction(tmp_path, capsys, monkeypatch):
    import sys
    from hedy.scaling import main
    torch.manual_seed(0)
    m = GPT(make_config(32, 2))
    torch.save({"config": m.cfg.__dict__, "model": m.state_dict()}, tmp_path / "model.pt")
    np.random.RandomState(0).randint(0, 1024, 20_000).astype(np.uint16).tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"val_tokens": 20_000, "val_bytes": 61_000}))
    (tmp_path / "pred.json").write_text(json.dumps({"predicted_val_loss_fixed_windows": 6.9,
        "fit_only_95pct_interval_bootstrap": [6.0, 8.0], "with_seed_noise_95pct_interval": [5.5, 8.5]}))
    monkeypatch.setattr(sys, "argv", ["scaling", "loss", "--ckpt", str(tmp_path), "--data_dir", str(tmp_path),
                                      "--prediction", str(tmp_path / "pred.json"), "--threads", "1"])
    main()
    out = capsys.readouterr().out
    assert "fixed-window validation loss" in out and "inside the fit-only interval" in out and "True" in out
