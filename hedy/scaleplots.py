"""Static figures for the scaling study (matplotlib). Palette: validated reference palette (slots 1-3
categorical, blue ordinal ramp for ordered compute levels); text in ink colors, never series colors."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}   # categorical slots 1-4
RAMP = ["#86b6ef", "#5598e7", "#256abf", "#104281"]                         # ordinal: light -> dark = low -> high compute


def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-"); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.tick_params(which="minor", labelsize=8, labelcolor=INK2, colors=INK2)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", fontweight="bold")


def _fig(w=7.2, h=4.4):
    fig = plt.figure(figsize=(w, h), dpi=160, facecolor=SURFACE)
    return fig


def _legend(ax, **kw):
    return ax.legend(**{"frameon": False, "fontsize": 9, "labelcolor": INK2, **kw})


def isoflop_figure(runs, table, path):
    """Loss vs parameters, one curve per compute budget, with the parabola fit and its minimum."""
    fig = _fig(); ax = fig.add_subplot(111)
    levels = sorted(table)
    for i, C in enumerate(levels):
        col = RAMP[min(i + (len(RAMP) - len(levels)), len(RAMP) - 1)] if len(levels) <= len(RAMP) else RAMP[-1]
        rs = sorted([r for r in runs if abs(math.log10(float(r["tag"].split(":")[-1])) - math.log10(C)) < 1e-6], key=lambda r: r["N"])
        N, L = np.array([r["N"] for r in rs], float), np.array([r["val_loss"] for r in rs])
        c2, c1, c0 = np.polyfit(np.log(N), L, 2)
        xs = np.linspace(np.log(N.min()), np.log(N.max()), 100)
        ax.plot(np.exp(xs), c2 * xs ** 2 + c1 * xs + c0, color=col, linewidth=1.6, zorder=2)
        ax.plot(N, L, "o", color=col, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=f"{C:.0e} FLOPs".replace("e+", "e"))
        n_star, l_star, d_star, inside, _ = table[C]
        if inside:
            ax.plot([n_star], [l_star], "D", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.4, zorder=4)
        ax.annotate(f"{C:.0e}".replace("e+", "e"), (N[-1], c2 * np.log(N[-1]) ** 2 + c1 * np.log(N[-1]) + c0),
                    textcoords="offset points", xytext=(8, 0), ha="left", va="center", fontsize=8.5, color=INK2)
    ax.plot([], [], "D", markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.4, markersize=6.5, label="fitted minimum")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi * (hi / lo) ** 0.12)      # room on the right for the direct labels
    _style(ax, "parameters N (log scale)", "validation loss (nats/token)", "IsoFLOP curves: for each compute budget, loss vs model size")
    _legend(ax, loc="upper left", title="compute budget (6·N·D)", title_fontsize=9)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def optimum_figure(table, fit_params, pl_N, pl_D, C_range, path, para=None):
    """Compute-optimal N* and D* vs C: IsoFLOP minima (solid = bracketed, hollow = bound only) + fits."""
    fig = _fig(9.2, 4.2)
    Cs = np.array(sorted(table))
    cr = np.logspace(np.log10(C_range[0]), np.log10(C_range[1]), 50)
    for k, (ax, key, ylabel, pl) in enumerate(((fig.add_subplot(121), 0, "optimal parameters N*", pl_N), (fig.add_subplot(122), 2, "optimal tokens D*", pl_D))):
        ys = np.array([table[C][key] for C in Cs]); inside = np.array([table[C][3] for C in Cs])
        ax.plot(cr, pl[1] * cr ** pl[0], color=CAT["blue"], linewidth=1.8, zorder=2, label=f"IsoFLOP power-law fit (slope {pl[0]:.2f})")
        ax.plot(Cs[inside], ys[inside], "o", color=CAT["blue"], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label="IsoFLOP minimum (bracketed)")
        if (~inside).any():
            ax.plot(Cs[~inside], ys[~inside], "o", markerfacecolor=SURFACE, markeredgecolor=CAT["blue"], markeredgewidth=1.5, markersize=6, zorder=3, label="IsoFLOP minimum (not bracketed: a bound)")
        if para is not None:
            vals = np.array([optimal_from(para, c)[0 if key == 0 else 1] for c in cr])
            ax.plot(cr, vals, color=CAT["orange"], linewidth=1.8, zorder=2, label=f"parametric fit (slope {para_slope(para, key):.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        _style(ax, "compute C = 6·N·D (FLOPs)", ylabel)
        if k == 0:
            _legend(ax, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def optimal_from(p, C):
    from .scalefit import optimal_allocation
    return optimal_allocation(p, C)[:2]


def para_slope(p, key):
    from .scalefit import exponents
    return exponents(p)[0 if key == 0 else 1]


def lr_figure(lr_runs, path):
    """Validation loss vs learning rate for each model size (same token budget); ring marks the best."""
    fig = _fig(); ax = fig.add_subplot(111)
    dims = sorted({r["d"] for r in lr_runs})
    cols = [CAT["blue"], CAT["orange"], CAT["aqua"], CAT["yellow"]]   # slots 1-4: adjacent-pair safe for lines
    ends = []
    assert len(dims) <= len(cols), "more series than validated categorical slots"
    for d, col in zip(dims, cols):
        rs = sorted([r for r in lr_runs if r["d"] == d], key=lambda r: r["lr"])
        lr, L = [r["lr"] for r in rs], [r["val_loss"] for r in rs]
        N = rs[0]["N"]
        ax.plot(lr, L, "-o", color=col, linewidth=1.8, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"d={d}  ({N / 1e6:.2f}M params)")
        b = int(np.argmin(L))
        ax.plot([lr[b]], [L[b]], "o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3, zorder=5)
        ends.append([lr[-1], L[-1], f"d={d}"])
    ends.sort(key=lambda e: e[1]); gap = 0.11 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 1.6   # keep end labels apart
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for x_, y_, lab in ends:
        ax.annotate(lab, (x_, y_), textcoords="offset points", xytext=(7, 0), va="center", fontsize=8.5, color=INK2)
    ax.plot([], [], "o", markersize=11, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3, label="best learning rate")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim(); ax.set_xlim(lo, hi * (hi / lo) ** 0.1)         # room for the end labels
    _style(ax, "peak learning rate (log scale)", "validation loss (nats/token)", "Learning-rate sweep by model size (same token budget)")
    _legend(ax, loc="lower right")
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)


def extrapolation_figure(pred, actual, labels, path, level):
    """Held-out compute level: predicted vs actual loss from a fit that never saw that level."""
    fig = _fig(5.4, 4.8); ax = fig.add_subplot(111)
    lo, hi = min(min(pred), min(actual)) - 0.03, max(max(pred), max(actual)) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.0, zorder=1, label="perfect prediction")
    ax.plot(actual, pred, "o", color=CAT["blue"], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3, label=f"models trained at {level:.0e} FLOPs".replace("e+", "e"))
    pts = sorted(zip(actual, pred, labels))
    crowded = [t for t in pts if t[0] < lo + 0.3 * (hi - lo)]
    for i, (a, p_, lab) in enumerate(crowded):         # fan out the cluster along an empty column, with leader lines
        ax.annotate(lab, xy=(a, p_), xytext=(lo + 0.5 * (hi - lo), lo + 0.1 * (hi - lo) + 0.075 * (hi - lo) * i), fontsize=8.5, color=INK2,
                    va="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6, shrinkA=2, shrinkB=4))
    for a, p_, lab in pts[len(crowded):]:
        ax.annotate(lab, (a, p_), textcoords="offset points", xytext=(8, -12), fontsize=8.5, color=INK2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    _style(ax, "actual validation loss", "loss predicted by the fit (never saw this budget)", "Extrapolation test")
    _legend(ax, loc="upper left")
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)
