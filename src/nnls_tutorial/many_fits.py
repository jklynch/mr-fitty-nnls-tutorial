"""Notebook 5's figures: the draws, where the time goes, and agreement with SciPy.

Unlike the earlier notebooks' figures, these have no widgets: they show measurements.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import plotting as P


def seconds(run, repeat=3):
    """The best of `repeat` timings of run(), in seconds."""
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def holdout_figure(spectra, held_out, shown=40):
    """One draw's held-out energies on the measured spectrum, and the first `shown` draws."""
    fig, (ax_s, ax_m) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5])
    )
    e = spectra.energies
    ax_s.plot(e, spectra.s, color=P.INK, lw=1.2, marker=".", ms=3)
    out = held_out[0]
    ax_s.plot(e[out], spectra.s[out], "o", color=P.NNLS_COLOR, ms=4, label="held out by draw 1")
    ax_s.set_ylabel("normalized absorption")
    ax_s.legend(loc="lower right")
    ax_s.set_title("Draw 1: fit on the black energies, score on the red ones", fontsize=11.5)
    ax_m.imshow(
        held_out[:shown],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap([P.GRID, P.NNLS_COLOR]),
        extent=(e[0], e[-1], shown + 0.5, 0.5),
    )
    ax_m.set_ylabel("draw")
    ax_m.set_xlabel("energy (eV)")
    ax_m.grid(False)
    ax_m.set_title(f"The held-out energies of the first {shown} draws", fontsize=11.5)
    return fig


def timing_figure(timings, fits):
    """Horizontal bars: seconds for the whole search, by method, with time per fit."""
    names = list(timings)
    values = np.array([timings[name] for name in names])
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, values, color=[P.INK_MUTED] * (len(names) - 1) + [P.NNLS_COLOR], height=0.6)
    for row, value in zip(y, values, strict=True):
        ax.annotate(
            f"{value:.3g} s, {value / fits * 1e6:.2g} µs per fit, {values[0] / value:.0f}× faster"
            if row
            else f"{value:.3g} s, {value / fits * 1e6:.2g} µs per fit",
            (value, row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.5,
        )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(values.min() / 2, values.max() * 40)
    ax.set_xlabel("seconds for the whole search (log scale)")
    ax.grid(axis="y", visible=False)
    return fig


def lessons_figure(lessons):
    """One small panel per lesson: the slow way against the fast way, in milliseconds."""
    fig, axes = plt.subplots(1, len(lessons), figsize=(5 * len(lessons), 3.2))
    for ax, (title, pair) in zip(axes, lessons.items(), strict=True):
        names = list(pair)
        values = 1000 * np.array([pair[name] for name in names])
        ax.barh([0, 1], values, color=[P.INK_MUTED, P.NNLS_COLOR], height=0.6)
        for row, value in enumerate(values):
            ax.annotate(
                f"{value:.2g} ms",
                (value, row),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
            )
        ax.set_yticks([0, 1], names)
        ax.invert_yaxis()
        ax.set_xlim(0, values.max() * 1.45)
        ax.set_xlabel("milliseconds")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{title}\n{values[0] / values[1]:.1f}× faster", fontsize=11)
    return fig


def agreement_figure(spectra, columns, W_scipy, W_fast, pe_differences):
    """Left: each draw's weights, compiled against SciPy, for one combination. Right: how far
    apart every draw's prediction error is, over the whole search."""
    fig, (ax_w, ax_d) = plt.subplots(1, 2, figsize=(12, 5))
    for i, j in enumerate(columns):
        ax_w.plot(
            W_scipy[:, i],
            W_fast[:, i],
            "o",
            ms=3,
            alpha=0.6,
            color=P.COLUMN_COLORS[i],
            label=spectra.labels[j],
        )
    top = max(W_scipy.max(), W_fast.max()) * 1.05
    ax_w.plot([0, top], [0, top], color=P.INK_MUTED, lw=0.8, zorder=0)
    ax_w.set_xlim(-0.02, top)
    ax_w.set_ylim(-0.02, top)
    ax_w.set_aspect("equal")
    ax_w.set_xlabel("weight, scipy.optimize.nnls")
    ax_w.set_ylabel("weight, compiled search")
    ax_w.legend(loc="upper left")
    ax_w.set_title("Every draw's weights, for one combination", fontsize=11.5)
    positive = pe_differences[pe_differences > 0]
    zeros = len(pe_differences) - len(positive)
    bins = np.logspace(-18, -11, 29)
    ax_d.hist(np.clip(positive, bins[0], bins[-1]), bins=bins, color=P.INK_MUTED)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("|difference in prediction error| (log scale)")
    ax_d.set_ylabel("draws")
    ax_d.set_title(
        f"All {len(pe_differences):,} fits of the search ({zeros:,} differ by exactly 0)",
        fontsize=11.5,
    )
    return fig
