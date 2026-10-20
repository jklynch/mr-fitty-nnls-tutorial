"""One visual style for every figure in the series.

Colors follow the entity, never its rank. Each reference column r_i keeps its color everywhere,
and so do the measured spectrum s and the two answers: the least-squares projection (onto the
span) and the NNLS projection (onto the cone). Those two also differ by line style and marker,
so color never carries the distinction alone.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# The categorical slots, in their validated order. Columns take them first to last.
COLUMN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

INK = "#0b0b0b"  # the measured spectrum s, and text
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"  # axes, annotations
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

LS_COLOR = "#6250d6"  # violet: least squares, the projection onto the span
NNLS_COLOR = "#e34948"  # red: NNLS, the projection onto the cone
LS_STYLE = dict(color=LS_COLOR, linestyle="--", marker="s", markerfacecolor="none")
NNLS_STYLE = dict(color=NNLS_COLOR, linestyle="-", marker="o")

# Light fills for the region maps: one per support of two columns.
REGION_FILLS = {
    (): "#ebeae5",  # x = 0: gray, "nothing"
    (0,): "#cde2fb",  # only a1: blue tint
    (1,): "#fbd9cb",  # only a2: orange tint
    (0, 1): "#c9ecdd",  # both: aqua tint
}


def use_style():
    """Apply the series style to matplotlib. Every notebook calls this once."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=COLUMN_COLORS),
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "font.size": 10.5,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "figure.constrained_layout.use": True,
        }
    )


def column_label(i):
    """$r_1$, $r_2$, ... for column i (0-based)."""
    return f"$r_{{{i + 1}}}$"


def arrow(ax, start, end, color, label=None, width=0.012, **kwargs):
    """A vector drawn as an arrow from start to end on a 2D axes."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, shrinkA=0, shrinkB=0),
        **kwargs,
    )
    if label is not None:
        ax.annotate(
            label,
            xy=end,
            xytext=(6, 6),
            textcoords="offset points",
            color=INK,
            fontsize=13,
        )


def arrow3d(ax, start, end, color, label=None, label_offset=0.08, **kwargs):
    """A vector drawn as a line with a cone-free arrowhead marker on a 3D axes."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    ax.plot(*zip(start, end, strict=True), color=color, lw=2.2, **kwargs)
    ax.scatter(*end, color=color, s=28, depthshade=False)
    if label is not None:
        tip = end + label_offset * (end - start) / max(np.linalg.norm(end - start), 1e-12)
        ax.text(*tip, label, color=INK, fontsize=13)


def square_limits(ax, radius):
    """Equal-aspect axes centered on the origin."""
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal")
    ax.axhline(0, color=INK_MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=INK_MUTED, lw=0.8, zorder=0)


def show(fig):
    """Display a figure and release it, so redraws inside a widget do not pile up."""
    plt.show()
    plt.close(fig)
