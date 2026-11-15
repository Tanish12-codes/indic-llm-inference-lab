"""One shared look for every plot. Plots are always drawn from CSVs, never typed numbers.

Palette: validated categorical slots (blue, orange, aqua, yellow) in fixed order,
so "GPT-2" is always blue, "Qwen2.5" always orange, "Param2" always aqua.
Aqua/yellow are low-contrast on white, so bars carry value labels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10,
        # Fallback chain: Latin glyphs from DejaVu, Devanagari from Nirmala UI (Windows) / Noto / Lohit.
        "font.family": _available_fonts(),
    })


def _available_fonts():
    """DejaVu for Latin plus the first installed Devanagari font (only names that exist, so no warnings)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    wanted = ["Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "Mangal"]
    return ["DejaVu Sans"] + [f for f in wanted if f in have][:1]


def color_for(name: str, order: list[str]) -> str:
    """Colour follows the entity, never its rank."""
    return SERIES[order.index(name) % len(SERIES)]


def grouped_bars(ax, groups, series, values, order=None, fmt="{:.2f}"):
    """values[s][g] -> bar. Groups on x, one bar per series, 2px gaps, value labels."""
    import numpy as np
    order = order or series
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        ys = [values[s].get(g, float("nan")) for g in groups]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, ys, width * 0.92, label=s,
                      color=color_for(s, order), edgecolor=SURFACE, linewidth=1)
        for b, y in zip(bars, ys):
            if y == y:  # not NaN
                ax.annotate(fmt.format(y), (b.get_x() + b.get_width() / 2, y),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, color=TEXT2)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    ax.legend()
