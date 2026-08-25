"""Figure styling: the dataviz palette, matplotlib defaults, and shared helpers.

Palette values come from the dataviz skill's reference instance. The categorical
slots used for the four method families were validated all-pairs in light mode
(worst CVD dE 9.2, worst normal-vision dE 16.3) with analysis/validate_palette.py --
figures are light-mode only, since they are bound for a paper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e6e5e1"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]          # blue, orange, aqua, violet
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#e34948"    # blue <-> red, gray midpoint

STATUS = {"ok": "#1baf7a", "OOM": "#e34948", "missing": "#d8d7d2"}

SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
DIV_CMAP = mpl.colors.LinearSegmentedColormap.from_list("div", [DIV_LO, DIV_MID, DIV_HI])

# --- method families -------------------------------------------------------
FAMILY = {
    "colmap/no_gt": "Classical SfM", "mpsfm": "Classical SfM",
    "vggt": "VGGT", "vggt-long": "VGGT", "vggt-slam": "VGGT", "vggt-omega": "VGGT",
    "da3": "DA3", "da3_streaming": "DA3",
    "pi3": "Pi3", "pi3x": "Pi3", "pi-long": "Pi3",
}
FAMILY_ORDER = ["Classical SfM", "VGGT", "DA3", "Pi3"]
FAMILY_COLOR = dict(zip(FAMILY_ORDER, CAT))

# every method that chunks/streams rather than one-shotting the whole sequence
CHUNKED = {"colmap/no_gt", "mpsfm", "da3_streaming", "vggt-long", "vggt-slam", "pi3x", "pi-long"}

METHOD_ORDER = ["colmap/no_gt", "mpsfm", "vggt", "vggt-long", "vggt-slam", "vggt-omega",
                "da3", "da3_streaming", "pi3", "pi3x", "pi-long"]

ASSET_ORDER = ["glass_house2", "old_house", "unfinished_build", "crane", "bridge", "ship", "office2"]
ASSET_LABEL = {"glass_house2": "glass_house2", "old_house": "old_house",
               "unfinished_build": "unfinished_build", "crane": "crane",
               "bridge": "bridge", "ship": "ship", "office2": "office2"}

ILLUM_ORDER = ["sunny", "overcast", "sunrise", "night"]
SOIL_ORDER = ["verylow", "low", "medium", "high"]
REFL_ORDER = ["0.0", "0.25", "0.5", "0.75", "1.0"]
STANDOFF_ORDER = ["near", "mid", "far", "trendspek"]


def mcolor(method):
    return FAMILY_COLOR[FAMILY[method]]


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "text.color": INK,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def family_legend(fig, ncol=4, y=0.0, loc="lower center"):
    h = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=3, label=f) for f in FAMILY_ORDER]
    fig.legend(handles=h, loc=loc, ncol=ncol, bbox_to_anchor=(0.5, y),
               handlelength=1.6, columnspacing=1.8)


def save(fig, name, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def annotate_cells(ax, M, fmt="{:.2f}", thresh=0.55, cmap=None, vmin=None, vmax=None, na="--"):
    """Value labels inside a heatmap, ink flipped against cell lightness."""
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, na, ha="center", va="center", color=INK3, fontsize=7)
                continue
            lum = mpl.colors.rgb_to_hsv(cmap(norm(v))[:3])[2] if cmap else 1.0
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="#ffffff" if lum < thresh else INK)


def titled(ax, title, subtitle=None, pad=30, sub_pt=13):
    """Title above the axes with a non-colliding subtitle beneath it.

    `pad` is the title's offset in points; the subtitle is placed `sub_pt` points
    above the axes, so keep pad > sub_pt + ~12.
    """
    ax.set_title(title, pad=pad)
    if subtitle:
        h_pt = ax.get_figure().get_size_inches()[1] * ax.get_position().height * 72
        ax.text(0, 1 + sub_pt / h_pt, subtitle, transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="bottom")


def fig_title(fig, title, subtitle=None, x=0.006, gap_pt=7, title_pt=13, sub_pt=9):
    """Figure-level title + subtitle, stacked above the axes without colliding.

    Call after tight_layout(); positions are computed in points off the figure
    height so they hold at any figure size, and savefig(bbox_inches="tight")
    grows the canvas to include them.
    """
    H = fig.get_size_inches()[1] * 72
    y = 1 + gap_pt / H
    if subtitle:
        fig.text(x, y, subtitle, fontsize=sub_pt, color=INK2, va="bottom", ha="left")
        y += (sub_pt * 1.35 + 5) / H
    fig.text(x, y, title, fontsize=title_pt, fontweight="bold", color=INK, va="bottom", ha="left")
