"""Not every method is scored on the same frames -- the comparability caveat."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import style as S
from common import loade, OUT

S.setup()
p, _ = loade("pose")
M = S.METHOD_ORDER
KEY = ["Asset", "Traj", "Illumination", "Soil", "Reflectivity"]

ok = p[p.Status == "ok"].copy()
ok["MaxFrames"] = ok.groupby(KEY)["Frames"].transform("max")
ok["Reg"] = ok.Frames / ok.MaxFrames


def fig_registration():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8),
                             gridspec_kw={"width_ratios": [1.05, 1.5]})

    # left: distribution of registration rate per method
    ax = axes[0]
    order = ok.groupby("Method").Reg.median().sort_values().index
    order = [m for m in order if m in M]
    for i, m in enumerate(order):
        v = ok[ok.Method == m].Reg.values
        ax.scatter(v, np.full(len(v), i) + np.random.RandomState(1).uniform(-.16, .16, len(v)),
                   s=5, color=S.mcolor(m), alpha=0.35, linewidth=0, zorder=3)
        ax.scatter([np.median(v)], [i], s=55, color=S.INK, zorder=5, marker="|", linewidth=2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("frames scored ÷ most any method scored on that same sequence")
    ax.set_xlim(0.25, 1.03)
    ax.axvline(1.0, color=S.INK3, lw=0.9, ls=(0, (4, 3)))
    ax.set_title("Two methods are scored on a subset of the frames", pad=8, fontsize=9.5)
    S.grid(ax, axis="x")

    # right: where COLMAP drops frames
    ax = axes[1]
    c = ok[ok.Method == "colmap/no_gt"]
    piv = (c.pivot_table(index="Asset", columns="Illumination", values="Reg", aggfunc="median")
           .reindex(columns=S.ILLUM_ORDER))
    piv = piv.reindex([a for a in S.ASSET_ORDER if a in piv.index]).dropna(how="all")
    A = piv.values.astype(float)
    ax.imshow(A, cmap=S.SEQ_CMAP, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.0%}", ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if np.isfinite(v) and v > 0.80 else S.INK)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8.5)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(np.arange(len(piv.columns) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(piv.index) + 1) - .5, minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.set_title("COLMAP registers fewest frames exactly where the scene is hardest",
                 pad=8, fontsize=9.5)

    fig.tight_layout()
    S.fig_title(fig, "Read every method comparison with this caveat",
                "COLMAP and VGGT-SLAM report poses only for the frames they registered, and the "
                "evaluator scores them on those frames alone; the feed-forward methods are scored "
                "on all of them. Their scores therefore sit on an easier, self-selected subset.")
    S.family_legend(fig, y=-0.07)
    S.save(fig, "22_frame_registration", OUT)


if __name__ == "__main__":
    fig_registration()
