"""Coverage and the scalability (OOM) story."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import style as S
from common import loade, OUT, CORE_ASSETS

S.setup()
OUTD = OUT

pose, _ = loade("pose")
depth, _ = loade("depth")
cloud, _ = loade("cloud")

ASSETS = [a for a in S.ASSET_ORDER if a in set(pose.Asset)]
METHODS = S.METHOD_ORDER

# median sequence length per asset, from whatever rows scored
seqlen = pose[pose.Status == "ok"].groupby("Asset").Frames.median()
# assets whose every method OOM'd have no ok row there; fall back across categories
for a in ASSETS:
    if a not in seqlen.index:
        seqlen[a] = np.nan


# ---------------------------------------------------------------- fig 1
def fig_coverage_grid():
    """Method x asset run status, one tile per cell, three metric categories."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    code = {"ok": 0, "OOM": 1, "missing": 2}
    cmap = mpl.colors.ListedColormap([S.STATUS["ok"], S.STATUS["OOM"], S.STATUS["missing"]])
    for ax, (name, df) in zip(axes, [("pose", pose), ("depth", depth), ("cloud", cloud)]):
        M = np.full((len(METHODS), len(ASSETS)), np.nan)
        frac = np.zeros_like(M)
        for i, m in enumerate(METHODS):
            for j, a in enumerate(ASSETS):
                s = df[(df.Method == m) & (df.Asset == a)].Status
                if not len(s):
                    continue
                top = s.value_counts().idxmax()
                M[i, j] = code[top]
                frac[i, j] = (s == "ok").mean()
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        for i in range(len(METHODS)):
            for j in range(len(ASSETS)):
                f = frac[i, j]
                if 0 < f < 1:            # partial: print the ok fraction
                    ax.text(j, i, f"{f:.0%}", ha="center", va="center",
                            fontsize=6.5, color="#ffffff")
        ax.set_xticks(range(len(ASSETS)))
        ax.set_xticklabels(ASSETS, rotation=40, ha="right")
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_title(name)
        ax.set_xticks(np.arange(len(ASSETS) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(METHODS) + 1) - .5, minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    S.fig_title(fig, "Run status by method and asset",
                "one tile per method x asset, coloured by that cell's dominant status; a "
                "percentage marks a partly-complete cell (share of its variants scored)")
    fig.legend(handles=[Patch(color=S.STATUS[k], label=v) for k, v in
                        [("ok", "scored"), ("OOM", "out of memory"), ("missing", "not run yet")]],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    S.save(fig, "01_coverage_grid", OUTD)


# ---------------------------------------------------------------- fig 2
def fig_oom_vs_length():
    """Why cells go red: sequence length, not scene content."""
    LEN = {"glass_house2": 97, "old_house": 99, "unfinished_build": 99,
           "crane": 522, "bridge": 566, "ship": 803, "office2": 1042}
    rows = []
    for m in METHODS:
        for a in ASSETS:
            st = pose[(pose.Method == m) & (pose.Asset == a)].Status
            if len(st):
                rows.append(dict(Method=m, Asset=a, status=st.value_counts().idxmax(), n=LEN[a]))
    d = pd.DataFrame(rows)

    chunked = [m for m in METHODS if m in S.CHUNKED]
    oneshot = [m for m in METHODS if m not in S.CHUNKED]
    top_to_bottom = chunked + oneshot                 # chunked block on top
    n = len(top_to_bottom)
    ypos = {m: n - 1 - i for i, m in enumerate(top_to_bottom)}

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for m in top_to_bottom:
        ax.axhline(ypos[m], color=S.GRID, lw=0.8, zorder=0)
    split = n - len(chunked) - 0.5                    # boundary between the two blocks
    ax.axhline(split, color=S.INK3, lw=0.9, ls=(0, (4, 3)), zorder=1)

    for stt, mk, lbl in [("ok", "o", "scored"), ("OOM", "X", "out of memory"),
                         ("missing", "s", "not run yet")]:
        sub = d[d.status == stt]
        ax.scatter(sub.n, [ypos[m] for m in sub.Method], s=85, marker=mk,
                   color=S.STATUS[stt], edgecolor=S.SURFACE, linewidth=1.5, zorder=3, label=lbl)

    ax.set_yticks([ypos[m] for m in top_to_bottom])
    ax.set_yticklabels(top_to_bottom)
    ax.set_xscale("log")
    ax.set_xlim(70, 1500)
    ax.set_ylim(-0.8, n - 0.3)
    ax.set_xticks([100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("sequence length (frames)")

    blend = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.015, (n - 1 + split) / 2, "chunked /\nstreaming", transform=blend,
            fontsize=8.5, color=S.INK2, va="center", linespacing=1.4, fontweight="bold")
    ax.text(1.015, split / 2, "one-shot", transform=blend,
            fontsize=8.5, color=S.INK2, va="center", fontweight="bold")

    for x, lbl, dy in [(98, "3 short assets\n(~100 fr)", 0), (522, "crane", 0),
                       (566, "bridge", -1), (803, "ship", 0), (1042, "office2", 0)]:
        ax.annotate(lbl, xy=(x, n - 0.55 + dy * 0.42), fontsize=7, color=S.INK3,
                    ha="center", va="bottom", linespacing=1.3)

    S.titled(ax, "Out-of-memory failure is set by sequence length, not by the scene",
             "every one-shot method dies between 100 and 800 frames; every chunked or "
             "streaming method clears all seven assets", pad=32)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.26))
    S.save(fig, "02_oom_vs_sequence_length", OUTD)


if __name__ == "__main__":
    fig_coverage_grid()
    fig_oom_vs_length()
