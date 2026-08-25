"""How performance moves along each axis of scene variation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import style as S
from common import loade, axis_slice, OUT, CORE_ASSETS

S.setup()
D = {c: loade(c)[0] for c in ("pose", "depth", "cloud")}
M = S.METHOD_ORDER

PANELS = [("pose", "AUC@30", "up", "pose · AUC@30"),
          ("depth", "Relative MAE", "down", "depth · relative MAE"),
          ("cloud", "F-score @0.25m", "up", "cloud · F-score @0.25 m")]

# axis -> (levels, reference level, assets it is defined on, label)
AXES = {
    "Illumination": (S.ILLUM_ORDER, "sunny", None, "illumination"),
    "Standoff":     (S.STANDOFF_ORDER, "near", None, "trajectory"),
    "Soil":         (S.SOIL_ORDER, "verylow", ["glass_house2"], "soiling"),
    "Reflectivity": (S.REFL_ORDER, "0.0", ["glass_house2"], "reflectivity"),
}


def levels_table(cat, metric, axis, assets):
    """Method x level medians, pooled over everything else."""
    levels, ref, forced, _ = AXES[axis]
    df = D[cat]
    ok = df[df.Status == "ok"]
    ok = ok[ok.Asset.isin(forced or assets)]
    ok = axis_slice(ok, "Soil" if axis == "Soil" else
                        "Reflectivity" if axis == "Reflectivity" else axis)
    if axis in ("Soil", "Reflectivity"):
        ok = ok[ok[axis] != ""]
    # median within asset first, then across assets, so a big asset cannot dominate
    per = ok.groupby(["Asset", "Method", axis])[metric].median()
    t = per.groupby(["Method", axis]).mean().unstack(axis)
    return t.reindex(index=M, columns=[c for c in levels if c in t.columns])


def pct_change(t, ref, direction):
    """% change from the reference level, oriented so positive always means worse."""
    base = t[ref]
    d = (t.sub(base, axis=0)).div(base.abs(), axis=0) * 100
    return d if direction == "down" else -d


# ---------------------------------------------------------------- heatmaps
def fig_axis_heatmap(axis, n, assets=None, tag=""):
    levels, ref, forced, label = AXES[axis]
    assets = forced or assets or CORE_ASSETS
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    for ax, (cat, metric, direction, title) in zip(axes, PANELS):
        t = levels_table(cat, metric, axis, assets)
        d = pct_change(t, ref, direction)
        A = d.values.astype(float)
        im = ax.imshow(A, cmap=S.DIV_CMAP, vmin=-60, vmax=60, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=6.5, color=S.INK3)
                    continue
                ax.text(j, i, "ref" if d.columns[j] == ref else f"{v:+.0f}%",
                        ha="center", va="center", fontsize=6.6,
                        color="#ffffff" if abs(v) > 42 else S.INK)
        ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(d.columns, fontsize=8)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M if ax is axes[0] else [""] * len(M), fontsize=7.5)
        ax.set_xticks(np.arange(A.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(M) + 1) - .5, minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=1.8)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, pad=8, fontsize=9.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015,
                      ticks=[-60, -30, 0, 30, 60])
    cb.ax.set_yticklabels(["−60%", "−30%", "0", "+30%", "+60%"], fontsize=7.5)
    cb.set_label("worse than reference  →", fontsize=8, color=S.INK2)
    cb.outline.set_visible(False)
    scope = ("glass_house2 only — the one asset that varies this axis" if forced else
             "the three assets every method completed" if assets == CORE_ASSETS else
             "all seven assets — method rows cover different asset sets, so compare within a row")
    S.fig_title(fig, f"Response to {label}: change from {ref}",
                f"positive = worse than the {ref} reference; median within asset, then averaged "
                f"across assets. Scope: {scope}. Clipped at \u00b160%.")
    S.save(fig, f"{n}_axis_{axis.lower()}_heatmap{tag}", OUT)


# ---------------------------------------------------------------- small multiples
def fig_axis_small_multiples(axis, n, cat="pose", metric="AUC@30", direction="up",
                             assets=None, tag=""):
    levels, ref, forced, label = AXES[axis]
    assets = forced or assets or CORE_ASSETS
    t = levels_table(cat, metric, axis, assets)
    fig, axes = plt.subplots(2, 6, figsize=(14.2, 5.0), sharex=True, sharey=True)
    lo, hi = np.nanmin(t.values), np.nanmax(t.values)
    pad = (hi - lo) * 0.12
    x = np.arange(len(t.columns))
    for ax, m in zip(axes.ravel(), M):
        y = t.loc[m].values.astype(float)
        ax.plot(x, y, color=S.mcolor(m), marker="o", markersize=5,
                markeredgecolor=S.SURFACE, markeredgewidth=1.2, zorder=3)
        ax.axhline(y[list(t.columns).index(ref)], color=S.INK3, lw=0.8,
                   ls=(0, (3, 3)), zorder=1)
        ax.set_title(m, fontsize=8.5, pad=5)
        ax.set_ylim(lo - pad, hi + pad)
        S.grid(ax)
        drop = (y[-1] - y[list(t.columns).index(ref)])
        if direction == "up":
            worst = t.columns[int(np.nanargmin(y))]
            ax.text(0.97, 0.06, f"worst: {worst}", transform=ax.transAxes, ha="right",
                    fontsize=6.8, color=S.INK2)
    for k in range(len(M), axes.size):                      # unused panels
        axes.ravel()[k].axis("off")
    ncol = axes.shape[1]
    for c in range(ncol):                                   # x labels on the lowest live panel
        col = [axes[r, c] for r in range(axes.shape[0]) if r * ncol + c < len(M)]
        bot = col[-1]
        bot.set_xticks(x)
        bot.set_xticklabels(t.columns, rotation=35, ha="right", fontsize=7.5)
        bot.tick_params(labelbottom=True)
    axes[0, 0].set_ylabel(metric); axes[1, 0].set_ylabel(metric)
    fig.tight_layout()
    scope = ("glass_house2" if forced else
             "the three assets every method completed" if assets == CORE_ASSETS else "all seven assets")
    S.fig_title(fig, f"{cat} · {metric} across {label}, one panel per method",
                f"shared y-axis, so panel height is comparable; dashed line = the {ref} "
                f"reference. {scope}.")
    S.family_legend(fig, y=-0.05)
    S.save(fig, f"{n}_axis_{axis.lower()}_by_method{tag}", OUT)


if __name__ == "__main__":
    for i, ax in enumerate(["Illumination", "Standoff", "Soil", "Reflectivity"]):
        fig_axis_heatmap(ax, f"{7 + i * 2:02d}")
        fig_axis_small_multiples(ax, f"{8 + i * 2:02d}")
    # appendix: the same two axes over every asset, where the long/large scenes are in play
    for i, ax in enumerate(("Illumination", "Standoff")):
        fig_axis_heatmap(ax, f"A{1 + i * 2}", assets=S.ASSET_ORDER, tag="_all_assets")
        fig_axis_small_multiples(ax, f"A{2 + i * 2}", assets=S.ASSET_ORDER, tag="_all_assets")
