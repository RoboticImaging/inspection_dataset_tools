"""Method strengths and weaknesses."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import style as S
from common import loade, OUT, CORE_ASSETS

S.setup()
D = {c: loade(c)[0] for c in ("pose", "depth", "cloud")}
M = S.METHOD_ORDER


def cell(cat, metric, assets=CORE_ASSETS):
    """(Asset, Method) median over that cell's variants."""
    df = D[cat]
    ok = df[(df.Status == "ok") & (df.Asset.isin(assets))]
    return ok.groupby(["Asset", "Method"])[metric].median()


def across(cat, metric, assets=CORE_ASSETS):
    """One number per method: mean of its per-asset medians (so a 144-variant
    asset does not outweigh a 16-variant one)."""
    return cell(cat, metric, assets).unstack("Method").mean()


PANELS = [("pose", "AUC@30", "up", "pose  ·  AUC@30", "higher is better"),
          ("depth", "Relative MAE", "down", "depth  ·  relative MAE", "lower is better"),
          ("cloud", "F-score @0.25m", "up", "cloud  ·  F-score @0.25 m", "higher is better"),
          ("cloud", "Chamfer (m)", "down", "cloud  ·  Chamfer distance (m)", "lower is better")]


# ---------------------------------------------------------------- fig 3
def fig_headline_ranking():
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.2))
    for ax, (cat, metric, direction, title, sub) in zip(axes, PANELS):
        v = across(cat, metric).reindex(M)
        per = cell(cat, metric).unstack("Method").reindex(columns=M)
        order = v.sort_values(ascending=(direction == "up")).index
        y = np.arange(len(order))
        ax.barh(y, v[order], color=[S.mcolor(m) for m in order], height=0.62, zorder=2)
        for j, m in enumerate(order):                       # per-asset spread
            ax.scatter(per[m].dropna(), [j] * per[m].notna().sum(), s=13,
                       color=S.INK, alpha=0.55, zorder=4, linewidth=0)
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_xlabel(sub)
        S.grid(ax, axis="x")
        ax.set_title(title, pad=10)
        ax.set_xlim(0, None)
        ax.get_yticklabels()[-1].set_fontweight("bold")
    fig.tight_layout()
    S.fig_title(fig, "Method ranking on the three assets every method completed",
                "bar = mean of the per-asset medians (glass_house2, old_house, "
                "unfinished_build); dots = the three individual asset medians")
    S.family_legend(fig, y=-0.06)
    S.save(fig, "03_method_ranking_headline", OUT)


# ---------------------------------------------------------------- fig 4
def fig_rot_vs_trans():
    """Rotation is nearly solved; translation is where methods separate."""
    rra = [across("pose", f"RRA@{t} %").reindex(M) for t in (5, 15, 30)]
    rta = [across("pose", f"RTA@{t} %").reindex(M) for t in (5, 15, 30)]
    order = rta[0].sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    for ax, t, r, tr in zip(axes, (5, 15, 30), rra, rta):
        y = np.arange(len(order))
        ax.barh(y + 0.20, r[order], height=0.32, color=S.CAT[0], zorder=2)
        ax.barh(y - 0.20, tr[order], height=0.32, color=S.CAT[1], zorder=2)
        ax.set_yticks(y); ax.set_yticklabels(order)
        ax.set_xlim(0, 105); ax.set_xlabel("% of camera pairs within threshold")
        ax.set_title(f"threshold {t}°", pad=10)
        S.grid(ax, axis="x")
    fig.tight_layout()
    fig.legend(handles=[mpl.patches.Patch(color=S.CAT[0], label="rotation (RRA)"),
                        mpl.patches.Patch(color=S.CAT[1], label="translation (RTA)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    S.fig_title(fig, "Rotation is close to solved; translation is what separates the methods",
                "relative pose accuracy on the three complete assets, mean of per-asset "
                "medians; sorted by RTA@5\u00b0")
    S.save(fig, "04_rotation_vs_translation", OUT)


# ---------------------------------------------------------------- fig 5
def fig_cloud_pr():
    """Precision/recall at the three F-score radii."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, t in zip(axes, ["0.05m", "0.1m", "0.25m"]):
        p = across("cloud", f"Precision @{t}").reindex(M)
        r = across("cloud", f"Recall @{t}").reindex(M)
        offs = [(0, 10), (0, -14), (12, 3), (-12, 3)]
        for k, m in enumerate(M):
            ax.scatter(r[m], p[m], s=90, color=S.mcolor(m), zorder=3,
                       edgecolor=S.SURFACE, linewidth=1.4)
            dx, dy = offs[k % len(offs)]
            ax.annotate(m, (r[m], p[m]), fontsize=6.8, color=S.INK2,
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right" if dx < 0 else "center")
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color=S.INK3, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_title(f"radius {t}", pad=10)
        ax.set_xlim(0, None); ax.set_ylim(0, None)
        S.grid(ax, axis="both")
    fig.suptitle("Predicted clouds recall the surface far better than they stay on it",
                 x=0.006, ha="left", fontsize=13, fontweight="semibold", y=1.05)
    fig.text(0.006, 0.995, "every method sits far below the dashed parity line at every radius: "
             "recall ≫ precision. Precision at 5 cm is at the floor for all of them.",
             ha="left", fontsize=9, color=S.INK2)
    fig.tight_layout()
    S.save(fig, "05_cloud_precision_recall", OUT)


# ---------------------------------------------------------------- fig 6
def fig_method_asset_heatmaps():
    """Every headline metric, method x asset, on all seven assets."""
    panels = [("pose", "AUC@30", "up"), ("depth", "Relative MAE", "down"),
              ("depth", "Accuracy @1.05", "up"), ("cloud", "F-score @0.25m", "up"),
              ("cloud", "Chamfer (m)", "down"), ("pose", "ATE RMSE (m)", "down")]
    assets = S.ASSET_ORDER
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for ax, (cat, metric, d) in zip(axes.ravel(), panels):
        piv = cell(cat, metric, assets).unstack("Method").reindex(index=assets, columns=M)
        A = piv.values.astype(float)
        # rank-normalise within each asset so scenes of different scale are comparable
        R = np.full_like(A, np.nan)
        for i in range(A.shape[0]):
            row = A[i]
            fin = np.isfinite(row)
            if fin.sum() > 1:
                r = pd.Series(row[fin]).rank(ascending=(d == "down"))
                R[i, fin] = (r.values - 1) / (fin.sum() - 1)
        im = ax.imshow(R, cmap=S.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        fmt = "{:.3g}"
        S.annotate_cells(ax, A, fmt=fmt, cmap=S.SEQ_CMAP, vmin=0, vmax=1)
        # re-do labels against the rank colour, not the raw value
        for t in ax.texts: t.remove()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    dark = R[i, j] > 0.62
                    ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=6.2,
                            color="#ffffff" if dark else S.INK)
                else:
                    st = D[cat][(D[cat].Asset == assets[i]) & (D[cat].Method == M[j])].Status
                    lbl = "OOM" if len(st) and (st == "OOM").any() else "--"
                    ax.text(j, i, lbl, ha="center", va="center", fontsize=6,
                            color=S.STATUS["OOM"] if lbl == "OOM" else S.INK3)
        ax.set_xticks(range(len(M))); ax.set_xticklabels(M, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(assets))); ax.set_yticklabels(assets, fontsize=7.5)
        ax.set_xticks(np.arange(len(M) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(assets) + 1) - .5, minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{cat}  ·  {metric}  ({'↑' if d == 'up' else '↓'})", pad=8, fontsize=9.5)
    fig.tight_layout()
    S.fig_title(fig, "Every headline metric, method by asset",
                "cell = median over that asset's variants; shading is the method's rank within "
                "the asset (dark = best), so scenes of different scale stay comparable. "
                "OOM = out of memory, -- = not run.")
    S.save(fig, "06_method_asset_heatmaps", OUT)


if __name__ == "__main__":
    fig_headline_ranking()
    fig_rot_vs_trans()
    fig_cloud_pr()
    fig_method_asset_heatmaps()
