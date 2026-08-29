"""Asset difficulty, the trajectory gap, and cross-cutting summaries."""
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
LEN = {"glass_house2": 97, "old_house": 99, "unfinished_build": 99,
       "crane": 522, "bridge": 566, "ship": 803, "office2": 1042}
# assets ordered by sequence length, which is also roughly the difficulty order
BY_LEN = sorted(LEN, key=LEN.get)

SCALE_FREE = [("pose", "AUC@30", "up", "pose · AUC@30"),
              ("depth", "Relative MAE", "down", "depth · relative MAE"),
              ("cloud", "F-score @0.25m", "up", "cloud · F-score @0.25 m")]


# ---------------------------------------------------------------- fig 15
def fig_asset_difficulty():
    """Only scale-free metrics, so a 60 m bridge and a 10 m house are comparable."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8))
    for ax, (cat, metric, d, title) in zip(axes, SCALE_FREE):
        ok = D[cat][D[cat].Status == "ok"]
        data, labels, ns = [], [], []
        for a in BY_LEN:
            v = ok[ok.Asset == a].groupby("Method")[metric].median().dropna()
            if len(v):
                data.append(v.values); labels.append(a); ns.append(len(v))
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color=S.INK, lw=1.6),
                        whiskerprops=dict(color=S.INK3, lw=1),
                        capprops=dict(color=S.INK3, lw=1), showfliers=False)
        for patch, a in zip(bp["boxes"], labels):
            patch.set(facecolor=S.SEQ[3] if LEN[a] < 200 else S.SEQ[8],
                      edgecolor="none", alpha=0.95)
        for i, (v, a) in enumerate(zip(data, labels)):
            ax.scatter(v, np.full(len(v), i + 1) + np.random.RandomState(0).uniform(-.13, .13, len(v)),
                       s=11, color=S.INK, alpha=0.5, zorder=4, linewidth=0)
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels([f"{a}\n{LEN[a]} fr · {n} methods" for a, n in zip(labels, ns)],
                           fontsize=7.5)
        ax.set_xlabel(f"{metric}  ({'higher' if d == 'up' else 'lower'} is better)")
        ax.set_title(title, pad=8, fontsize=9.5)
        S.grid(ax, axis="x")
        if d == "down":
            ax.set_xscale("log")
    fig.tight_layout()
    fig.legend(handles=[mpl.patches.Patch(color=S.SEQ[3], label="short sequence (~100 frames)"),
                        mpl.patches.Patch(color=S.SEQ[8], label="long sequence (500–1050 frames)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    S.fig_title(fig, "Asset difficulty, on scale-free metrics only",
                "one dot per method (its median over that asset's variants), box over the "
                "methods that ran. Assets ordered by sequence length. Note the survivorship: "
                "the long assets are scored only by the methods that survived them, and are "
                "still far worse.")
    S.save(fig, "15_asset_difficulty", OUT)


# ---------------------------------------------------------------- fig 16
def fig_trendspek_gap():
    """The real inspection trajectory against the synthetic stand-off orbits."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for ax, (cat, metric, d, title) in zip(axes, SCALE_FREE):
        ok = D[cat][D[cat].Status == "ok"]
        rows = []
        for a in BY_LEN:
            sub = ok[ok.Asset == a]
            prox = sub[sub.Standoff != "trendspek"].groupby("Method")[metric].median()
            tren = sub[sub.Standoff == "trendspek"].groupby("Method")[metric].median()
            j = pd.concat([prox.rename("proxy"), tren.rename("trendspek")], axis=1).dropna()
            if len(j):
                rows.append((a, j.proxy.median(), j.trendspek.median(), len(j)))
        y = np.arange(len(rows))
        for i, (a, p, t, n) in enumerate(rows):
            # offset the pair vertically so a near-tie cannot hide one marker behind the other
            ax.plot([p, t], [i + 0.12, i - 0.12], color=S.INK3, lw=1.4, zorder=2)
            ax.scatter([p], [i + 0.12], s=60, color=S.CAT[0], zorder=3,
                       edgecolor=S.SURFACE, linewidth=1.2)
            ax.scatter([t], [i - 0.12], s=60, color=S.CAT[1], zorder=3,
                       edgecolor=S.SURFACE, linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}  ({LEN[a]} fr)" for a, *_ in rows], fontsize=8)
        ax.set_xlabel(f"{metric}  ({'higher' if d == 'up' else 'lower'} is better)")
        ax.set_title(title, pad=8, fontsize=9.5)
        if d == "down":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0]))
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        S.grid(ax, axis="x")
    fig.tight_layout()
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=S.CAT[0],
                                   label="proxy-surface orbits (median of the three stand-offs)"),
                        plt.Line2D([], [], marker="o", ls="", color=S.CAT[1],
                                   label="trendspek (the real inspection trajectory)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10))
    S.fig_title(fig, "The real inspection trajectory is much harder than the synthetic orbits",
                "median across the methods that completed the asset; the gap widens with "
                "sequence length and scene size.")
    S.save(fig, "16_trendspek_vs_proxy", OUT)


# ---------------------------------------------------------------- fig 17
def _eta_sq(df, metric, factors):
    """Share of total variance each main effect explains, on a balanced full-factorial block."""
    d = df.dropna(subset=[metric])
    y = d[metric].astype(float)
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()
    out = {}
    for f in factors:
        g = d.groupby(f)[metric]
        out[f] = float((g.count() * (g.mean() - gm) ** 2).sum() / sst) if sst > 0 else np.nan
    out["residual / interactions"] = max(0.0, 1 - sum(out.values()))
    return out


def fig_variance_share():
    """Which knob actually moves the number: variance decomposition on glass_house2,
    the only asset that varies all four axes."""
    blocks = [("soiling block\n(4 traj × 4 illum × 4 soil)", "Soil"),
              ("reflectivity block\n(4 traj × 4 illum × 5 reflectivity)", "Reflectivity")]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4))
    for r, (bname, third) in enumerate(blocks):
        for c, (cat, metric, d, title) in enumerate(SCALE_FREE):
            ax = axes[r, c]
            ok = D[cat][(D[cat].Status == "ok") & (D[cat].Asset == "glass_house2")]
            ok = axis_slice(ok, third)
            ok = ok[ok[third] != ""]
            facs = ["Method", "Standoff", "Illumination", third]
            e = _eta_sq(ok, metric, facs)
            names = facs + ["residual / interactions"]
            vals = [e[k] * 100 for k in names]
            cols = [S.CAT[0], S.CAT[1], S.CAT[2], S.CAT[3], S.GRID]
            ypos = np.arange(len(names))[::-1]
            ax.barh(ypos, vals, color=cols, height=0.6, zorder=2)
            for yv, v in zip(ypos, vals):
                ax.text(v + 1.5, yv, f"{v:.1f}%", va="center", fontsize=7.5, color=S.INK2)
            ax.set_yticks(ypos)
            ax.set_yticklabels(["method"] + ["trajectory", "illumination",
                                             third.lower()] + ["residual /\ninteractions"],
                               fontsize=8)
            ax.set_xlim(0, 100)
            ax.set_title(title if r == 0 else "", pad=8, fontsize=9.5)
            S.grid(ax, axis="x")
            if c == 0:
                ax.text(-0.42, 0.5, bname, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.5, color=S.INK2, linespacing=1.5)
            if r == 1:
                ax.set_xlabel("% of total variance explained")
    fig.tight_layout()
    S.fig_title(fig, "Which knob moves the number: variance shares on glass_house2",
                "one-way main effects on the two balanced full-factorial blocks. The choice of "
                "method dominates; soiling and reflectivity are almost inert.")
    S.save(fig, "17_variance_share", OUT)


# ---------------------------------------------------------------- fig 18
def fig_robustness():
    """Typical performance against worst-case over the appearance axes."""
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    for ax, (cat, metric, d, title) in zip(axes, SCALE_FREE):
        ok = D[cat][(D[cat].Status == "ok") & (D[cat].Asset.isin(CORE_ASSETS))]
        per = ok.groupby(["Method", "Asset", "Illumination"])[metric].median()
        typ = per.groupby(["Method", "Asset"]).median().groupby("Method").mean()
        worst = (per.groupby(["Method", "Asset"]).min() if d == "up"
                 else per.groupby(["Method", "Asset"]).max()).groupby("Method").mean()
        # label offsets alternate around the marker so the dense cluster stays readable
        offs = [(0, 11), (0, -15), (13, 4), (-13, 4)]
        for k, m in enumerate([m for m in M if m in typ]):
            ax.scatter(typ[m], worst[m], s=95, color=S.mcolor(m), zorder=3,
                       edgecolor=S.SURFACE, linewidth=1.4)
            dx, dy = offs[k % len(offs)]
            ax.annotate(m, (typ[m], worst[m]), fontsize=6.8, color=S.INK2,
                        xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else "right" if dx < 0 else "center")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=S.INK3, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"typical {metric}")
        ax.set_ylabel(f"worst illumination")
        ax.set_title(title, pad=8, fontsize=9.5)
        S.grid(ax, axis="both")
    fig.tight_layout()
    S.fig_title(fig, "Robustness: typical performance against the worst illumination",
                "on the diagonal = illumination costs nothing; far off it = the method has a "
                "lighting condition it cannot handle. Core assets only.")
    S.family_legend(fig, y=-0.07)
    S.save(fig, "18_robustness_vs_typical", OUT)


# ---------------------------------------------------------------- fig 19
def fig_constancy():
    """Where the numbers do not move at all along an axis."""
    axes_list = ["Standoff", "Illumination", "Soil", "Reflectivity"]
    rows = []
    for cat in ("pose", "depth", "cloud"):
        df = D[cat]
        ok = df[df.Status == "ok"]
        metrics = [c for c in df.columns
                   if c not in ("Asset", "Method", "Traj", "Illumination", "Soil",
                                "Reflectivity", "Source", "Cloud source", "Status",
                                "Category", "Family", "StandoffM", "Standoff")]
        for k in metrics:
            rec = {"cat": cat, "metric": f"{cat} · {k}"}
            for axis in axes_list:
                s = axis_slice(ok, axis)
                if axis in ("Soil", "Reflectivity"):
                    s = s[s[axis] != ""]
                others = [a for a in axes_list if a != axis]
                g = s.groupby(["Asset", "Method"] + others)[k]
                rng, n = g.max() - g.min(), g.count()
                rng = rng[n > 1]
                rec[axis] = float((rng <= 1e-12).mean()) if len(rng) else np.nan
            rows.append(rec)
    t = pd.DataFrame(rows).set_index("metric")
    t = t[t[axes_list].max(axis=1) > 0.05].sort_values("Soil", ascending=False)
    A = t[axes_list].values.astype(float) * 100

    fig, ax = plt.subplots(figsize=(7.4, 0.30 * len(t) + 2.6))
    im = ax.imshow(A, cmap=S.SEQ_CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color="#ffffff" if np.isfinite(v) and v > 58 else S.INK)
    ax.set_xticks(range(len(axes_list)))
    ax.set_xticklabels(["trajectory", "illumination", "soiling", "reflectivity"], fontsize=8.5)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index, fontsize=7.5)
    ax.set_xticks(np.arange(len(axes_list) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(t) + 1) - .5, minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    S.fig_title(fig, "Metrics that do not move at all along an axis",
                "% of (asset, method, other-axis) cells whose value is bit-identical across "
                "every level of the axis. Only metrics flat somewhere are listed.")
    S.save(fig, "19_constancy_map", OUT)


# ---------------------------------------------------------------- fig 20
def fig_pose_vs_geometry():
    """Does a good trajectory buy good geometry?"""
    p = D["pose"]; c = D["cloud"]; d = D["depth"]
    key = ["Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity"]
    j = (p[p.Status == "ok"].set_index(key)[["AUC@30"]]
         .join(c[c.Status == "ok"].set_index(key)[["F-score @0.25m"]], how="inner")
         .join(d[d.Status == "ok"].set_index(key)[["Relative MAE"]], how="inner")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, (yk, ylab) in zip(axes, [("F-score @0.25m", "cloud F-score @0.25 m"),
                                     ("Relative MAE", "depth relative MAE")]):
        for m in M:
            s = j[j.Method == m]
            ax.scatter(s["AUC@30"], s[yk], s=7, color=S.mcolor(m), alpha=0.35, linewidth=0)
        r = j["AUC@30"].corr(j[yk], method="spearman")
        ax.set_xlabel("pose AUC@30"); ax.set_ylabel(ylab)
        ax.set_title(f"Spearman ρ = {r:+.2f}", pad=8, fontsize=9.5)
        if yk == "Relative MAE":
            ax.set_yscale("log")
        S.grid(ax, axis="both")
    fig.tight_layout()
    S.fig_title(fig, "Pose quality predicts geometry quality, but only loosely",
                f"one point per scored (asset, method, variant) row, n = {len(j):,}; colour = "
                "method family.")
    S.family_legend(fig, y=-0.07)
    S.save(fig, "20_pose_vs_geometry", OUT)


# ---------------------------------------------------------------- fig 21
def fig_sunrise_collapse():
    """A single condition on which every learned method fails at once - and COLMAP does not."""
    p = D["pose"]
    ok = p[(p.Status == "ok") & (p.Asset == "glass_house2")]
    blocks = [("reflectivity block  (soil held at its default, reflectivity swept 0.0–1.0)",
               "Reflectivity"),
              ("soiling block  (reflectivity held at its default, soil swept verylow–high)",
               "Soil")]
    cols = [(il, so) for il in S.ILLUM_ORDER for so in S.STANDOFF_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.6))
    for ax, (title, third) in zip(axes, blocks):
        b = axis_slice(ok, third)
        b = b[b[third] != ""]
        t = b.groupby(["Method", "Illumination", "Standoff"])["AUC@30"].median()
        A = np.array([[t.get((m, il, so), np.nan) for il, so in cols] for m in M], float)
        ax.imshow(A, cmap=S.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                v = A[i, j]
                ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.2f}", ha="center",
                        va="center", fontsize=6.2,
                        color="#ffffff" if np.isfinite(v) and v > 0.62 else S.INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([so for _, so in cols], rotation=45, ha="right", fontsize=6.8)
        ax.set_yticks(range(len(M))); ax.set_yticklabels(M, fontsize=7.5)
        ax.set_xticks(np.arange(len(cols) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(M) + 1) - .5, minor=True)
        ax.grid(which="minor", color=S.SURFACE, linewidth=1.6)
        ax.tick_params(which="minor", length=0)
        for k, il in enumerate(S.ILLUM_ORDER):          # illumination group headers
            ax.text(k * 4 + 1.5, -0.95, il, ha="center", va="bottom",
                    fontsize=8.5, color=S.INK, fontweight="bold")
            if k:
                ax.axvline(k * 4 - 0.5, color=S.INK3, lw=1.1)
        ax.set_title(title, pad=26, fontsize=9.5)
        if third == "Reflectivity":                      # ring the collapsed cells
            ax.add_patch(mpl.patches.Rectangle((7.5, -0.5), 2, len(M),
                                               fill=False, edgecolor=S.STATUS["OOM"],
                                               lw=2.0, zorder=5))
    fig.tight_layout(h_pad=3.2)
    S.fig_title(fig, "One condition breaks every learned method at once — and COLMAP shrugs",
                "pose AUC@30 on glass_house2, median over the swept third axis. In the "
                "reflectivity block, sunrise at the near and mid stand-offs (boxed) collapses "
                "all ten learned methods to 0.01–0.10 while COLMAP holds 0.98. The soiling "
                "block, same scene and same trajectories, shows nothing of the kind.")
    S.save(fig, "21_sunrise_collapse", OUT)


if __name__ == "__main__":
    fig_asset_difficulty()
    fig_trendspek_gap()
    fig_variance_share()
    fig_robustness()
    fig_constancy()
    fig_pose_vs_geometry()
    fig_sunrise_collapse()


