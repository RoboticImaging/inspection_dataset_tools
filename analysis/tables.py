"""Every table, written as markdown (review) and LaTeX (paper) side by side."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import style as S
from common import loade, axis_slice, TAB, CORE_ASSETS

D = {c: loade(c)[0] for c in ("pose", "depth", "cloud")}
M = S.METHOD_ORDER
LEN = {"glass_house2": 97, "old_house": 99, "unfinished_build": 99,
       "crane": 522, "bridge": 566, "ship": 803, "office2": 1042}
TAB.mkdir(parents=True, exist_ok=True)
OUT_MD, OUT_TEX = [], []


def emit(name, df, caption, note="", float_fmt="{:.3g}"):
    OUT_MD.append(f"\n## {name}\n\n**{caption}**\n\n"
                  + (f"_{note}_\n\n" if note else "")
                  + df.to_markdown(floatfmt=".3g") + "\n")
    tex = df.to_latex(float_format=lambda x: float_fmt.format(x), escape=True,
                      caption=caption, label=f"tab:{name.split('.')[0].lower()}")
    OUT_TEX.append(tex)
    print(f"\n### {name}: {caption}")
    print(df.to_string(float_format=lambda x: float_fmt.format(x)))


def cellmed(cat, metric, assets):
    ok = D[cat][(D[cat].Status == "ok") & (D[cat].Asset.isin(assets))]
    return ok.groupby(["Asset", "Method"])[metric].median().unstack("Method")


def across(cat, metric, assets=CORE_ASSETS):
    return cellmed(cat, metric, assets).mean()


# ---------------------------------------------------------------- T1
def t1_main():
    cols = {
        "AUC@30 ↑": across("pose", "AUC@30"),
        "RTA@5 ↑": across("pose", "RTA@5 %"),
        "RRA@5 ↑": across("pose", "RRA@5 %"),
        "rel. MAE ↓": across("depth", "Relative MAE"),
        "Acc@1.05 ↑": across("depth", "Accuracy @1.05"),
        "F@0.25m ↑": across("cloud", "F-score @0.25m"),
        "Chamfer ↓": across("cloud", "Chamfer (m)"),
    }
    t = pd.DataFrame(cols).reindex(M)
    st = D["pose"].groupby(["Method", "Asset"]).Status.apply(lambda s: s.value_counts().idxmax()).unstack()
    t["assets done"] = [f"{int((st.loc[m] == 'ok').sum())}/7" for m in M]
    t["OOM on"] = [int((st.loc[m] == "OOM").sum()) for m in M]
    t.index.name = "method"
    emit("T1_main_results", t.round(3),
         "Headline scores on the three assets every method completed",
         "mean of the per-asset medians (glass_house2, old_house, unfinished_build). "
         "The last two columns come from all seven assets and show how far each method scales. "
         "COLMAP and VGGT-SLAM are scored only on the frames they registered (see T6).")
    return t


# ---------------------------------------------------------------- T2
def t2_scaling():
    st = D["pose"].groupby(["Method", "Asset"]).Status.apply(lambda s: s.value_counts().idxmax()).unstack()
    rows = []
    for m in M:
        okA = [a for a in st.columns if st.loc[m, a] == "ok"]
        oomA = [a for a in st.columns if st.loc[m, a] == "OOM"]
        rows.append(dict(method=m,
                         mode="chunked / streaming" if m in S.CHUNKED else "one-shot",
                         longest_ok=max([LEN[a] for a in okA], default=0),
                         shortest_OOM=min([LEN[a] for a in oomA], default=np.nan),
                         assets_ok=len(okA), assets_OOM=len(oomA),
                         assets_not_run=int((st.loc[m] == "missing").sum())))
    t = pd.DataFrame(rows).set_index("method").sort_values(["mode", "longest_ok"],
                                                          ascending=[True, False])
    emit("T2_scaling", t, "How far each method scales, in frames",
         "longest sequence scored and shortest sequence that ran out of GPU memory. "
         "Every one-shot method has a ceiling; no chunked or streaming method hit one.",
         "{:.0f}")
    return t


# ---------------------------------------------------------------- T3
def t3_assets():
    rows = []
    for a in sorted(LEN, key=LEN.get):
        r = {"asset": a, "frames": LEN[a]}
        st = D["pose"][D["pose"].Asset == a].groupby("Method").Status.apply(
            lambda s: s.value_counts().idxmax())
        r["methods scored"] = int((st == "ok").sum())
        r["methods OOM"] = int((st == "OOM").sum())
        for cat, metric, d, lbl in [("pose", "AUC@30", "up", "AUC@30"),
                                    ("depth", "Relative MAE", "down", "rel. MAE"),
                                    ("cloud", "F-score @0.25m", "up", "F@0.25m")]:
            v = cellmed(cat, metric, [a])
            if v.empty or v.isna().all().all():
                r[f"best {lbl}"] = "--"; r[f"median {lbl}"] = np.nan
                continue
            s = v.iloc[0].dropna()
            best = s.idxmax() if d == "up" else s.idxmin()
            r[f"best {lbl}"] = f"{best} ({s[best]:.3g})"
            r[f"median {lbl}"] = float(s.median())
        rows.append(r)
    t = pd.DataFrame(rows).set_index("asset")
    emit("T3_asset_difficulty", t, "Asset difficulty, ordered by sequence length",
         "'median' is over the methods that completed that asset - so the long assets are "
         "already scored by a survivor set of the more capable methods, and are still far worse.")
    return t


# ---------------------------------------------------------------- T4
def t4_axis_sensitivity():
    axes_ = [("Standoff", "trajectory", S.STANDOFF_ORDER, "near", None),
             ("Illumination", "illumination", S.ILLUM_ORDER, "sunny", None),
             ("Soil", "soiling", S.SOIL_ORDER, "verylow", ["glass_house2"]),
             ("Reflectivity", "reflectivity", S.REFL_ORDER, "0.0", ["glass_house2"])]
    rows = []
    for axis, label, levels, ref, forced in axes_:
        for cat, metric, d in [("pose", "AUC@30", "up"), ("depth", "Relative MAE", "down"),
                               ("cloud", "F-score @0.25m", "up")]:
            ok = D[cat][D[cat].Status == "ok"]
            ok = ok[ok.Asset.isin(forced or CORE_ASSETS)]
            ok = axis_slice(ok, axis)
            if axis in ("Soil", "Reflectivity"):
                ok = ok[ok[axis] != ""]
            per = ok.groupby(["Asset", "Method", axis])[metric].median()
            t = per.groupby(["Method", axis]).mean().unstack(axis)
            t = t.reindex(columns=[c for c in levels if c in t.columns])
            base = t[ref]
            pct = (t.sub(base, axis=0)).div(base.abs(), axis=0) * 100
            if d == "up":
                pct = -pct
            worst_level = pct.mean().idxmax()
            worst_method = pct.max(axis=1).idxmax()
            rows.append(dict(axis=label, metric=f"{cat} · {metric}",
                             median_worst_pct=float(pct.max(axis=1).median()),
                             mean_pct_at_worst_level=float(pct[worst_level].mean()),
                             worst_level=worst_level,
                             most_affected=f"{worst_method} ({pct.max(axis=1).max():+.0f}%)"))
    t = pd.DataFrame(rows).set_index(["axis", "metric"])
    t.columns = ["median worst-level cost %", "mean cost at worst level %",
                 "worst level", "most-affected method"]
    emit("T4_axis_sensitivity", t.round(1),
         "How much each axis of scene variation costs, relative to its reference level",
         "positive = worse than reference. Trajectory and illumination move the numbers; "
         "soiling and reflectivity move almost nothing except for one method each.")
    return t


# ---------------------------------------------------------------- T5
def t5_robustness():
    rows = []
    for m in M:
        r = {"method": m}
        for axis, levels, ref, forced, lbl in [
                ("Illumination", S.ILLUM_ORDER, "sunny", None, "illumination"),
                ("Soil", S.SOIL_ORDER, "verylow", ["glass_house2"], "soiling"),
                ("Reflectivity", S.REFL_ORDER, "0.0", ["glass_house2"], "reflectivity")]:
            ok = D["pose"][(D["pose"].Status == "ok") & (D["pose"].Method == m)]
            ok = ok[ok.Asset.isin(forced or CORE_ASSETS)]
            ok = axis_slice(ok, axis)
            if axis in ("Soil", "Reflectivity"):
                ok = ok[ok[axis] != ""]
            s = ok.groupby(["Asset", axis])["AUC@30"].median().groupby(axis).mean()
            s = s.reindex([x for x in levels if x in s.index])
            if s.empty or ref not in s:
                r[f"{lbl} worst"] = "--"; continue
            drop = (s[ref] - s.min()) / abs(s[ref]) * 100
            r[f"{lbl} worst"] = f"{s.idxmin()} (−{drop:.0f}%)"
        rows.append(r)
    t = pd.DataFrame(rows).set_index("method")
    emit("T5_robustness", t,
         "Worst level on each appearance axis, and what it costs in pose AUC@30",
         "core assets for illumination; glass_house2 for soiling and reflectivity, the only "
         "asset that varies them.")
    return t


# ---------------------------------------------------------------- T6
def t6_frames():
    p = D["pose"]
    ok = p[p.Status == "ok"].copy()
    key = ["Asset", "Traj", "Illumination", "Soil", "Reflectivity"]
    ok["Reg"] = ok.Frames / ok.groupby(key)["Frames"].transform("max")
    t = ok.groupby("Method").Reg.agg(median="median", mean="mean", min="min",
                                     below_90pct=lambda s: float((s < 0.9).mean()))
    t = t.reindex(M)
    t["below_90pct"] = (t["below_90pct"] * 100).round(1)
    t.columns = ["median reg. rate", "mean", "min", "% of runs below 90%"]
    t.index.name = "method"
    emit("T6_frame_registration", t.round(3),
         "Share of each sequence's frames a method is actually scored on",
         "1.0 = scored on every frame any method saw. COLMAP and VGGT-SLAM drop frames they "
         "cannot register and are then scored only on the rest, so their scores are not "
         "measured on the same footing as the rest.")
    return t


# ---------------------------------------------------------------- T7
def t7_completion():
    rows = []
    for cat in ("pose", "depth", "cloud"):
        df = D[cat]
        v = df.Status.value_counts()
        rows.append(dict(category=cat, scored=int(v.get("ok", 0)), OOM=int(v.get("OOM", 0)),
                         not_run=int(v.get("missing", 0)), total=len(df),
                         pct_scored=100 * v.get("ok", 0) / len(df)))
    t = pd.DataFrame(rows).set_index("category")
    emit("T7_completion", t.round(1), "How much of the benchmark grid is filled in",
         "the grid is 7 assets x 11 methods x that asset's variants = 4,048 rows per category.")
    return t


if __name__ == "__main__":
    t1_main(); t2_scaling(); t3_assets(); t4_axis_sensitivity()
    t5_robustness(); t6_frames(); t7_completion()
    (TAB / "tables.md").write_text("# Result tables\n" + "\n".join(OUT_MD))
    (TAB / "tables.tex").write_text("\n\n".join(OUT_TEX))
    print(f"\nwrote {TAB/'tables.md'} and {TAB/'tables.tex'}")
