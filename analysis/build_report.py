"""Builds the HTML report, embedding the figures and generating every table from the data."""
import base64, sys, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import style as S
from common import loade, axis_slice, CORE_ASSETS

HERE = Path(__file__).resolve().parent
WEB = HERE / "figures_web"
OUT = Path("/tmp/claude-1003/-home-ngoncharov-repos-inspection-dataset-tools/"
           "c532a466-9d24-4f87-a620-ffe2f16d0d20/scratchpad/inspection_benchmark_report.html")

D = {c: loade(c)[0] for c in ("pose", "depth", "cloud")}
M = S.METHOD_ORDER
LEN = {"glass_house2": 97, "old_house": 99, "unfinished_build": 99,
       "crane": 522, "bridge": 566, "ship": 803, "office2": 1042}


def img(stem):
    b = (WEB / f"{stem}.jpg").read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def fig(stem, num, title, caption):
    return f"""
<figure class="plate" id="fig{num}">
  <img src="{img(stem)}" alt="{html.escape(title)}" loading="lazy">
  <figcaption><span class="fignum">Figure {num}</span> {caption}</figcaption>
</figure>"""


def table(df, caption, note="", highlight_col=None, index_name=None):
    """A DataFrame as an HTML table; numbers get tabular figures, index gets the row label style."""
    idx = index_name or (df.index.name or "")
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for i, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                txt = "&mdash;" if pd.isna(v) else (f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 10
                                                    else f"{v:.3g}")
                cells.append(f'<td class="num">{txt}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        cls = ' class="hl"' if highlight_col and i == highlight_col else ""
        rows.append(f"<tr{cls}><th scope='row'>{html.escape(str(i))}</th>{''.join(cells)}</tr>")
    return f"""
<div class="tablewrap">
  <table>
    <caption><span class="tabnum">{caption}</span>{f'<span class="tabnote">{note}</span>' if note else ''}</caption>
    <thead><tr><th scope="col">{html.escape(idx)}</th>{head}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""


# ---------------------------------------------------------------- data for tables
def across(cat, k, assets=CORE_ASSETS):
    ok = D[cat][(D[cat].Status == "ok") & (D[cat].Asset.isin(assets))]
    return ok.groupby(["Asset", "Method"])[k].median().unstack("Method").mean()


def t_main():
    st = D["pose"].groupby(["Method", "Asset"]).Status.apply(
        lambda s: s.value_counts().idxmax()).unstack()
    t = pd.DataFrame({
        "AUC@30 ↑": across("pose", "AUC@30"),
        "RTA@5 ↑": across("pose", "RTA@5 %"),
        "rel. MAE ↓": across("depth", "Relative MAE"),
        "Acc@1.05 ↑": across("depth", "Accuracy @1.05"),
        "F@0.25 m ↑": across("cloud", "F-score @0.25m"),
        "Chamfer ↓": across("cloud", "Chamfer (m)"),
    }).reindex(M).round(3)
    t["assets"] = [f"{int((st.loc[m]=='ok').sum())}/7" for m in M]
    t["OOM"] = [int((st.loc[m] == "OOM").sum()) or "—" for m in M]
    t.index.name = "method"
    return t


def t_scaling():
    st = D["pose"].groupby(["Method", "Asset"]).Status.apply(
        lambda s: s.value_counts().idxmax()).unstack()
    rows = []
    for m in M:
        okA = [a for a in st.columns if st.loc[m, a] == "ok"]
        oomA = [a for a in st.columns if st.loc[m, a] == "OOM"]
        rows.append(dict(method=m, inference="chunked / streaming" if m in S.CHUNKED else "one-shot",
                         **{"longest scored": max([LEN[a] for a in okA], default=0),
                            "first OOM at": min([LEN[a] for a in oomA], default=None),
                            "assets OOM": len(oomA) or "—"}))
    t = pd.DataFrame(rows).set_index("method")
    return t.sort_values(["inference", "longest scored"], ascending=[True, False])


def t_assets():
    rows = []
    for a in sorted(LEN, key=LEN.get):
        st = D["pose"][D["pose"].Asset == a].groupby("Method").Status.apply(
            lambda s: s.value_counts().idxmax())
        r = {"asset": a, "frames": LEN[a], "scored": int((st == "ok").sum()),
             "OOM": int((st == "OOM").sum()) or "—"}
        for cat, k, d, lbl in [("pose", "AUC@30", "up", "AUC@30"),
                               ("depth", "Relative MAE", "down", "rel. MAE"),
                               ("cloud", "F-score @0.25m", "up", "F@0.25 m")]:
            ok = D[cat][(D[cat].Status == "ok") & (D[cat].Asset == a)]
            s = ok.groupby("Method")[k].median().dropna()
            if s.empty:
                r[f"median {lbl}"] = np.nan; r[f"best {lbl}"] = "—"; continue
            b = s.idxmax() if d == "up" else s.idxmin()
            r[f"median {lbl}"] = round(float(s.median()), 3)
            r[f"best {lbl}"] = f"{b} · {s[b]:.3g}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("asset")


def t_axis():
    specs = [("Standoff", "trajectory", S.STANDOFF_ORDER, "near", None),
             ("Illumination", "illumination", S.ILLUM_ORDER, "sunny", None),
             ("Soil", "soiling", S.SOIL_ORDER, "verylow", ["glass_house2"]),
             ("Reflectivity", "reflectivity", S.REFL_ORDER, "0.0", ["glass_house2"])]
    rows = []
    for axis, label, levels, ref, forced in specs:
        rec = {"axis": label, "reference": ref}
        for cat, k, d, lbl in [("pose", "AUC@30", "up", "pose"),
                               ("depth", "Relative MAE", "down", "depth"),
                               ("cloud", "F-score @0.25m", "up", "cloud")]:
            ok = D[cat][D[cat].Status == "ok"]
            ok = ok[ok.Asset.isin(forced or CORE_ASSETS)]
            ok = axis_slice(ok, axis)
            if axis in ("Soil", "Reflectivity"):
                ok = ok[ok[axis] != ""]
            per = ok.groupby(["Asset", "Method", axis])[k].median()
            t = per.groupby(["Method", axis]).mean().unstack(axis)
            t = t.reindex(columns=[c for c in levels if c in t.columns])
            pct = (t.sub(t[ref], axis=0)).div(t[ref].abs(), axis=0) * 100
            if d == "up":
                pct = -pct
            worst = pct.max(axis=1)
            rec[f"{lbl}: median cost"] = f"{worst.median():+.0f}%"
            rec[f"{lbl}: worst method"] = f"{worst.idxmax()} {worst.max():+.0f}%"
        rows.append(rec)
    return pd.DataFrame(rows).set_index("axis")


def t_frames():
    ok = D["pose"][D["pose"].Status == "ok"].copy()
    key = ["Asset", "Traj", "Illumination", "Soil", "Reflectivity"]
    ok["Reg"] = ok.Frames / ok.groupby(key)["Frames"].transform("max")
    t = ok.groupby("Method").Reg.agg(median="median", mean="mean", worst="min")
    t["runs below 90%"] = ok.groupby("Method").Reg.apply(lambda s: f"{(s < 0.9).mean():.0%}")
    return t.reindex(M).round(3).rename_axis("method")


def t_completion():
    rows = []
    for cat in ("pose", "depth", "cloud"):
        v = D[cat].Status.value_counts()
        rows.append(dict(metric=cat, scored=int(v.get("ok", 0)), OOM=int(v.get("OOM", 0)),
                         **{"not run": int(v.get("missing", 0)),
                            "grid": len(D[cat]),
                            "% filled": f"{100*v.get('ok',0)/len(D[cat]):.0f}%"}))
    return pd.DataFrame(rows).set_index("metric")


# ---------------------------------------------------------------- page
CSS = """
:root{
  color-scheme: light;
  --ground:#f6f6f3; --plate:#fcfcfb; --sunk:#efeeea;
  --ink:#15171c; --ink2:#54565e; --ink3:#8b8d93;
  --rule:#dedcd6; --rule2:#e9e7e2;
  --accent:#1c5cab; --accent-soft:#e3ecf9;
  --caution:#9e2b28; --caution-soft:#f8eae9;
  --good:#146b4e;
  /* raster figures carry a light background of their own; the mount matches it in both
     themes deliberately, so a light plate never sits on a dark ground with a seam */
  --fig-mount:#fcfcfb; --fig-edge:#dedcd6;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --ground:#141519; --plate:#1b1d22; --sunk:#212429;
    --ink:#eceae5; --ink2:#a8aab1; --ink3:#7c7f87;
    --rule:#2c2f36; --rule2:#25282e;
    --accent:#6da7ec; --accent-soft:#1b2a3f;
    --caution:#e88a88; --caution-soft:#33211f;
    --good:#4fbf95;
    --fig-mount:#fcfcfb; --fig-edge:#3a3d45;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --ground:#141519; --plate:#1b1d22; --sunk:#212429;
  --ink:#eceae5; --ink2:#a8aab1; --ink3:#7c7f87;
  --rule:#2c2f36; --rule2:#25282e;
  --accent:#6da7ec; --accent-soft:#1b2a3f;
  --caution:#e88a88; --caution-soft:#33211f;
  --good:#4fbf95;
  --fig-mount:#fcfcfb; --fig-edge:#3a3d45;
}

*{box-sizing:border-box;}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Serif",Georgia,"Times New Roman",serif;
  font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.sans{font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.mono,code,.num,.chip,.fignum,.tabnum,.eyebrow,th,figcaption,caption{
  font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}

/* ---- shell ---------------------------------------------------- */
.wrap{max-width:1240px; margin:0 auto; padding:0 28px 120px;}
.col{max-width:68ch; margin-inline:auto;}
.plate-wide{max-width:1180px; margin-inline:auto;}

/* ---- masthead ------------------------------------------------- */
header.masthead{
  border-bottom:1px solid var(--rule);
  padding:64px 0 40px; margin-bottom:8px;
}
.eyebrow{
  font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--ink3); display:block; margin-bottom:20px;
}
h1{
  font-family:"IBM Plex Sans",sans-serif; font-weight:600;
  font-size:clamp(30px,4.6vw,50px); line-height:1.08; letter-spacing:-.021em;
  margin:0 0 20px; text-wrap:balance; max-width:20ch;
}
.standfirst{font-size:19.5px; line-height:1.55; color:var(--ink2); max-width:60ch; margin:0;}

.factbar{
  display:flex; flex-wrap:wrap; gap:0; margin-top:40px;
  border-top:1px solid var(--rule);
}
.fact{
  flex:1 1 150px; padding:18px 22px 16px; border-right:1px solid var(--rule);
}
.fact:last-child{border-right:0;}
.fact .v{
  font-family:"IBM Plex Sans",sans-serif; font-weight:600; font-size:27px;
  letter-spacing:-.02em; display:block; line-height:1.1;
  font-variant-numeric:tabular-nums;
}
.fact .k{font-size:11.5px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink3); font-family:"IBM Plex Mono",monospace; display:block; margin-top:7px;}

/* ---- sections -------------------------------------------------- */
section{padding-top:64px;}
h2{
  font-family:"IBM Plex Sans",sans-serif; font-weight:600;
  font-size:clamp(23px,2.7vw,31px); line-height:1.18; letter-spacing:-.017em;
  margin:0 0 8px; text-wrap:balance;
}
h2 .n{
  font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--accent);
  display:block; letter-spacing:.12em; margin-bottom:12px; font-weight:400;
}
h3{
  font-family:"IBM Plex Sans",sans-serif; font-weight:600; font-size:19px;
  line-height:1.3; letter-spacing:-.008em; margin:44px 0 6px; text-wrap:balance;
}
h4{
  font-family:"IBM Plex Mono",monospace; font-weight:500; font-size:12px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink3);
  margin:36px 0 10px;
}
p{margin:0 0 18px;}
a{color:var(--accent); text-underline-offset:2px; text-decoration-thickness:1px;}
strong{font-weight:600; color:var(--ink);}
.lede{font-size:18px; color:var(--ink2);}

ul,ol{margin:0 0 18px; padding-left:1.15em;}
li{margin-bottom:9px;}
li::marker{color:var(--ink3);}

/* ---- finding block --------------------------------------------- */
.finding{
  border-top:2px solid var(--ink); padding-top:14px; margin:46px 0 26px;
}
.finding h3{margin-top:0;}
.scope{
  display:flex; flex-wrap:wrap; gap:7px; margin:12px 0 16px;
}
.chip{
  font-size:11px; letter-spacing:.045em; padding:3px 9px 4px;
  border:1px solid var(--rule); border-radius:2px; color:var(--ink2);
  background:var(--plate); white-space:nowrap;
}
.chip.ev{border-color:var(--accent); color:var(--accent); background:var(--accent-soft);}
.chip.warn{border-color:var(--caution); color:var(--caution); background:var(--caution-soft);}

/* ---- caveat ---------------------------------------------------- */
.caveat{
  border-left:3px solid var(--caution); background:var(--caution-soft);
  padding:22px 26px 6px; margin:30px 0;
}
.caveat h3{margin-top:0; color:var(--caution); font-size:17px;}
.caveat p:last-child{margin-bottom:16px;}

.note{
  border-left:2px solid var(--rule); padding:2px 0 2px 20px; margin:26px 0;
  color:var(--ink2); font-size:15.5px;
}

/* ---- figures ---------------------------------------------------- */
figure.plate{margin:44px 0 40px;}
figure.plate img{
  display:block; width:100%; height:auto; border-radius:2px;
  background:var(--fig-mount); border:1px solid var(--fig-edge);
}
figcaption{
  font-size:12.5px; line-height:1.62; color:var(--ink2);
  margin-top:12px; max-width:96ch;
}
.fignum{
  color:var(--accent); letter-spacing:.06em; margin-right:9px; white-space:nowrap;
}

/* ---- tables ----------------------------------------------------- */
.tablewrap{overflow-x:auto; margin:34px 0 38px; -webkit-overflow-scrolling:touch;}
table{border-collapse:collapse; width:100%; font-size:13px; min-width:560px;
      background:var(--plate); font-family:"IBM Plex Mono",monospace;}
caption{
  caption-side:top; text-align:left; font-size:12.5px; color:var(--ink2);
  padding:0 0 12px; line-height:1.6;
}
.tabnum{color:var(--ink); font-weight:500;}
.tabnote{display:block; color:var(--ink3); margin-top:4px; font-size:12px;}
thead th{
  text-align:right; font-weight:500; font-size:11px; letter-spacing:.05em;
  color:var(--ink3); border-bottom:1px solid var(--ink); padding:9px 11px;
  white-space:nowrap; text-transform:none;
}
thead th:first-child{text-align:left;}
tbody th{
  text-align:left; font-weight:500; color:var(--ink); padding:8px 11px;
  white-space:nowrap; border-bottom:1px solid var(--rule2);
}
tbody td{
  padding:8px 11px; border-bottom:1px solid var(--rule2); color:var(--ink2);
  text-align:right; white-space:nowrap;
}
td.num{font-variant-numeric:tabular-nums;}
tbody tr:hover{background:var(--sunk);}
tbody tr.hl th, tbody tr.hl td{background:var(--accent-soft); color:var(--ink);}

/* ---- suggestion cards -------------------------------------------- */
.cards{display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
       grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); margin:30px 0 10px;}
.card{background:var(--plate); padding:22px 24px 20px;}
.card .where{
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink3); display:block; margin-bottom:9px;
}
.card h4{margin:0 0 8px; font-family:"IBM Plex Sans",sans-serif; font-size:16px;
  text-transform:none; letter-spacing:-.005em; color:var(--ink); font-weight:600;}
.card p{font-size:14.5px; line-height:1.55; margin:0; color:var(--ink2);}

/* ---- footer ------------------------------------------------------ */
footer{border-top:1px solid var(--rule); margin-top:80px; padding-top:32px;
  font-size:13.5px; color:var(--ink3);}
footer code{font-size:12.5px; background:var(--sunk); padding:1px 5px; border-radius:2px;
  color:var(--ink2);}

hr.rule{border:0; border-top:1px solid var(--rule); margin:0;}

@media (max-width:720px){
  .wrap{padding:0 18px 80px;}
  body{font-size:16px;}
  .fact{flex-basis:50%; border-right:0; border-bottom:1px solid var(--rule);}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important;}}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px;}
"""


def build():
    T = {}
    T["main"] = table(t_main(), "Table 1 — Headline scores, on the three assets every method completed",
                      "Mean of the per-asset medians over glass_house2, old_house and unfinished_build. "
                      "The last two columns are over all seven assets. ↑ higher is better, ↓ lower is better.",
                      highlight_col="da3_streaming", index_name="method")
    T["scaling"] = table(t_scaling(), "Table 2 — How far each method scales, in frames",
                         "Longest sequence scored, and the shortest sequence on which it ran out of GPU memory. "
                         "mpsfm has no OOM because it was never launched on a long asset.",
                         index_name="method")
    T["assets"] = table(t_assets(), "Table 3 — Asset difficulty, ordered by sequence length",
                        "'median' is taken over the methods that completed the asset, so the long assets are "
                        "already scored by a survivor set of the more capable methods.", index_name="asset")
    T["axis"] = table(t_axis(), "Table 4 — What each axis of variation costs",
                      "Median over methods of each method's worst level, as a percentage change from the "
                      "reference. Positive = worse. Trajectory and illumination scope: the three complete "
                      "assets. Soiling and reflectivity scope: glass_house2, the only asset that varies them.",
                      index_name="axis")
    T["frames"] = table(t_frames(), "Table 5 — Share of each sequence's frames a method is scored on",
                        "1.00 = scored on every frame any method saw on that sequence.", index_name="method")
    T["completion"] = table(t_completion(), "Table 6 — How much of the benchmark grid is filled in",
                            "The grid is 7 assets × 11 methods × that asset's variants = 4,048 rows per "
                            "metric category.", index_name="metric")

    body = f"""
<div class="wrap">

<header class="masthead">
  <div class="col">
    <span class="eyebrow">Inspection dataset · benchmark analysis · 25 August 2026</span>
    <h1>What the inspection benchmark results actually say</h1>
    <p class="standfirst">Eleven reconstruction methods, seven assets, four axes of scene variation.
    Two findings dominate everything else: sequence length decides which methods run at all, and the
    axes the benchmark was built to probe &mdash; soiling and reflectivity &mdash; turn out to move almost
    nothing.</p>
  </div>
  <div class="factbar plate-wide">
    <div class="fact"><span class="v">57%</span><span class="k">of the grid scored</span></div>
    <div class="fact"><span class="v">11</span><span class="k">methods</span></div>
    <div class="fact"><span class="v">7</span><span class="k">assets</span></div>
    <div class="fact"><span class="v">4</span><span class="k">variation axes</span></div>
    <div class="fact"><span class="v">2,312</span><span class="k">scored runs per metric</span></div>
  </div>
</header>

<section id="caveats">
  <div class="col">
    <h2><span class="n">00 &middot; Read first</span>Two things gate every number below</h2>
    <p class="lede">Neither is a reason to distrust the results, but both change what a given
    comparison is allowed to claim. Anything that goes into the paper should carry them.</p>

    <div class="caveat">
      <h3>Only 57% of the grid is filled, and the gaps are not random</h3>
      <p>Of 4,048 possible rows per metric, 2,312 are scored, 880 ran out of GPU memory, and 856 were
      never run. The empty cells cluster: <strong>office2 is effectively absent</strong> &mdash; nine of
      eleven methods have no row at all, and the two that do cover 14% and 25% of its variants. Read no
      office2 conclusion from this data. <strong>mpsfm</strong> was only ever run on the three short
      assets. Out-of-memory failures are not scattered either; they are a clean function of sequence
      length (&sect;01).</p>
      <p>Every method-vs-method comparison in this report is therefore restricted by default to
      <strong>glass_house2, old_house and unfinished_build</strong> &mdash; the three assets all eleven
      methods completed. Where a figure widens to all seven, it says so.</p>
    </div>

    <div class="caveat">
      <h3>COLMAP and VGGT-SLAM are not scored on the same frames as everyone else</h3>
      <p>Both report poses only for the frames they successfully register, and the evaluator scores
      them on those frames alone. COLMAP's median coverage is 96%, but its mean is 83% and its worst
      run is 31% &mdash; and it drops most frames exactly where the scene is hardest (48% of crane at
      sunrise, 76% of bridge at sunrise). The feed-forward methods are scored on every frame.</p>
      <p>So COLMAP's leading pose numbers in Table&nbsp;1, and its apparent indifference to
      illumination in &sect;03, are measured on a self-selected subset. The rank-order correlation
      between its coverage and its AUC is essentially zero (&rho; = &minus;0.02), so this is a
      comparability caveat rather than a demonstrated inflation &mdash; but a paper claiming
      &ldquo;COLMAP is the most accurate&rdquo; needs to say on what.</p>
    </div>
    {T["frames"]}
  </div>
  {fig("22_frame_registration", 1, "Frame registration rates",
       "Left: the share of each sequence's frames each method is scored on, one dot per run. "
       "Nine methods sit at 1.0. Right: where COLMAP drops frames &mdash; the pattern tracks scene "
       "difficulty, not method settings.")}
</section>

<hr class="rule">

<section id="scaling">
  <div class="col">
    <h2><span class="n">01 &middot; Coverage</span>Sequence length, not scene content, decides which methods run</h2>

    <div class="finding">
      <h3>Every one-shot method has a frame ceiling; no chunked method hit one</h3>
      <div class="scope">
        <span class="chip ev">7 assets · 11 methods</span>
        <span class="chip">880 OOM rows per metric</span>
        <span class="chip">Fig 2–3, Tables 2 &amp; 6</span>
      </div>
      <p>Out-of-memory failure is an <em>asset-level</em> property: when a method dies on an asset it
      dies on all sixteen (or all 144) of its variants, never some. Sorting the assets by frame count
      lines the failures up exactly. <strong>da3</strong> and <strong>vggt-omega</strong> clear ~100
      frames and die at 522. <strong>vggt</strong> and <strong>pi3</strong> clear 522 and die at
      566&nbsp;/&nbsp;803. Every method that chunks or streams &mdash; colmap, da3_streaming, vggt-long,
      vggt-slam, pi3x, pi-long &mdash; clears all of them, with pi3x the single exception at office2's
      1,042 frames.</p>
      <p>This is the most practically consequential result in the set. For an inspection workload,
      where sequences are hundreds to thousands of frames, four of the eleven benchmarked methods are
      simply not candidates on this hardware, whatever their accuracy on a hundred frames.</p>
    </div>
  </div>
  {fig("01_coverage_grid", 2, "Run status by method and asset",
       "The same picture holds for all three metric categories. Note the office2 column: it is almost "
       "entirely empty, and the two partial cells are 14% and 25% complete.")}
  {fig("02_oom_vs_sequence_length", 3, "OOM against sequence length",
       "Every red cross falls to the right of the method's last green dot, and the split between the "
       "two blocks is exactly the chunked/one-shot split.")}
  <div class="col">
    {T["scaling"]}
    {T["completion"]}
  </div>
</section>

<hr class="rule">

<section id="methods">
  <div class="col">
    <h2><span class="n">02 &middot; Methods</span>Strengths and weaknesses</h2>

    <div class="finding">
      <h3>da3_streaming is the only method that is both accurate and scalable</h3>
      <div class="scope">
        <span class="chip ev">3 assets · 11 methods</span>
        <span class="chip">mean of per-asset medians</span>
        <span class="chip">Fig 4, Table 1</span>
      </div>
      <p>On the three complete assets, <strong>da3</strong> wins depth (relative MAE 0.027) and cloud
      quality (F@0.25&nbsp;m 0.658, Chamfer 0.282&nbsp;m) and comes second on pose. <strong>colmap</strong>
      wins pose (AUC@30 0.992) but is second-worst on depth (0.148) &mdash; and carries the frame-coverage
      caveat. The useful result is that <strong>da3_streaming gives up almost nothing to da3</strong>
      (AUC 0.979 vs 0.979; relative MAE 0.031 vs 0.027; F@0.25&nbsp;m 0.630 vs 0.658) while scaling to
      six of seven assets instead of three.</p>
      <p>At the other end, <strong>mpsfm</strong> is the weakest method on every metric family and is
      also the one that collapses under appearance change (&sect;03). <strong>vggt-slam</strong> and
      <strong>vggt-long</strong> scale well but pay for it: their cloud F-scores (0.28) are less than
      half da3's, and vggt-slam's Chamfer (0.744&nbsp;m) is the worst of anything that isn't mpsfm.</p>
    </div>
    {T["main"]}

    <div class="finding">
      <h3>Rotation is close to solved; translation is the only discriminator left</h3>
      <div class="scope">
        <span class="chip ev">3 assets · 11 methods</span>
        <span class="chip">Fig 5</span>
      </div>
      <p>Across the scored runs the median RRA is 97.9% at 5&deg; and 98.0% at 30&deg; &mdash; it barely
      moves with the threshold, because it is already at the ceiling. RTA moves from 82.8% to 97.9%
      over the same range. At 15&deg; and 30&deg; every method is saturated on both and the panel says
      nothing; <strong>only RTA@5&deg; separates the field</strong>, spreading from 42.7% (vggt) to
      99.3% (colmap).</p>
      <p>Consequence for the paper: reporting RRA/RTA at three thresholds spends a third of a table on
      two saturated columns. RTA@5&deg; plus AUC@30 carries the same information.</p>
    </div>
  </div>
  {fig("04_rotation_vs_translation", 4, "Rotation vs translation accuracy at three thresholds",
       "The 15&deg; and 30&deg; panels are almost uniformly saturated. Everything that distinguishes the "
       "methods is in the 5&deg; panel, and almost all of it is in the translation bar.")}
  {fig("03_method_ranking_headline", 5, "Method ranking on four headline metrics",
       "Dots are the three individual per-asset medians behind each bar. Where the dots are spread "
       "(mpsfm on depth, colmap on cloud) the bar is a poor summary and the per-asset numbers should "
       "be quoted instead.")}
  <div class="col">
    <div class="finding">
      <h3>Predicted clouds recall the surface far better than they stay on it</h3>
      <div class="scope">
        <span class="chip ev">3 assets · 11 methods</span>
        <span class="chip warn">metric-design finding</span>
        <span class="chip">Fig 6</span>
      </div>
      <p>At every radius, every method sits well below the precision = recall line. At 5&nbsp;cm the
      median precision across all scored runs is 0.23%, against a median recall of 9.6% &mdash; precision
      is at the floor for the whole field and discriminates nothing. This is largely a sampling
      artefact: precision is measured against a point cloud sampled off the GT mesh, so at a 5&nbsp;cm
      radius the GT sampling density, not the prediction, sets the ceiling.</p>
      <p>Only the 25&nbsp;cm radius produces a usable spread. If the paper reports one cloud F-score,
      it should be F@0.25&nbsp;m; the 5&nbsp;cm columns are better dropped or moved to an appendix with
      the sampling caveat attached.</p>
    </div>
  </div>
  {fig("05_cloud_precision_recall", 6, "Cloud precision against recall at three radii",
       "The dashed line is precision = recall. The 5&nbsp;cm panel is compressed against the x-axis "
       "for every method.")}
  {fig("06_method_asset_heatmaps", 7, "Every headline metric, method by asset",
       "Shading is the method's rank within each asset, so a 60&nbsp;m bridge and a 10&nbsp;m house are "
       "comparable row to row. Raw values are printed in the cells. This is the figure to mine for "
       "per-asset claims.")}
  <div class="col">
    {fig("20_pose_vs_geometry", 8, "Pose quality against geometry quality",
        "Pose quality tracks geometry quality closely (Spearman &rho; = +0.74 against cloud F-score, "
        "&minus;0.67 against relative MAE, n = 2,299) but does not determine it: the vertical spread at "
        "high AUC is a band of runs that localise well and still reconstruct poorly.")}
  </div>
</section>

<hr class="rule">

<section id="assets">
  <div class="col">
    <h2><span class="n">03 &middot; Assets</span>The relative challenge of the asset classes</h2>

    <div class="finding">
      <h3>The large outdoor structures are an order of magnitude harder, on scale-free metrics</h3>
      <div class="scope">
        <span class="chip ev">7 assets</span>
        <span class="chip">survivor-biased in the long assets' favour</span>
        <span class="chip">Fig 9, Table 3</span>
      </div>
      <p>Judged only on metrics that carry no units &mdash; AUC@30, relative MAE, F-score at a fixed
      radius &mdash; the three short assets (glass_house2, old_house, unfinished_build) sit at median
      AUC 0.87&ndash;0.94 and F@0.25&nbsp;m 0.38&ndash;0.68. The long ones do not come close:
      <strong>crane</strong> is the hardest asset in the benchmark at median AUC 0.058, relative MAE
      0.86 and F@0.25&nbsp;m 0.017 &mdash; its <em>best</em> method scores 0.246 AUC. bridge and ship sit
      between, at median AUC 0.47 and 0.76.</p>
      <p>The bias runs the wrong way for a &ldquo;they're just longer&rdquo; explanation: the long
      assets are scored only by the methods that survived them, i.e. the more capable half of the
      field, and they are still far worse. Two effects are confounded here and the data cannot
      separate them &mdash; the long assets are also physically much larger, so a fixed 25&nbsp;cm
      F-score radius is relatively tighter on a bridge than on a house. Relative MAE and AUC@30 have no
      such dependence, and they show the same ordering, which is the stronger evidence.</p>
    </div>
  </div>
  {fig("15_asset_difficulty", 9, "Asset difficulty on scale-free metrics",
       "One dot per method. crane is the outlier in every panel; office2 has only two methods behind "
       "its box and should not be read as a result.")}
  <div class="col">
    {T["assets"]}

    <div class="finding">
      <h3>The real inspection trajectory is much harder than the synthetic orbits</h3>
      <div class="scope">
        <span class="chip ev">6 assets</span>
        <span class="chip">Fig 10–11, Table 4</span>
      </div>
      <p>On the short assets, trendspek costs a few percent against the proxy-surface orbits. On the
      long ones it is decisive: <strong>ship falls from AUC 0.78 to 0.16, bridge from 0.51 to
      0.09</strong>. Within the proxy orbits themselves the ordering is consistent &mdash; standing
      further off the surface helps, and the near stand-off is the worst of the three for almost every
      method.</p>
      <p>Both effects point the same way. The closer and less regular the camera path, the less
      inter-frame overlap and parallax context a method has, and the feed-forward methods degrade
      faster than COLMAP does. This is the axis the benchmark should lean on hardest &mdash; it produces
      the largest, most systematic effects of the four.</p>
    </div>
  </div>
  {fig("16_trendspek_vs_proxy", 10, "The real trajectory against the synthetic orbits",
       "Each pair is a single asset: blue is the median over the three proxy stand-offs, orange is "
       "trendspek. The gap widens down the page as sequences get longer and scenes larger.")}
  {fig("09_axis_standoff_heatmap", 11, "Response to trajectory, per method",
       "Change from the near stand-off, oriented so positive is worse. The trendspek column is the "
       "one that reddens.")}
</section>

<hr class="rule">

<section id="axes">
  <div class="col">
    <h2><span class="n">04 &middot; Variation axes</span>How the axes of scene variation change performance</h2>
    <p class="lede">This is the question the benchmark was built to answer, and the answer is
    lopsided. Ranked by how much they move the numbers: <strong>trajectory &gt; illumination
    &raquo; reflectivity &gt; soiling</strong>. Two of the four axes are, for nine of the eleven
    methods, inert.</p>
  </div>
  {fig("17_variance_share", 12, "Variance shares on glass_house2",
       "One-way main effects on the two balanced full-factorial blocks. Soiling explains 0.0% of the "
       "variance in every metric; reflectivity explains 0.0&ndash;0.4%. The choice of method explains "
       "45&ndash;68% in the soiling block. The large residual is interaction &mdash; most of it the one "
       "described in &sect;05.")}
  <div class="col">
    {T["axis"]}

    <div class="finding">
      <h3>Soiling changes nothing at all &mdash; except for mpsfm, which it destroys</h3>
      <div class="scope">
        <span class="chip ev">glass_house2 · 11 methods</span>
        <span class="chip">4 soil levels × 4 traj × 4 illum</span>
        <span class="chip">Fig 13</span>
      </div>
      <p>Going from verylow to high soiling moves pose AUC@30 by under <strong>2% for nine of the
      eleven methods</strong> and by under 0.6% for seven of them &mdash; pi3 by 0.21%, da3_streaming
      by 0.23%, pi-long by 0.26%, colmap by 0.36%. mpsfm is the sole real exception and it is not a
      small one: AUC 0.815 &rarr; 0.343, relative MAE +97%, cloud F-score &minus;64%. vggt-slam is a
      distant second at 7%.</p>
      <p>The clean reading is that soiling is a texture-only perturbation. It changes local appearance
      without changing scene structure, and the learned feed-forward models are indifferent to it,
      while the one method whose front end is classical sparse feature matching loses its
      correspondences.</p>
    </div>
  </div>
  {fig("11_axis_soil_heatmap", 13, "Response to soiling, per method",
       "Change from the verylow reference. One row is red; the other ten are white.")}
  <div class="col">
    <div class="finding">
      <h3>Reflectivity behaves the same way, but hits the correspondence-based methods too</h3>
      <div class="scope">
        <span class="chip ev">glass_house2 · 11 methods</span>
        <span class="chip">5 reflectivity levels</span>
        <span class="chip">Fig 14</span>
      </div>
      <p>DA3, Pi3, pi-long and vggt-omega are flat to within 1&ndash;2% across the full 0.0&nbsp;&rarr;
      1.0 sweep. The methods that do degrade, degrade <em>monotonically</em>, which is the signature of
      a real effect rather than noise: mpsfm 0.947 &rarr; 0.249, vggt-slam 0.781 &rarr; 0.590,
      vggt 0.681 &rarr; 0.554, vggt-long 0.757 &rarr; 0.652. COLMAP holds its pose but its depth
      relative MAE roughly doubles (0.107 &rarr; 0.222).</p>
      <p>So the split is not learned-vs-classical but <em>how the method establishes
      correspondence</em>. A mirror surface moves with the camera; anything that matches image features
      across frames is misled by it, and the methods that regress geometry directly are not.</p>
    </div>
  </div>
  {fig("13_axis_reflectivity_heatmap", 14, "Response to reflectivity, per method",
       "Change from reflectivity 0.0. The degradation is monotone in the affected rows &mdash; a real "
       "effect, not scatter.")}
  <div class="col">
    <div class="finding">
      <h3>Illumination sensitivity is method-specific, and sunrise &mdash; not night &mdash; is the killer</h3>
      <div class="scope">
        <span class="chip ev">3 assets · 11 methods</span>
        <span class="chip">Fig 15–16, Table 4</span>
      </div>
      <p>There is no benchmark-wide illumination effect to report; there is a per-method one. Against
      the sunny reference, <strong>da3 and da3_streaming lose 1&ndash;2%</strong> at their worst
      condition and <strong>vggt loses 33%, mpsfm 29%, vggt-long 21%</strong>. The worst condition is
      sunrise for eight of the eleven &mdash; night, the condition one would expect to be hardest, is
      milder for almost everyone. Low-angle sun produces long shadows and specular glare that break
      correspondence; darkness merely lowers contrast uniformly.</p>
      <p>Depth is more sensitive than pose to the same change: vggt-long's relative MAE rises 107% at
      sunrise while its AUC falls 21%.</p>
    </div>
  </div>
  {fig("07_axis_illumination_heatmap", 15, "Response to illumination, per method",
       "Change from the sunny reference on the three complete assets. The sunrise column is the "
       "reddest in all three metric families.")}
  {fig("08_axis_illumination_by_method", 16, "Illumination response, one panel per method",
       "Shared y-axis, so panel-to-panel comparison is fair. The DA3 panels are almost flat lines; "
       "the VGGT panels have a visible notch at sunrise.")}
  <div class="col">
    {fig("18_robustness_vs_typical", 17, "Robustness against typical performance",
        "Distance below the diagonal is what a method's worst lighting condition costs it. da3, "
        "da3_streaming, pi3 and vggt-omega sit almost on the line; mpsfm and vggt are far off it.")}
  </div>
</section>

<hr class="rule">

<section id="collapse">
  <div class="col">
    <h2><span class="n">05 &middot; Anomaly</span>One condition breaks every learned method at once</h2>

    <div class="finding">
      <h3>glass_house2, sunrise, near and mid stand-off &mdash; ten methods collapse, COLMAP does not</h3>
      <div class="scope">
        <span class="chip warn">verify before publishing</span>
        <span class="chip">40 sequences</span>
        <span class="chip">Fig 18</span>
      </div>
      <p>In the reflectivity block of glass_house2, at sunrise, on the near and mid stand-offs, all ten
      learned methods drop to AUC@30 between <strong>0.01 and 0.11</strong> while COLMAP scores
      <strong>0.98&ndash;1.00</strong>. It holds at every one of the five reflectivity levels, including
      0.0, so it is not a reflectivity effect. The far stand-off is degraded but not collapsed
      (0.27&ndash;0.67) and trendspek is unaffected (0.23&ndash;0.96).</p>
      <p>The soiling block &mdash; same scene, same trajectories, same illumination, a different render
      set &mdash; shows nothing of the kind: sunrise/near there is 0.63&ndash;0.96. A benchmark-wide
      scan finds 12 cells where COLMAP clears 0.8 and every learned method falls below 0.35; ten of
      those twelve are this one condition.</p>
      <p>Two readings are consistent with the data, and they need different responses. Either this is a
      genuine and rather striking failure mode &mdash; low sun on a glass building at close range
      defeating every feed-forward model while classical matching copes &mdash; in which case it is one
      of the more interesting results in the benchmark and deserves its own figure. Or something is
      wrong with those forty renders, in which case the reflectivity block's sunrise rows should be
      excluded. <strong>The renders should be looked at before either claim is made.</strong> The fact
      that DA3, Pi3 and vggt-omega &mdash; architecturally unrelated &mdash; land within 0.01 of each
      other argues for the second.</p>
    </div>
  </div>
  {fig("21_sunrise_collapse", 18, "The sunrise collapse, and its absence in the soiling block",
       "Pose AUC@30 on glass_house2, laid out illumination by stand-off. The boxed columns are the "
       "collapse. The lower panel is the same scene under the same trajectories from the other render "
       "block.")}
</section>

<hr class="rule">

<section id="constant">
  <div class="col">
    <h2><span class="n">06 &middot; Constants</span>Where the numbers do not move at all</h2>
    <p class="lede">Flat is a finding. Several metrics are bit-identical across every level of an axis,
    and each constancy says something specific.</p>
    <ul>
      <li><strong>Valid predicted-pixel ratio is constant in ~90% of cells on every axis.</strong> The
      feed-forward methods emit a fully dense depth map no matter what they are shown; they never
      abstain. Depth &ldquo;completeness&rdquo; therefore measures the GT mask, not the method.</li>
      <li><strong>RRA@30 is constant across soiling in 68% of cells</strong>, RRA@15 in 62%. That is
      saturation, not robustness &mdash; the metric is pinned at its ceiling and cannot register a
      change.</li>
      <li><strong>Frame counts and camera-pair counts are constant across illumination, soiling and
      reflectivity in 80&ndash;87% of cells but never across trajectory.</strong> That is the expected
      sanity check: the appearance axes are re-renders of the same camera path. The 13&ndash;20% that
      are <em>not</em> constant are the COLMAP and VGGT-SLAM rows from &sect;00.</li>
      <li><strong>Cloud &ldquo;cropped %&rdquo; is constant in ~90% of cells and is 0 in most of
      them</strong> &mdash; the bounding-box crop almost never fires, so the column carries no
      information and can be dropped from the results tables.</li>
      <li><strong>No headline metric is ever exactly constant.</strong> AUC@30, relative MAE and
      F@0.25&nbsp;m are flat in 0% of cells on all four axes &mdash; changing soiling or reflectivity
      always perturbs them a little. The finding in &sect;04 is that the perturbation is not
      <em>meaningful</em>, which is a different and weaker claim than exact invariance, and the paper
      should make it in those terms.</li>
    </ul>
  </div>
  {fig("19_constancy_map", 19, "Metrics that do not move along an axis",
       "Percentage of (asset, method, other-axis) cells whose value is identical across every level. "
       "Read the top block as instrumentation, the RRA/RTA block as saturation.")}
</section>

<hr class="rule">

<section id="tables">
  <div class="col">
    <h2><span class="n">07 &middot; For the paper</span>Tables worth building</h2>
    <p class="lede">Six suggestions, in the order I would argue for them. The first three carry the
    findings that survive every caveat.</p>
  </div>
  <div class="cards plate-wide">
    <div class="card">
      <span class="where">Main body · headline</span>
      <h4>Accuracy × scalability, one table</h4>
      <p>Table 1's six metric columns plus Table 2's &ldquo;longest scored&rdquo; and
      &ldquo;first OOM at&rdquo;. Putting the frame ceiling next to the accuracy makes the
      da3 / da3_streaming trade the reader's first impression instead of a footnote.</p>
    </div>
    <div class="card">
      <span class="where">Main body · the benchmark's own question</span>
      <h4>Axis cost, four rows</h4>
      <p>One row per axis, columns for pose / depth / cloud, each cell the median worst-level cost and
      the single most-affected method (Table 4). Four rows carry the whole
      &ldquo;trajectory &gt; illumination &raquo; reflectivity &gt; soiling&rdquo; result.</p>
    </div>
    <div class="card">
      <span class="where">Main body · asset design</span>
      <h4>Asset card</h4>
      <p>Per asset: frames, scene extent, which axes it varies, how many methods completed it, and the
      median plus best score on the three scale-free metrics (Table 3). Doubles as the dataset
      description and as the difficulty result.</p>
    </div>
    <div class="card">
      <span class="where">Main body or appendix</span>
      <h4>Robustness table</h4>
      <p>Per method, the worst level on each appearance axis and its cost in AUC@30. Reads as a
      one-glance robustness ranking and makes the mpsfm and VGGT-family weaknesses concrete without a
      figure.</p>
    </div>
    <div class="card">
      <span class="where">Appendix · honesty</span>
      <h4>Evaluation coverage</h4>
      <p>Table 5 and Table 6 together: what fraction of the grid is scored, and what fraction of each
      sequence's frames each method is scored on. Anticipates the reviewer question rather than
      waiting for it.</p>
    </div>
    <div class="card">
      <span class="where">Appendix · metric design</span>
      <h4>Why F@0.25 m</h4>
      <p>Precision and recall at all three radii for all methods, showing the 5&nbsp;cm columns pinned
      at the floor. Justifies reporting a single radius in the main body in three lines instead of
      inviting the question.</p>
    </div>
  </div>
  <div class="col">
    <h4>Columns to drop</h4>
    <ul>
      <li><strong>RRA/RTA at 15&deg; and 30&deg;</strong> &mdash; saturated for every method on every
      asset that matters. Keep 5&deg; and AUC@30.</li>
      <li><strong>Precision / recall / F at 0.05&nbsp;m</strong> &mdash; floored by GT sampling density,
      not by the predictions.</li>
      <li><strong>Cropped %</strong> &mdash; zero almost everywhere.</li>
      <li><strong>Depth completeness at 1.05</strong> unless reported next to the valid-pixel ratio;
      on its own it reads as a method property when it is a GT-mask property.</li>
    </ul>
  </div>
</section>

<hr class="rule">

<section id="open">
  <div class="col">
    <h2><span class="n">08 &middot; Open</span>What would most improve the result set</h2>
    <ol>
      <li><strong>Finish office2, or drop it.</strong> It is 144 variants &times; 11 methods of the grid
      and currently contributes nothing but empty cells.</li>
      <li><strong>Run mpsfm on the long assets.</strong> It is the most appearance-sensitive method in
      the set, which makes it the most interesting one to see under a long real trajectory &mdash; and
      it currently has no OOM data at all.</li>
      <li><strong>Check the forty glass_house2 sunrise renders</strong> behind &sect;05 before that
      result is written up either way.</li>
      <li><strong>Record scene extent per asset.</strong> Without it, the metric-radius confound in
      &sect;03 cannot be closed, and absolute-distance cloud metrics cannot be normalised across a
      bridge and a house.</li>
      <li><strong>Consider evaluating COLMAP and VGGT-SLAM on the full frame set</strong>, scoring
      unregistered frames as failures, alongside the current registered-only numbers. That turns
      &sect;00's caveat into a measurement.</li>
      <li><strong>Vary soiling and reflectivity on a second asset.</strong> Both currently rest on
      glass_house2 alone, and a glass building is an unusually favourable case for a reflectivity
      sweep.</li>
    </ol>
  </div>
</section>

<hr class="rule">

<section id="appendix">
  <div class="col">
    <h2><span class="n">09 &middot; Appendix</span>The rest of the figures</h2>
    <p class="lede">Seven more plates, kept out of the argument above but generated from the same
    data and available at full resolution as PNG and PDF. The per-method small multiples are the
    easiest ones to lift a single panel out of.</p>
  </div>
  {fig("10_axis_standoff_by_method", 20, "Trajectory response, one panel per method",
       "Shared y-axis. The trendspek point is the last on each x-axis; note how many panels drop "
       "there and how few drop between near, mid and far.")}
  {fig("12_axis_soil_by_method", 21, "Soiling response, one panel per method",
       "Ten flat lines and one that falls off. This is the single clearest plate for the soiling "
       "result if a figure is preferred over Table 4.")}
  {fig("14_axis_reflectivity_by_method", 22, "Reflectivity response, one panel per method",
       "The monotone decline in the mpsfm, vggt, vggt-long and vggt-slam panels is what distinguishes "
       "this axis from soiling.")}
  {fig("A1_axis_illumination_heatmap_all_assets", 23,
       "Illumination response over all seven assets",
       "The same computation as Figure 15 widened to every asset. Rows now cover different asset sets, "
       "so compare within a row rather than between rows &mdash; the reason the main text uses the "
       "three-asset version.")}
  {fig("A2_axis_illumination_by_method_all_assets", 24,
       "Illumination response per method, all seven assets",
       "The absolute levels drop against Figure 16 because the hard long assets are now in the pool.")}
  {fig("A3_axis_standoff_heatmap_all_assets", 25,
       "Trajectory response over all seven assets",
       "The trendspek column reddens considerably against the three-asset version in Figure 11 &mdash; "
       "that widening is the &sect;03 result.")}
  {fig("A4_axis_standoff_by_method_all_assets", 26,
       "Trajectory response per method, all seven assets",
       "The near &rarr; mid &rarr; far improvement is visible in most panels; the trendspek drop is "
       "visible in nearly all of them.")}
</section>

<footer class="col">
  <p>Built from <code>collacted_results/{{cloud,depth,pose}}.csv</code> &mdash; 4,048 rows per metric
  category, collated by <code>collate_results.py</code>. Analysis scripts, all 26 figures as PNG and
  PDF, and the tables as Markdown and LaTeX are in <code>analysis/</code>; every figure here is
  regenerated by <code>venv/bin/python analysis/fig_*.py</code> and every table by
  <code>analysis/tables.py</code>. Chart palette validated for colour-vision deficiency with
  <code>analysis/validate_palette.py</code>.</p>
  <p>Appendix figures &mdash; the illumination and trajectory responses computed over all seven assets
  rather than the three complete ones &mdash; are <code>A1</code>&ndash;<code>A4</code> in
  <code>analysis/figures/</code>.</p>
</footer>

</div>"""

    page = f"""<title>Inspection Benchmark Findings</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{CSS}</style>
{body}"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    build()
