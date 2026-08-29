"""Shared loading for the collated eval results."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "collacted_results"
OUT = ROOT / "analysis" / "figures"
TAB = ROOT / "analysis" / "tables"

LABELS = ["Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity"]

# assets that vary only traj x illumination (soil pinned at verylow)
SIMPLE_ASSETS = ["bridge", "crane", "old_house", "ship", "unfinished_build"]
RICH_ASSETS = ["glass_house2", "office2"]

ILLUM_ORDER = ["sunny", "overcast", "sunrise", "night"]
SOIL_ORDER = ["verylow", "low", "medium", "high"]
REFL_ORDER = ["0.0", "0.25", "0.5", "0.75", "1.0"]

METHOD_ORDER = ["colmap/no_gt", "mpsfm", "vggt", "vggt-long", "vggt-slam", "vggt-omega",
                "da3", "da3_streaming", "pi3", "pi3x", "pi-long"]


def load(cat):
    """One collated category ('cloud'|'depth'|'pose') with a Status column.

    Status: 'ok' (numbers present), 'OOM' (gpu ran out of memory), 'missing' (not run yet).
    Metric cells are numeric (NaN where OOM/missing) so nothing non-numeric leaks into a plot.
    """
    df = pd.read_csv(RES / f"{cat}.csv", dtype=str, keep_default_na=False)
    meta = [c for c in df.columns if c in LABELS or c in ("Source", "Cloud source")]
    metrics = [c for c in df.columns if c not in meta]
    oom = (df[metrics] == "OOM").any(axis=1)
    empty = (df[metrics] == "").all(axis=1)
    df["Status"] = np.where(oom, "OOM", np.where(empty, "missing", "ok"))
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Reflectivity/Soil stay categorical strings; '' means "axis not varied on this asset"
    df["Category"] = cat
    return df, metrics


def load_all():
    out = {}
    for c in ("cloud", "depth", "pose"):
        df, m = load(c)
        out[c] = (df, m)
    return out


def axis_slice(df, axis):
    """Rows where `axis` is the axis being varied.

    Soil rows: Reflectivity blank. Reflectivity rows: Soil blank.
    Traj/Illumination: everything (they are crossed with the rest).
    """
    if axis == "Soil":
        return df[(df.Soil != "") & (df.Reflectivity == "")]
    if axis == "Reflectivity":
        return df[df.Reflectivity != ""]
    return df


# --- derived columns -------------------------------------------------------

def enrich(df):
    """Adds Family, Standoff (near/mid/far/trendspek) and StandoffM to a loaded frame.

    Each asset has its own three `proxy_surface_<n>` offsets (bridge 5/10/15 m,
    crane 0/2/4 m, ...), so the raw name is not comparable across assets. Ranking
    the three within the asset gives near/mid/far, which is; `trendspek` is the
    real inspection trajectory and stays its own level.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import FAMILY
    df = df.copy()
    df["Family"] = df.Method.map(FAMILY)
    df["StandoffM"] = pd.to_numeric(
        df.Traj.str.replace("proxy_surface_", "", regex=False), errors="coerce")
    rank = {}
    for asset, g in df.groupby("Asset"):
        vals = sorted(v for v in g.StandoffM.dropna().unique())
        for i, v in enumerate(vals):
            rank[(asset, v)] = ["near", "mid", "far"][i] if len(vals) == 3 else f"L{i}"
    df["Standoff"] = [
        "trendspek" if t == "trendspek" else rank.get((a, s), "?")
        for a, t, s in zip(df.Asset, df.Traj, df.StandoffM)]
    return df


def loade(cat):
    df, m = load(cat)
    return enrich(df), m


HEADLINE = {
    "pose":  [("AUC@30", "up"), ("RTA@5 %", "up"), ("RRA@5 %", "up"), ("ATE RMSE (m)", "down")],
    "depth": [("Relative MAE", "down"), ("Accuracy @1.05", "up"),
              ("Scale invariant MSE", "down"), ("Completeness @1.05", "up")],
    "cloud": [("F-score @0.25m", "up"), ("Chamfer (m)", "down"),
              ("Accuracy (m)", "down"), ("Completeness (m)", "down")],
}
ALL_HEADLINE = [(c, k, d) for c, v in HEADLINE.items() for k, d in v]

# assets where every one of the 11 methods produced a full set of rows
CORE_ASSETS = ["glass_house2", "old_house", "unfinished_build"]


def per_asset_method(df, metric, assets=None, agg="median"):
    """(Asset, Method) -> aggregated metric over that cell's variants, ok rows only."""
    ok = df[df.Status == "ok"]
    if assets is not None:
        ok = ok[ok.Asset.isin(assets)]
    return ok.groupby(["Asset", "Method"])[metric].agg(agg).unstack("Method")
