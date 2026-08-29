#!/usr/bin/env python3
"""Plots pose/point-cloud/depth metrics against the three scene properties eval.py's variants
sweep -- soiling, reflectivity, and illumination -- comparing every method on the same axes.

Usage:
    python plot_variant_metrics.py glass_house2 --out plots/glass_house2
    python plot_variant_metrics.py glass_house2 --out plots/glass_house2 --methods da3,vggt,mpsfm

Reads an `eval.py --save-hierarchical` output tree (see `eval.py`'s module docstring and
`latex_results_table.py`, which reads the same tree): `<results_root>/<asset>/<method>/
{cloud,depth,pose}[_<timestamp>].csv`.

Each asset's variants vary exactly one of Soil/Reflectivity at a time (the other left blank);
Illumination and Traj are always set. So:
  - the soiling sweep = rows with Reflectivity blank, grouped by Soil, averaged over Traj and
    Illumination
  - the reflectivity sweep = rows with Soil blank, grouped by Reflectivity, averaged over Traj
    and Illumination
  - the illumination sweep = rows with Soil blank and Reflectivity == 0.0 (the clean,
    non-reflective baseline both other sweeps start from), grouped by Illumination, averaged
    over Traj

Soiling and reflectivity are plotted as line charts (both are ordered/continuous); illumination
-- a set of distinct lighting types, not a continuous property -- as a grouped bar chart instead.
An "OOM" cell (see `inspection_eval.runner`'s OOM handling) is excluded from the average, same as
`latex_results_table.py`'s `_metric_cell`. Writes one PNG per (property, metric-category) pair,
9 total, to --out.
"""

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- method labels & styling, kept consistent with latex_results_table.py's METHOD_LABELS ------

METHOD_LABELS = {
    "colmap/no_gt": "COLMAP",
    "mpsfm": "MP-SfM",
    "vggt": "VGGT",
    "da3": "DA3",
    "vggt-omega": "VGGT-Ω",
    "pi3": "π³",
    "pi3x": "π-χ",
    "pi-long": "Pi-Long",
    "da3_streaming": "DA3-Streaming",
    "vggt-long": "VGGT-Long",
    "vggt-slam": "VGGT-SLAM 2.0",
}

# Fixed color+marker per method, assigned in a fixed order and reused identically across every
# figure -- never re-cycled/re-assigned based on which methods happen to be present in a given
# plot, so the same method always reads the same color/marker everywhere.
_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
    "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", "#1B9E77",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
METHOD_ORDER = list(METHOD_LABELS)  # fixed order the palette/markers are assigned in


def method_style(method: str) -> tuple[str, str]:
    i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return _PALETTE[i % len(_PALETTE)], _MARKERS[i % len(_MARKERS)]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# --- reading eval.py --save-hierarchical's output ------------------------------------------

_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}-\d{6}$")


def _latest_metric_csvs(method_dir: Path) -> dict[str, Path]:
    """The most recent csv for each metric under one asset/method's hierarchical output folder --
    same rule eval.py/check_completion_status.py/latex_results_table.py all use, duplicated
    rather than imported (see latex_results_table.py's module docstring: avoids pulling in the
    heavier inspection_viz dependency chain just to find a filename)."""
    if not method_dir.is_dir():
        return {}
    by_metric: dict[str, list[Path]] = {}
    for path in method_dir.glob("*.csv"):
        by_metric.setdefault(_TIMESTAMP_SUFFIX.sub("", path.stem), []).append(path)

    def latest(paths: list[Path]) -> Path:
        bare = [p for p in paths if not _TIMESTAMP_SUFFIX.search(p.stem)]
        return bare[0] if bare else max(paths, key=lambda p: p.stem)

    return {metric: latest(paths) for metric, paths in by_metric.items()}


def load_rows(results_root: Path, asset: str, method: str, category: str) -> list[dict]:
    """Every row of `method`'s `category` ("cloud"/"depth"/"pose") csv, or [] if that method has
    none yet."""
    folder = method.replace("/", "-")  # matches eval.py write_hierarchical's flattening
    csvs = _latest_metric_csvs(results_root / asset / folder)
    if category not in csvs:
        return []
    with csvs[category].open(newline="") as handle:
        return list(csv.DictReader(handle))


def discover_methods(results_root: Path, asset: str) -> list[str]:
    """Every method with a results folder under results_root/asset, named back the way
    METHOD_LABELS/eval.py expect ("colmap-no_gt" -> "colmap/no_gt" -- the one folder name eval.py
    flattens, per write_hierarchical), in METHOD_ORDER's fixed order first (so styling stays
    stable run to run) with any unlisted method appended after."""
    asset_dir = results_root / asset
    if not asset_dir.is_dir():
        return []
    found = set()
    for child in asset_dir.iterdir():
        if child.is_dir():
            name = "colmap/no_gt" if child.name == "colmap-no_gt" else child.name
            found.add(name)
    ordered = [m for m in METHOD_ORDER if m in found]
    ordered += sorted(found - set(ordered))
    return ordered


# --- aggregation ------------------------------------------------------------------------------

def _numbers(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        v = row.get(key, "")
        if v in ("", "OOM", None):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _fixed_select(predicate):
    """Wrap a static row predicate as a select_factory (see SWEEPS) that ignores the data and
    always returns the same predicate -- for soil/reflectivity, whose selection criteria don't
    depend on what's actually in the data."""
    return lambda all_rows: (predicate, None)


def _illumination_baseline(all_rows: list[dict]) -> tuple[str, str]:
    """The (Soil, Reflectivity) combo the illumination sweep should be read off of: among every
    combo actually present in the data, the one covering the most distinct Illumination values --
    ties broken toward the canonical clean/non-reflective baseline (blank Soil, Reflectivity
    "0.0"), the one the soil/reflectivity sweeps are themselves built from (see module docstring).
    That canonical baseline is what most assets (e.g. glass_house2, office2) use. But an asset
    that never ran the reflectivity sweep -- e.g. crane, whose data only ever has Soil="verylow",
    Reflectivity="" -- swept Illumination at that fixed level instead, so this falls back to
    whatever combo Illumination actually varies at."""
    by_combo: dict[tuple[str, str], set[str]] = {}
    for row in all_rows:
        illum = row.get("Illumination", "")
        if not illum:
            continue
        combo = (row.get("Soil", ""), row.get("Reflectivity", ""))
        by_combo.setdefault(combo, set()).add(illum)
    if not by_combo:
        return "", "0.0"
    return max(by_combo, key=lambda combo: (len(by_combo[combo]), combo == ("", "0.0")))


def _illumination_select(all_rows: list[dict]):
    soil, reflectivity = _illumination_baseline(all_rows)
    predicate = lambda row: row.get("Soil", "") == soil and row.get("Reflectivity", "") == reflectivity
    if (soil, reflectivity) == ("", "0.0"):
        return predicate, None
    note = f"Soil={soil or '(none)'}, Reflectivity={reflectivity or '(none)'} -- no clean baseline in this asset's data"
    return predicate, note


SWEEPS = {
    "soil": {
        "label": "Soiling",
        "column": "Soil",
        "order": ["verylow", "low", "medium", "high"],
        "select_factory": _fixed_select(
            lambda row: row.get("Reflectivity", "") == "" and row.get("Soil", "") != ""),
        "kind": "line",
        "numeric_x": False,
    },
    "reflectivity": {
        "label": "Reflectivity",
        "column": "Reflectivity",
        "order": None,  # sorted numerically at aggregation time, from whatever's present
        "select_factory": _fixed_select(
            lambda row: row.get("Soil", "") == "" and row.get("Reflectivity", "") != ""),
        "kind": "line",
        "numeric_x": True,
    },
    "illumination": {
        "label": "Illumination",
        "column": "Illumination",
        "order": ["night", "sunrise", "overcast", "sunny"],
        "select_factory": _illumination_select,
        "kind": "bar",
        "numeric_x": False,
    },
}


def sweep_levels(sweep: dict, rows: list[dict]) -> list[str]:
    """The x-axis levels for one sweep, in display order: the sweep's fixed order if it has one
    (soil/illumination), else every value actually present, sorted numerically (reflectivity)."""
    if sweep["order"] is not None:
        return sweep["order"]
    values = {row[sweep["column"]] for row in rows if row.get(sweep["column"], "") != ""}
    return sorted(values, key=float)


def aggregate(rows: list[dict], column: str, select, metric: str) -> dict[str, tuple[float, float, int]]:
    """{level: (mean, std, n)} for one method's rows, restricted to `select` and grouped by
    `column` -- averaged over every other variant axis (Traj, and Illumination/Soil/Reflectivity
    where `select` doesn't fix them). A level with no usable numbers (every row missing or OOM)
    is simply absent from the result, rather than a fabricated 0."""
    selected = [row for row in rows if select(row)]
    by_level: dict[str, list[dict]] = {}
    for row in selected:
        by_level.setdefault(row[column], []).append(row)
    result = {}
    for level, level_rows in by_level.items():
        nums = _numbers(level_rows, metric)
        if nums:
            mean = statistics.mean(nums)
            std = statistics.stdev(nums) if len(nums) > 1 else 0.0
            result[level] = (mean, std, len(nums))
    return result


# --- metrics per category ----------------------------------------------------------------------

METRICS = {
    "cloud": ["Accuracy (m)", "Completeness (m)", "Chamfer (m)",
              "F-score @0.05m", "F-score @0.1m", "F-score @0.25m"],
    "depth": ["RMSE (m)", "MAE (m)", "Relative MAE", "Scale invariant MSE",
              "Accuracy @1.05", "Completeness @1.05"],
    "pose": ["AUC@30", "ATE RMSE (m)", "ATE median (m)",
             "RRA@15 %", "RTA@15 %", "Abs rot err mean (deg)"],
}

# Whether higher or lower values are better, per metric name -- used to annotate each subplot
# title with a small arrow ("↑ higher is better" / "↓ lower is better").
_HIGHER_IS_BETTER = {
    "F-score @0.05m", "F-score @0.1m", "F-score @0.25m",
    "Accuracy @1.05", "Completeness @1.05",
    "AUC@30", "RRA@15 %", "RTA@15 %",
}
_LOWER_IS_BETTER = {
    "Accuracy (m)", "Completeness (m)", "Chamfer (m)",
    "RMSE (m)", "MAE (m)", "Relative MAE", "Scale invariant MSE",
    "ATE RMSE (m)", "ATE median (m)", "Abs rot err mean (deg)",
}
assert not (_HIGHER_IS_BETTER & _LOWER_IS_BETTER)
assert _HIGHER_IS_BETTER | _LOWER_IS_BETTER == {m for ms in METRICS.values() for m in ms}


def metric_title(metric: str) -> str:
    if metric in _HIGHER_IS_BETTER:
        return f"{metric} ↑"
    if metric in _LOWER_IS_BETTER:
        return f"{metric} ↓"
    return metric


# --- plotting -----------------------------------------------------------------------------------

def plot_figure(asset: str, category: str, sweep_name: str, sweep: dict,
                 method_rows: dict[str, list[dict]], out_dir: Path, dpi: int) -> Path | None:
    metrics = METRICS[category]
    all_rows = [row for rows in method_rows.values() for row in rows]
    levels = sweep_levels(sweep, all_rows)
    if not levels:
        return None
    select, select_note = sweep["select_factory"](all_rows)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x_pos = [float(l) for l in levels] if sweep["numeric_x"] else list(range(len(levels)))
    n_methods = len(method_rows)
    bar_width = 0.8 / max(n_methods, 1)

    legend_entries: dict[str, object] = {}  # method -> handle, insertion order = METHOD_ORDER
    for ax, metric in zip(axes.flat, metrics):
        any_data = False
        for i, (method, rows) in enumerate(method_rows.items()):
            agg = aggregate(rows, sweep["column"], select, metric)
            if not agg:
                continue
            any_data = True
            color, marker = method_style(method)
            means = [agg[l][0] if l in agg else float("nan") for l in levels]
            stds = [agg[l][1] if l in agg else 0.0 for l in levels]
            if sweep["kind"] == "line":
                handle = ax.errorbar(x_pos, means, yerr=stds, label=method_label(method),
                                     color=color, marker=marker, markersize=5, linewidth=1.5,
                                     capsize=2, elinewidth=0.75, alpha=0.9)
            else:  # grouped bar
                offsets = [xp + (i - (n_methods - 1) / 2) * bar_width for xp in x_pos]
                handle = ax.bar(offsets, means, width=bar_width, yerr=stds,
                                label=method_label(method), color=color, edgecolor="white",
                                linewidth=0.5, capsize=2, error_kw={"elinewidth": 0.75})
            legend_entries.setdefault(method, handle)
        ax.set_title(metric_title(metric), fontsize=10)
        ax.set_xlabel(sweep["label"])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(levels)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                   color="gray")

    if legend_entries:
        ordered = [m for m in METHOD_ORDER if m in legend_entries]
        ordered += [m for m in legend_entries if m not in ordered]
        handles = [legend_entries[m] for m in ordered]
        labels = [method_label(m) for m in ordered]
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
                  bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    title = f"{asset} — {category} metrics vs. {sweep['label'].lower()}"
    if select_note:
        title += f"\n({select_note})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = out_dir / f"{sweep_name}_{category}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- CLI --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("asset", nargs="?", default="glass_house2",
                   help="asset folder name under --results-root (default: %(default)s)")
    p.add_argument("--results-root", type=Path, default=Path("/mnt/jgrayHDD/eccv_2026_results"),
                   help="an eval.py --save-hierarchical output root (default: %(default)s)")
    p.add_argument("--out", type=Path, required=True,
                   help="directory the 9 PNGs are written into (created if missing)")
    p.add_argument("--methods", default=None,
                   help="comma-separated subset of methods (as under --results-root/<asset>/, "
                        "e.g. da3,vggt,colmap/no_gt); default: every method found there")
    p.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    methods = (args.methods.split(",") if args.methods
              else discover_methods(args.results_root, args.asset))
    if not methods:
        print(f"no method result folders found under {args.results_root / args.asset}",
             file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for category in METRICS:
        method_rows = {m: load_rows(args.results_root, args.asset, m, category) for m in methods}
        method_rows = {m: rows for m, rows in method_rows.items() if rows}
        if not method_rows:
            print(f"no {category} results found for any method, skipping", file=sys.stderr)
            continue
        for sweep_name, sweep in SWEEPS.items():
            path = plot_figure(args.asset, category, sweep_name, sweep, method_rows, args.out,
                               args.dpi)
            if path:
                written.append(path)
                print(f"wrote {path}")

    if not written:
        print("no figures were written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
