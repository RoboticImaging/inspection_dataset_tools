#!/usr/bin/env python3
"""Collates every results csv under an `eval.py --save-hierarchical` output tree
(`<results_root>/<asset>/<method>/<metric>[_<timestamp>].csv` -- see `eval.py`'s module
docstring, and `latex_results_table.py`/`plot_variant_metrics.py`, which read the same tree) into
one combined csv per metric category ("cloud"/"depth"/"pose", as of writing, though nothing here
hardcodes that): every row, from every method, of every asset, that category has, stacked into one
table with a "Source" column giving that row's original csv's path (relative to --results-root)
for traceability back to a specific run.

Values are taken as-is -- an "OOM" cell (see `inspection_eval.runner`'s OOM handling) included
unchanged -- and nothing is aggregated or filtered the way `plot_variant_metrics.py`/
`latex_results_table.py` do. This is meant as raw input for further analysis (pandas, a
spreadsheet, ...), not a human-facing summary.

Different runs of the same category can differ in which columns they wrote (e.g. a metric added
later), so each category's header is the union across every csv seen for it: the common label
columns (Asset, Method, Traj, Illumination, Soil, Reflectivity) first, then every metric column in
first-seen order, then Source last. A row is blank in any column its own source csv didn't have.

--include-missing pads every category out to the same asset/method/variant grid, adding a blank
row (Asset/Method/Traj/Illumination/Soil/Reflectivity filled in, every metric + Source empty) for
any (asset, method, variant) combo that category has no row for -- e.g. a baseline that hasn't
been run/evaluated yet on some asset, or one only partially evaluated (some variants present,
others not). This makes the three categories end up with matching row counts for the variants
they share (see `check_completion_status.py`'s docstring for why they otherwise legitimately
don't: a metric can skip a variant another metric doesn't, e.g. no GT mesh means no cloud row).
The universe of "should exist" asset/method combos it fills gaps against is just every asset
folder and every method folder actually seen somewhere under --results-root (any
--assets/--methods filter narrows both sides) -- not a canonical baseline registry, so a method
with no folder under *any* asset never gets a placeholder row. Each asset's true variant list is
read from `inspection_viz.dataset.AssetDataset` under --data-root -- the same source
`latex_results_table.py`'s own completeness check uses -- so it pulls in that package
(cv2/trimesh/OpenEXR/PIL) unlike the rest of this script; run it from an environment that has it
(e.g. the `inspection_dataset_tools` conda env) if the plain interpreter doesn't.

Usage:
    python collate_results.py --out results_csvs/
    python collate_results.py --out results_csvs/ --results-root /mnt/jgrayHDD/eccv_2026_results
    python collate_results.py --out results_csvs/ --assets crane,glass_house2
    python collate_results.py --out results_csvs/ --methods da3,vggt,colmap/no_gt
    python collate_results.py --out results_csvs/ --include-missing
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# --- reading eval.py --save-hierarchical's output ------------------------------------------

_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}-\d{6}$")

# The label columns every results csv starts with (see inspection_eval.runner.LABEL_COLUMNS /
# EvalCase.label) -- reproduced here rather than imported, same reason check_completion_status.py
# gives: keeps this script to stdlib csv, no inspection_eval/inspection_viz import needed.
LABEL_COLUMNS = ("Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity")

BASE_FIELDS = ["Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity"]


def _latest_metric_csvs(method_dir: Path) -> dict[str, Path]:
    """The most recent csv for each metric under one asset/method's hierarchical output folder --
    same rule eval.py/check_completion_status.py/latex_results_table.py/plot_variant_metrics.py
    all use, duplicated rather than imported (see module docstring)."""
    if not method_dir.is_dir():
        return {}
    by_metric: dict[str, list[Path]] = {}
    for path in method_dir.glob("*.csv"):
        by_metric.setdefault(_TIMESTAMP_SUFFIX.sub("", path.stem), []).append(path)

    def latest(paths: list[Path]) -> Path:
        bare = [p for p in paths if not _TIMESTAMP_SUFFIX.search(p.stem)]
        return bare[0] if bare else max(paths, key=lambda p: p.stem)

    return {metric: latest(paths) for metric, paths in by_metric.items()}


def discover_method_dirs(results_root: Path, assets: set[str] | None) -> list[Path]:
    """Every `<results_root>/<asset>/<method_folder>` directory actually present, optionally
    restricted to `assets` (asset folder names). Doesn't care what the folder itself is named --
    the asset/method identifying a row comes from that row's own Asset/Method columns, not the
    path, so this is unaffected by write_hierarchical's "/" -> "-" flattening of a method name
    like "colmap/no_gt" (see module docstring)."""
    method_dirs = []
    for asset_dir in sorted(results_root.iterdir()):
        if not asset_dir.is_dir() or (assets and asset_dir.name not in assets):
            continue
        method_dirs += [d for d in sorted(asset_dir.iterdir()) if d.is_dir()]
    return method_dirs


def discover_universe(results_root: Path, assets: set[str] | None,
                      methods: set[str] | None) -> tuple[list[str], list[str]]:
    """(assets, methods) -- every asset folder name (filtered by `assets`) and every method name
    found under any of them (filtered by `methods`), for --include-missing to know which
    asset/method combos should exist. A method folder is unflattened the same way
    write_hierarchical flattens a slash-bearing method name for its folder -- "colmap-no_gt" ->
    "colmap/no_gt", the one case that needs it -- so this lines up with the Method values rows are
    filtered against elsewhere (see `collate_rows`)."""
    all_assets = []
    all_methods = set()
    for asset_dir in sorted(results_root.iterdir()):
        if not asset_dir.is_dir() or (assets and asset_dir.name not in assets):
            continue
        all_assets.append(asset_dir.name)
        for method_dir in sorted(asset_dir.iterdir()):
            if method_dir.is_dir():
                name = "colmap/no_gt" if method_dir.name == "colmap-no_gt" else method_dir.name
                all_methods.add(name)
    if methods:
        all_methods &= methods
    return all_assets, sorted(all_methods)


def add_missing_variants(by_category: dict[str, tuple[list[str], list[dict]]],
                         all_assets: list[str], all_methods: list[str], data_root: Path) -> None:
    """--include-missing's fill-in: for every category, every (asset, method) in `all_assets` x
    `all_methods`, and every variant that asset actually has (per `AssetDataset`, see module
    docstring), add one blank placeholder row (Asset/Method/Traj/Illumination/Soil/Reflectivity
    filled in, every metric + Source empty) for any such variant that category has no row for.
    Mutates `by_category`'s row lists in place. An asset not found under `data_root` is skipped
    (with a warning) rather than crashing -- same reasoning as `latex_results_table.py`'s
    `_asset_variants`."""
    try:
        from inspection_viz.dataset import AssetDataset  # opt-in heavier dep -- see module docstring
    except ImportError as e:
        raise SystemExit(
            "--include-missing needs inspection_viz (cv2/trimesh/OpenEXR/PIL), which "
            f"isn't importable here ({e}). Run this from an environment that has it, e.g.:\n"
            "  conda run -n inspection_dataset_tools python3 collate_results.py ..."
        ) from e

    variants_by_asset = {}
    for asset in all_assets:
        try:
            variants_by_asset[asset] = list(AssetDataset(data_root / asset).variants)
        except FileNotFoundError:
            print(f"warning: {asset!r} not found under {data_root}, can't confirm its true "
                 f"variant list -- skipping --include-missing for it", file=sys.stderr)
            variants_by_asset[asset] = []

    for fieldnames, rows in by_category.values():
        present = {(row.get("Asset", ""), row.get("Method", ""), row.get("Traj", ""),
                   row.get("Illumination", ""), row.get("Soil", ""), row.get("Reflectivity", ""))
                  for row in rows}
        for asset in all_assets:
            for method in all_methods:
                for v in variants_by_asset[asset]:
                    traj, illum = v.traj or "", v.illumination or ""
                    soil, reflectivity = v.soil or "", v.reflectivity or ""
                    if (asset, method, traj, illum, soil, reflectivity) in present:
                        continue
                    rows.append({"Asset": asset, "Method": method, "Traj": traj,
                                "Illumination": illum, "Soil": soil, "Reflectivity": reflectivity})


def collate_rows(results_root: Path, assets: set[str] | None, methods: set[str] | None,
                 ) -> dict[str, tuple[list[str], list[dict]]]:
    """Every row from every metric csv under `results_root`, each tagged with its Source (see
    module docstring), grouped by category. `methods` filters against each row's own Method
    column, not the folder name -- see `discover_method_dirs`. Returns {category: (fieldnames,
    rows)}: fieldnames is BASE_FIELDS plus the union of that category's metric columns seen, in
    first-seen order, plus "Source"."""
    metric_fields: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    rows: dict[str, list[dict]] = {}
    for method_dir in discover_method_dirs(results_root, assets):
        for category, path in sorted(_latest_metric_csvs(method_dir).items()):
            cat_seen = seen.setdefault(category, set(BASE_FIELDS) | {"Source"})
            cat_fields = metric_fields.setdefault(category, [])
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for name in reader.fieldnames or []:
                    if name not in cat_seen and name not in LABEL_COLUMNS:
                        cat_fields.append(name)
                        cat_seen.add(name)
                for row in reader:
                    if methods and row.get("Method", "") not in methods:
                        continue
                    row["Source"] = str(path.relative_to(results_root))
                    rows.setdefault(category, []).append(row)
    return {category: (BASE_FIELDS + metric_fields[category] + ["Source"], category_rows)
            for category, category_rows in rows.items()}


# --- CLI --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("/mnt/jgrayHDD/eccv_2026_results"),
                   help="an eval.py --save-hierarchical output root (default: %(default)s)")
    p.add_argument("--out", type=Path, required=True,
                   help="directory the collated csvs are written to, one per category "
                        "(<out>/cloud.csv, <out>/depth.csv, <out>/pose.csv, ...; created if "
                        "missing)")
    p.add_argument("--assets", default=None,
                   help="comma-separated subset of asset folder names (default: every asset "
                        "found under --results-root)")
    p.add_argument("--methods", default=None,
                   help="comma-separated subset of methods, matched against each row's own "
                        "Method column (e.g. da3,vggt,colmap/no_gt); default: every method found")
    p.add_argument("--include-missing", action="store_true",
                   help="pad every category out to the same asset/method/variant grid, adding a "
                        "blank row (label columns filled in, every metric + Source empty) for "
                        "any combo a category has no row for, instead of just omitting it -- "
                        "needs inspection_viz (--data-root); see module docstring")
    p.add_argument("--data-root", type=Path, default=Path("/home/shared/eccv_2026_data"),
                   help="asset root a true variant list is read from for --include-missing "
                        "(default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.results_root.is_dir():
        print(f"no such results root: {args.results_root}", file=sys.stderr)
        return 1

    assets = set(args.assets.split(",")) if args.assets else None
    methods = set(args.methods.split(",")) if args.methods else None
    by_category = collate_rows(args.results_root, assets, methods)
    if not by_category:
        print(f"no results found under {args.results_root}", file=sys.stderr)
        return 1
    if args.include_missing:
        all_assets, all_methods = discover_universe(args.results_root, assets, methods)
        add_missing_variants(by_category, all_assets, all_methods, args.data_root)

    args.out.mkdir(parents=True, exist_ok=True)
    for category, (fieldnames, rows) in sorted(by_category.items()):
        out_path = args.out / f"{category}.csv"
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows ({len(fieldnames)} columns) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
