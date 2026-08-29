#!/usr/bin/env python3
"""Prints ready-to-paste LaTeX rows for the paper's per-scene results tables, read straight off
an `eval.py --save-hierarchical` output tree (`<results_root>/<asset>/<method>/<metric>[_<ts>].csv`
-- see `eval.py`'s module docstring).

Usage:
    python latex_results_table.py /mnt/jgrayHDD/eccv_2026_results vggt-omega da3 pi-long
    python latex_results_table.py RESULTS_ROOT pi3 --table depth
    python latex_results_table.py RESULTS_ROOT colmap/no_gt --decimals 2

Each method becomes one `Label & c1 & ... & c7 \\\\` line, one cell per scene column, values
averaged across every variant (illumination/soil/traj/reflectivity) that scene has:

    "cloud" table row: AUC30 (pose.csv "AUC@30") / Chamfer Distance (cloud.csv "Chamfer (m)")
    "depth" table row: sMSE(log) (depth.csv "Scale invariant MSE") / AbsRel (depth.csv "Relative MAE")

A cell is `OOM` if every variant of that scene hit a GPU OOM, blank if the scene hasn't been
*completely* evaluated for that method yet (no rows at all, or fewer than the scene's real
variant count -- see `--data-root` below), `left / right` if every variant scored normally, and
`[left / right]` (the average of the non-OOM variants only) if some but not all did -- possible
when a hierarchical run is resumed (see `eval.py --resume --save-hierarchical`) and only some of
a scene's variants had already OOM'd on an earlier attempt.

A combo isn't averaged at all until every variant its scene actually has is accounted for in its
csvs (real or OOM) -- otherwise a still-partway-through eval (e.g. an interrupted
`--save-hierarchical` run, or `crane/pi-long` scoring only 1 of its 16 variants as of writing)
would silently average just the rows that happen to exist, and read as a finished result. This
needs to know each scene's true variant count, via `--data-root` (default
`/home/shared/eccv_2026_data`) and `inspection_viz.dataset.AssetDataset` -- the same source
`eval.py`'s own `incomplete_targets` judges a method's *predictions* against, just applied here to
`eval.py`'s *scoring* instead. Pulls in `inspection_viz` (and so cv2/trimesh/OpenEXR/PIL) for
that reason -- `check_completion_status.py` already pays the same cost for the same reason; this
script still avoids `baselines.*`/`inspection_eval`, which is the heavier half.

Pass `--full-table` instead of method names to print every method's row, grouped into the three
sections the templates use (`SECTIONS`), instead of naming methods one at a time.
"""

import argparse
import csv
import re
import statistics
import sys
from functools import lru_cache
from pathlib import Path

from inspection_viz.dataset import AssetDataset, Variant

# --- scene columns, in table order -----------------------------------------------------------

# Column header -> the asset folder eval.py was pointed at for that scene. Bridge/Ship/Crane/Old
# House match their folder name outright; Office/Glass House/Unfin. don't (see the config's
# `asset:` path in baselines/configs/data/*.yaml, and eccv_2026_data itself -- office2 and
# glass_house2 are the ones with actual predictions/ trees, and per the user "Office" pins to
# office2 specifically even though it currently has no predictions/ of its own yet).
COLUMNS = [
    ("Office", "office2"),
    ("Bridge", "bridge"),
    ("Ship", "ship"),
    ("Crane", "crane"),
    ("Old House", "old_house"),
    ("Glass House", "glass_house2"),
    ("Unfin.", "unfinished_build"),
]

# Method key (what you'd pass eval.py / find under predictions/, "colmap/no_gt" for the colmap
# case) -> the row label the paper's tables use. A method not listed here still works (see
# `_label`): its raw key is used verbatim, with a warning, so a new baseline needs no code change
# here to try the script.
METHOD_LABELS = {
    "colmap/no_gt": "COLMAP",
    "mpsfm": "MP-SfM",
    "vggt": "VGGT",
    "da3": "DA3",
    "vggt-omega": r"VGGT-$\Omega$",
    "pi3": r"$\pi^3$",
    "pi3x": r"$\pi - \chi$",
    "pi-long": "Pi-Long",
    "da3_streaming": "DA3-Streaming",
    "vggt-long": "VGGT-Long",
    "vggt-slam": "VGGT-SLAM 2.0",
}

# --full-table's method order: the three sections the templates group methods into, each with its
# ordered method keys -- exactly `METHOD_LABELS`'s 11 entries, split out this way.
SECTIONS = [
    ("Structure from Motion", ["colmap/no_gt", "mpsfm"]),
    ("End to End Feed-Forward", ["vggt", "da3", "vggt-omega", "pi3", "pi3x"]),
    ("Chunk and SLAM Based E2E", ["pi-long", "da3_streaming", "vggt-long", "vggt-slam"]),
]

# table name -> the 1-2 (metric csv, column) pairs that make up one cell, left-to-right.
TABLE_SPECS = {
    "cloud": [("pose", "AUC@30"), ("cloud", "Chamfer (m)")],
    "depth": [("depth", "Scale invariant MSE"), ("depth", "Relative MAE")],
}
TABLE_TITLES = {"cloud": "point cloud table", "depth": "depth table"}


# --- reading eval.py --save-hierarchical's output ---------------------------------------------

_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}-\d{6}$")


def _latest_metric_csvs(method_dir: Path) -> dict[str, Path]:
    """The most recent csv for each metric under one asset/method's hierarchical output folder
    (see `eval.py`'s `write_hierarchical`): grouped by name with any timestamp suffix stripped,
    keeping a bare <metric>.csv (always current -- a --no-timestamp run overwrites it in place
    rather than adding another file) or else the lexicographically-latest timestamped one (they
    sort correctly as text). The same rule `eval.py`'s and `check_completion_status.py`'s own
    copies of this function use, duplicated rather than imported -- see the module docstring."""
    if not method_dir.is_dir():
        return {}
    by_metric: dict[str, list[Path]] = {}
    for path in method_dir.glob("*.csv"):
        by_metric.setdefault(_TIMESTAMP_SUFFIX.sub("", path.stem), []).append(path)

    def latest(paths: list[Path]) -> Path:
        bare = [p for p in paths if not _TIMESTAMP_SUFFIX.search(p.stem)]
        return bare[0] if bare else max(paths, key=lambda p: p.stem)

    return {metric: latest(paths) for metric, paths in by_metric.items()}


def _column_values(path: Path, column: str) -> list[str]:
    """The raw text of `column` from every row of one results csv, in file order."""
    with path.open(newline="") as handle:
        return [row.get(column, "") for row in csv.DictReader(handle)]


# --- has this combo's eval actually finished? ---------------------------------------------

@lru_cache(maxsize=None)
def _asset_variants(data_root: Path, asset_folder: str) -> frozenset | None:
    """Every variant `asset_folder` really has, per its own rgb data under `data_root` -- the
    ground truth `eval.py`'s own `incomplete_targets` judges a method's predictions against, here
    used the same way to judge `eval.py`'s *scoring*. `None` if the asset can't be loaded there
    (not present on this machine, say), so a scene that just isn't reachable degrades to "can't
    confirm this is complete" rather than crashing. Cached: one `AssetDataset` per asset actually
    asked about, not one per method/table call."""
    try:
        return frozenset(AssetDataset(data_root / asset_folder).variants)
    except FileNotFoundError:
        return None


def _touched_variants(method_dir: Path) -> set:
    """Every variant with a row in *any* metric csv under one asset/method's hierarchical folder
    -- value irrelevant, so an OOM row counts as "reached" too (an all-OOM combo already gets a
    row for every variant -- see `inspection_eval.runner.evaluate`'s OOM handling). Same
    Traj/Illumination/Soil/Reflectivity -> `Variant` conversion `check_completion_status.py`'s
    `eval_variants` uses."""
    touched = set()
    for path in _latest_metric_csvs(method_dir).values():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                touched.add(Variant(row.get("Traj") or None, row.get("Illumination") or None,
                                    row.get("Soil") or None, row.get("Reflectivity") or None))
    return touched


@lru_cache(maxsize=None)
def _is_complete(data_root: Path, method_dir: Path) -> bool:
    """Has `eval.py` actually reached every variant `method_dir`'s asset has, or is this combo's
    eval only partway through (an interrupted --save-hierarchical run, or one resumed since but
    not finished)? A subset check, not equality, so stale rows left over from a since-shrunk
    asset don't wrongly read as incomplete."""
    total = _asset_variants(data_root, method_dir.parent.name)
    return total is not None and total <= _touched_variants(method_dir)


# --- one cell -----------------------------------------------------------------------------

def _metric_cell(method_dir: Path, metric: str, column: str) -> tuple[str, float | None]:
    """One (metric, column)'s contribution to a cell: `("missing", None)` if that scene was never
    evaluated for this metric (no csv, or a csv with no rows), `("oom", None)` if every row is
    OOM, or `("value", average)` -- the mean of whichever rows have a usable number, which may be
    every row (all scored normally) or only some of them (a mix -- see the module docstring)."""
    csvs = _latest_metric_csvs(method_dir)
    if metric not in csvs:
        return "missing", None
    values = _column_values(csvs[metric], column)
    if not values:
        return "missing", None
    numbers = [float(v) for v in values if v not in ("", "OOM")]
    if not numbers:
        return ("oom", None) if any(v == "OOM" for v in values) else ("missing", None)
    kind = "value" if all(v != "OOM" for v in values) else "partial"
    return kind, statistics.mean(numbers)


def _format_cell(method_dir: Path, specs: list[tuple[str, str]], decimals: int) -> str:
    """The finished `"left / right"` (or single-metric `"left"`) text for one scene cell, per
    `_metric_cell`'s per-metric results: blank if every metric is missing, `OOM` if every present
    metric is fully OOM, brackets around the joined text if any metric was only a partial OOM."""
    results = [_metric_cell(method_dir, metric, column) for metric, column in specs]
    if all(kind == "missing" for kind, _ in results):
        return ""
    parts = []
    for kind, value in results:
        if kind == "missing":
            parts.append("")
        elif kind == "oom":
            parts.append("OOM")
        else:
            parts.append(f"{value:.{decimals}f}")
    if all(kind in ("missing", "oom") for kind, _ in results):
        return "OOM"
    text = " / ".join(parts)
    return f"[{text}]" if any(kind == "partial" for kind, _ in results) else text


# --- one row --------------------------------------------------------------------------------

def _label(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    print(f"warning: no known LaTeX label for method {method!r}; using it as-is", file=sys.stderr)
    return method


def build_row(results_root: Path, data_root: Path, method: str, table: str, decimals: int) -> str:
    """One `Label & c1 & ... & c7 \\\\` line for `method`, for `table` ("cloud" or "depth")."""
    specs = TABLE_SPECS[table]
    folder = method.replace("/", "-")  # same flattening eval.py's write_hierarchical uses
    cells = []
    for _, asset in COLUMNS:
        method_dir = results_root / asset / folder
        cells.append(_format_cell(method_dir, specs, decimals)
                     if _is_complete(data_root, method_dir) else "")
    return f"{_label(method)} & " + " & ".join(cells) + r" \\"


def build_table(results_root: Path, data_root: Path, table: str, decimals: int) -> str:
    """The whole table body for `table`: every method from `SECTIONS`, in its section, with the
    same \\hline / colored \\multicolumn section headers the templates use. Everything between
    (not including) the column-header row and \\end{tabular} -- the static wrapper around it
    (\\begin{table*}...\\end{table*}, caption, label) is already written and isn't regenerated."""
    lines = []
    for title, methods in SECTIONS:
        lines.append(r"\hline")
        lines.append(r"\multicolumn{8}{c}{\cellcolor{catgray}\textbf{" + title + r"}} \\")
        lines.append(r"\hline")
        lines += [build_row(results_root, data_root, method, table, decimals)
                 for method in methods]
    lines.append(r"\hline")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_root", type=Path,
                   help="an eval.py --save-hierarchical output root (--out/--name)")
    p.add_argument("methods", nargs="*",
                   help="method key(s) to emit a row for, e.g. vggt-omega, pi-long, colmap/no_gt "
                        "-- omit and pass --full-table instead to print every method's row")
    p.add_argument("--full-table", action="store_true",
                   help="print every method's row, grouped into the templates' three sections "
                        "(see SECTIONS), instead of one row per name in `methods`")
    p.add_argument("--data-root", type=Path, default=Path("/home/shared/eccv_2026_data"),
                   help="asset root a scene's true variant count is read from, to tell a "
                        "not-yet-completely-evaluated combo from a finished one (default: "
                        "%(default)s)")
    p.add_argument("--table", choices=("cloud", "depth", "both"), default="both",
                   help="which table's row shape to print (default: %(default)s)")
    p.add_argument("--decimals", type=int, default=3,
                   help="decimal places for each value (default: %(default)s)")
    args = p.parse_args(argv)
    if args.full_table and args.methods:
        p.error("--full-table can't be combined with explicit method names")
    if not args.full_table and not args.methods:
        p.error("give at least one method, or pass --full-table")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = ("cloud", "depth") if args.table == "both" else (args.table,)
    for i, table in enumerate(tables):
        if len(tables) > 1:
            print(f"% --- {TABLE_TITLES[table]} ---")
        if args.full_table:
            print(build_table(args.results_root, args.data_root, table, args.decimals))
        else:
            for method in args.methods:
                print(build_row(args.results_root, args.data_root, method, table, args.decimals))
        if len(tables) > 1 and i < len(tables) - 1:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
