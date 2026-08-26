#!/usr/bin/env python3
"""Scans an eccv_2026_data-style root and reports, for every registered baseline
(baselines/registry.py) and every asset found there, whether that baseline has completed every
sequence variant the asset has -- judged purely by whether the fused point cloud
(predictions/<method>/<illumination>/clouds/...ply) exists for that variant.

This is a coarse but reliable signal: `out.write_cloud` (baselines.common.runner.SequenceOutput)
is the very last thing a baseline's run() does for a sequence, after every per-frame depth/cam has
already been written -- so a saved cloud implies the whole sequence finished, without having to
count individual depth frames or guess what frame range a given run actually asked for. It does
assume the baseline config had `output.cloud: true` (the default for every baseline config under
baselines/configs/ as of writing); a variant run with clouds disabled will always show as
NOT STARTED here even if its depth/cam output is complete.

If a baseline hit a GPU OOM (see baselines.common.runner.run_baseline) on any variant of an asset,
that overrides everything else for that asset/baseline: the summary just reads OOM, since the
variants it did finish before the OOM aren't a meaningful COMPLETE/PARTIAL/NOT STARTED picture of
that asset for that baseline. --verbose lists which variant(s) triggered it.

colmap is checked alongside every registered baseline, as its own "colmap/no_gt" column --
predictions found under <asset>/colmap/no_gt (every asset currently on disk only ever has a
"no_gt" mode; see `AssetDataset`'s COLMAP_PREFIX). It isn't in baselines/registry.py (it's run via
its own colmap_*.sh scripts, not baselines.common.runner), but `AssetDataset` already treats a
colmap run as just another "method" -- a fused cloud at colmap/no_gt/.../dense/fused.ply reads the
same as predictions/<method>/.../clouds/....ply -- so the exact same COMPLETE/PARTIAL/NOT
STARTED/OOM accounting applies to it unchanged. --baseline colmap/no_gt selects only it, same as
any registered name.

--eval-folder adds a second table alongside the prediction one: the same COMPLETE/PARTIAL/NOT
STARTED/OOM picture, but for whether `eval.py` has actually scored each asset/baseline combo yet,
read off an `eval.py --save-hierarchical` run rooted at the given folder (that folder is exactly
what was passed as eval.py's --out/--name -- see eval.py's `hierarchical_root`). A variant counts
as OOM there if any metric's row for it reads "OOM" (see inspection_eval.runner's OOM handling),
and as done if any metric scored it at all -- metrics can legitimately skip different variants
(no gt mesh means no cloud row, say), so requiring every metric to agree would under-count.

Usage:
    python check_completion_status.py
    python check_completion_status.py --eccv-root /home/shared/eccv_2026_data
    python check_completion_status.py --baseline vggt --baseline pi3   # only these baselines
    python check_completion_status.py --asset old_house                # only this asset
    python check_completion_status.py --exclude-asset old_house        # every asset but this
    python check_completion_status.py --verbose                        # list missing variants too
    python check_completion_status.py --example                        # only the example sequence
    python check_completion_status.py --eval-folder eval_results/myrun # also check eval status

Needs only this repo's own light dependencies (inspection_viz -- see requirements.txt), not any
baseline's own heavy ones (torch, cv2, ...), so it runs fine directly on the host. --eval-folder
reads results csvs with the stdlib `csv` module rather than importing `inspection_eval` for the
same reason: that package pulls in cv2/trimesh transitively.
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import registry
from baselines.common.config import example_filters
from inspection_viz.dataset import AssetDataset, Variant

DEFAULT_ECCV_ROOT = Path("/home/shared/eccv_2026_data")

# COMPLETE/PARTIAL/NOT STARTED/OOM colors for the printed tables/--verbose detail (see Status.color
# below). Off entirely when stdout isn't a terminal (e.g. redirected to a log file, as the mpsfm
# crash-capture workflow does) or NO_COLOR is set (https://no-color.org), so piped/saved output
# stays plain text rather than full of escape codes.
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_GREEN, _YELLOW, _RED, _BLUE, _RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[34m", "\033[0m") if _USE_COLOR else ("",) * 5


def _colorize(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if color else text

# colmap isn't a registered baseline (baselines/registry.py) -- it's run separately, via
# colmap_*.sh, not baselines.common.runner -- but AssetDataset already folds it into the same
# "method" concept as every other prediction source (see inspection_viz.dataset.AssetDataset,
# COLMAP_PREFIX), so it's checked here the exact same way, just not drawn from that registry.
# "no_gt" is the only mode any asset currently has on disk (checked across every asset's
# colmap/ dir); a future second mode would need its own entry here.
COLMAP_METHOD = "colmap/no_gt"

# The label columns an eval.py results csv starts with (see inspection_eval.runner.LABEL_COLUMNS
# / EvalCase.label) -- reproduced here rather than imported, to keep --eval-folder's csv reading
# free of inspection_eval's heavier dependencies (see module docstring).
LABEL_COLUMNS = ("Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity")


@dataclass
class Status:
    """COMPLETE/PARTIAL/NOT STARTED/OOM for one asset/baseline combo -- what `compute_status`
    reports for predictions, and what `compute_eval_status` reports for --eval-folder's
    evaluation-status table, from otherwise unrelated inputs (a cloud file vs. a results csv)."""

    total: int             # every variant the asset has
    done: list[Variant]     # variants with output (a saved cloud, or a scored eval row)
    oom: list[Variant]      # variants with no output, but a marker saying why: a GPU OOM

    @property
    def state(self) -> str:
        if not self.done:
            return "NOT STARTED"
        if len(self.done) == self.total:
            return "COMPLETE"
        return "PARTIAL"

    @property
    def summary(self) -> str:
        if self.oom:
            return "OOM"
        return f"{self.state} ({len(self.done)}/{self.total})"

    @property
    def color(self) -> str:
        """The ANSI color `summary`/`state` should print in: blue for OOM (checked first, same as
        `summary`), else green/yellow/red for COMPLETE/PARTIAL/NOT STARTED."""
        if self.oom:
            return _BLUE
        return {"COMPLETE": _GREEN, "PARTIAL": _YELLOW, "NOT STARTED": _RED}[self.state]


def discover_assets(eccv_root: Path, only: set[str] | None,
                    exclude: set[str] | None = None) -> list[AssetDataset]:
    """Every subdirectory of `eccv_root` that's actually an inspection asset (has at least one
    sequence). eccv_2026_data also holds loose files at the same level (colmap_*.sh,
    upload_script.py, ...); those, and any directory with no matching sequences, are skipped.
    `exclude` drops assets by name after `only` is applied, for omitting a few noisy assets
    from an otherwise-default (or --asset-filtered) run without having to list every other one.
    """
    assets = []
    for path in sorted(eccv_root.iterdir()):
        if not path.is_dir() or (only and path.name not in only):
            continue
        if exclude and path.name in exclude:
            continue
        try:
            ds = AssetDataset(path)
        except Exception as e:
            print(f"  skipping {path.name}: {e}", file=sys.stderr)
            continue
        if not ds.variants:
            continue
        assets.append(ds)
    return assets


def compute_status(ds: AssetDataset, method: str, variants: list[Variant]) -> Status:
    done = [v for v in variants if ds.cloud_path(v, method) is not None]
    oom = [v for v in variants if v not in done and ds.oom_path(v, method) is not None]
    return Status(total=len(variants), done=done, oom=oom)


_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}-\d{6}$")


def _latest_metric_csvs(method_dir: Path) -> list[Path]:
    """The most recent csv for each metric under one asset/method's `eval.py --save-hierarchical`
    folder. `write_hierarchical` names files `<metric>_<timestamp>.csv`, or plain `<metric>.csv`
    under `--no-timestamp` -- e.g. `crane/da3/cloud_20260821-144854.csv` -- so a folder evaluated
    more than once can hold several csvs per metric. Grouping by the name with any timestamp
    suffix stripped and keeping the newest per group picks the right one either way: a plain
    `<metric>.csv` is always current (a later `--no-timestamp` run overwrites it in place, rather
    than adding another file), and otherwise the lexicographically-largest timestamp wins (they
    sort correctly as text)."""
    if not method_dir.is_dir():
        return []
    by_metric: dict[str, list[Path]] = {}
    for path in method_dir.glob("*.csv"):
        metric = _TIMESTAMP_SUFFIX.sub("", path.stem)
        by_metric.setdefault(metric, []).append(path)

    def latest(paths: list[Path]) -> Path:
        bare = [p for p in paths if not _TIMESTAMP_SUFFIX.search(p.stem)]
        return bare[0] if bare else max(paths, key=lambda p: p.stem)

    return [latest(paths) for paths in by_metric.values()]


def eval_variants(method_dir: Path) -> tuple[set[Variant], set[Variant]]:
    """(done, oom) variants found across every metric csv under one asset/method's hierarchical
    eval folder (see module docstring). A missing folder, or one with no csvs yet, yields two
    empty sets -- `compute_eval_status` then reports that combo as NOT STARTED, same as a
    baseline with no cloud file."""
    done: set[Variant] = set()
    oom: set[Variant] = set()
    for path in _latest_metric_csvs(method_dir):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                variant = Variant(row.get("Traj") or None, row.get("Illumination") or None,
                                  row.get("Soil") or None, row.get("Reflectivity") or None)
                values = [v for k, v in row.items() if k not in LABEL_COLUMNS]
                (oom if "OOM" in values else done).add(variant)
    return done, oom


def compute_eval_status(eval_folder: Path, asset: str, method: str,
                        variants: list[Variant]) -> Status:
    # a colmap method's name has a "/" in it (e.g. "colmap/no_gt"); flattened here the same way
    # eval.py's write_hierarchical does, so this lands on the one real folder
    # (<eval_folder>/<asset>/colmap-no_gt) instead of nesting a nonexistent extra directory level.
    folder = method.replace("/", "-")
    done_v, oom_v = eval_variants(eval_folder / asset / folder)
    done = [v for v in variants if v in done_v]
    oom = [v for v in variants if v in oom_v]
    return Status(total=len(variants), done=done, oom=oom)


def print_status_table(assets: list[AssetDataset], methods: list[str], results: dict) -> None:
    """One row per asset, one column per baseline; sized from what actually got computed, since
    an OOM annotation can make a summary longer than a bare "COMPLETE (00/00)"."""
    name_w = max(len(ds.root.name) for ds in assets)
    col_w = max(len("COMPLETE (00/00)"), *(len(m) for m in methods),
               *(len(s.summary) for s in results.values())) + 1

    header = " " * name_w + "  " + "  ".join(m.ljust(col_w) for m in methods)
    print(header)
    print("-" * len(header))
    for ds in assets:
        row = [ds.root.name.ljust(name_w)]
        for method in methods:
            status = results[ds.root.name, method]
            # pad the plain text to col_w *before* adding color codes -- ljust'ing the colorized
            # string would count the (invisible) escape sequence characters as part of the width
            # and under-pad every cell.
            row.append(_colorize(status.summary.ljust(col_w), status.color))
        print("  ".join(row))


def print_missing(results: dict, variants_by_asset: dict, missing_label: str) -> None:
    """--verbose detail for one results dict: for every non-COMPLETE combo, either the OOM
    variant(s) that explain it or the variants still missing `missing_label` (e.g. "clouds",
    "eval rows")."""
    missing_any = False
    for (asset, method), status in results.items():
        if status.state == "COMPLETE":
            continue
        missing_any = True
        if status.oom:
            print(f"\n{asset} / {method}: {_colorize('OOM', status.color)}, hit on:")
            for v in sorted(status.oom, key=str):
                print(f"  - {v.as_dict()}")
            continue
        missing = sorted(set(variants_by_asset[asset]) - set(status.done), key=str)
        print(f"\n{asset} / {method}: {_colorize(status.summary, status.color)}, "
             f"missing {missing_label} for:")
        for v in missing:
            print(f"  - {v.as_dict()}")
    if not missing_any:
        print("\nEverything checked is COMPLETE.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eccv-root", type=Path, default=DEFAULT_ECCV_ROOT,
                   help=f"asset root to scan (default: {DEFAULT_ECCV_ROOT})")
    p.add_argument("--baseline", action="append", dest="baselines", metavar="NAME",
                   help=f"only check this baseline (repeatable), or {COLMAP_METHOD!r} for colmap; "
                        f"default: every registered one plus {COLMAP_METHOD!r}")
    p.add_argument("--asset", action="append", dest="assets", metavar="NAME",
                   help="only check this asset by folder name (repeatable); default: every one found")
    p.add_argument("--exclude-asset", action="append", dest="exclude_assets", metavar="NAME",
                   help="omit this asset by folder name from the printout (repeatable); "
                        "applied after --asset")
    p.add_argument("--verbose", action="store_true",
                   help="list the specific variants missing a cloud for any non-COMPLETE result")
    p.add_argument("--example", action="store_true",
                   help="only check the one sequence pinned in "
                        "baselines/configs/data_example/<asset>.yaml (traj/illumination/soil), "
                        "instead of every variant found")
    p.add_argument("--eval-folder", type=Path, default=None,
                   help="also report evaluation status, read from an eval.py --save-hierarchical "
                        "run rooted here (the exact folder passed as eval.py's --out/--name -- "
                        "pass the already-suffixed folder yourself if that run also used "
                        "--example); default: don't check evaluation status")
    args = p.parse_args(argv)

    methods = args.baselines or [*registry.available(), COLMAP_METHOD]
    unknown = set(methods) - set(registry.available()) - {COLMAP_METHOD}
    if unknown:
        raise SystemExit(f"unknown baseline(s) {sorted(unknown)}; registered: "
                         f"{', '.join(registry.available())}, plus {COLMAP_METHOD!r}")

    only = set(args.assets) if args.assets else None
    exclude = set(args.exclude_assets) if args.exclude_assets else None
    print(f"scanning {args.eccv_root} ...")
    assets = discover_assets(args.eccv_root, only, exclude)
    if not assets:
        raise SystemExit(f"no assets with sequences found under {args.eccv_root}"
                         + (f" matching {sorted(only)}" if only else ""))

    filters = example_filters() if args.example else {}
    variants_by_asset = {}
    kept = []
    for ds in assets:
        variants = [v for v in ds.variants
                    if all(getattr(v, f) == val for f, val in filters.items() if val is not None)]
        if args.example and not variants:
            print(f"  skipping {ds.root.name}: no variant matches the example filter",
                  file=sys.stderr)
            continue
        variants_by_asset[ds.root.name] = variants
        kept.append(ds)
    assets = kept
    if not assets:
        raise SystemExit(f"no assets have a variant matching the example filter under "
                         f"{args.eccv_root}")
    print()

    results = {}  # (asset, method) -> Status
    for ds in assets:
        for method in methods:
            results[ds.root.name, method] = compute_status(ds, method, variants_by_asset[ds.root.name])
    print_status_table(assets, methods, results)

    eval_results = None
    if args.eval_folder:
        eval_results = {}
        for ds in assets:
            for method in methods:
                eval_results[ds.root.name, method] = compute_eval_status(
                    args.eval_folder, ds.root.name, method, variants_by_asset[ds.root.name])
        print(f"\nEvaluation status ({args.eval_folder}):")
        print_status_table(assets, methods, eval_results)

    if args.verbose:
        print_missing(results, variants_by_asset, "clouds")
        if eval_results is not None:
            print_missing(eval_results, variants_by_asset, "eval rows")


if __name__ == "__main__":
    main()
