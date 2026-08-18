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

Usage:
    python check_completion_status.py
    python check_completion_status.py --eccv-root /home/shared/eccv_2026_data
    python check_completion_status.py --baseline vggt --baseline pi3   # only these baselines
    python check_completion_status.py --asset old_house                # only this asset
    python check_completion_status.py --verbose                        # list missing variants too

Needs only this repo's own light dependencies (inspection_viz -- see requirements.txt), not any
baseline's own heavy ones (torch, cv2, ...), so it runs fine directly on the host.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import registry
from inspection_viz.dataset import AssetDataset, Variant

DEFAULT_ECCV_ROOT = Path("/home/shared/eccv_2026_data")


@dataclass
class Status:
    total: int             # every variant the asset has
    done: list[Variant]     # variants with a saved cloud from this method

    @property
    def state(self) -> str:
        if not self.done:
            return "NOT STARTED"
        if len(self.done) == self.total:
            return "COMPLETE"
        return "PARTIAL"

    @property
    def summary(self) -> str:
        return f"{self.state} ({len(self.done)}/{self.total})"


def discover_assets(eccv_root: Path, only: set[str] | None) -> list[AssetDataset]:
    """Every subdirectory of `eccv_root` that's actually an inspection asset (has at least one
    sequence). eccv_2026_data also holds loose files at the same level (colmap_*.sh,
    upload_script.py, ...); those, and any directory with no matching sequences, are skipped.
    """
    assets = []
    for path in sorted(eccv_root.iterdir()):
        if not path.is_dir() or (only and path.name not in only):
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


def compute_status(ds: AssetDataset, method: str) -> Status:
    done = [v for v in ds.variants if ds.cloud_path(v, method) is not None]
    return Status(total=len(ds.variants), done=done)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eccv-root", type=Path, default=DEFAULT_ECCV_ROOT,
                   help=f"asset root to scan (default: {DEFAULT_ECCV_ROOT})")
    p.add_argument("--baseline", action="append", dest="baselines", metavar="NAME",
                   help="only check this baseline (repeatable); default: every registered one")
    p.add_argument("--asset", action="append", dest="assets", metavar="NAME",
                   help="only check this asset by folder name (repeatable); default: every one found")
    p.add_argument("--verbose", action="store_true",
                   help="list the specific variants missing a cloud for any non-COMPLETE result")
    args = p.parse_args(argv)

    methods = args.baselines or registry.available()
    unknown = set(methods) - set(registry.available())
    if unknown:
        raise SystemExit(f"unknown baseline(s) {sorted(unknown)}; registered: "
                         f"{', '.join(registry.available())}")

    only = set(args.assets) if args.assets else None
    print(f"scanning {args.eccv_root} ...")
    assets = discover_assets(args.eccv_root, only)
    if not assets:
        raise SystemExit(f"no assets with sequences found under {args.eccv_root}"
                         + (f" matching {sorted(only)}" if only else ""))
    print()

    # One row per asset, one column per baseline.
    name_w = max(len(ds.root.name) for ds in assets)
    col_w = max(len("COMPLETE (00/00)"), *(len(m) for m in methods)) + 1

    header = " " * name_w + "  " + "  ".join(m.ljust(col_w) for m in methods)
    print(header)
    print("-" * len(header))

    results = {}  # (asset, method) -> Status
    for ds in assets:
        row = [ds.root.name.ljust(name_w)]
        for method in methods:
            status = compute_status(ds, method)
            results[ds.root.name, method] = status
            row.append(status.summary.ljust(col_w))
        print("  ".join(row))

    if args.verbose:
        missing_any = False
        for (asset, method), status in results.items():
            if status.state == "COMPLETE":
                continue
            missing_any = True
            ds = next(d for d in assets if d.root.name == asset)
            missing = sorted(set(ds.variants) - set(status.done), key=str)
            print(f"\n{asset} / {method}: {status.summary}, missing clouds for:")
            for v in missing:
                print(f"  - {v.as_dict()}")
        if not missing_any:
            print("\nEverything checked is COMPLETE.")


if __name__ == "__main__":
    main()
