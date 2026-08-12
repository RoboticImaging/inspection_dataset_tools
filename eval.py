"""Evaluate baseline predictions against ground truth.

    venv/bin/python eval.py /data/old_house/predictions/da3
    venv/bin/python eval.py /data/old_house/predictions/*        # every method, one table each

Each prediction folder is scored on every parameter combination it contains, one row per
variant. See `inspection_eval/metrics/base.py` for how to add a metric.
"""

import argparse
import sys
from pathlib import Path

from inspection_eval import evaluate, metrics, targets_from_paths
from inspection_eval.case import ALIGN_MODES
from inspection_eval.report import FORMATS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", nargs="*", type=Path,
                   help="one or more <asset>/predictions/<method> folders; an asset root or "
                        "its predictions/ dir stands for every method inside it")
    p.add_argument("--out", type=Path, default=Path("eval_results"),
                   help="where the tables are written (default: %(default)s)")
    p.add_argument("--metrics", default=None,
                   help=f"comma-separated subset of: {','.join(metrics.REGISTRY)}")
    p.add_argument("--align", default="sim3", choices=ALIGN_MODES,
                   help="how predictions are put in the gt gauge, fitted to the camera centers: "
                        "sim3 (default) also solves the unknown scale, se3 assumes the "
                        "prediction is already metric, none takes the poses as they are")
    p.add_argument("--format", default="csv,md",
                   help=f"comma-separated output formats: {','.join(FORMATS)} "
                        "(default: %(default)s)")
    p.add_argument("--set", action="append", default=[], metavar="METRIC.OPTION=VALUE",
                   help="override a metric option, e.g. --set depth.scale=median "
                        "--set cloud.thresholds=0.1,0.5 (repeatable)")
    for field in ("traj", "illumination", "soil", "reflectivity"):
        p.add_argument(f"--{field}", default=None, help=f"only evaluate variants with this {field}")
    p.add_argument("--list-metrics", action="store_true",
                   help="print the registered metrics with their options and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="only print the tables")
    return p.parse_args(argv)


def parse_overrides(items) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        name, _, option = key.partition(".")
        if not (name and option and value):
            raise SystemExit(f"--set expects METRIC.OPTION=VALUE, got {item!r}")
        if name not in metrics.REGISTRY:
            raise SystemExit(f"--set: unknown metric {name!r}; known: {sorted(metrics.REGISTRY)}")
        out.setdefault(name, {})[option] = value
    return out


def list_metrics() -> None:
    import dataclasses
    for metric in metrics.REGISTRY.values():
        needs = ", ".join(metric.needs) or "nothing"
        print(f"{metric.name}  (needs: {needs})")
        for field in dataclasses.fields(metric.options) if metric.options else ():
            print(f"    {metric.name}.{field.name} = {getattr(metric.default_options(), field.name)!r}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_metrics:
        list_metrics()
        return 0
    if not args.predictions:
        raise SystemExit("give at least one <asset>/predictions/<method> folder (see --help)")

    targets = targets_from_paths(args.predictions)
    names = [n.strip() for n in args.metrics.split(",")] if args.metrics else None
    filters = {"traj": args.traj, "illumination": args.illumination,
               "soil": args.soil, "reflectivity": args.reflectivity}

    def on_event(kind, message):
        if not args.quiet:
            print(message, flush=True)

    results = evaluate(targets, names, args.align, parse_overrides(args.set), filters, on_event)

    for name, table in results.tables.items():
        if len(table):
            print(f"\n## {name}\n")
            print(table.to_markdown())
    if not any(len(t) for t in results.tables.values()):
        print("no results: nothing could be evaluated", file=sys.stderr)
        for skip in results.skips:
            print(f"  {skip.case.method} / {skip.case.variant} / {skip.metric}: {skip.reason}",
                  file=sys.stderr)
        return 1

    written = results.write(args.out, [f.strip() for f in args.format.split(",")])
    print("\n" + "\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
