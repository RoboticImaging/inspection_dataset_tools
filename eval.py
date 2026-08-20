"""Evaluate baseline predictions against ground truth.

    venv/bin/python eval.py /data/old_house/predictions/da3
    venv/bin/python eval.py /data/old_house/predictions/*        # every method, one table each

Each prediction folder is scored on every parameter combination it contains, one row per
variant. See `inspection_eval/metrics/base.py` for how to add a metric.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from inspection_eval import evaluate, frame_gaps, metrics, targets_from_paths
from inspection_eval.case import ALIGN_MODES
from inspection_eval.report import FORMATS, read_tables
from inspection_eval.runner import case_key


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", nargs="*", type=Path,
                   help="one or more <asset>/predictions/<method> folders; an asset root or "
                        "its predictions/ dir stands for every method inside it")
    p.add_argument("--out", type=Path, default=Path("eval_results"),
                   help="where the tables are written, in a subfolder named after what was "
                        "evaluated and when (default: %(default)s/<asset>-<method>-<timestamp>)")
    p.add_argument("--name", default=None,
                   help="name the subfolder yourself (--out/<NAME>) instead of letting it be "
                        "named after what was evaluated and when; point a later run at the same "
                        "name to carry on filling it")
    p.add_argument("--no-timestamp", action="store_true",
                   help="write straight into --out instead of a named, timestamped subfolder")
    existing = p.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true",
                          help="if the output folder already holds results, keep them and only "
                               "score the cases they are missing, without asking")
    existing.add_argument("--overwrite", action="store_true",
                          help="if the output folder already holds results, delete them and "
                               "evaluate everything again, without asking")
    p.add_argument("--metrics", default=None,
                   help=f"comma-separated subset of: {','.join(metrics.REGISTRY)}")
    p.add_argument("--allow-missing-frames", action="store_true",
                   help="don't fail when an rgb frame has no matching depth or pose file; "
                        "silently score only the frames that are complete instead (metrics "
                        "already skip a frame that is missing on either side, so this just "
                        "waives the upfront check)")
    p.add_argument("--align", default="sim3", choices=ALIGN_MODES,
                   help="how predictions are put in the gt gauge, fitted to the camera centers: "
                        "sim3 (default) also solves the unknown scale, se3 assumes the "
                        "prediction is already metric, none takes the poses as they are")
    p.add_argument("--format", default="csv,md",
                   help=f"comma-separated output formats: {','.join(FORMATS)} "
                        "(default: %(default)s)")
    p.add_argument("--no-stream", action="store_true",
                   help="only write the tables once, when everything has been scored; by default "
                        "they are rewritten after every case, so a long run can be read (and "
                        "interrupted) while it is still going")
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


def run_label(targets) -> str:
    """A short, readable tag for what `targets` covers, e.g. "old_house-da3" or, for a wildcard
    run over several methods, "old_house-da3+vggt". Used to name the output folder so that
    `eval_results/` stays browsable instead of filling up with bare timestamps."""
    assets = sorted({t.dataset.root.name for t in targets})
    # a colmap method's name has a "/" in it (e.g. "colmap/no_gt"); flattened here so it names one
    # folder rather than nesting one, the only place that slash would otherwise reach a filesystem
    methods = sorted({t.method.replace("/", "-") for t in targets})
    return f"{_join(assets, 'assets')}-{_join(methods, 'methods')}"


def _join(names: list, plural: str, limit: int = 3) -> str:
    if len(names) <= limit:
        return "+".join(names)
    return f"{len(names)}-{plural}"


def out_folder(args, targets) -> Path:
    """--out/<NAME> when a name was given, else --out itself or a folder named after this run."""
    if args.name:
        return args.out / args.name
    if args.no_timestamp:
        return args.out
    return args.out / f"{run_label(targets)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def start_from(out_dir: Path, args) -> dict:
    """The tables in `out_dir` to carry into this run, or {} to fill it from scratch.

    Only ever asked when a folder is reused (`--name`, `--no-timestamp`), and asked up front so a
    long run never stops on a question halfway through."""
    tables = {name: t for name, t in read_tables(out_dir).items() if len(t)} \
        if out_dir.is_dir() else {}
    if not tables:
        return {}
    cases = len({case_key(row) for table in tables.values() for row in table.rows})

    if args.overwrite:
        mode = "overwrite"
    elif args.resume:
        mode = "continue"
    elif not sys.stdin.isatty():
        mode = "continue"  # nothing gets deleted on a question nobody was there to answer
        print(f"{out_dir} already has {cases} scored case(s); continuing from there "
              f"(--overwrite to redo them)")
    else:
        mode = ask_about(out_dir, cases)

    if mode == "overwrite":
        removed = [p for name in tables for fmt in FORMATS
                   if (p := out_dir / f"{name}.{fmt}").exists()]
        for path in removed:
            path.unlink()
        print(f"removed {len(removed)} table file(s) from {out_dir}")
        return {}
    return tables


def ask_about(out_dir: Path, cases: int) -> str:
    print(f"\n{out_dir} already has results for {cases} case(s):"
          f"\n  [c] continue — keep them and score only what is missing"
          f"\n  [o] overwrite — delete them and evaluate everything again"
          f"\n  [q] quit")
    while True:
        try:
            answer = input("choice [c]: ").strip().lower()
        except EOFError:
            return "continue"
        if answer in ("", "c", "continue"):
            return "continue"
        if answer in ("o", "overwrite"):
            return "overwrite"
        if answer in ("q", "quit"):
            raise SystemExit("aborted, nothing was changed")


def describe_gaps(gaps) -> str:
    lines = [f"  {g.dataset.root.name} / {g.source} / {g.variant}: {g.field} missing for "
            f"frame(s) {', '.join(map(str, g.indices))}" for g in gaps]
    return ("some rgb frames have no matching depth or pose file:\n" + "\n".join(lines) +
           "\n\neither add the missing files, or pass --allow-missing-frames to evaluate only "
           "the frames that are complete")


BLOCKS = ("█", "░") if (sys.stdout.encoding or "").lower().startswith("utf") else ("#", "-")


def progress_bar(index: int, total: int, width: int = 20) -> str:
    """"[ 3/12] ██████░░░░░░░░░░░░░░" — how far through the cases the run is."""
    filled = round(width * index / total) if total else width
    return (f"[{index:>{len(str(total))}}/{total}] "
            f"{BLOCKS[0] * filled}{BLOCKS[1] * (width - filled)}")


class Progress:
    """Renders `evaluate`'s events. A normal run gets one counter line per case with that case's
    metric results underneath; a quiet run on a terminal gets a single bar rewritten in place, so
    `-q` still shows life without burying the tables. Quiet and piped stays silent."""

    def __init__(self, quiet: bool, stream=None):
        self.quiet = quiet
        self.stream = stream or sys.stdout
        self.live = quiet and self.stream.isatty()

    def __call__(self, kind, message, **info):
        if kind in ("case", "done"):
            line = progress_bar(info["index"], info["total"])
            if kind == "case":
                line += f" {message}"
            if self.live:
                self._live(line, last=kind == "done")
            elif kind == "case":
                self._say(line)
        elif kind != "done":
            self._say(message)

    def _say(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)

    def _live(self, line: str, last: bool) -> None:
        width = shutil.get_terminal_size().columns - 1
        self.stream.write("\r" + line[:width].ljust(width) + ("\n" if last else ""))
        self.stream.flush()


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

    out_dir = out_folder(args, targets)
    resume = start_from(out_dir, args)  # asked before anything long runs

    if not args.allow_missing_frames:
        gaps = frame_gaps(targets, filters)
        if gaps:
            raise SystemExit(describe_gaps(gaps))

    formats = [f.strip() for f in args.format.split(",")]
    flush = None if args.no_stream else lambda partial: partial.write(out_dir, formats)

    results = evaluate(targets, names, args.align, parse_overrides(args.set), filters,
                       Progress(args.quiet), flush, resume)

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

    written = results.write(out_dir, formats)
    print("\n" + "\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
