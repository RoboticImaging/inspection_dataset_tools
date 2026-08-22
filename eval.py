"""Evaluate baseline predictions against ground truth.

    venv/bin/python eval.py /data/old_house/predictions/da3
    venv/bin/python eval.py /data/old_house/predictions/*        # every method, one table each

Each prediction folder is scored on every parameter combination it contains, one row per
variant. See `inspection_eval/metrics/base.py` for how to add a metric.

An asset/method combo is only scored once it's COMPLETE -- a prediction for every variant its
asset has -- so a still-in-progress run isn't evaluated as though its partial output were the
whole picture (see `inspection_eval.incomplete_targets`). --example is the one exception: it only
ever asks about the single sequence it pins, so completeness there means just that one being
present.

By default every table mixes every asset/method combo's rows together, one file per metric;
--save-hierarchical instead writes one subfolder per combo (see `write_hierarchical`), which is
easier to browse when a run covers many assets or methods. --resume works with it too: a combo
that's already fully scored is skipped outright, and one that stopped partway through picks up
from its last completed case (see `start_from_hierarchical`).
"""

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from baselines.common.config import example_filters
from inspection_eval import (evaluate, frame_gaps, incomplete_targets, metrics, oom_variants,
                            split_by, targets_from_paths)
from inspection_eval.case import ALIGN_MODES
from inspection_eval.report import FORMATS, Table, read_table, read_tables
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
                               "score the cases they are missing, without asking; combined with "
                               "--save-hierarchical, this also skips straight past any "
                               "asset/method combo whose hierarchical folder already has every "
                               "case scored, instead of walking through it to confirm")
    existing.add_argument("--overwrite", action="store_true",
                          help="if the output folder already holds results, delete them and "
                               "evaluate everything again, without asking; not combinable with "
                               "--save-hierarchical")
    p.add_argument("--save-hierarchical", action="store_true",
                   help="write --out/<asset>/<method>/<metric>.<format> instead of one "
                        "table per metric mixing every asset/method combo together, so "
                        "it's easy to find a specific combo's files; --name still picks "
                        "the root (--out/<NAME>/<asset>/<method>/...), and \"_example\" is "
                        "appended to it when --example is set. There's no longer one "
                        "run-labeled, timestamped folder, so instead each file gets the "
                        "run's timestamp in its own name -- unless --no-timestamp, in "
                        "which case a later run just replaces that combo's files. Combine "
                        "with --resume to continue an interrupted hierarchical run: cases "
                        "already scored (in any combo's most recent timestamp) are kept, "
                        "and the run's files keep that same timestamp instead of starting "
                        "a new one. Not combinable with --overwrite")
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
    p.add_argument("--example", action="store_true",
                   help="evaluate only the one sequence pinned in "
                        "baselines/configs/data_example/<asset>.yaml "
                        "(traj/illumination/soil); can't be combined with an explicit "
                        "--traj/--illumination/--soil/--reflectivity")
    for field in ("traj", "illumination", "soil", "reflectivity"):
        p.add_argument(f"--{field}", default=None, help=f"only evaluate variants with this {field}")
    p.add_argument("--list-metrics", action="store_true",
                   help="print the registered metrics with their options and exit")
    p.add_argument("-q", "--quiet", action="store_true", help="only print the tables")
    args = p.parse_args(argv)
    if args.example and any(getattr(args, f) is not None
                            for f in ("traj", "illumination", "soil", "reflectivity")):
        p.error("--example can't be combined with --traj/--illumination/--soil/--reflectivity")
    if args.overwrite and args.save_hierarchical:
        p.error("--overwrite can't be combined with --save-hierarchical")
    return args


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


def run_label(targets, example: bool = False) -> str:
    """A short, readable tag for what `targets` covers, e.g. "old_house-da3" or, for a wildcard
    run over several methods, "old_house-da3+vggt" -- "-example" appended when `--example` picked
    the variant instead of a real sweep, so that run isn't mistaken for a full one later. Used to
    name the output folder so that `eval_results/` stays browsable instead of filling up with bare
    timestamps."""
    assets = sorted({t.dataset.root.name for t in targets})
    # a colmap method's name has a "/" in it (e.g. "colmap/no_gt"); flattened here so it names one
    # folder rather than nesting one, the only place that slash would otherwise reach a filesystem
    methods = sorted({t.method.replace("/", "-") for t in targets})
    label = f"{_join(assets, 'assets')}-{_join(methods, 'methods')}"
    return f"{label}-example" if example else label


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
    return args.out / f"{run_label(targets, args.example)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def hierarchical_root(args) -> Path:
    """--save-hierarchical's output root: --out/<NAME> when named, else --out itself, with
    "_example" appended when --example picked the variant instead of a real sweep -- same reason
    as run_label's "-example", just at the root instead of a run-labeled folder, since hierarchical
    mode has no such folder (see `write_hierarchical`)."""
    root = args.out / args.name if args.name else args.out
    return root.with_name(f"{root.name}_example") if args.example else root


def write_hierarchical(results, root: Path, formats, timestamp: str | None) -> list[Path]:
    """Write `results` under root/<asset>/<method>/<metric>.<format>, one subfolder per
    asset/method combo instead of one table per metric mixing every combo's rows together -- so
    it's obvious at a glance which files belong to which combo, instead of having to open a table
    and filter its Asset/Method columns to find out.

    There's no single run-labeled folder to carry a timestamp the way `out_folder` does, so
    `timestamp` (when given -- omit it for --no-timestamp) goes in each file's own name instead:
    root/<asset>/<method>/<metric>_<timestamp>.<format>. Without it, a later run on the same
    asset/method combo simply replaces that combo's files. A fresh hierarchical run otherwise
    doesn't merge across runs the way --resume does for the flat layout -- --resume is the one
    exception, and it works by reusing the timestamp it resumed from (see `start_from_hierarchical`)
    rather than by merging here, so this function itself never needs to know the difference.
    """
    suffix = f"_{timestamp}" if timestamp else ""
    written = []
    for table in results.tables.values():
        if not len(table):
            continue
        for (asset, method), sub in split_by(table, ("Asset", "Method")).items():
            # same treatment as run_label: a colmap method's name has a "/" in it, which would
            # otherwise nest a nonexistent extra directory level
            out_dir = root / asset / method.replace("/", "-")
            written += sub.write(out_dir, formats, suffix=suffix)
    return written


_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}-\d{6}$")


def _latest_metric_csvs(method_dir: Path) -> dict[str, Path]:
    """The most recent csv for each metric under one asset/method's hierarchical output folder
    (see `write_hierarchical`): grouped by name with any timestamp suffix stripped, keeping a
    bare <metric>.csv (always current -- a --no-timestamp run overwrites it in place rather than
    adding another file) or else the lexicographically-latest timestamped one (they sort
    correctly as text). `check_completion_status.py` reads this same layout the same way, kept
    separate here since it deliberately avoids importing `inspection_eval`."""
    if not method_dir.is_dir():
        return {}
    by_metric: dict[str, list[Path]] = {}
    for path in method_dir.glob("*.csv"):
        by_metric.setdefault(_TIMESTAMP_SUFFIX.sub("", path.stem), []).append(path)

    def latest(paths: list[Path]) -> Path:
        bare = [p for p in paths if not _TIMESTAMP_SUFFIX.search(p.stem)]
        return bare[0] if bare else max(paths, key=lambda p: p.stem)

    return {metric: latest(paths) for metric, paths in by_metric.items()}


def start_from_hierarchical(out_root: Path, targets) -> tuple[dict, str | None]:
    """--resume's counterpart to `start_from`, for --save-hierarchical.

    Every already-scored row found under out_root/<asset>/<method>/ for each of `targets` (the
    combo's most recent timestamp only -- see `_latest_metric_csvs`), merged into one
    {metric: Table}, the shape `evaluate`'s `resume` expects: `case_key` covers Asset/Method too,
    so rows from every combo merge safely into the same table, and a combo whose folder is
    already complete ends up with every one of its cases in `done` -- `evaluate` then skips
    straight past it, exactly as if it had been left out of `targets` -- while a combo that's
    only partway through resumes from whichever case it stopped at.

    Paired with the latest timestamp found across every combo's files, so a combo already on that
    generation has its files updated in place rather than growing a new timestamped copy each
    time --resume is used. A combo left behind on an older generation (say, evaluated in an
    earlier, separate sweep) instead gains a fresh file under the now-current timestamp, next to
    its older one -- same "no merging between generations" rule a bare --save-hierarchical run
    already has, just applied per combo instead of per run. {}, None when nothing on disk matches
    any of `targets` (a fresh hierarchical run).
    """
    tables: dict[str, Table] = {}
    timestamps = set()
    for target in targets:
        method_dir = out_root / target.dataset.root.name / target.method.replace("/", "-")
        for metric, path in _latest_metric_csvs(method_dir).items():
            match = _TIMESTAMP_SUFFIX.search(path.stem)
            if match:
                timestamps.add(match.group()[1:])
            merged = tables.setdefault(metric, Table(metric))
            for row in read_table(path).rows:
                merged.add(row)
    return tables, (max(timestamps) if timestamps else None)


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


def describe_incomplete(incomplete) -> str:
    lines = [f"  {i.dataset.root.name} / {i.method}: {i.done}/{i.total} variant(s) have a cloud"
            for i in incomplete]
    return (f"{len(incomplete)} asset/method combo(s) aren't complete yet, so are skipped rather "
           "than evaluated partially (pass --example to evaluate just the pinned single-sequence "
           "case instead):\n" + "\n".join(lines))


def describe_oom(oom) -> str:
    pairs = sorted({(o.dataset.root.name, o.method) for o in oom})
    lines = [f"  {asset} / {method}" for asset, method in pairs]
    return (f"{len(pairs)} asset/method combo(s) hit a GPU OOM: every variant of each is reported "
           "as OOM below rather than scored:\n" + "\n".join(lines))


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
    filters = example_filters() if args.example else {
        "traj": args.traj, "illumination": args.illumination,
        "soil": args.soil, "reflectivity": args.reflectivity,
    }

    if not args.example:
        incomplete = incomplete_targets(targets)
        if incomplete:
            print(describe_incomplete(incomplete), file=sys.stderr)
            skip = {(i.dataset.root, i.method) for i in incomplete}
            targets = [t for t in targets if (t.dataset.root, t.method) not in skip]
        if not targets:
            raise SystemExit("nothing left to evaluate: every asset/method combo given is "
                             "incomplete (pass --example to evaluate the pinned single-sequence "
                             "case instead, or finish the run for a full one)")

    oom = oom_variants(targets, filters)
    if oom:
        print(describe_oom(oom), file=sys.stderr)

    formats = [f.strip() for f in args.format.split(",")]

    if args.save_hierarchical:
        out_root = hierarchical_root(args)
        if args.resume:
            resume, prev_timestamp = start_from_hierarchical(out_root, targets)
        else:
            resume, prev_timestamp = {}, None  # a fresh hierarchical run stands on its own
        timestamp = None if args.no_timestamp else prev_timestamp or \
            datetime.now().strftime("%Y%m%d-%H%M%S")
        flush = None if args.no_stream else \
            lambda partial: write_hierarchical(partial, out_root, formats, timestamp)
    else:
        out_dir = out_folder(args, targets)
        resume = start_from(out_dir, args)  # asked before anything long runs
        flush = None if args.no_stream else lambda partial: partial.write(out_dir, formats)

    if not args.allow_missing_frames:
        gaps = frame_gaps(targets, filters)
        if gaps:
            raise SystemExit(describe_gaps(gaps))

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

    written = write_hierarchical(results, out_root, formats, timestamp) if args.save_hierarchical \
        else results.write(out_dir, formats)
    print("\n" + "\n".join(f"wrote {p}" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
