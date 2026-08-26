"""Turning `<asset>/predictions/<method>` folders into results tables.

The unit of work is one (asset, method, variant): a prediction folder holds every parameter
combination a method was run on, and each is scored separately, one row per variant per metric.
A metric that cannot run on a case (no fused cloud, no gt depth, ...) is skipped for that case
alone and reported at the end, so a partially-populated prediction folder still yields a table.

The exception is a GPU OOM (see `baselines.common.runner`): once any variant of an (asset, method)
has hit one, every variant of that pair is reported as OOM instead of being scored -- literally,
every metric-value cell of its row reads "OOM" rather than a number -- on the view that a run known
to have failed partway through an asset shouldn't have the variants it did finish read as the whole
picture. See `oom_variants`.
"""

import contextlib
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

from inspection_viz.config import Layout
from inspection_viz.dataset import COLMAP_PREFIX, AssetDataset, Variant

from . import metrics as metrics_pkg
from .case import EvalCase
from .report import Table

PREDICTIONS_DIR = "predictions"
COLMAP_DIR = "colmap"  # <asset>/colmap/<mode> stands for a method the same way predictions/<method> does


@dataclass
class Target:
    """One method's predictions inside one asset."""

    dataset: AssetDataset
    method: str


@dataclass
class Skip:
    case: EvalCase
    metric: str
    reason: str


@dataclass
class FrameGap:
    """An rgb frame of `source` that has no matching depth or cam file, in a variant that
    otherwise has that field. `EvalCase` already tolerates this — it just drops the frame from
    whichever of `depth_frames`/`cam_frames` needed it — so a gap is not fatal by itself, only
    worth surfacing before a run silently scores less than the whole sequence."""

    dataset: AssetDataset
    source: str  # "gt", or a method name
    variant: Variant
    field: str  # "depth" or "cam"
    indices: list  # list[int], the rgb frames missing that field


@dataclass
class OomVariant:
    """A variant of `method` that hit a GPU OOM while being predicted (see
    `baselines.common.runner`) and so has no output to score -- unlike a variant that was simply
    never run, there's a marker saying why."""

    dataset: AssetDataset
    method: str
    variant: Variant


@dataclass
class Incomplete:
    """One (asset, method) that hasn't got a prediction for every variant its asset has -- the
    "COMPLETE" a run needs to reach before `evaluate` will score it by default (see
    `incomplete_targets`)."""

    dataset: AssetDataset
    method: str
    done: int   # variants with a prediction
    total: int  # every variant the asset has


@dataclass
class Results:
    tables: dict  # metric name -> Table
    skips: list  # list[Skip]

    def write(self, out_dir: Path, formats=("csv", "md")) -> list:
        return [p for t in self.tables.values() if len(t) for p in t.write(out_dir, formats)]


def _normpath(path: str | Path) -> Path:
    """Absolute and `.`/`..`-collapsed, like `Path.resolve()` -- but unlike it, symlinks are left
    alone. `predictions/` is sometimes symlinked out to separate storage (a bigger disk, say), and
    `target_from_path` finds the asset root by going `.parent.parent` from a method folder; fully
    resolving first would walk through that symlink and land two levels up *inside the other
    storage*, not at the asset root the symlink itself lives under."""
    return Path(os.path.abspath(Path(path).expanduser()))


def target_from_path(path: str | Path, layout: Layout = Layout()) -> Target:
    """`.../old_house/predictions/da3` -> the asset dataset plus the method name "da3".
    `.../old_house/colmap/no_gt` -> the same, with method name "colmap/no_gt"."""
    path = _normpath(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    if path.parent.name == PREDICTIONS_DIR:
        return Target(AssetDataset(path.parent.parent, layout), path.name)
    if path.parent.name == COLMAP_DIR:
        return Target(AssetDataset(path.parent.parent, layout), f"{COLMAP_PREFIX}{path.name}")
    raise ValueError(
        f"{path} does not look like a prediction folder: expected "
        f"<asset>/{PREDICTIONS_DIR}/<method> or <asset>/{COLMAP_DIR}/<mode>")


def targets_from_paths(paths, layout: Layout = Layout()) -> list[Target]:
    """Every method named by `paths`. A path may be one method's folder, the `predictions` or
    `colmap` dir, or the asset root above them -- the latter two stand for every method/mode
    found inside."""
    out, seen = [], set()
    for raw in paths:
        path = _normpath(raw)
        for method_dir in _expand(path):
            target = target_from_path(method_dir, layout)
            key = (target.dataset.root, target.method)
            if key not in seen:
                seen.add(key)
                out.append(target)
    return out


def _expand(path: Path) -> list[Path]:
    if path.name in (PREDICTIONS_DIR, COLMAP_DIR):
        found = sorted(d for d in path.iterdir() if d.is_dir())
        if not found:
            raise FileNotFoundError(f"no methods under {path}")
        return found
    groups = [path / name for name in (PREDICTIONS_DIR, COLMAP_DIR) if (path / name).is_dir()]
    if not groups:
        return [path]
    found = sorted(d for g in groups for d in g.iterdir() if d.is_dir())
    if not found:
        raise FileNotFoundError(f"no methods under {', '.join(str(g) for g in groups)}")
    return found


def variants(target: Target, **filters) -> list[Variant]:
    """Variants this method predicted, keeping only those matching the given field filters
    (e.g. `traj="trendspek"`); a filter of None matches everything."""
    out = target.dataset.prediction_variants(target.method)
    for field, wanted in filters.items():
        if wanted is not None:
            out = [v for v in out if getattr(v, field) == wanted]
    return out


def frame_gaps(targets, filters: dict = None) -> list[FrameGap]:
    """Every rgb frame across `targets` that is missing depth or cam data it ought to have.

    "Ought to have" means: some other frame of that same (source, variant) does have the field,
    so the field is one the source provides, and this frame is a hole in it rather than data the
    source never captures at all — an asset with no gt depth is normal, an asset missing frame
    x's depth is a gap. gt is checked once per (asset, variant) even though it is shared by every
    method's Target, so a gap in it is not reported once per method.
    """
    out = []
    seen_gt = set()
    for target in targets:
        for variant in variants(target, **(filters or {})):
            sources = [(target.method, target.dataset.sequence(variant, target.method))]
            gt_key = (target.dataset.root, variant)
            if gt_key not in seen_gt:
                seen_gt.add(gt_key)
                sources.append(("gt", target.dataset.sequence(variant, "gt")))
            for source, seq in sources:
                for field, path_of in (("depth", lambda f: f.depth_path),
                                       ("cam", lambda f: f.cam_path)):
                    if not any(path_of(f) is not None for f in seq):
                        continue  # this source doesn't provide the field at all; not a gap
                    missing = [f.index for f in seq if path_of(f) is None]
                    if missing:
                        out.append(FrameGap(target.dataset, source, variant, field, missing))
    return out


def _matches(variant: Variant, filters: dict) -> bool:
    """Does `variant` satisfy every field `filters` sets? (`None` = no constraint on that field.)"""
    return not any(wanted is not None and getattr(variant, field) != wanted
                  for field, wanted in (filters or {}).items())


def oom_variants(targets, filters: dict = None) -> list[OomVariant]:
    """Variants of `targets` that hit a GPU OOM during prediction and so have no output to score,
    among the ones `filters` would otherwise select (same `{field: wanted}` shape as `variants`)."""
    out = []
    for target in targets:
        predicted = set(target.dataset.prediction_variants(target.method))
        for variant in target.dataset.variants:
            if variant in predicted or not _matches(variant, filters):
                continue
            if target.dataset.oom_path(variant, target.method) is not None:
                out.append(OomVariant(target.dataset, target.method, variant))
    return out


def incomplete_targets(targets) -> list[Incomplete]:
    """`targets` missing a prediction (depth/cam, a fused cloud, or a colmap reconstruction -- see
    `AssetDataset.prediction_variants`) for some variant of their asset, with no GPU OOM marker to
    explain why -- always judged against every variant the asset has, not whatever a filter narrows
    a run to, since "complete" describes the prediction folder on disk, not one run's slice of it.
    Deliberately the same notion of "predicted" `variants`/`evaluate` already score by -- not
    `cloud_path`, since a fused cloud is an optional output (`output.cloud: false` is valid), not
    proof a variant was run.

    A target that's incomplete *because* of an OOM is not included here: `oom_variants`/`evaluate`
    already report those (as an "OOM" row, not silence), rather than dropping them."""
    out = []
    for target in targets:
        total = len(target.dataset.variants)
        predicted = set(target.dataset.prediction_variants(target.method))
        missing = [v for v in target.dataset.variants if v not in predicted]
        if not missing:
            continue
        if any(target.dataset.oom_path(v, target.method) is not None for v in missing):
            continue
        out.append(Incomplete(target.dataset, target.method, total - len(missing), total))
    return out


LABEL_COLUMNS = ("Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity")


def case_key(row: dict) -> tuple:
    """What identifies a case across runs: its label columns, as text. Works on an `EvalCase`'s
    `label` and on a row read back from a `.csv`, so a resumed run can tell what it already has —
    a column the case does not use is absent in one and blank in the other, and both already
    read "" via `.get`'s own default. Deliberately *not* `row.get(column, "") or ""`: a resumed
    numeric label -- Reflectivity 0.0, say -- comes back from `report._typed` as the *falsy*
    float 0.0, and `or ""` would collapse that into the same key as a case that never had the
    column at all, breaking --resume's dedup for reflectivity 0.0 specifically."""
    return tuple(str(row.get(column, "")) for column in LABEL_COLUMNS)


@dataclass
class _CaseResult:
    """One case's outcome, safe to send back across a process boundary: no `EvalCase`, `Table`,
    or exception object, only plain data -- what `_score_case` returns and `evaluate`'s loop
    would otherwise have computed inline. `rows` holds a metric's returned dict for every metric
    that succeeded; `skips` holds (metric name, "skip"|"error", reason) for the rest, the same
    two outcomes the loop already distinguishes via different `say(...)` events."""

    rows: dict    # metric name -> row
    skips: list   # list[(metric name, "skip" | "error", reason)]


def _score_case(target: Target, variant: Variant, align_mode: str, metric_names: list,
                options: dict) -> _CaseResult:
    """Score one case against every metric in `metric_names`, building its own `EvalCase` from
    `(target, variant)`. This is the only thing allowed to touch an `EvalCase`'s cached
    properties, so it must never be handed one that's already touched them -- a colmap case's
    frames wrap `pycolmap.Image` objects once loaded, which are not picklable, so an EvalCase
    that crossed a process boundary already populated would blow up on the way back.

    Called directly, in-process, for the serial (`workers <= 1`) path, and via `_pool_worker`
    inside each worker process for the parallel path -- one function, so the two paths can never
    disagree about what a case's outcome is."""
    case = EvalCase(target.dataset, target.method, variant, align_mode)
    rows, skips = {}, []
    for name in metric_names:
        metric = metrics_pkg.get(name)
        missing = case.missing(metric.needs)
        if missing:
            skips.append((name, "skip", f"no {', '.join(missing)}"))
            continue
        try:
            rows[name] = metric.fn(case, options[name])
        except Exception as exc:  # one bad variant must not sink the whole run
            skips.append((name, "error", f"{type(exc).__name__}: {exc}"))
    return _CaseResult(rows, skips)


_pool_context: dict = {}  # populated by _pool_init, in each worker process's own memory only


def _pool_init(metric_names: list, options: dict, align_mode: str) -> None:
    """`Pool(initializer=...)`: runs once per worker process rather than once per task, so the
    run-wide constants (metric names/options/align mode) are pickled once at pool startup instead
    of with every case. Also caps OpenCV's own thread pool -- `geometry.nearest_distances`
    already asks scipy for every thread it can get (`workers=-1`); N worker *processes* each
    doing that too would oversubscribe the CPU further, and `cv2.setNumThreads` (unlike an env
    var, which BLAS/OpenMP only read at library-load time) still takes effect after a fork."""
    import cv2
    cv2.setNumThreads(1)
    _pool_context.update(metric_names=metric_names, options=options, align_mode=align_mode)


def _pool_worker(task: tuple) -> _CaseResult:
    """`Pool.imap`'s per-task entry point: unpacks the one thing that varies per case,
    `(target, variant)`, and delegates to `_score_case` with the constants `_pool_init` stashed."""
    target, variant = task
    ctx = _pool_context
    return _score_case(target, variant, ctx["align_mode"], ctx["metric_names"], ctx["options"])


def evaluate(targets, metric_names=None, align_mode: str = "sim3", overrides: dict = None,
             filters: dict = None, on_event=None, after_case=None, resume: dict = None,
             workers: int = 1) -> Results:
    """Score every variant of every target with every metric.

    `overrides` maps a metric name to `{option: "value as text"}` (the CLI's `--set`).
    `on_event(kind, message, **info)` is called with progress ("plan", "case", "metric", "skip",
    "error", "done"). Every case is enumerated before any scoring starts, so "plan" carries the
    `total` and each "case" carries `index`/`total` — enough to draw a progress bar.
    `after_case(results)` is handed the results so far after each case, so a caller can persist
    them as the run goes instead of only at the end.
    `resume` is a previous run's `{metric: Table}` (see `report.read_tables`): those rows are
    carried into this run's tables and the cases they cover are not scored again. Rows are only
    written once a case has been through every metric, so a case present in any table is one that
    finished — which makes "everything up to the last completed case" exactly what is skipped.

    A (asset, method) that hit a GPU OOM on any variant is not scored at all: every variant it
    would otherwise cover gets a row in every metric's table with "OOM" in place of every value
    that metric would otherwise have reported, since a baseline that OOM'd partway through an
    asset shouldn't be read as if the variants it did finish are the whole picture.

    `workers` scores that many cases at once, each in its own process (default 1: sequential,
    in-process, unchanged from before this parameter existed) -- an already-resumed or OOM case
    is never dispatched to a worker (see `to_score` below), so a run that is entirely resumed or
    entirely OOM never touches a pool at all. Opt-in only: this is never auto-detected from
    `os.cpu_count()`, since a run commonly shares a box with other people's jobs.
    """
    names = list(metric_names or metrics_pkg.REGISTRY)
    chosen = [metrics_pkg.get(n) for n in names]
    options = {m.name: metrics_pkg.apply_overrides(m.default_options(),
                                                   (overrides or {}).get(m.name, {}))
               for m in chosen}
    tables = {m.name: (resume or {}).get(m.name) or Table(m.name) for m in chosen}
    done = {case_key(row) for table in tables.values() for row in table.rows}
    skips = []
    results = Results(tables, skips)
    say = on_event or (lambda kind, message, **info: None)

    oom_targets = {(o.dataset.root, o.method) for o in oom_variants(targets, filters)}
    oom_keys = set()
    plan = []
    for t in targets:
        if (t.dataset.root, t.method) in oom_targets:
            # the whole asset/method is reported as OOM, so every one of its variants belongs in
            # the plan -- not just the ones that happened to get real output before the OOM hit.
            vs = [v for v in t.dataset.variants if _matches(v, filters)]
        else:
            vs = variants(t, **(filters or {}))
        plan.extend((t, v) for v in vs)
    total = len(plan)
    say("plan", f"{total} case(s) x {len(chosen)} metric(s)"
                + (f", {len(done)} already scored" if done else ""),
        total=total, metrics=names, done=len(done))

    # Only a case that is neither already-done (resumed) nor OOM ever needs a metric actually run
    # on it -- those two are handled inline below with zero dispatch, exactly as before `workers`
    # existed. Precomputing `to_score` up front lets the serial and parallel paths share one
    # definition of what gets scored, and lets `imap` line results up with `plan`'s order.
    to_score = [(t, v) for t, v in plan
               if (t.dataset.root, t.method) not in oom_targets
               and case_key(EvalCase(t.dataset, t.method, v, align_mode).label) not in done]

    if workers > 1 and to_score:
        # fork explicitly, not whatever the platform default is: workers must inherit
        # metrics_pkg.REGISTRY exactly as it stands right now, including anything registered at
        # runtime -- spawn would re-import the metrics package fresh and could lose that.
        pool = multiprocessing.get_context("fork").Pool(
            workers, initializer=_pool_init, initargs=(names, options, align_mode))
        results_iter = pool.imap(_pool_worker, to_score)  # ordered: matches plan's own order
    else:
        pool = contextlib.nullcontext()
        results_iter = (_score_case(t, v, align_mode, names, options) for t, v in to_score)

    with pool:
        for index, (target, variant) in enumerate(plan, 1):
            case = EvalCase(target.dataset, target.method, variant, align_mode)
            is_oom = (target.dataset.root, target.method) in oom_targets
            say("case", f"{target.dataset.root.name} / {target.method} / {variant}"
                        + (" [OOM]" if is_oom else ""), index=index, total=total)
            if case_key(case.label) in done:
                say("resumed", "  already scored, kept from the previous run")
                continue
            if is_oom:
                oom_keys.add(case_key(case.label))
                for metric in chosen:
                    # A metric that declares `columns` (see `metrics.base`) tells us its row shape
                    # up front, from options alone -- so the row is "OOM" in full immediately, even
                    # if every case in this run turns out to be OOM and `fn` is never called at all.
                    # One without it falls back on the backfill below, same as before.
                    cols = metric.columns(options[metric.name]) if metric.columns else ()
                    tables[metric.name].add({**case.label, **{c: "OOM" for c in cols}})
                say("oom", "  OOM: this method hit a GPU OOM on this asset, nothing was scored")
            else:
                result = next(results_iter)
                skip_by_name = {name: (kind, reason) for name, kind, reason in result.skips}
                for metric in chosen:
                    if metric.name in result.rows:
                        tables[metric.name].add({**case.label, **result.rows[metric.name]})
                        say("metric", f"  {metric.name}: ok")
                    else:
                        kind, reason = skip_by_name[metric.name]
                        skips.append(Skip(case, metric.name, reason))
                        verb = "skipped" if kind == "skip" else "failed"
                        say(kind, f"  {metric.name}: {verb} ({reason})")
            # A metric without declared `columns` gets its OOM row's non-label cells filled in here
            # instead of up front, once some other case has taught the table about columns it didn't
            # have before -- so a number and an OOM are never visually confusable in the same column.
            # (A metric that does declare `columns` already got this in full above; this is a no-op
            # for it.) Redone after every case (not just once at the end) so a streamed or
            # interrupted run's on-disk table is never left showing a blank cell where "OOM" belongs.
            for table in tables.values():
                for row in table.rows:
                    if case_key(row) in oom_keys:
                        for column in table.columns:
                            if column not in LABEL_COLUMNS:
                                row[column] = "OOM"
            if after_case:
                after_case(results)
        say("done", f"{total} case(s) scored", index=total, total=total)

    return results
