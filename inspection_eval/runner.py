"""Turning `<asset>/predictions/<method>` folders into results tables.

The unit of work is one (asset, method, variant): a prediction folder holds every parameter
combination a method was run on, and each is scored separately, one row per variant per metric.
A metric that cannot run on a case (no fused cloud, no gt depth, ...) is skipped for that case
alone and reported at the end, so a partially-populated prediction folder still yields a table.

The exception is a GPU OOM (see `baselines.common.runner`): once any variant of an (asset, method)
has hit one, every variant of that pair is reported as a flat "OOM" row instead of being scored,
on the view that a run known to have failed partway through an asset shouldn't have the variants
it did finish read as the whole picture. See `oom_variants`.
"""

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
class Results:
    tables: dict  # metric name -> Table
    skips: list  # list[Skip]

    def write(self, out_dir: Path, formats=("csv", "md")) -> list:
        return [p for t in self.tables.values() if len(t) for p in t.write(out_dir, formats)]


def target_from_path(path: str | Path, layout: Layout = Layout()) -> Target:
    """`.../old_house/predictions/da3` -> the asset dataset plus the method name "da3".
    `.../old_house/colmap/no_gt` -> the same, with method name "colmap/no_gt"."""
    path = Path(path).expanduser().resolve()
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
        path = Path(raw).expanduser().resolve()
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


LABEL_COLUMNS = ("Asset", "Method", "Traj", "Illumination", "Soil", "Reflectivity")


def case_key(row: dict) -> tuple:
    """What identifies a case across runs: its label columns, as text. Works on an `EvalCase`'s
    `label` and on a row read back from a `.csv`, so a resumed run can tell what it already has —
    a column the case does not use is absent in one and blank in the other, and both read ""."""
    return tuple(str(row.get(column, "") or "") for column in LABEL_COLUMNS)


def evaluate(targets, metric_names=None, align_mode: str = "sim3", overrides: dict = None,
             filters: dict = None, on_event=None, after_case=None, resume: dict = None) -> Results:
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
    would otherwise cover is written as a single "OOM" row instead (in every metric's table alike),
    since a baseline that OOM'd partway through an asset shouldn't be read as if the variants it
    did finish are the whole picture.
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

    for index, (target, variant) in enumerate(plan, 1):
        case = EvalCase(target.dataset, target.method, variant, align_mode)
        is_oom = (target.dataset.root, target.method) in oom_targets
        say("case", f"{target.dataset.root.name} / {target.method} / {variant}"
                    + (" [OOM]" if is_oom else ""), index=index, total=total)
        if case_key(case.label) in done:
            say("resumed", "  already scored, kept from the previous run")
            continue
        if is_oom:
            for metric in chosen:
                tables[metric.name].add({**case.label, "OOM": "OOM"})
            say("oom", "  OOM: this method hit a GPU OOM on this asset, nothing was scored")
        else:
            for metric in chosen:
                missing = case.missing(metric.needs)
                if missing:
                    skips.append(Skip(case, metric.name, f"no {', '.join(missing)}"))
                    say("skip", f"  {metric.name}: skipped (no {', '.join(missing)})")
                    continue
                try:
                    row = metric.fn(case, options[metric.name])
                except Exception as exc:  # one bad variant must not sink the whole run
                    skips.append(Skip(case, metric.name, f"{type(exc).__name__}: {exc}"))
                    say("error", f"  {metric.name}: failed ({type(exc).__name__}: {exc})")
                    continue
                tables[metric.name].add({**case.label, **row})
                say("metric", f"  {metric.name}: ok")
        if after_case:
            after_case(results)
    say("done", f"{total} case(s) scored", index=total, total=total)

    return results
