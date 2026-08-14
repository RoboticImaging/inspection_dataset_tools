"""Turning `<asset>/predictions/<method>` folders into results tables.

The unit of work is one (asset, method, variant): a prediction folder holds every parameter
combination a method was run on, and each is scored separately, one row per variant per metric.
A metric that cannot run on a case (no fused cloud, no gt depth, ...) is skipped for that case
alone and reported at the end, so a partially-populated prediction folder still yields a table.
"""

from dataclasses import dataclass
from pathlib import Path

from inspection_viz.config import Layout
from inspection_viz.dataset import AssetDataset, Variant

from . import metrics as metrics_pkg
from .case import EvalCase
from .report import Table

PREDICTIONS_DIR = "predictions"


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
class Results:
    tables: dict  # metric name -> Table
    skips: list  # list[Skip]

    def write(self, out_dir: Path, formats=("csv", "md")) -> list:
        return [p for t in self.tables.values() if len(t) for p in t.write(out_dir, formats)]


def target_from_path(path: str | Path, layout: Layout = Layout()) -> Target:
    """`.../old_house/predictions/da3` -> the asset dataset plus the method name "da3"."""
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if path.parent.name != PREDICTIONS_DIR:
        raise ValueError(
            f"{path} does not look like a prediction folder: expected "
            f"<asset>/{PREDICTIONS_DIR}/<method>")
    return Target(AssetDataset(path.parent.parent, layout), path.name)


def targets_from_paths(paths, layout: Layout = Layout()) -> list[Target]:
    """Every method named by `paths`. A path may be one method's folder, or the `predictions`
    dir / the asset root above it, which stand for every method inside."""
    out, seen = [], set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.name == PREDICTIONS_DIR or (path / PREDICTIONS_DIR).is_dir():
            root = path if path.name == PREDICTIONS_DIR else path / PREDICTIONS_DIR
            found = sorted(d for d in root.iterdir() if d.is_dir())
            if not found:
                raise FileNotFoundError(f"no methods under {root}")
        else:
            found = [path]
        for method_dir in found:
            target = target_from_path(method_dir, layout)
            key = (target.dataset.root, target.method)
            if key not in seen:
                seen.add(key)
                out.append(target)
    return out


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


def evaluate(targets, metric_names=None, align_mode: str = "sim3", overrides: dict = None,
             filters: dict = None, on_event=None) -> Results:
    """Score every variant of every target with every metric.

    `overrides` maps a metric name to `{option: "value as text"}` (the CLI's `--set`).
    `on_event(kind, message)` is called with progress ("case", "metric", "skip", "error").
    """
    names = list(metric_names or metrics_pkg.REGISTRY)
    chosen = [metrics_pkg.get(n) for n in names]
    options = {m.name: metrics_pkg.apply_overrides(m.default_options(),
                                                   (overrides or {}).get(m.name, {}))
               for m in chosen}
    tables = {m.name: Table(m.name) for m in chosen}
    skips = []
    say = on_event or (lambda kind, message: None)

    for target in targets:
        for variant in variants(target, **(filters or {})):
            case = EvalCase(target.dataset, target.method, variant, align_mode)
            say("case", f"{target.dataset.root.name} / {target.method} / {variant}")
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

    return Results(tables, skips)
