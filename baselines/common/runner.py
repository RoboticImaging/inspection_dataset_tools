"""Driving a baseline over every sequence a data config selects.

`run_baseline` owns everything that is the same whichever baseline is running: resolving the
data config against the asset on disk, skipping sequences that already have predictions, loading
the model once for the whole run, and recording what the run was configured with under
`predictions/<method>/config.yaml`. A baseline supplies only `setup()` (load the model) and
`run()` (predict one sequence and write it through the `SequenceOutput` it is handed).

Prediction files themselves are written by `baselines.common.writer`, in the same formats the
ground truth uses — see that module's docstring for the camera-json convention.
"""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from inspection_viz.config import Layout, fill_template
from inspection_viz.dataset import AssetDataset, Frame, Variant

from .config import OutputSpec, RunConfig, add_config_args, config_from_args
from .writer import cloud_path, frame_stem, output_dirs, write_cam, write_cloud, write_depth


def variant_label(variant: Variant) -> str:
    """A compact one-line name for a variant, e.g. "traj=trendspek,illumination=sunny,soil=low"."""
    return ",".join(f"{k}={v}" for k, v in variant.as_dict().items())


@dataclasses.dataclass
class SequenceOutput:
    """Where one sequence's predictions go, and how they are written.

    Handed to `Baseline.run` so a baseline never has to know the folder layout: it hands back
    per-frame depth + camera (and a fused cloud if it makes one) and this puts them where the
    viewer and the evaluator look for them.
    """

    root: Path
    layout: Layout
    method: str
    variant: Variant
    spec: OutputSpec

    def __post_init__(self):
        binds = self.variant.as_dict()
        self.depth_dir, self.cam_dir = output_dirs(self.root, self.layout, self.method, binds)
        self.cloud_path = cloud_path(self.root, self.layout, self.method, binds)

    @property
    def label(self) -> str:
        return variant_label(self.variant)

    def exists(self) -> bool:
        """Has a previous run already written this sequence? Any depth file counts: a run that
        died halfway still leaves a dir behind, and re-running it is the point of `overwrite`."""
        return self.depth_dir.is_dir() and any(self.depth_dir.iterdir())

    def scratch_dir(self, name: str) -> Path:
        """A working dir beside the fused cloud, for a model's own exporter to write into."""
        return self.cloud_path.parent / f"{self.cloud_path.stem}_{name}"

    def write_frame(self, index: int, depth: np.ndarray, c2w: np.ndarray, K: np.ndarray,
                    intrinsics_width: int, conf: np.ndarray | None = None) -> None:
        """One frame: planar z-depth in meters at the rgb resolution, plus its camera.

        `intrinsics_width` is the width `K` was measured against (not necessarily the width of
        `depth`). `conf` is the model's confidence map, same size as `depth`, needed only when
        `output.min_confidence` is set — pixels below it are written as nan.
        """
        depth = np.asarray(depth, dtype=np.float32)
        if self.spec.min_confidence is not None:
            if conf is None:
                raise SystemExit("output.min_confidence is set but this baseline produced no "
                                 "confidence map")
            depth = np.where(np.asarray(conf) >= self.spec.min_confidence, depth, np.nan) \
                      .astype(np.float32)
        stem = frame_stem(index)
        write_depth(self.depth_dir / f"{stem}.exr", depth)
        write_cam(self.cam_dir / f"{stem}.json", c2w, K, width=intrinsics_width)

    def write_cloud(self, points: np.ndarray, colors: np.ndarray | None = None) -> None:
        """The one fused world-space cloud for this sequence."""
        write_cloud(self.cloud_path, points, colors)


class Baseline:
    """One baseline: a model plus the glue that turns a sequence of frames into predictions.

    Subclass this in `baselines/<name>/run_sequence.py`, set `name` and `Config` (the dataclass
    of the baseline's own knobs, loaded from the `model:` section of the baseline config), and
    add one line to `baselines/registry.py` so `baselines/run.py` can find it.
    """

    name: str = ""
    Config: type = None

    def __init__(self, config):
        self.config = config

    def setup(self) -> None:
        """Load the model. Called once per run, before any sequence, and only if there is at
        least one sequence to run — so a fully-skipped run never touches the GPU."""

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        """Predict one sequence and write it through `out`."""
        raise NotImplementedError


def run_baseline(baseline: Baseline, cfg: RunConfig, layout: Layout = Layout(),
                 say=print) -> list[Variant]:
    """Run `baseline` over every sequence `cfg.data` selects. Returns the variants it wrote.

    Sequences that already have output are skipped unless `output.overwrite` is set, so a run
    interrupted halfway (or extended to more variants later) can simply be started again.
    """
    ds = AssetDataset(cfg.data.asset, layout)
    method = cfg.output.method or baseline.name
    if not method:
        raise SystemExit("no output.method, and this baseline has no registered name")

    variants = cfg.data.variants(ds)
    if not variants:
        raise SystemExit(f"no sequences of {ds.root} match this data config; it has:\n"
                         + "\n".join(f"  {variant_label(v)}" for v in ds.variants))

    todo, skipped = [], []
    for variant in variants:
        out = SequenceOutput(ds.root, layout, method, variant, cfg.output)
        (todo if cfg.output.overwrite or not out.exists() else skipped).append(out)
    for out in skipped:
        say(f"skipping {out.label}: already has output (set output.overwrite: true to redo)")
    if not todo:
        raise SystemExit(f"nothing to run: all {len(variants)} selected sequence(s) already have "
                         f"predictions under predictions/{method}/")

    say(f"{len(todo)} sequence(s) to run for method {method!r} in {ds.root}")
    baseline.setup()

    done = []
    for i, out in enumerate(todo, start=1):
        frames = cfg.data.select(ds.sequence(out.variant))
        if not frames:
            say(f"[{i}/{len(todo)}] {out.label}: no frames left after the frame range/stride, "
                "skipping")
            continue
        say(f"[{i}/{len(todo)}] {out.label}: {len(frames)} frames")
        baseline.run(frames, out)
        done.append(out.variant)
        say(f"  wrote frames {frames[0].index}..{frames[-1].index}")

    if done:
        path = write_run_config(ds.root, layout, method, baseline, cfg, done,
                                kept=[o.variant for o in skipped])
        say(f"done: {len(done)} sequence(s) -> predictions/{method}/ under {ds.root}")
        say(f"  run config: {path.relative_to(ds.root)}")
    return done


def write_run_config(root: Path, layout: Layout, method: str, baseline: Baseline,
                     cfg: RunConfig, variants: list[Variant], kept: list[Variant] = ()) -> Path:
    """Record what produced `predictions/<method>/`, next to the predictions themselves.

    One file per method, rewritten by each run: it describes the predictions currently on disk,
    not a history of how they got there. `kept` are the variants this run skipped because they
    already had output — they are still in the folder, but were written by an earlier run whose
    config this file has now replaced, so keep separate `output.method` names for configurations
    you want to tell apart later.
    """
    data = {k: v for k, v in dataclasses.asdict(cfg.data).items() if k != "root"}
    payload = {
        "baseline": baseline.name,
        "method": method,
        "ran_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data": data,
        "variants": [v.as_dict() for v in variants],
        "model": dataclasses.asdict(cfg.model),
        "output": {**dataclasses.asdict(cfg.output), "method": method},
    }
    if kept:
        payload["variants_kept"] = [v.as_dict() for v in kept]  # left as an earlier run wrote them
    path = root / fill_template(layout.pred_config_file, method=method)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def run_from_args(baseline_cls: type, args) -> list[Variant]:
    """Run `baseline_cls` from parsed `add_config_args` arguments — the tail end of both
    entry points (a baseline's own `run_sequence.py` and `baselines/run.py`)."""
    cfg = config_from_args(args, baseline_cls.Config)
    return run_baseline(baseline_cls(cfg.model), cfg)


def cli_main(baseline_cls: type, argv=None, description: str = None) -> list[Variant]:
    """The `main()` of a baseline's own `run_sequence.py`: parse configs, run, write."""
    p = argparse.ArgumentParser(description=description or baseline_cls.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    return run_from_args(baseline_cls, p.parse_args(argv))
