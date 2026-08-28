"""Yaml -> dataclass config loading, shared by every baseline runner.

A run is described by two halves, each in its own file so each can be reused independently:

  data config     — which asset, which of its sequences, which frames (`DataConfig`). Named after
                    the asset, not the baseline: the same file drives every baseline.
  baseline config — the baseline's own knobs (its `ModelConfig` dataclass under `model:`) plus
                    the `output:` toggles (`OutputSpec`). Named after the baseline, not the
                    asset: the same file drives every asset.

Both halves may also live in a single file, under `dataset:` / `model:` / `output:` keys — the
shape every config had before the split. `load_run_config` accepts either, so older configs and
runners that call `load_config(path, Config)` themselves keep working unchanged.

Every knob is a typed, defaulted field in a dataclass; the yaml only overrides what it wants to.

One-off tweaks don't need a new yaml file: every runner's `--set data.trajectory=foo` /
`--set model.ka.strategy=basic` (repeatable, dotted path into data/model/output) patches the
loaded yaml in place before it's turned into dataclasses -- see `add_config_args`.
"""

# Postponed annotation evaluation (PEP 563): lets `str | Path`/`str | list | None` etc. below run
# on Python < 3.10 (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu 20.04's
# default Python 3.8) without needing `typing.Optional`/`typing.Union` -- annotations are stored
# as strings instead of evaluated at class-definition time, so `type.__or__` (added in 3.10) is
# never actually called.
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

from inspection_viz.dataset import AssetDataset, Sequence, Variant

# Data-config field -> the `Variant` field it filters. Both names are kept: "trajectory" reads
# better in a config, "traj" is what the dir templates (and so `Variant`) call it.
AXES = {"trajectory": "traj", "illumination": "illumination",
        "soil": "soil", "reflectivity": "reflectivity"}


def load_config(path: str | Path, cls):
    """Populate a (possibly nested) dataclass `cls` from a yaml file. Unknown keys raise, to
    catch typos rather than silently ignoring them."""
    return _from_dict(cls, _read_yaml(path))


def _read_yaml(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")
    return raw


def _resolve_field_type(cls, field):
    """Best-effort resolution of one dataclass field's annotation back to a real type object.

    With `from __future__ import annotations` active anywhere in this module's chain (needed for
    `str | Path` etc. on Python < 3.10 -- see the top of this file), `field.type` is the
    unevaluated *string* the annotation was written as (e.g. "ImageReaderConfig"), and
    `dataclasses.is_dataclass` on a string is always False -- silently skipping every
    nested-dataclass conversion in `_from_dict` below instead of erroring, so nested config
    sections would stay plain dicts.

    Deliberately NOT `typing.get_type_hints(cls)`: that resolves every field's annotation on the
    class at once, and this repo's config dataclasses have genuine `X | Y` (PEP 604) fields
    (e.g. `DataConfig.trajectory: str | list | None`) that cannot be evaluated on Python < 3.10 at
    all, ever, postponed or not -- `type.__or__` simply doesn't exist there. get_type_hints would
    raise on those before ever reaching the one field actually being asked about here. Resolving
    field-by-field instead means only the *specific* annotation being asked about needs to
    evaluate; failing to evaluate a union field is fine, since a union is never itself a dataclass
    the caller would need to recurse into.
    """
    if isinstance(field.type, type):
        return field.type
    module = sys.modules.get(cls.__module__)
    try:
        return eval(field.type, vars(module) if module else {})
    except TypeError:
        return None


def _from_dict(cls, raw: dict):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, field in fields.items():
        if name not in raw:
            continue
        val = raw[name]
        field_type = _resolve_field_type(cls, field)
        if field_type is not None and dataclasses.is_dataclass(field_type) and isinstance(val, dict):
            val = _from_dict(field_type, val)
        kwargs[name] = val
    return cls(**kwargs)


def _matches(value: str | None, spec) -> bool:
    """Does a variant's value on one axis satisfy the config's spec for that axis?

    None (or an absent key) means "every value this asset has", a list means any of them, and a
    bare scalar means exactly that one. Specs are compared as text, so `reflectivity: 0.0` in
    yaml still matches the `ref=0.0` on disk.
    """
    if spec is None:
        return True
    wanted = [str(s) for s in spec] if isinstance(spec, (list, tuple)) else [str(spec)]
    return value in wanted


@dataclasses.dataclass
class DataConfig:
    """Which sequences of one asset a run covers, and which frames of them.

    Each axis takes a single value (`illumination: sunny`), a list (`[sunny, overcast]`), or
    null — which means every value the asset has on that axis. So one file names one concrete
    sequence, the whole cross product of the asset's variants, or anything in between; what
    "every value" resolves to is discovered from the asset itself (see `variants`).
    """

    asset: str = ""
    root: str = ""  # the pre-`asset` name for the same thing, still accepted
    trajectory: str | list | None = None
    illumination: str | list | None = None
    soil: str | list | None = None
    reflectivity: str | list | None = None
    frame_start: int = 0
    frame_end: int | None = None  # exclusive; None = to the end
    frame_stride: int = 1

    def __post_init__(self):
        if self.asset and self.root and Path(self.asset) != Path(self.root):
            raise ValueError(f"config sets both asset: {self.asset!r} and root: {self.root!r}; "
                             "they name the same thing, so keep only `asset`")
        self.asset = self.asset or self.root
        self.root = self.asset  # both spellings stay live, for runners written against either

    def variant(self) -> Variant:
        """The single variant this selects, for runners that predate multi-sequence runs. An
        axis left null still means "unspecified" here, not "all" — `variants` is what expands
        a null or a list against the asset."""
        vals = [getattr(self, field) for field in AXES]
        if any(isinstance(v, (list, tuple)) for v in vals):
            raise ValueError("this config selects several sequences; use variants(dataset)")
        return Variant(*vals)

    def variants(self, ds: AssetDataset) -> list[Variant]:
        """Every variant of `ds` this config selects, in the dataset's own order."""
        return [v for v in ds.variants
                if all(_matches(getattr(v, vfield), getattr(self, cfield))
                       for cfield, vfield in AXES.items())]

    def select(self, seq: Sequence) -> list:
        """Frames of `seq` chosen by frame_start/frame_end/frame_stride."""
        end = self.frame_end if self.frame_end is not None else len(seq)
        return seq.frames[self.frame_start:end:self.frame_stride]


DatasetSelection = DataConfig  # the pre-split name; `root` and `variant()` still work on it

EXAMPLE_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "data_example"


def example_filters(configs_dir: Path = EXAMPLE_CONFIGS_DIR) -> dict:
    """The traj/illumination/soil/reflectivity filter every baselines/configs/data_example/*.yaml
    agrees on -- the one sequence those configs pin per asset, for tools that want to run or check
    just that quick case instead of a full variant sweep. Raises if the configs don't actually
    agree, since a single filter dict can't express per-asset values."""
    paths = sorted(configs_dir.glob("*.yaml"))
    if not paths:
        raise SystemExit(f"no example configs found under {configs_dir}")
    configs = [load_config(p, DataConfig) for p in paths]
    filters = {vfield: getattr(configs[0], cfield) for cfield, vfield in AXES.items()}
    for path, cfg in zip(paths[1:], configs[1:]):
        for cfield, vfield in AXES.items():
            if getattr(cfg, cfield) != filters[vfield]:
                raise SystemExit(
                    f"{path.name} sets {cfield}: {getattr(cfg, cfield)!r}, but {paths[0].name} "
                    f"sets it to {filters[vfield]!r}; --example needs every example config to "
                    "agree on one filter")
    return filters


@dataclasses.dataclass
class OutputSpec:
    """Where predictions land and how they're filtered before writing."""

    # Subfolder name: predictions/{method}/... (also the viewer's Source label). None = the
    # registered name of the baseline being run, which is what you want unless you are writing
    # two configurations of one baseline side by side (e.g. "da3" and "da3_knownpose").
    method: str | None = None
    overwrite: bool = False
    min_confidence: float | None = None  # pixels below this get depth=nan; None = keep everything
    # A percentile (0-100) of a baseline's own lowest-confidence *cloud* points to drop before
    # fusing -- unrelated to min_confidence above (an absolute per-pixel depth cutoff): this only
    # ever thins the point cloud a baseline's run_sequence.py builds itself, never touches depth,
    # and isn't consumed by write_frame at all. None = keep every point. Split into its own field
    # (rather than overloading min_confidence for both meanings) because the two aren't
    # interchangeable -- a percentile isn't a depth threshold, and mixing them previously meant
    # any baseline using this percentile convention without also constructing a per-pixel conf
    # array crashed on its very first write_frame call (min_confidence set, conf=None).
    cloud_confidence_percentile: float | None = None
    # One fused .ply per variant, unprojected from the predicted depth + poses. Confidence
    # filtering and the point cap are left to whatever post-processing defaults the model's own
    # exporter uses, so the cloud matches what that model ships as its reference output.
    cloud: bool = True
    # Read every depth frame and fused cloud back right after writing it, retrying the write
    # (write_attempts times total) if it doesn't decode -- catches a process kill (the OS
    # OOM-killer, a GPU driver crash, node preemption) or a storage hiccup leaving a truncated or
    # corrupt file behind, which otherwise only surfaces much later, and much more confusingly, as
    # a decode error during eval. Writes are always atomic regardless of this flag (see
    # baselines.common.writer._atomic_write); this only controls the extra read-back-to-confirm
    # step, whose cost -- a small decode per frame, milliseconds -- is negligible next to a
    # frame's model inference time. Set False only if that cost genuinely matters somewhere.
    verify_writes: bool = True
    write_attempts: int = 3


@dataclasses.dataclass
class RunConfig:
    """One run: the data half, the baseline's own knobs, and the output toggles."""

    data: DataConfig
    model: object
    output: OutputSpec


BASELINE_KEYS = ("model", "output")
DATA_KEYS = ("dataset", "data")


def load_run_config(model_cls, config=None, data=None, model=None, overrides=()) -> RunConfig:
    """Load a run config from either shape.

    `config` is one combined file (`dataset:` + `model:` + `output:`); `data` and `model` are the
    split pair. Either of the split files may be given alone, and either overrides the matching
    half of `config` — so a legacy config can be re-pointed at another asset with just `--data`.

    `overrides` is `--set section.path=value` strings (see `add_config_args`/`apply_overrides`),
    applied last -- after the yaml files, so a `--set` always wins over whatever a file loaded.
    """
    raw_data, raw_baseline = {}, {}
    if config is not None:
        raw = _read_yaml(config)
        raw_data = next((raw.pop(k) for k in DATA_KEYS if k in raw), {}) or {}
        raw_baseline = raw
    if data is not None:
        raw_data = _unwrap(_read_yaml(data), DATA_KEYS)
    if model is not None:
        raw_baseline = _read_yaml(model)
    unknown = set(raw_baseline) - set(BASELINE_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in the baseline config: {sorted(unknown)}; "
                         f"expected only {list(BASELINE_KEYS)}"
                         + (" (dataset selection belongs in the data config)"
                            if unknown & set(DATA_KEYS) else ""))
    raw_baseline.setdefault("model", {})
    raw_baseline.setdefault("output", {})
    apply_overrides({"data": raw_data, "model": raw_baseline["model"],
                     "output": raw_baseline["output"]}, overrides)
    return RunConfig(
        data=_from_dict(DataConfig, raw_data),
        model=_from_dict(model_cls, raw_baseline["model"]),
        output=_from_dict(OutputSpec, raw_baseline["output"]),
    )


def _unwrap(raw: dict, keys) -> dict:
    """A data config written flat, or still wrapped in the `dataset:` key it had when it was a
    section of a combined config — splitting one by copy-paste shouldn't have to unindent it."""
    if len(raw) == 1 and isinstance(val := next(iter(raw.values())), dict) \
            and next(iter(raw)) in keys:
        return val
    return raw


def _set_override_path(section: dict, path: list, value, spec: str) -> None:
    """`section[path[0]][path[1]]...[path[-1]] = value`, creating intermediate dicts as needed
    (so `--set model.ka.strategy=basic` works whether or not the yaml even has a `ka:` block)."""
    d = section
    for key in path[:-1]:
        d = d.setdefault(key, {})
        if not isinstance(d, dict):
            raise SystemExit(f"--set {spec}: {key!r} is not a mapping in the loaded config, "
                             "can't override into it")
    d[path[-1]] = value


def apply_overrides(sections: dict, overrides) -> None:
    """Patch `--set section.path.to.field=value` overrides (see `add_config_args`) onto the raw
    `{"data": ..., "model": ..., "output": ...}` dicts in place, before `_from_dict` turns them
    into dataclasses -- so a typo'd field name still surfaces via `_from_dict`'s own "unknown
    config key(s)" check, exactly as if it had been a typo in the yaml file itself, rather than
    silently setting an attribute `_from_dict` never looks at.

    The value is parsed with `yaml.safe_load`, the same as any yaml scalar, so
    `--set model.ka.apply=false` produces the bool `False` (not the string `"false"`), numbers
    parse as numbers, `null` clears an Optional field, and flow-style lists/mappings
    (`--set data.trajectory=[a,b]`) work too.
    """
    for spec in overrides:
        if "=" not in spec:
            raise SystemExit(f"--set {spec!r}: expected KEY=VALUE, e.g. model.ka.strategy=basic")
        key, _, value = spec.partition("=")
        path = key.split(".")
        if path[0] not in sections:
            raise SystemExit(f"--set {spec!r}: unknown section {path[0]!r}, "
                             f"expected one of {sorted(sections)}")
        if len(path) < 2:
            raise SystemExit(f"--set {spec!r}: need a field under {path[0]!r}, "
                             f"e.g. {path[0]}.some_field=...")
        _set_override_path(sections[path[0]], path[1:], yaml.safe_load(value), spec)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """The config arguments every baseline runner takes."""
    parser.add_argument("config", nargs="?",
                        help="a single file holding both halves (dataset:/model:/output:), the "
                             "pre-split shape; --data/--model override its matching half")
    parser.add_argument("--data", metavar="YAML",
                        help="data config: which asset and which of its sequences to run on, "
                             "e.g. ../configs/data/old_house.yaml")
    parser.add_argument("--model", metavar="YAML",
                        help="baseline config: this baseline's own model:/output: knobs, "
                             "e.g. ../configs/da3.yaml (omit to run every default)")
    parser.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                        help="override one field of the loaded config without a new yaml file, "
                             "e.g. --set model.ka.strategy=basic or --set data.trajectory=foo "
                             "(dotted path, first segment must be data/model/output; repeatable; "
                             "value is parsed as yaml, so true/false/null/numbers/[a,b] all work)")


def config_from_args(args, model_cls) -> RunConfig:
    if not (args.config or args.data):
        raise SystemExit("nothing to run: pass --data <data config> (and usually --model "
                         "<baseline config>), or one combined config file")
    return load_run_config(model_cls, args.config, args.data, args.model, args.set)
