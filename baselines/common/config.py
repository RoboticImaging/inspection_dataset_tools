"""Yaml -> dataclass config loading, shared by every baseline runner.

Each baseline's `run_sequence.py` defines its own model dataclass and a top-level `Config`
that nests `DatasetSelection` / `OutputSpec` alongside it, then calls `load_config(path, Config)`.
Every knob is a typed, defaulted field in a dataclass; the yaml only overrides what it wants to.
"""

import dataclasses
from pathlib import Path

import yaml

from inspection_viz.dataset import Variant, Sequence


def load_config(path: str | Path, cls):
    """Populate a (possibly nested) dataclass `cls` from a yaml file. Unknown keys raise, to
    catch typos rather than silently ignoring them."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return _from_dict(cls, raw)


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
        if dataclasses.is_dataclass(field.type) and isinstance(val, dict):
            val = _from_dict(field.type, val)
        kwargs[name] = val
    return cls(**kwargs)


@dataclasses.dataclass
class DatasetSelection:
    """Which sequence to run the baseline on, and which frames of it."""

    root: str = ""
    trajectory: str | None = None
    illumination: str | None = None
    soil: str | None = None
    reflectivity: str | None = None
    frame_start: int = 0
    frame_end: int | None = None  # exclusive; None = to the end
    frame_stride: int = 1

    def variant(self) -> Variant:
        return Variant(self.trajectory, self.illumination, self.soil, self.reflectivity)

    def select(self, seq: Sequence) -> list:
        """Frames of `seq` chosen by frame_start/frame_end/frame_stride."""
        end = self.frame_end if self.frame_end is not None else len(seq)
        return seq.frames[self.frame_start:end:self.frame_stride]


@dataclasses.dataclass
class OutputSpec:
    """Where predictions land and how they're filtered before writing."""

    method: str = "da3"  # subfolder name: predictions/{method}/... (also the viewer's Source label)
    overwrite: bool = False
    min_confidence: float | None = None  # pixels below this get depth=nan; None = keep everything
