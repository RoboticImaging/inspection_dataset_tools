import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import io
from .camera import Camera
from .config import Layout, segment_patterns


@dataclass(frozen=True)
class Variant:
    traj: str | None = None
    illumination: str | None = None
    soil: str | None = None
    reflectivity: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Frame:
    def __init__(self, index: int, rgb_path: Path,
                 depth_path: Path | None, cam_path: Path | None, layout: Layout):
        self.index = index
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.cam_path = cam_path
        self._layout = layout

    def rgb(self) -> np.ndarray:
        return io.read_rgb(self.rgb_path)

    def depth(self) -> np.ndarray | None:
        if self.depth_path is None:
            return None
        return io.read_depth(self.depth_path) * self._layout.depth_scale

    def camera(self) -> Camera | None:
        if self.cam_path is None:
            return None
        w, h = io.image_size(self.rgb_path)
        return Camera.from_blender(io.read_cam(self.cam_path), w, h)


class Sequence:
    """Frames of one variant, ordered by frame index."""

    def __init__(self, variant: Variant, frames: list[Frame]):
        self.variant = variant
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i) -> Frame:
        return self.frames[i]

    def __iter__(self):
        return iter(self.frames)


def _walk_template(root: Path, template: str | tuple[str, ...]) -> list[tuple[Path, dict]]:
    """Dirs under root matching the template(s), with placeholder bindings.

    A template may be a tuple of alternatives (e.g. soil-varying vs. reflectivity-varying
    paths that can never both apply); hits from every alternative are merged.
    """
    if not isinstance(template, str):
        hits, seen = [], set()
        for t in template:
            for d, binds in _walk_template(root, t):
                if d not in seen:
                    seen.add(d)
                    hits.append((d, binds))
        return hits

    hits = []

    def rec(d: Path, pats, binds):
        if not pats:
            hits.append((d, binds))
            return
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            m = pats[0].match(child.name)
            if m:
                rec(child, pats[1:], {**binds, **m.groupdict()})

    if root.is_dir():
        rec(root, segment_patterns(template), {})
    return hits


def _index_frames(d: Path | None, exts, frame_id: re.Pattern) -> dict[int, Path]:
    if d is None or not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.iterdir()):
        ids = frame_id.findall(f.stem)
        if f.is_file() and f.suffix.lower() in exts and ids:
            out[int(ids[-1])] = f
    return out


class AssetDataset:
    """One asset folder (e.g. .../glass_house) holding rgb/depth/cam data for many variants."""

    def __init__(self, root: str | Path, layout: Layout = Layout()):
        self.root = Path(root)
        self.layout = layout
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._rgb_dirs = _walk_template(self.root, layout.rgb_dir)
        self._depth_dirs = _walk_template(self.root, layout.depth_dir)
        self._cam_dirs = _walk_template(self.root, layout.cam_dir)
        self._pred_depth_dirs = _walk_template(self.root, layout.pred_depth_dir)
        self._pred_cam_dirs = _walk_template(self.root, layout.pred_cam_dir)
        mesh_pat = segment_patterns(layout.mesh_file)[0]
        self.meshes = {
            m.groupdict().get("kind", f.stem): f
            for f in sorted(self.root.iterdir())
            if f.is_file() and (m := mesh_pat.match(f.name))
        }
        self.variants = sorted(
            {Variant(b.get("traj"), b.get("illumination"), b.get("soil"), b.get("reflectivity"))
             for _, b in self._rgb_dirs},
            key=str,
        )
        self._frame_id = re.compile(layout.frame_id)

    def _values(self, field: str) -> list[str]:
        return sorted({v for v in (getattr(x, field) for x in self.variants) if v is not None})

    @property
    def trajectories(self) -> list[str]:
        return self._values("traj")

    @property
    def illuminations(self) -> list[str]:
        return self._values("illumination")

    @property
    def soils(self) -> list[str]:
        return self._values("soil")

    @property
    def reflectivities(self) -> list[str]:
        return self._values("reflectivity")

    @property
    def methods(self) -> list[str]:
        """Baseline prediction sources found under predictions/ (e.g. ["da3"])."""
        return sorted({b["method"] for _, b in self._pred_depth_dirs if "method" in b})

    def axes(self, traj: str | None, illumination: str | None) -> dict[str, list[str]]:
        """Which of {soil, reflectivity} can be varied for this (traj, illumination), and
        with which values. Soil and reflectivity are never both varied at once, so this is
        how the UI discovers which single axis is on offer (sometimes both, sometimes one)."""
        matches = [v for v in self.variants
                   if (traj is None or v.traj == traj)
                   and (illumination is None or v.illumination == illumination)]
        out = {}
        for field in ("soil", "reflectivity"):
            vals = sorted({getattr(v, field) for v in matches if getattr(v, field) is not None})
            if vals:
                out[field] = vals
        return out

    def has(self, variant: Variant) -> bool:
        return self._dir(self._rgb_dirs, variant.as_dict(), exact=True) is not None

    def sequence(self, variant: Variant, source: str = "gt") -> Sequence:
        """Frames for `variant`. `source` is "gt" or one of `self.methods`, selecting whether
        depth/cam come from ground truth or a baseline's predictions/ output (rgb is always gt —
        baselines consume it, they don't produce it)."""
        rgb_dir = self._dir(self._rgb_dirs, variant.as_dict(), exact=True)
        if rgb_dir is None:
            raise KeyError(f"no data for {variant}")
        lay = self.layout
        rgb = _index_frames(rgb_dir, lay.rgb_exts, self._frame_id)
        if source == "gt":
            depth_dir = self._dir(self._depth_dirs, variant.as_dict())
            cam_dir = self._dir(self._cam_dirs, variant.as_dict())
        else:
            want = {**variant.as_dict(), "method": source}
            depth_dir = self._dir(self._pred_depth_dirs, want)
            cam_dir = self._dir(self._pred_cam_dirs, want)
            if depth_dir is None and cam_dir is None:
                raise KeyError(f"no predictions for method={source!r}, {variant}")
        depth = _index_frames(depth_dir, lay.depth_exts, self._frame_id)
        cam = _index_frames(cam_dir, lay.cam_exts, self._frame_id)
        frames = [Frame(i, rgb[i], depth.get(i), cam.get(i), lay) for i in sorted(rgb)]
        return Sequence(variant, frames)

    @staticmethod
    def _dir(entries: list[tuple[Path, dict]], want: dict, exact: bool = False) -> Path | None:
        for d, binds in entries:
            if binds == want if exact else all(want.get(k) == v for k, v in binds.items()):
                return d
        return None
