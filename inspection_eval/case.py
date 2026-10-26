"""One unit of evaluation: an asset's ground truth vs. one method's prediction for one variant.

`EvalCase` is the only thing a metric sees. It owns the boring, shared half of evaluation —
pairing gt and predicted frames, fitting the gauge transform, scaling and size-matching depth,
sampling gt meshes — so that a metric is just a function of prepared arrays, and a new metric
needs no plumbing of its own. Everything is lazy and cached: asking for poses never loads a
depth map, and a metric can ask twice for free.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from inspection_viz.camera import Camera
from inspection_viz.dataset import AssetDataset, Frame, Variant

from . import geometry
from .align import Sim3, fit_cameras

ALIGN_MODES = ("sim3", "se3", "none")

# What a case can offer; metrics declare which of these they need (see metrics.base.register).
CAP_CAMERAS = "cameras"
CAP_DEPTH = "depth"
CAP_CLOUD = "cloud"


def match_size(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample whichever of two maps is larger to the other's size, nearest-neighbour so
    invalid (nan) pixels never bleed into valid ones."""
    (ha, wa), (hb, wb) = a.shape[:2], b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    if ha * wa > hb * wb:
        return cv2.resize(a, (wb, hb), interpolation=cv2.INTER_NEAREST), b
    return a, cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST)


@dataclass
class EvalCase:
    dataset: AssetDataset
    method: str
    variant: Variant
    align_mode: str = "sim3"
    cache_dir: Path = geometry.CACHE_DIR  # for gt clouds sampled off the (large) gt meshes

    def __post_init__(self):
        if self.align_mode not in ALIGN_MODES:
            raise ValueError(f"align_mode must be one of {ALIGN_MODES}, got {self.align_mode!r}")

    # --- identity -------------------------------------------------------------------------

    @property
    def label(self) -> dict[str, str]:
        """The index columns identifying this case in a results table -- always all of them
        (blank for whichever axis this variant doesn't use), so every metric's table gets the
        same column order regardless of which assets/methods happened to touch it first, and a
        blank cell and a genuinely-absent one are never ambiguous (see `runner.case_key`)."""
        v = self.variant
        return {"Asset": self.dataset.root.name, "Method": self.method, "Traj": v.traj or "",
                "Illumination": v.illumination or "", "Soil": v.soil or "",
                "Reflectivity": v.reflectivity or ""}

    # --- frames ---------------------------------------------------------------------------

    @cached_property
    def gt_frames(self) -> dict[int, Frame]:
        return {f.index: f for f in self.dataset.sequence(self.variant, "gt")}

    @cached_property
    def pred_frames(self) -> dict[int, Frame]:
        return {f.index: f for f in self.dataset.sequence(self.variant, self.method)}

    # --- cameras --------------------------------------------------------------------------

    @cached_property
    def gt_cams(self) -> dict[int, Camera]:
        return {i: c for i, f in self.gt_frames.items() if _usable_cam(c := f.camera())}

    @cached_property
    def raw_pred_cams(self) -> dict[int, Camera]:
        """Predicted cameras as written, i.e. still in the method's own gauge."""
        return {i: c for i, f in self.pred_frames.items() if _usable_cam(c := f.camera())}

    @cached_property
    def cam_frames(self) -> list[int]:
        return sorted(self.gt_cams.keys() & self.raw_pred_cams.keys())

    @cached_property
    def transform(self) -> Sim3:
        """Gauge transform prediction -> gt, fitted to the shared camera centers."""
        if self.align_mode == "none":
            return Sim3()
        return fit_cameras(self.raw_pred_cams, self.gt_cams, with_scale=self.align_mode == "sim3")

    @cached_property
    def pred_cams(self) -> dict[int, Camera]:
        """Predicted cameras in the gt gauge. Intrinsics are untouched — a similarity transform
        of the world does not change what the camera saw."""
        t = self.transform
        return {i: replace(c, c2w=t.c2w(c.c2w)) for i, c in self.raw_pred_cams.items()}

    # --- depth ----------------------------------------------------------------------------

    @cached_property
    def depth_frames(self) -> list[int]:
        have_gt = {i for i, f in self.gt_frames.items() if f.depth_path is not None}
        have_pred = {i for i, f in self.pred_frames.items() if f.depth_path is not None}
        return sorted(have_gt & have_pred)

    def depth_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """(gt, predicted) depth for one frame, in meters, same size, invalid pixels nan.

        The prediction is *not* rescaled here — how to resolve the depth scale is the depth
        metric's business (see `DepthOptions.scale`), and other metrics want the raw map.
        """
        gt = _clean(self.gt_frames[index].depth(), far=_far(self.gt_cams.get(index)))
        pred = _clean(self.pred_frames[index].depth(), far=_far(self.raw_pred_cams.get(index)))
        return match_size(gt, pred)

    def mask_available(self, kind: str, frames) -> bool:
        """True only if *every one of* `frames` has a `kind` mask ("mask" or "window").
        All-or-nothing on purpose: silently evaluating a region on a subset of frames would
        bias it."""
        frames = list(frames)
        return bool(frames) and all(
            self.gt_frames[i].mask_paths.get(kind) is not None for i in frames)

    def mask(self, index: int, kind: str, shape: tuple[int, int]) -> np.ndarray:
        """Boolean region mask for one gt frame, resized (nearest) to `shape` = (h, w)."""
        m = self.gt_frames[index].mask(kind)
        if m is None:
            raise FileNotFoundError(f"no {kind} mask for frame {index}")
        if m.shape != tuple(shape):
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        return m

    # --- clouds ---------------------------------------------------------------------------

    @cached_property
    def fused_cloud_path(self):
        """The single .ply a method fused for this variant, if it wrote one."""
        return self.dataset.cloud_path(self.variant, self.method)

    def pred_cloud(self, source: str = "auto", stride: int = 4,
                   max_points: int | None = 2_000_000) -> np.ndarray:
        """The predicted surface as (N,3) points in the gt gauge.

        `source`: "fused" uses the method's own fused .ply, "unprojected" unprojects its
        per-frame depth with its own poses, "auto" prefers the .ply and falls back.
        """
        from inspection_viz import io

        if source not in ("auto", "fused", "unprojected"):
            raise ValueError(f"unknown cloud source {source!r}")
        use_fused = source == "fused" or (source == "auto" and self.fused_cloud_path is not None)
        if use_fused:
            if self.fused_cloud_path is None:
                raise FileNotFoundError(f"no fused cloud for {self.method} / {self.variant}")
            pts, _ = io.read_cloud(self.fused_cloud_path)
        else:
            chunks = []
            for i in self.depth_frames:
                cam = self.raw_pred_cams.get(i)
                if cam is None:
                    continue
                depth = _clean(self.pred_frames[i].depth(), far=_far(cam))
                chunks.append(cam.unproject(depth, stride=stride)[0])
            if not chunks:
                raise FileNotFoundError(f"no depth+poses to unproject for {self.method}")
            pts = np.concatenate(chunks)
        # Filtered before the cap so `max_points` counts usable points, not nan padding.
        pts = geometry.subsample(geometry.finite(pts), max_points)
        return self.transform.points(pts)

    def gt_cloud(self, kind: str, n_points: int = 1_000_000) -> np.ndarray:
        """(n_points, 3) sampled off the gt mesh of that `kind` (see `AssetDataset.meshes`)."""
        return geometry.sample_mesh(self.dataset.meshes[kind], n_points, self.cache_dir)

    def gt_mesh(self, *kinds: str) -> str | None:
        """The first of `kinds` this asset actually ships, else any mesh it has."""
        for kind in kinds:
            if kind in self.dataset.meshes:
                return kind
        return next(iter(self.dataset.meshes), None)

    # --- what metrics can run -------------------------------------------------------------

    @cached_property
    def capabilities(self) -> set[str]:
        caps = set()
        if len(self.cam_frames) >= 2:
            caps.add(CAP_CAMERAS)
        if self.depth_frames:
            caps.add(CAP_DEPTH)
        has_pred_surface = self.fused_cloud_path is not None or (
            self.depth_frames and self.raw_pred_cams)
        if self.dataset.meshes and has_pred_surface:
            caps.add(CAP_CLOUD)
        return caps

    def missing(self, needs) -> list[str]:
        return [n for n in needs if n not in self.capabilities]


def _usable_cam(cam: Camera | None) -> bool:
    """A pose with a nan or inf anywhere in it would put nan through the gauge fit and from
    there into every metric of the case, so the frame is dropped as if it had never been
    written — the same treatment a missing cam json gets."""
    return cam is not None and bool(np.all(np.isfinite(cam.c2w)))


def _far(cam: Camera | None) -> float | None:
    """A gt render's clip_end is background, not surface — anything at or past it is invalid."""
    return None if cam is None or cam.far is None else cam.far * 0.999


def _clean(depth: np.ndarray, far: float | None) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    invalid = ~np.isfinite(d) | (d <= 0)
    if far is not None:
        invalid |= d >= far
    return np.where(invalid, np.nan, d)
