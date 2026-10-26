"""File readers, keyed by extension. Add entries to the registries to support new formats."""

# Postponed annotation evaluation (PEP 563): lets `np.ndarray | None` etc. below run on
# Python < 3.10 (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu 20.04's
# default Python 3.8) without needing `typing.Optional`/`typing.Union` -- annotations are stored
# as strings instead of evaluated at class-definition time, so `type.__or__` (added in 3.10) is
# never actually called.
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import trimesh
from PIL import Image


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def _read_exr(path: Path) -> np.ndarray:
    channels = OpenEXR.File(str(path)).parts[0].channels
    ch = channels.get("Z") or channels.get("V") or next(iter(channels.values()))
    px = ch.pixels
    return (px[..., 0] if px.ndim == 3 else px).astype(np.float32)


def _read_depth_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read depth: {path}")
    return (img[..., 0] if img.ndim == 3 else img).astype(np.float32)


DEPTH_READERS = {".exr": _read_exr, ".png": _read_depth_img, ".tiff": _read_depth_img, ".tif": _read_depth_img}


def read_depth(path: Path) -> np.ndarray:
    return DEPTH_READERS[path.suffix.lower()](path)


def _read_mask_img(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read mask: {path}")
    return img[..., 0] if img.ndim == 3 else img


def _read_mask_npy(path: Path) -> np.ndarray:
    m = np.load(path)
    return m[..., 0] if m.ndim == 3 else m


MASK_READERS = {".png": _read_mask_img, ".tiff": _read_mask_img, ".tif": _read_mask_img,
                ".npy": _read_mask_npy}


def read_mask(path: Path) -> np.ndarray:
    """An evaluation-region mask as a boolean (H,W) array, True = consider the pixel.

    Masks hold 0 for "ignore" and 1 (or close to it, e.g. 0.99 after resampling) for "consider";
    thresholding at 0.5 keeps both of those -- and 0/255 8-bit pngs -- working, without letting a
    faint nonzero residue of resampling count as inside the region."""
    return MASK_READERS[path.suffix.lower()](path) > 0.5


def read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """A point cloud file (.ply): (N,3) float32 xyz, plus (N,3) uint8 rgb or None if uncolored."""
    cloud = trimesh.load(path, process=False)
    pts = np.asarray(cloud.vertices, dtype=np.float32)
    colors = np.asarray(getattr(cloud.visual, "vertex_colors", ()))  # (N,4) rgba, or empty
    if colors.ndim != 2 or len(colors) != len(pts):
        return pts, None
    return pts, colors[:, :3].astype(np.uint8)


def read_cam(path: Path) -> dict:
    return json.loads(Path(path).read_text())
