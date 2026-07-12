"""File readers, keyed by extension. Add entries to the registries to support new formats."""

import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
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


def read_cam(path: Path) -> dict:
    return json.loads(Path(path).read_text())
