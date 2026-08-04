"""Writes baseline predictions (depth + camera) back into the asset folder, in exactly the
format `inspection_viz` already reads for ground truth — no reader changes needed, and the
viewer's Source toggle picks up a new method as soon as its files exist.

Camera convention: predictions are plain OpenCV pinhole cameras (fx, fy, cx=w/2, cy=h/2), but we
reuse the Blender-style cam json schema (`loc` + `rot` WXYZ camera-to-world quaternion,
`focal_length`/`sensor_width`) that `Camera.from_blender` already decodes for gt. Any fx can be
represented losslessly in that schema by picking sensor_width=1 and focal_length=fx/width; that
ratio is scale-invariant, so it's correct however wide the frame `fx` was measured against was,
as long as it's the same frame `width` used to write it (see `write_cam`). fy is not stored —
gt cams are square-pixel too (`Camera.from_blender` only ever derives one focal length), so a
predicted fy is silently dropped; fine for the near-square-pixel models these baselines produce.
"""

import json
from pathlib import Path

import numpy as np
import OpenEXR

from inspection_viz.camera import rot_to_quat
from inspection_viz.config import Layout


def frame_stem(index: int) -> str:
    return f"{index:08d}"


def output_dirs(root: Path, layout: Layout, method: str, variant_dict: dict) -> tuple[Path, Path]:
    """Depth dir, cam dir for `method` + a variant (as `Variant.as_dict()`), under `root`."""
    kw = {"method": method, **variant_dict}
    return root / layout.pred_depth_dir.format(**kw), root / layout.pred_cam_dir.format(**kw)


def write_depth(path: Path, depth: np.ndarray) -> None:
    """Planar z-depth in meters, matching the gt exr convention (single channel "Z")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File(
        {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
        {"Z": np.ascontiguousarray(depth, dtype=np.float32)},
    ).write(str(path))


def write_cam(path: Path, c2w: np.ndarray, K: np.ndarray, width: int,
              clip_end: float | None = None) -> None:
    """`c2w`: (3,4) or (4,4) OpenCV camera-to-world. `K`: (3,3) pinhole intrinsics for a frame
    that is `width` px wide (must be the width the intrinsics were measured against, not
    necessarily the width of the depth map written alongside — see module docstring)."""
    r_blender = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    payload = {
        "loc": np.asarray(c2w[:3, 3]).tolist(),
        "rot": rot_to_quat(r_blender).tolist(),
        "focal_length": float(K[0, 0]) / width,
        "sensor_width": 1.0,
    }
    if clip_end is not None:
        payload["clip_end"] = float(clip_end)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
