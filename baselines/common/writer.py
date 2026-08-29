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

# Postponed annotation evaluation (PEP 563): lets `str | None`/`np.ndarray | None` etc. below run
# on Python < 3.10 (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu 20.04's
# default Python 3.8) without needing `typing.Optional`/`typing.Union` -- annotations are stored
# as strings instead of evaluated at class-definition time, so `type.__or__` (added in 3.10) is
# never actually called.
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import OpenEXR
import trimesh

from inspection_viz.camera import rot_to_quat
from inspection_viz.config import Layout, fill_template


def frame_stem(index: int) -> str:
    return f"{index:08d}"


def ensure_dir(path: Path) -> None:
    """`path.mkdir(parents=True, exist_ok=True)`, but tolerant of a stale/broken symlink sitting
    at `path` or one of its ancestors -- e.g. `predictions/` symlinked out to shared storage that
    isn't mounted right now. `Path.mkdir` sees that dangling entry as "already there" and raises
    `FileExistsError` instead of creating a real directory, even though `is_dir()`/`exists()` both
    report it as absent -- so a run against detached storage looks like a fresh run, then crashes
    the moment it tries to write. Clear each broken symlink first, outermost ancestor first, so
    the normal `mkdir` below can build a real directory tree in its place.
    """
    for ancestor in reversed([path, *path.parents]):
        if ancestor.is_symlink() and not ancestor.exists():
            ancestor.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, write_fn) -> None:
    """Call `write_fn(tmp_path)` to write into a same-directory temp file, then atomically replace
    `path` with it. Without this, a process killed mid-write -- the OS OOM-killer (a *system*
    memory OOM, not the GPU one `run_baseline` already catches -- SIGKILL can't be caught by any
    Python exception handler at all), a GPU driver crash, node preemption on shared compute, a
    hiccup on network-mounted storage (predictions/ is sometimes symlinked out to separate storage;
    see `check_completion_status.py`'s notes on that) -- leaves a truncated file sitting at the
    real path: a valid-looking header with an incomplete/corrupt data chunk, which then fails to
    decode much later (and much more confusingly) when eval tries to read it. With this, `path`
    either still holds whatever full file was there before, or holds a complete new one; the
    temp file is same-directory so the replace is a same-filesystem rename, and so atomic.
    """
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op once the replace above has moved it into place


def _write_verified(path: Path, write_fn, verify_fn, attempts: int) -> None:
    """`_atomic_write`, plus (when `attempts` > 1) reading the result back through `verify_fn` and
    retrying the whole write if it doesn't check out -- catching corruption that atomicity alone
    doesn't, e.g. a storage hiccup that corrupts the write itself rather than truncating it. Each
    retry redoes the write from the same in-memory data, so it's cheap and always safe to redo.
    If every attempt fails, whatever was written is removed (a known-bad file must not sit there
    looking like valid output) and the last verification error is raised.
    """
    if attempts <= 1:
        _atomic_write(path, write_fn)
        return
    last_exc = None
    for _ in range(attempts):
        _atomic_write(path, write_fn)
        try:
            verify_fn(path)
            return
        except Exception as exc:
            last_exc = exc
    path.unlink(missing_ok=True)
    raise IOError(f"{path}: didn't read back valid after {attempts} attempt(s): {last_exc}")


def _method(method: str | None) -> str:
    """`output.method` defaults to null, meaning "the registered name of the baseline being run"
    — a fallback only `run_baseline` can apply, so reaching a writer without one is a bug."""
    if not method:
        raise ValueError("no method to write under: set output.method in the config, or run the "
                         "baseline through baselines.common.runner.run_baseline, which falls "
                         "back to its registered name")
    return method


def output_dirs(root: Path, layout: Layout, method: str, variant_dict: dict) -> tuple[Path, Path]:
    """Depth dir, cam dir for `method` + a variant (as `Variant.as_dict()`), under `root`.

    Predictions are keyed by the *full* variant, not just the trajectory: a baseline infers depth
    from rgb, so each illumination/soil combination is a distinct prediction with its own dir.
    """
    kw = {"method": _method(method), **variant_dict}
    return (root / fill_template(layout.pred_depth_dir, **kw),
            root / fill_template(layout.pred_cam_dir, **kw))


def cloud_path(root: Path, layout: Layout, method: str, variant_dict: dict) -> Path:
    """The single fused .ply for `method` + a variant, under `root`."""
    return root / fill_template(layout.pred_cloud_file, method=_method(method), **variant_dict)


def oom_path(root: Path, layout: Layout, method: str, variant_dict: dict) -> Path:
    """Where the GPU-OOM marker for `method` + a variant goes, under `root`."""
    return root / fill_template(layout.pred_oom_file, method=_method(method), **variant_dict)


def write_depth(path: Path, depth: np.ndarray, verify: bool = True, attempts: int = 3) -> None:
    """Planar z-depth in meters, matching the gt exr convention (single channel "Z").

    Written atomically (see `_atomic_write`); when `verify` is set (the default), also read back
    immediately to confirm it decodes, retrying up to `attempts` times before giving up loudly --
    see `_write_verified`. The read-back is a single small decode, milliseconds at most, negligible
    next to a frame's model inference time; this is exactly the check that would otherwise only
    happen much later, at eval time, as `OSError: ... EXR_ERR_CORRUPT_CHUNK ...`.
    """
    depth = np.ascontiguousarray(depth, dtype=np.float32)

    def _write(tmp: Path) -> None:
        OpenEXR.File(
            {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
            {"Z": depth},
        ).write(str(tmp))

    def _verify(written: Path) -> None:
        from inspection_viz.io import read_depth  # lazy: keeps this module's import list light
        got = read_depth(written)
        if got.shape != depth.shape:
            raise ValueError(f"read back shape {got.shape}, wrote {depth.shape}")

    _write_verified(path, _write, _verify, attempts if verify else 1)


def write_oom(path: Path, message: str) -> None:
    """Record that this variant hit a GPU OOM while running, with the exception text for context."""
    ensure_dir(path.parent)
    path.write_text(message)


def clear_oom(path: Path) -> None:
    """Remove a stale OOM marker once the variant it names has predictions again."""
    path.unlink(missing_ok=True)


def write_cloud(path: Path, points: np.ndarray, colors: np.ndarray | None = None,
                verify: bool = True, attempts: int = 3) -> None:
    """Fused world-space point cloud as a binary .ply. `points` (N,3) float, `colors` (N,3) uint8.

    Written atomically and (unless `verify` is False) read back to confirm it parses, retrying up
    to `attempts` times -- see `write_depth` for why this is worth the (here, once-per-sequence)
    cost."""
    points = np.asarray(points, dtype=np.float32)
    colors = None if colors is None else np.asarray(colors, dtype=np.uint8)

    def _write(tmp: Path) -> None:
        cloud = trimesh.PointCloud(points, colors=colors)
        tmp.write_bytes(cloud.export(file_type="ply"))

    def _verify(written: Path) -> None:
        from inspection_viz.io import read_cloud
        got, _ = read_cloud(written)
        if len(got) != len(points):
            raise ValueError(f"read back {len(got)} point(s), wrote {len(points)}")

    _write_verified(path, _write, _verify, attempts if verify else 1)


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
    # No read-back verification here (unlike write_depth/write_cloud): a truncated json simply
    # fails to parse rather than silently reading back wrong, so atomicity alone already prevents
    # a partial write from ever landing at the real path.
    _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(payload)))
