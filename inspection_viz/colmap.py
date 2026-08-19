"""A COLMAP run as a prediction source, alongside the usual `predictions/<method>` layout.

Every colmap run this repo writes shares the same shape under `<colmap_dir>` (see
`Layout.colmap_dir` for how that folder is found):

    dense/
        sparse/             # the undistorted model: matches dense/images and the depth maps
            cameras.bin, images.bin, points3D.bin, rigs.bin, frames.bin
        images/<frame>.jpg  # undistorted rgb -- not used here, rgb still comes from gt
        stereo/depth_maps/<frame>.jpg.{geometric,photometric}.bin
        fused.ply           # the one fused point cloud for the run, if dense fusion ran

`sparse/`, `sparse_merge/`, `database.db` etc. also live alongside `dense/` (intermediate,
pre-undistortion state) but are not read here -- `dense/sparse` is the model actually consistent
with `dense/images` and the depth maps, so it is the one whose poses mean anything paired with
them. A folder missing `dense/` (registration failed, or dense stereo was never run) simply
yields no cameras/depth for that variant, the same as any other method that wrote nothing.

pycolmap is only imported inside `Reconstruction.load`, so it stays an optional dependency for
anyone who never points these tools at a colmap prediction folder.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import Layout
from .dataset import Frame

DEPTH_SUFFIXES = (".geometric.bin", ".photometric.bin")  # geometric preferred when both exist


def read_array(path: Path) -> np.ndarray:
    """COLMAP's dense depth/normal map binary format: a "width&height&channels&" ascii header
    followed by a column-major float32 array. 0 marks an unestimated pixel."""
    with open(path, "rb") as f:
        header = b""
        while header.count(b"&") < 3:
            b = f.read(1)
            if not b:
                raise EOFError(f"truncated header in {path}")
            header += b
        w, h, c = (int(x) for x in header.decode("ascii").rstrip("&").split("&"))
        arr = np.fromfile(f, np.float32)
    return np.transpose(arr.reshape((w, h, c), order="F"), (1, 0, 2)).squeeze()


@dataclass
class Reconstruction:
    """One colmap run's `dense/` output for one variant: one shared sparse model plus a depth
    map and rgb-sized camera per registered frame. `images` maps frame index (parsed off the
    colmap image name the same way every other source's filenames are) to the pycolmap Image."""

    model_dir: Path  # informational: marks that a frame has *a* pose to look up, nothing else
    depth_dir: Path
    cloud_path: Path | None
    images: dict  # frame index -> pycolmap.Image

    @classmethod
    @lru_cache(maxsize=8)
    def load(cls, colmap_dir: Path, frame_id_pattern: str) -> "Reconstruction":
        import pycolmap

        dense = colmap_dir / "dense"
        model_dir = dense / "sparse"
        rec = pycolmap.Reconstruction(model_dir)
        frame_id = re.compile(frame_id_pattern)
        images = {}
        for image in rec.images.values():
            ids = frame_id.findall(Path(image.name).stem)
            if ids:
                images[int(ids[-1])] = image
        cloud_path = dense / "fused.ply"
        return cls(model_dir, dense / "stereo" / "depth_maps",
                   cloud_path if cloud_path.is_file() else None, images)

    def camera(self, index: int, width: int, height: int) -> Camera | None:
        """The frame's pose and intrinsics, resized to (width, height) -- gt rgb's size, so it
        lines up with the depth this class hands back for the same frame."""
        image = self.images.get(index)
        if image is None or not image.has_pose:
            return None
        cam = image.camera
        K = cam.calibration_matrix().astype(np.float64)
        K[0, :] *= width / cam.width
        K[1, :] *= height / cam.height
        w2c = np.eye(4)
        w2c[:3, :4] = image.cam_from_world().matrix()
        return Camera(K, np.linalg.inv(w2c), width, height)

    def _depth_path(self, index: int) -> Path | None:
        image = self.images.get(index)
        if image is None:
            return None
        for suffix in DEPTH_SUFFIXES:
            p = self.depth_dir / f"{image.name}{suffix}"
            if p.is_file():
                return p
        return None

    def depth(self, index: int, width: int, height: int) -> np.ndarray | None:
        path = self._depth_path(index)
        if path is None:
            return None
        d = read_array(path)
        d = np.where(d > 0, d, np.nan).astype(np.float32)
        if d.shape[:2] != (height, width):
            d = cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)
        return d


class ColmapFrame(Frame):
    """A gt rgb frame paired with the camera and depth colmap reconstructed for it, instead of
    the per-frame files a regular method writes -- `camera()`/`depth()` read off the one shared
    `Reconstruction` for the variant rather than a path of their own."""

    def __init__(self, index: int, rgb_path: Path, recon: Reconstruction, layout: Layout):
        # depth_path/cam_path carry no data of their own here; set only so the rest of the
        # codebase's "is this field present for this frame" checks (frame_gaps, EvalCase) see the
        # same true/false they would for any other source.
        has_frame = index in recon.images
        depth_path = recon._depth_path(index) if has_frame else None
        super().__init__(index, rgb_path, depth_path, recon.model_dir if has_frame else None, layout)
        self._recon = recon
        self._size = None

    def _wh(self) -> tuple:
        if self._size is None:
            from . import io
            self._size = io.image_size(self.rgb_path)
        return self._size

    def camera(self) -> Camera | None:
        w, h = self._wh()
        return self._recon.camera(self.index, w, h)

    def depth(self) -> np.ndarray | None:
        w, h = self._wh()
        return self._recon.depth(self.index, w, h)
