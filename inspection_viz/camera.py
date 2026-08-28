# Postponed annotation evaluation (PEP 563): lets `float | None` etc. below run on Python < 3.10
# (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu 20.04's default Python 3.8)
# without needing `typing.Optional`/`typing.Union` -- annotations are stored as strings instead of
# evaluated at class-definition time, so `type.__or__` (added in 3.10) is never actually called.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    return np.array(q)


@dataclass
class Camera:
    """Pinhole camera. c2w is camera-to-world in OpenCV convention (+x right, +y down, +z forward)."""

    K: np.ndarray
    c2w: np.ndarray
    width: int
    height: int
    far: float | None = None

    @classmethod
    def from_blender(cls, params: dict, width: int, height: int) -> "Camera":
        # params: Blender export — loc, rot (WXYZ quat, cam looks down -z), focal_length/sensor_width in mm
        fx = width * params["focal_length"] / params["sensor_width"]
        K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])
        c2w = np.eye(4)
        c2w[:3, :3] = quat_to_rot(params["rot"]) @ np.diag([1.0, -1.0, -1.0])
        c2w[:3, 3] = params["loc"]
        return cls(K, c2w, width, height, far=params.get("clip_end"))

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def wxyz(self) -> np.ndarray:
        return rot_to_quat(self.c2w[:3, :3])

    @property
    def fov_y(self) -> float:
        return 2 * np.arctan(self.height / (2 * self.K[1, 1]))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def unproject(self, depth: np.ndarray, rgb: np.ndarray | None = None,
                  stride: int = 1, max_depth: float | None = None):
        """Planar z-depth map -> (N,3) world points and (N,3) colors (or None)."""
        z = depth[::stride, ::stride]
        v, u = np.mgrid[: depth.shape[0]: stride, : depth.shape[1]: stride] + 0.5
        far = max_depth if max_depth is not None else (self.far and self.far * 0.999)
        valid = np.isfinite(z) & (z > 0)
        if far:
            valid &= z < far
        x = (u - self.K[0, 2]) / self.K[0, 0] * z
        y = (v - self.K[1, 2]) / self.K[1, 1] * z
        pts = np.stack([x[valid], y[valid], z[valid]], -1)
        pts = pts @ self.c2w[:3, :3].T + self.c2w[:3, 3]
        colors = rgb[::stride, ::stride][valid] if rgb is not None else None
        return pts, colors
