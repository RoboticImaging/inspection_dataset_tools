"""Sim(3) alignment of a prediction into the ground-truth gauge.

A model that estimates its own poses reconstructs the scene up to a similarity transform: the
world origin, orientation and *scale* are arbitrary. Every metric that compares absolute
quantities (pose error, depth in meters, point-to-point distance) is therefore meaningless until
the prediction is put in the gt gauge, and the honest way to do that is to fit one transform to
the camera centers and then apply it to *everything* the method produced (poses, depth, clouds) —
which is what `EvalCase` does with the `Sim3` returned here.

Relative metrics (RRA/RTA) are invariant to this, so they read the same either way.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sim3:
    """The map `p_gt = scale * R @ p_pred + t`."""

    scale: float = 1.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def points(self, pts: np.ndarray) -> np.ndarray:
        """Transform (N,3) points."""
        return self.scale * (np.asarray(pts, dtype=float) @ self.R.T) + self.t

    def c2w(self, c2w: np.ndarray) -> np.ndarray:
        """Transform a (4,4) camera-to-world. Rotation is unscaled — a scaled world still has
        unit-length camera axes; only positions and distances stretch."""
        out = np.eye(4)
        out[:3, :3] = self.R @ c2w[:3, :3]
        out[:3, 3] = self.points(c2w[:3, 3])
        return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform taking `src` (N,3) onto `dst` (N,3).

    Umeyama 1991, *Least-squares estimation of transformation parameters between two point
    patterns* — the same closed form evo's `traj.align(correct_scale=True)` uses, minus the
    dependency.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N,3) arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"need >= 3 correspondences to fit a rotation, got {len(src)}")

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_src, dst - mu_dst
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:  # reflection: flip the least-significant axis
        s[2, 2] = -1
    rot = u @ s @ vt
    var = (a**2).sum() / len(src)
    scale = float((d * np.diag(s)).sum() / var) if with_scale and var > 0 else 1.0
    return Sim3(scale, rot, mu_dst - scale * rot @ mu_src)


def rotation_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between two stacks of (...,3,3) rotations."""
    tr = np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
