"""Sim(3) alignment: the transform that puts one set of camera poses into another's gauge.

A model that estimates its own poses reconstructs the scene up to a similarity transform: the
world origin, orientation and *scale* are arbitrary. Comparing anything absolute (pose error,
depth in meters, point-to-point distance) -- or just looking at two sources side by side --
is meaningless until one is put in the other's gauge, and the honest way to do that is to fit
one transform to the camera centers they share and apply it to everything (poses, depth,
clouds). `inspection_eval.case.EvalCase` uses this to score a prediction; `inspection_viz.Viewer`
uses the same fit so a colmap/da3/etc. source lines up with gt on screen instead of needing a
manual re-navigate every time you switch sources.
"""

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera


@dataclass(frozen=True)
class Sim3:
    """The map `p_dst = scale * R @ p_src + t`."""

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
    dependency. Needs >= 3 non-collinear points: two points fix translation and scale but leave
    one rotational degree of freedom (the twist about the axis between them) undetermined.
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


def fit_cameras(src: dict[int, Camera], dst: dict[int, Camera], with_scale: bool = True) -> Sim3:
    """The Sim3 mapping `src`'s camera positions onto `dst`'s, fitted to whatever frame indices
    the two dicts share. Identity if fewer than 3 are shared -- too few to fix a unique rotation
    (see `umeyama`), which is also the sane fallback when the two sources don't overlap at all."""
    shared = sorted(src.keys() & dst.keys())
    if len(shared) < 3:
        return Sim3()
    src_pts = np.array([src[i].position for i in shared])
    dst_pts = np.array([dst[i].position for i in shared])
    return umeyama(src_pts, dst_pts, with_scale=with_scale)
