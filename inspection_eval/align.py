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

import torch

from . import device
from .device import DTYPE, tensor


@dataclass(frozen=True)
class Sim3:
    """The map `p_gt = scale * R @ p_pred + t`."""

    scale: float = 1.0
    R: torch.Tensor = field(
        default_factory=lambda: torch.eye(3, dtype=DTYPE, device=device.DEVICE))
    t: torch.Tensor = field(
        default_factory=lambda: torch.zeros(3, dtype=DTYPE, device=device.DEVICE))

    def __post_init__(self):
        # A transform may be handed in as numpy (a hand-written gauge, a test); everything that
        # uses one assumes its two tensors are already on the eval device.
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "R", tensor(self.R, DTYPE))
        object.__setattr__(self, "t", tensor(self.t, DTYPE))

    def points(self, pts) -> torch.Tensor:
        """Transform (N,3) points. A float32 cloud stays float32 — it is the one array here big
        enough that doubling it would cost real time and memory."""
        pts = tensor(pts)
        return self.scale * (pts @ self.R.T.to(pts.dtype)) + self.t.to(pts.dtype)

    def c2w(self, c2w) -> torch.Tensor:
        """Transform a (4,4) camera-to-world. Rotation is unscaled — a scaled world still has
        unit-length camera axes; only positions and distances stretch."""
        c2w = tensor(c2w, DTYPE)
        out = torch.eye(4, dtype=DTYPE, device=c2w.device)
        out[:3, :3] = self.R @ c2w[:3, :3]
        out[:3, 3] = self.points(c2w[:3, 3])
        return out


def umeyama(src, dst, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform taking `src` (N,3) onto `dst` (N,3).

    Umeyama 1991, *Least-squares estimation of transformation parameters between two point
    patterns* — the same closed form evo's `traj.align(correct_scale=True)` uses, minus the
    dependency.
    """
    src = tensor(src, DTYPE)
    dst = tensor(dst, DTYPE)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N,3) arrays, got {tuple(src.shape)} and "
                         f"{tuple(dst.shape)}")
    if len(src) < 3:
        raise ValueError(f"need >= 3 correspondences to fit a rotation, got {len(src)}")

    mu_src, mu_dst = src.mean(dim=0), dst.mean(dim=0)
    a, b = src - mu_src, dst - mu_dst
    u, d, vt = torch.linalg.svd(b.T @ a / len(src))
    s = torch.eye(3, dtype=src.dtype, device=src.device)
    if torch.det(u) * torch.det(vt) < 0:  # reflection: flip the least-significant axis
        s[2, 2] = -1
    rot = u @ s @ vt
    var = (a**2).sum() / len(src)
    scale = float((d * torch.diagonal(s)).sum() / var) if with_scale and var > 0 else 1.0
    return Sim3(scale, rot, mu_dst - scale * rot @ mu_src)


def rotation_angles(a, b) -> torch.Tensor:
    """Geodesic angle in degrees between two stacks of (...,3,3) rotations."""
    a, b = tensor(a), tensor(b)
    tr = torch.diagonal(a.transpose(-1, -2) @ b, dim1=-2, dim2=-1).sum(dim=-1)
    return torch.rad2deg(torch.arccos(torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)))
