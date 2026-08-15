"""Camera pose metrics.

Two families, deliberately kept side by side because they answer different questions:

- **Relative** (RRA / RTA / AUC) — how well does the method get the geometry *between* frames?
  Invariant to the gauge transform, so they are the fair thing to quote for a method that
  estimates its own poses. Following PoseDiffusion (Wang et al., ICCV 2023): for a pair (i, j),
  RRA is the angle of the relative rotation error and RTA the angle between the relative
  translation directions; we report the percentage of pairs under a threshold and the AUC of the
  max of the two out to `auc_max` degrees.
- **Absolute** (ATE, absolute rotation error) — how well does the trajectory sit in the world
  once aligned? These read the gauge transform, and are only meaningful after alignment.
"""

from dataclasses import dataclass

import torch

from .. import device
from ..align import rotation_angles
from ..case import CAP_CAMERAS, EvalCase
from ..device import stack
from . import stats
from .base import register


@dataclass
class PoseOptions:
    thresholds: tuple = (5, 15, 30)  # degrees, for RRA/RTA percentages
    auc_max: int = 30  # degrees; AUC is over max(RRA, RTA) out to here
    pairs: str = "consecutive"  # "consecutive" (as the original eval did) or "all"


@register("pose", needs=(CAP_CAMERAS,), options=PoseOptions)
def pose_metrics(case: EvalCase, o: PoseOptions) -> dict:
    frames = case.cam_frames
    gt = [case.gt_cams[i] for i in frames]
    pred = [case.pred_cams[i] for i in frames]

    i_idx, j_idx = _pairs(len(frames), o.pairs)
    n_pairs = len(i_idx)
    if n_pairs == 0:
        raise ValueError("need at least two shared frames for pose metrics")

    r_gt = stack([c.c2w[:3, :3] for c in gt])
    r_pred = stack([c.c2w[:3, :3] for c in pred])
    t_gt = stack([c.position for c in gt])
    t_pred = stack([c.position for c in pred])

    rra = rotation_angles(_relative(r_gt, i_idx, j_idx), _relative(r_pred, i_idx, j_idx))
    rta = _direction_angles(t_gt[j_idx] - t_gt[i_idx], t_pred[j_idx] - t_pred[i_idx])

    # A pair whose baseline is zero in either trajectory has no relative translation direction,
    # so its RTA is undefined rather than bad. Such pairs leave the denominator instead of being
    # scored as failures, which would quietly punish a method for a stationary stretch of the
    # sequence; `Undefined pairs` says how many were dropped.
    err = stats.finite(torch.maximum(rra, rta))
    n_scored = int(err.numel())

    row = {}
    for tau in o.thresholds:
        row[f"RRA@{tau} %"] = 100.0 * stats.fraction_below(rra, tau)
    for tau in o.thresholds:
        row[f"RTA@{tau} %"] = 100.0 * stats.fraction_below(rta, tau)

    if n_scored:
        # one bin per degree out to auc_max, errors past it left out, as np.histogram had it
        hist = torch.histc(err, bins=o.auc_max, min=0, max=o.auc_max)
        row[f"AUC@{o.auc_max}"] = float(torch.cumsum(hist / n_scored, dim=0).mean())
    else:
        row[f"AUC@{o.auc_max}"] = float("nan")

    offsets = torch.linalg.norm(t_pred - t_gt, dim=1)
    abs_rot = rotation_angles(r_gt, r_pred)
    row.update({
        "ATE RMSE (m)": stats.rms(offsets),
        "ATE median (m)": stats.median(offsets),
        "Abs rot err mean (deg)": stats.mean(abs_rot),
        "Align scale": float(case.transform.scale),
        "Frames": len(frames),
        "Cam pairs": n_pairs,
        "Undefined pairs": n_pairs - n_scored,
    })
    return row


def _pairs(n: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "consecutive":
        i = torch.arange(n - 1, device=device.DEVICE)
        return i, i + 1
    if mode == "all":
        return torch.triu_indices(n, n, offset=1, device=device.DEVICE).unbind()
    raise ValueError(f"pairs must be 'consecutive' or 'all', got {mode!r}")


def _relative(rots: torch.Tensor, i_idx, j_idx) -> torch.Tensor:
    """R_i @ R_j^T for each pair — the rotation between two views."""
    return rots[i_idx] @ rots[j_idx].transpose(1, 2)


def _direction_angles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle in degrees between corresponding (N,3) vectors; scale-free, so a wrong baseline
    length costs nothing here (that is what ATE and the depth metrics are for)."""
    den = torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1)
    cos = (a * b).sum(dim=1) / torch.where(den > 0, den, torch.nan)
    return torch.rad2deg(torch.arccos(torch.clamp(cos, -1.0, 1.0)))
