"""Depth-map metrics, accumulated pixel-wise over every frame the method and the gt share.

Ported from the dataset repo's `eval.depth_metrics`, with the same definitions so numbers stay
comparable:

- **RMSE / MAE / Relative MAE** over pixels valid in both maps.
- **Accuracy / Completeness @ tau** — the fraction of pixels whose depth ratio
  `max(d_gt/d_est, d_est/d_gt)` is under tau, over the valid *estimated* pixels (accuracy) and
  over the valid *gt* pixels (completeness). So a method that predicts little but predicts it
  well scores high on accuracy and low on completeness. (Schöps et al., *Depth Map Evaluation
  for Multi-View Stereo*.)
- **Scale invariant MSE** — squared log error after removing the best per-frame scale, i.e. what
  is left once the one thing a monocular method cannot know is forgiven.

Scale: a predicted depth map is only meaningful in the gt gauge, so by default the whole
sequence is multiplied by the scale of the pose alignment (`scale: "align"`), which keeps depth
consistent with the poses. `"median"` instead fits a per-frame median ratio — the usual
monocular-depth convention, and the right choice when only relative depth quality is of
interest; `"none"` takes the prediction at face value (use it with gt-conditioned runs).
"""

from dataclasses import dataclass

import numpy as np

from ..case import CAP_DEPTH, EvalCase
from .base import register
from .stats import positive_depth


@dataclass
class DepthOptions:
    thresholds: tuple = (1.05, 1.01, 1.005)  # ratio thresholds tau for accuracy/completeness
    scale: str = "align"  # "align" | "median" | "none"
    max_frames: int | None = None  # evaluate at most this many frames (evenly spread)


def _columns(o: DepthOptions) -> list[str]:
    """The keys `depth_metrics` returns for `o`, without touching a case -- see
    `metrics.base`'s docstring for why this exists. Kept in lockstep with `depth_metrics` by
    `test_depth_columns_match_what_the_metric_actually_returns`."""
    cols = ["RMSE (m)", "MAE (m)", "Relative MAE", "Scale invariant MSE"]
    cols += [f"Accuracy @{tau}" for tau in o.thresholds]
    cols += [f"Completeness @{tau}" for tau in o.thresholds]
    cols += ["Valid px ratio (est)", "Valid px ratio (gt)", "Depth scale", "Frames"]
    return cols


@register("depth", needs=(CAP_DEPTH,), options=DepthOptions, columns=_columns)
def depth_metrics(case: EvalCase, o: DepthOptions) -> dict:
    frames = _select(case.depth_frames, o.max_frames)
    taus = np.asarray(o.thresholds, dtype=float)

    n_px = n_valid = n_valid_gt = n_valid_est = 0
    sum_abs = sum_sq = sum_rel = sum_si = 0.0
    within = np.zeros(len(taus))

    for index in frames:
        gt, pred = case.depth_pair(index)
        pred = pred * _frame_scale(case, o, gt, pred)

        # `depth_pair` already hands over nan for everything unusable, but this metric applies a
        # scale of its own afterwards and is the thing that divides by depth and logs it, so it
        # re-checks rather than trusting that.
        valid_gt = positive_depth(gt)
        valid_est = positive_depth(pred)
        valid = valid_gt & valid_est

        n_px += gt.size
        n_valid_gt += int(np.count_nonzero(valid_gt))
        n_valid_est += int(np.count_nonzero(valid_est))
        n_valid += int(np.count_nonzero(valid))
        if not np.any(valid):
            continue

        g, e = gt[valid].astype(np.float64), pred[valid].astype(np.float64)
        abs_diff = np.abs(g - e)
        sum_abs += float(abs_diff.sum())
        sum_sq += float((abs_diff**2).sum())
        sum_rel += float((abs_diff / g).sum())

        ratio = np.maximum(g / e, e / g)
        within += (ratio[:, None] < taus[None, :]).sum(axis=0)

        log_err = np.log(g) - np.log(e)
        sum_si += float(((log_err - log_err.mean()) ** 2).sum())

    if n_valid == 0:
        raise ValueError("no pixel is valid in both the gt and the predicted depth maps")

    row = {
        "RMSE (m)": np.sqrt(sum_sq / n_valid),
        "MAE (m)": sum_abs / n_valid,
        "Relative MAE": sum_rel / n_valid,
        "Scale invariant MSE": sum_si / n_valid,
    }
    for tau, count in zip(o.thresholds, within):
        row[f"Accuracy @{tau}"] = count / n_valid_est if n_valid_est else float("nan")
    for tau, count in zip(o.thresholds, within):
        row[f"Completeness @{tau}"] = count / n_valid_gt if n_valid_gt else float("nan")
    row.update({
        "Valid px ratio (est)": n_valid_est / n_px,
        "Valid px ratio (gt)": n_valid_gt / n_px,
        "Depth scale": _sequence_scale(case, o),
        "Frames": len(frames),
    })
    return row


def _select(frames: list, max_frames: int | None) -> list:
    if max_frames is None or len(frames) <= max_frames:
        return frames
    return [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]


def _sequence_scale(case: EvalCase, o: DepthOptions) -> float:
    """The one scale applied to every frame, or nan when it is fitted per frame."""
    if o.scale == "align":
        scale = float(case.transform.scale)
        # A degenerate gauge fit (collinear or coincident camera centers) would multiply every
        # depth map by nan or flip its sign; every number below would then be nan or nonsense,
        # so the metric fails loudly here instead of reporting them.
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"the gauge alignment gave an unusable depth scale ({scale}); "
                             "use scale=median or scale=none for this case")
        return scale
    if o.scale == "none":
        return 1.0
    if o.scale == "median":
        return float("nan")
    raise ValueError(f"scale must be 'align', 'median' or 'none', got {o.scale!r}")


def _frame_scale(case: EvalCase, o: DepthOptions, gt: np.ndarray, pred: np.ndarray) -> float:
    if o.scale != "median":
        return _sequence_scale(case, o)
    valid = positive_depth(gt) & positive_depth(pred)
    if not np.any(valid):
        return 1.0
    scale = float(np.median(gt[valid] / pred[valid]))
    # One unusable frame should cost that frame, not the sequence: leaving it at 1.0 keeps it in
    # the aggregate as the (large) error it is, rather than nan-ing every pixel of it.
    return scale if np.isfinite(scale) and scale > 0 else 1.0
