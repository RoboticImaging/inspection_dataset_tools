"""Point-cloud / surface metrics against the gt meshes.

Distances are one-sided nearest-neighbour distances between the predicted cloud and points
sampled off the gt mesh:

- **Accuracy** — predicted point to nearest gt point ("is what it reconstructed real?").
- **Completeness** — gt point to nearest predicted point ("did it reconstruct everything?").
- **Chamfer** — the mean of the two.
- **Precision / Recall / F-score @ tau** — the same two distances thresholded, which is what
  benchmarks like ETH3D and Tanks & Temples report and what stays comparable across scenes of
  different size.

Assets ship two gt meshes and the two directions use different ones: `{asset}_accuracy.glb` (the
full scene — anything the method reconstructed should lie on it) for the accuracy direction, and
`{asset}_completeness.glb` (the asset of interest) for the completeness direction, so a method is
not punished for skipping scenery it was never asked to reconstruct. Assets with a single mesh
use it for both.

The prediction is cropped to the gt bounding box first (`crop: "gt_bbox"`): a method that hangs
a cloud of sky points behind the asset would otherwise post an arbitrarily bad accuracy that
says nothing about the surface it did reconstruct.
"""

from dataclasses import dataclass

import torch

from .. import geometry
from ..case import CAP_CLOUD, EvalCase
from . import stats
from .base import register


@dataclass
class CloudOptions:
    thresholds: tuple = (0.05, 0.1, 0.25)  # meters, for precision/recall/F-score
    gt_points: int = 1_000_000  # points sampled off each gt mesh
    max_points: int | None = 2_000_000  # cap on the predicted cloud (random subset)
    source: str = "auto"  # "auto" | "fused" (method's own .ply) | "unprojected" (from depth)
    unproject_stride: int = 4  # pixel stride, when unprojecting
    crop: str = "gt_bbox"  # "gt_bbox" | "none"
    crop_margin: float = 0.05  # fraction of the gt bbox diagonal, added on every side
    accuracy_mesh: str = "accuracy"
    completeness_mesh: str = "completeness"


@register("cloud", needs=(CAP_CLOUD,), options=CloudOptions)
def cloud_metrics(case: EvalCase, o: CloudOptions) -> dict:
    acc_kind = case.gt_mesh(o.accuracy_mesh)
    comp_kind = case.gt_mesh(o.completeness_mesh)
    if acc_kind is None or comp_kind is None:
        raise FileNotFoundError(f"no gt mesh at {case.dataset.root}")

    pred = case.pred_cloud(o.source, stride=o.unproject_stride, max_points=o.max_points)
    gt_acc = case.gt_cloud(acc_kind, o.gt_points)
    gt_comp = case.gt_cloud(comp_kind, o.gt_points) if comp_kind != acc_kind else gt_acc

    n_raw = len(pred)
    if n_raw == 0:
        raise ValueError("the method produced no points at all for this variant")
    if o.crop == "gt_bbox":
        pred = geometry.crop(pred, *geometry.bounds(gt_acc, o.crop_margin))
    elif o.crop != "none":
        raise ValueError(f"crop must be 'gt_bbox' or 'none', got {o.crop!r}")

    # Nothing left inside the evaluated region is a result, not an error: distances are
    # undefined but the method scored zero on everything thresholded.
    empty = len(pred) == 0
    nan = torch.full((1,), float("nan"), dtype=geometry.POINT_DTYPE, device=pred.device)
    d_acc = geometry.nearest_distances(pred, gt_acc) if not empty else nan
    d_comp = (geometry.nearest_distances(gt_comp, pred) if not empty
              else nan.expand(len(gt_comp)))

    # Distances are reduced over the defined ones only, so a stray point cannot nan out the
    # headline number; when *none* is defined (the empty case above) the mean stays nan, because
    # "how far off was it" has no answer, while the thresholded scores below correctly read 0.
    acc, comp = stats.mean(d_acc), stats.mean(d_comp)
    row = {
        "Accuracy (m)": acc,
        "Accuracy median (m)": stats.median(d_acc),
        "Completeness (m)": comp,
        "Completeness median (m)": stats.median(d_comp),
        "Chamfer (m)": (acc + comp) / 2,
    }
    for tau in o.thresholds:
        precision = stats.fraction_below(d_acc, tau, default=0.0)
        recall = stats.fraction_below(d_comp, tau, default=0.0)
        f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        row[f"Precision @{tau}m"] = precision
        row[f"Recall @{tau}m"] = recall
        row[f"F-score @{tau}m"] = f
    row.update({
        "Pred points": len(pred),
        "Cropped %": 100.0 * (n_raw - len(pred)) / n_raw if n_raw else 0.0,
        "Cloud source": "fused" if _used_fused(case, o) else "unprojected",
    })
    return row


def _used_fused(case: EvalCase, o: CloudOptions) -> bool:
    return o.source == "fused" or (o.source == "auto" and case.fused_cloud_path is not None)
