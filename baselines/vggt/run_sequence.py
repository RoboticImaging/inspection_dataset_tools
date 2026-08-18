"""Run VGGT on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in — so inspection_viz's Source toggle picks them up unchanged.

This is plain feed-forward VGGT (facebookresearch/vggt), not VGGT-SLAM's submap/loop-closure
pipeline — one joint inference call over the whole given sequence, same shape as
baselines/depth_anything_3/run_sequence.py.

Usage — inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/vggt.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/vggt install.
"""

import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

from vggt.models.vggt import VGGT as VGGTModel  # aliased: the Baseline below owns the name
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


@dataclasses.dataclass
class ModelConfig:
    device: str = "cuda"
    checkpoint_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Accepts (3,4) or (4,4) OpenCV world-to-camera; returns (4,4) camera-to-world."""
    if w2c.shape[0] == 3:
        w2c = np.vstack([w2c, [0, 0, 0, 1]])
    return np.linalg.inv(w2c)


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts VGGT's world gauge onto the gt one, as far as one
    frame can. VGGT reasons in its own arbitrary gauge (world frame anchored to camera 0, at an
    unrecoverable scale) -- so raw predictions land nowhere near the gt trajectory even though
    the *shape* of the trajectory is right. This is the relative transform between the gt and
    predicted pose of frame 0, so applying it lands frame 0 exactly on its gt pose; it only fixes
    position/orientation (6 of the 7 similarity-transform DOF) -- one frame can't disambiguate
    VGGT's scale too, so depth and inter-frame translations stay at VGGT's own scale. Same
    rationale as depth_anything_3/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_vggt_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                    vggt_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a VGGT 3x3 intrinsic matrix -- measured against the (width, height) VGGT actually
    processed -- back to a frame's original (width, height), so it's expressed in the same pixel
    grid as the depth map written alongside it (which is resized to orig_image_size below).

    `write_cam` only ever stores the scale-invariant ratio fx/width, so this is a no-op for the
    written cam_params json as long as `width` is passed consistently with whatever scale K is
    in -- but keeping K itself in real, original-pixel units avoids a footgun for anything that
    reads K directly instead of through write_cam, and matches
    baselines/VGGT-SLAM/run_sequence.py's own version of this function.

    VGGT's default "crop" preprocessing (load_and_preprocess_images's default `mode`) uniformly
    scales width to 518px and only center-crops height when the source is taller than it is wide;
    for the landscape video these baselines target, that crop never fires, so a uniform scale
    recovered from the *actual* processed shape is exact.
    """
    w_orig, h_orig = orig_image_size
    w_vggt, h_vggt = vggt_image_size
    scale_x, scale_y = w_orig / w_vggt, h_orig / h_vggt

    K_orig = K_pred.copy()
    K_orig[0, 0] *= scale_x  # fx
    K_orig[1, 1] *= scale_y  # fy
    K_orig[0, 2] *= scale_x  # cx
    K_orig[1, 2] *= scale_y  # cy
    return K_orig


def run_inference(model: VGGTModel, frames: list[Frame], device: str) -> dict:
    """One joint feed-forward call over the whole sequence -- VGGT reasons across all given
    frames at once, which is the only way its poses/depths come out mutually consistent. Long
    sequences will simply run out of GPU memory; there is no chunking fallback, because
    per-chunk predictions would live in unrelated world gauges and scales (same caveat as DA3).

    Mirrors vggt_slam.solver.Solver.run_predictions's own call into the model (`model(images)`,
    pose_encoding_to_extri_intri, then squeeze the batch dim off every tensor), minus everything
    submap/loop-closure specific.
    """
    images = load_and_preprocess_images([str(f.rgb_path) for f in frames]).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"], predictions["intrinsic"] = extrinsic, intrinsic
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            predictions[key] = val.float().cpu().numpy().squeeze(0)  # drop the batch dim
    return predictions


class VGGT(Baseline):
    """Plain feed-forward VGGT, one joint inference call per sequence."""

    name = "vggt"
    Config = ModelConfig

    def setup(self) -> None:
        cfg = self.config
        print(f"loading VGGT on {cfg.device} ...")
        self.model = VGGTModel()
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(cfg.checkpoint_url))
        self.model.eval().to(cfg.device)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        predictions = run_inference(self.model, frames, self.config.device)

        # VGGT's own processing (width, height) -- see load_and_preprocess_images -- as opposed to
        # each frame's original size below.
        proc_image_size = (predictions["images"].shape[-1], predictions["images"].shape[-2])
        c2ws = np.stack([_w2c_to_c2w(predictions["extrinsic"][j]) for j in range(len(frames))])
        align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
        c2ws = align @ c2ws

        if out.spec.cloud:
            # Unprojected once over the whole batch, at VGGT's own processing resolution -- each
            # frame's extrinsic/intrinsic are only meaningful together at that resolution, before
            # the per-frame depth gets resized back to its own original size below.
            world_pts = unproject_depth_map_to_point_map(
                predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
            )  # (S, h, w, 3), VGGT's own world gauge
            world_pts = trimesh.transform_points(world_pts.reshape(-1, 3), align)
            cols = predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3)  # (S,3,h,w) -> (S*h*w,3), [0,1]
            cols = (cols * 255).clip(0, 255).astype(np.uint8)
            cloud_mask = np.isfinite(world_pts).all(axis=1)

            if out.spec.min_confidence is not None:
                # A percentile (0-100) of lowest-confidence points to drop from the cloud, not an
                # absolute cutoff -- raw confidence isn't comparable across runs/models the way an
                # absolute probability is. This only ever filters the cloud, not per-pixel depth
                # (unlike SequenceOutput.write_frame's own min_confidence, which is an absolute
                # cutoff on depth -- not used here, so depth is never nan-masked by confidence).
                confs = predictions["depth_conf"].reshape(-1)
                conf_thresh_absolute = np.percentile(confs, out.spec.min_confidence)
                cloud_mask &= (confs >= conf_thresh_absolute) & (confs >= 1e-5)

        for j, f in enumerate(frames):
            width, height = image_size(f.rgb_path)
            depth = cv2.resize(predictions["depth"][j].squeeze(-1), (width, height), interpolation=cv2.INTER_LINEAR)
            K = rescale_vggt_intrinsics_from_K(predictions["intrinsic"][j], (width, height), proc_image_size)
            out.write_frame(f.index, depth, c2ws[j], K, intrinsics_width=width)

        if out.spec.cloud:
            out.write_cloud(world_pts[cloud_mask], cols[cloud_mask])
            print(f"  wrote {cloud_mask.sum():,} points -> {out.cloud_path.relative_to(out.root)}")


if __name__ == "__main__":
    cli_main(VGGT, description=__doc__)
