"""Run Pi3X on one sequence of an inspection asset and write depth + camera predictions back into
the asset folder (predictions/{method}/...), in the same format the ground truth is stored in --
so inspection_viz's Source toggle picks them up unchanged.

Pi3X (yyfz/Pi3, same submodule as ../Pi3 -- see baselines/Pi3X/Dockerfile's header) is Pi3's
multimodal, metric-scale successor, driven through its own chunked VO pipeline
(pi3.pipe.pi3x_vo.Pi3XVO) instead of one joint call over the whole sequence: frames are processed
`chunk_size` at a time with `overlap` frames shared between consecutive chunks, each chunk aligned
onto the last via a masked Sim(3) fit -- the same shape as VGGT-Long's chunking (see
baselines/VGGT-Long/run_sequence.py), just done in-process in plain PyTorch, with no subprocess,
no disk-backed intermediates and no separate weights-download step. Unlike Pi3, Pi3X's own output
is already metric-scale (camera_poses/points have real-world scale baked in via its `metric`
head), though this baseline still only fixes gt's position/orientation, not scale -- see
_gt_frame0_alignment.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/pi3x.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/pi3 install.
"""

import dataclasses
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

from pi3.models.pi3x import Pi3X as Pi3XModel  # aliased: the Baseline below owns the name
from pi3.pipe.pi3x_vo import Pi3XVO
from pi3.utils.geometry import recover_intrinsic_from_rays_d


@dataclasses.dataclass
class ModelConfig:
    device: str = "cuda"
    hf_repo: str = "yyfz233/Pi3X"    # from_pretrained id, matches Pi3's own example_vo.py; ignored if ckpt is set
    ckpt: str | None = None          # local .pt/.safetensors checkpoint, overrides hf_repo
    # Total-pixel budget frames get resized into before inference -- Pi3's own default (see
    # pi3.utils.basic.load_images_as_tensor/load_multimodal_data). Frames are resized to the
    # largest (w, h) that are both multiples of the model's 14px patch size and fit this budget,
    # preserving aspect ratio.
    pixel_limit: int = 255_000
    # Pi3XVO's own chunking knobs (pi3.pipe.pi3x_vo.Pi3XVO.__call__'s own defaults) -- how many
    # frames go through the model at once, and how many of those overlap with the next chunk to
    # give the Sim(3) alignment fit something to align against.
    chunk_size: int = 16
    overlap: int = 6
    # Confidence floor Pi3XVO uses internally to decide which points are trustworthy enough to
    # align consecutive chunks against -- separate from output.min_confidence below, which is our
    # own point-cloud export filter applied after the fact.
    conf_thre: float = 0.05


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts Pi3X's world gauge onto the gt one, as far as one frame
    can. Pi3X reasons in its own arbitrary gauge (world frame anchored to camera 0 of the first
    chunk) -- so raw predictions land nowhere near the gt trajectory even though the *shape* of the
    trajectory is right. This is the relative transform between the gt and predicted pose of frame
    0, so applying it lands frame 0 exactly on its gt pose; it only fixes position/orientation (6
    of the 7 similarity-transform DOF). Unlike Pi3, Pi3X predicts an already-metric scale (its
    `metric` head), so in principle that 7th DOF doesn't need fixing here the way it does for
    Pi3/VGGT -- this baseline doesn't yet assume or verify that, though, and still only applies the
    6-DOF fix, same rationale as baselines/vggt/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_pi3x_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                    proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a Pi3X 3x3 intrinsic matrix -- recovered at the (width, height) Pi3X actually
    processed -- back to a frame's original (width, height), so it's expressed in the same pixel
    grid as the depth map written alongside it. Same rationale and math as
    baselines/Pi3/run_sequence.py's rescale_pi3_intrinsics_from_K.
    """
    w_orig, h_orig = orig_image_size
    w_proc, h_proc = proc_image_size
    scale_x, scale_y = w_orig / w_proc, h_orig / h_proc

    K_orig = K_pred.copy()
    K_orig[0, 0] *= scale_x  # fx
    K_orig[1, 1] *= scale_y  # fy
    K_orig[0, 2] *= scale_x  # cx
    K_orig[1, 2] *= scale_y  # cy
    return K_orig


def load_frames_as_tensor(frames: list[Frame], pixel_limit: int) -> torch.Tensor:
    """Loads `frames`' rgb into a single (N, 3, H, W) [0, 1] tensor, all resized to one target
    size shared by the whole batch. Same resize-to-a-pixel-budget-multiple-of-14 algorithm as
    Pi3's own pi3.utils.basic loaders, just driven by our own already-selected frame list instead
    of a directory glob -- see baselines/Pi3/run_sequence.py's version of this function (verbatim
    copy: Pi3X shares the same image-loading convention as plain Pi3)."""
    images = [Image.open(f.rgb_path).convert("RGB") for f in frames]
    w_orig, h_orig = images[0].size
    scale = math.sqrt(pixel_limit / (w_orig * h_orig))
    k, m = round(w_orig * scale / 14), round(h_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_orig / h_orig:
            k -= 1
        else:
            m -= 1
    target_w, target_h = max(1, k) * 14, max(1, m) * 14

    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(img.resize((target_w, target_h), Image.Resampling.LANCZOS)) for img in images]
    return torch.stack(tensors, dim=0)


def load_pi3x_model(cfg: ModelConfig) -> Pi3XModel:
    if cfg.ckpt is None:
        return Pi3XModel.from_pretrained(cfg.hf_repo)
    model = Pi3XModel()
    if cfg.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(cfg.ckpt)
    else:
        weight = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(weight, strict=False)  # matches Pi3's own example_vo.py
    return model


def run_inference(model: Pi3XModel, frames: list[Frame], cfg: ModelConfig) -> dict:
    """Runs the whole sequence through Pi3XVO's own chunk/align/stitch pipeline (see module
    docstring) -- it manages its own torch.no_grad()/autocast internally, so this doesn't need to.

    Pi3XVO's merged output only carries 'points' (world-space), 'camera_poses' (c2w) and 'conf'
    (already sigmoid-applied, unlike plain Pi3's raw logit -- see Pi3XVO.__call__) -- no per-frame
    camera-space depth or intrinsics, since those only ever exist transiently per-chunk inside the
    pipe. Both are reconstructed here instead, straight from what *is* returned: transforming each
    frame's world points into its own camera frame via that frame's own (inverse) camera_poses
    recovers exactly the camera-space (x*z, y*z, z) points Pi3's own `local_points` would have
    been -- so depth is just its z, and it doubles as a ray-direction map for
    recover_intrinsic_from_rays_d, the same trick baselines/Pi3/run_sequence.py uses on the real
    local_points.
    """
    imgs = load_frames_as_tensor(frames, cfg.pixel_limit).to(cfg.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    pipe = Pi3XVO(model)
    pred = pipe(imgs[None], chunk_size=cfg.chunk_size, overlap=cfg.overlap,
                conf_thre=cfg.conf_thre, dtype=dtype)

    n_frames = pred["points"].shape[1]
    if n_frames != len(frames):
        raise SystemExit(
            f"Pi3XVO merged {n_frames} frames, expected {len(frames)} -- its last chunk is "
            f"dropped whenever it would be <= overlap ({cfg.overlap}) frames long; try a "
            "different chunk_size/overlap, or a frame_end that avoids the short remainder"
        )

    with torch.no_grad():
        w2cs = torch.linalg.inv(pred["camera_poses"][0])  # (N, 4, 4)
        points_h = torch.cat([pred["points"][0], torch.ones_like(pred["points"][0, ..., :1])], dim=-1)
        local_points = torch.einsum("nij,nhwj->nhwi", w2cs, points_h)[..., :3]  # (N, H, W, 3), camera-space
        intrinsic = recover_intrinsic_from_rays_d(local_points, force_center_principal_point=True)

    return {
        "points": pred["points"][0],
        "camera_poses": pred["camera_poses"][0],
        "conf": pred["conf"][0],
        "depth": local_points[..., 2],
        "intrinsic": intrinsic,
        "images": imgs,
    }


class Pi3X(Baseline):
    """Pi3X, driven through its own chunked VO pipeline (see module docstring)."""

    name = "pi3x"
    Config = ModelConfig

    def setup(self) -> None:
        cfg = self.config
        print(f"loading Pi3X on {cfg.device} ...")
        self.model = load_pi3x_model(cfg)
        self.model.eval().to(cfg.device)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        print(f"running {len(frames)} frames (chunk_size={cfg.chunk_size}, overlap={cfg.overlap}) ...")
        predictions = run_inference(self.model, frames, cfg)
        for key, val in predictions.items():
            predictions[key] = val.float().cpu().numpy()

        # Pi3X's own processing (width, height) -- see load_frames_as_tensor -- as opposed to each
        # frame's original size below.
        proc_image_size = (predictions["images"].shape[-1], predictions["images"].shape[-2])
        c2ws = predictions["camera_poses"]  # Pi3X predicts camera-to-world directly, OpenCV convention
        align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
        c2ws = align @ c2ws
        conf = predictions["conf"]  # already a probability -- Pi3XVO applies sigmoid itself, unlike plain Pi3

        if out.spec.cloud:
            # Pi3X hands back world-space points directly (camera_poses already applied), in its
            # own world gauge -- align moves them onto the gt one, same as it moves the poses.
            world_pts = trimesh.transform_points(predictions["points"].reshape(-1, 3), align)
            cols = predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3)  # (S,3,h,w) -> (S*h*w,3), [0,1]
            cols = (cols * 255).clip(0, 255).astype(np.uint8)
            cloud_mask = np.isfinite(world_pts).all(axis=1)
            if out.spec.min_confidence is not None:
                # A percentile (0-100) of lowest-confidence points to drop, not an absolute cutoff
                # -- matches baselines/vggt/run_sequence.py's own point-cloud confidence filter
                # (percentile relative to this run's own conf distribution, since raw confidence
                # isn't comparable across runs/models the way an absolute probability is). Only
                # ever filters the cloud, not per-pixel depth (unlike SequenceOutput.write_frame's
                # own min_confidence, which is an absolute cutoff -- not used here).
                confs = conf.reshape(-1)
                conf_thresh_absolute = np.percentile(confs, out.spec.min_confidence)
                cloud_mask &= (confs >= conf_thresh_absolute) & (confs >= 1e-5)

        for j, f in enumerate(frames):
            width, height = image_size(f.rgb_path)
            depth = cv2.resize(predictions["depth"][j], (width, height), interpolation=cv2.INTER_LINEAR)
            K = rescale_pi3x_intrinsics_from_K(predictions["intrinsic"][j], (width, height), proc_image_size)
            out.write_frame(f.index, depth, c2ws[j], K, intrinsics_width=width)

        if out.spec.cloud:
            out.write_cloud(world_pts[cloud_mask], cols[cloud_mask])
            print(f"  wrote {cloud_mask.sum():,} points -> {out.cloud_path.relative_to(out.root)}")


if __name__ == "__main__":
    cli_main(Pi3X, description=__doc__)