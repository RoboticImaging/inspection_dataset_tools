"""Run Pi3 on one sequence of an inspection asset and write depth + camera predictions back into
the asset folder (predictions/{method}/...), in the same format the ground truth is stored in --
so inspection_viz's Source toggle picks them up unchanged.

Pi3 (yyfz/Pi3) is plain feed-forward, like VGGT -- one joint inference call over the whole given
sequence returns per-frame depth, camera-to-world pose and confidence that are already mutually
consistent, so this follows the exact same shape as baselines/vggt/run_sequence.py (not DA3's,
which instead calls into the model's own glb exporter).

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py ../configs/pi3/pi3-example-config.yaml

Paths are the same inside the container as on the host, so an absolute config path works too.
Do not run this outside the container: it needs the image's torch/pi3 install.
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines.common

from baselines.common.config import DatasetSelection, OutputSpec, load_config
from baselines.common.writer import (
    cloud_path, frame_stem, output_dirs, write_cam, write_cloud, write_depth,
)
from inspection_viz.dataset import AssetDataset, Frame
from inspection_viz.io import image_size

from pi3.models.pi3 import Pi3
from pi3.utils.geometry import recover_intrinsic_from_rays_d


@dataclasses.dataclass
class ModelConfig:
    device: str = "cuda"
    hf_repo: str = "yyfz233/Pi3"     # from_pretrained id, matches Pi3's own example.py; ignored if ckpt is set
    ckpt: str | None = None          # local .pt/.safetensors checkpoint, overrides hf_repo
    # Total-pixel budget frames get resized into before inference -- Pi3's own default (see
    # pi3.utils.basic.load_images_as_tensor). Frames are resized to the largest (w, h) that are
    # both multiples of the model's 14px patch size and fit this budget, preserving aspect ratio.
    pixel_limit: int = 255_000


@dataclasses.dataclass
class Config:
    dataset: DatasetSelection = dataclasses.field(default_factory=DatasetSelection)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    output: OutputSpec = dataclasses.field(default_factory=OutputSpec)


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts Pi3's world gauge onto the gt one, as far as one frame
    can. Pi3 reasons in its own arbitrary gauge (world frame anchored to camera 0, at an
    unrecoverable scale) -- so raw predictions land nowhere near the gt trajectory even though the
    *shape* of the trajectory is right. This is the relative transform between the gt and predicted
    pose of frame 0, so applying it lands frame 0 exactly on its gt pose; it only fixes
    position/orientation (6 of the 7 similarity-transform DOF) -- one frame can't disambiguate
    Pi3's scale too, so depth and inter-frame translations stay at Pi3's own scale. Same rationale
    as baselines/vggt/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_pi3_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                   proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a Pi3 3x3 intrinsic matrix -- recovered at the (width, height) Pi3 actually
    processed -- back to a frame's original (width, height), so it's expressed in the same pixel
    grid as the depth map written alongside it (which is resized to orig_image_size below).

    Same rationale and same uniform-scale math as
    baselines/vggt/run_sequence.py's rescale_vggt_intrinsics_from_K: `write_cam` only ever stores
    the scale-invariant ratio fx/width, so this is a no-op for the written cam_params json as long
    as `width` is passed consistently with whatever scale K is in -- but keeping K itself in real,
    original-pixel units avoids a footgun for anything that reads K directly. Pi3's loader
    (load_frames_as_tensor below) always resizes to a uniform target size derived from the whole
    frame (no cropping), so a uniform scale recovered from the actual processed shape is exact.
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
    size shared by the whole batch.

    Same resize-to-a-pixel-budget-multiple-of-14 algorithm as Pi3's own
    pi3.utils.basic.load_images_as_tensor (sized off the first frame, then shrunk one 14px step at
    a time off whichever axis is furthest from budget until under `pixel_limit`), just driven by
    our own already-selected frame list instead of a directory glob -- so
    dataset.frame_start/frame_end/frame_stride selection is respected instead of Pi3's own
    every-file-in-a-dir + interval sampling.
    """
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


def load_pi3_model(cfg: ModelConfig) -> Pi3:
    if cfg.ckpt is None:
        return Pi3.from_pretrained(cfg.hf_repo)
    model = Pi3()
    if cfg.ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(cfg.ckpt)
    else:
        weight = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(weight)
    return model


def run_inference(model: Pi3, frames: list[Frame], device: str, pixel_limit: int) -> dict:
    """One joint feed-forward call over the whole sequence -- Pi3 reasons across all given frames
    at once, which is the only way its poses/depths come out mutually consistent. Long sequences
    will simply run out of GPU memory; there is no chunking fallback, because per-chunk predictions
    would live in unrelated world gauges and scales (same caveat as VGGT/DA3).

    Mirrors Pi3's own example.py call into the model (`model(imgs[None])`, add then drop the batch
    dim), plus recovering per-frame intrinsics -- Pi3 doesn't predict K directly, but `local_points`
    (its per-pixel camera-space points, xy already scaled by z) doubles exactly as an unnormalized
    ray direction map, so pi3.utils.geometry.recover_intrinsic_from_rays_d fits a pinhole K from it.
    """
    imgs = load_frames_as_tensor(frames, pixel_limit).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(imgs[None])  # add the batch dim
    with torch.no_grad():
        predictions["intrinsic"] = recover_intrinsic_from_rays_d(
            predictions["local_points"].float(), force_center_principal_point=True
        )
    predictions["images"] = imgs[None]
    for key, val in predictions.items():
        predictions[key] = val.float().cpu().numpy().squeeze(0)  # drop the batch dim
    return predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="path to a run config yaml, e.g. ../configs/pi3/pi3-example-config.yaml")
    args = p.parse_args()
    cfg = load_config(args.config, Config)

    ds = AssetDataset(cfg.dataset.root)
    variant = cfg.dataset.variant()
    frames = cfg.dataset.select(ds.sequence(variant))  # gt sequence: rgb input
    if not frames:
        raise SystemExit(f"no frames selected for {variant} in {cfg.dataset.root}")

    depth_dir, cam_dir = output_dirs(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    ply_path = cloud_path(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    if not cfg.output.overwrite and depth_dir.exists() and any(depth_dir.iterdir()):
        raise SystemExit(f"{depth_dir} already has output; set output.overwrite: true to replace")

    print(f"loading Pi3 on {cfg.model.device} ...")
    model = load_pi3_model(cfg.model)
    model.eval().to(cfg.model.device)

    print(f"running {len(frames)} frames in a single inference call")
    predictions = run_inference(model, frames, cfg.model.device, cfg.model.pixel_limit)

    # Pi3's own processing (width, height) -- see load_frames_as_tensor -- as opposed to each
    # frame's original size below.
    proc_image_size = (predictions["images"].shape[-1], predictions["images"].shape[-2])
    c2ws = predictions["camera_poses"]  # Pi3 predicts camera-to-world directly, OpenCV convention
    align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
    c2ws = align @ c2ws
    conf = 1 / (1 + np.exp(-predictions["conf"]))  # conf_head outputs a logit -- see pi3/models/pi3.py

    if cfg.output.cloud:
        # Pi3 hands back world-space points directly (camera_poses already applied), in its own
        # world gauge -- align moves them onto the gt one, same as it moves the poses.
        world_pts = trimesh.transform_points(predictions["points"].reshape(-1, 3), align)
        cols = predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3)  # (S, 3, h, w) -> (S*h*w, 3), [0,1]
        cols = (cols * 255).clip(0, 255).astype(np.uint8)
        cloud_mask = np.isfinite(world_pts).all(axis=1)
        if cfg.output.min_confidence is not None:
            # min_confidence here is a percentile (0-100) of lowest-confidence points to drop, not
            # an absolute cutoff -- matches baselines/vggt/run_sequence.py's own point-cloud
            # confidence filter (percentile relative to this run's own conf distribution, since raw
            # confidence isn't comparable across runs/models the way an absolute probability is).
            confs = conf.reshape(-1)
            conf_thresh_absolute = np.percentile(confs, cfg.output.min_confidence)
            cloud_mask &= (confs >= conf_thresh_absolute) & (confs >= 1e-5)

    for j, f in enumerate(frames):
        width, height = image_size(f.rgb_path)
        depth = cv2.resize(predictions["local_points"][j, ..., 2], (width, height), interpolation=cv2.INTER_LINEAR)
        K = rescale_pi3_intrinsics_from_K(predictions["intrinsic"][j], (width, height), proc_image_size)
        stem = frame_stem(f.index)
        write_depth(depth_dir / f"{stem}.exr", depth)
        write_cam(cam_dir / f"{stem}.json", c2ws[j], K, width=width)
    print(f"  wrote frames {frames[0].index}..{frames[-1].index}")

    if cfg.output.cloud:
        write_cloud(ply_path, world_pts[cloud_mask], cols[cloud_mask])
        print(f"  wrote {cloud_mask.sum():,} points -> {ply_path.relative_to(ds.root)}")

    print(f"done: predictions/{cfg.output.method}/ under {ds.root}")


if __name__ == "__main__":
    main()
