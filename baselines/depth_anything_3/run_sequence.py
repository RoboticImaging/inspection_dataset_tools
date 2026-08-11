"""Run Depth Anything 3 on one sequence of an inspection asset and write depth + camera
predictions back into the asset folder (predictions/{method}/...), in the same format the
ground truth is stored in — so inspection_viz's Source toggle picks them up unchanged.

Usage (inside the container built by ../Dockerfile — see run_container.sh):
    python baselines/depth_anything_3/run_sequence.py --config baselines/configs/da3_glass_house.yaml
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines.common

from baselines.common.config import DatasetSelection, OutputSpec, load_config
from baselines.common.writer import (
    cloud_path, frame_stem, output_dirs, write_cam, write_cloud, write_depth,
)
from inspection_viz.camera import Camera
from inspection_viz.dataset import AssetDataset, Frame
from inspection_viz.io import image_size

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction


@dataclasses.dataclass
class ModelConfig:
    hf_repo: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    device: str = "cuda"
    process_res: int = 504
    # Frames per inference call. DA3 reasons jointly over everything it's given in one call, so
    # a chunk's poses/depths are only consistent with each other, not across chunks — fine for
    # short sequences (one chunk) or per-chunk viewing; for long sequences that need one globally
    # consistent trajectory, use DA3-Streaming instead (not wired up here).
    chunk_size: int = 32
    use_known_poses: bool = False   # feed gt cam params as conditioning instead of predicting them
    use_ray_pose: bool = False
    ref_view_strategy: str = "saddle_balanced"


@dataclasses.dataclass
class Config:
    dataset: DatasetSelection = dataclasses.field(default_factory=DatasetSelection)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    output: OutputSpec = dataclasses.field(default_factory=OutputSpec)


def _known_pose_arrays(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    exts, ints = [], []
    for f in frames:
        cam = f.camera()
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3] = np.linalg.inv(cam.c2w)[:3]
        exts.append(w2c)
        ints.append(cam.K.astype(np.float32))
    return np.stack(exts), np.stack(ints)


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Accepts (3,4) or (4,4) OpenCV world-to-camera; returns (4,4) camera-to-world."""
    if w2c.shape[0] == 3:
        w2c = np.vstack([w2c, [0, 0, 0, 1]])
    return np.linalg.inv(w2c)


def _align_chunk_to_gt_frame0(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """DA3 reasons in its own arbitrary world gauge when it isn't conditioned on known poses
    (world frame anchored to whichever frame `ref_view_strategy` picked as reference, at an
    unrecoverable scale) -- so raw predicted poses land nowhere near the gt trajectory even
    though the *shape* of the trajectory is right. Rigidly rotate+translate the whole chunk so
    its first frame's pose exactly matches the known gt pose for that frame; this only fixes
    position/orientation (6 of the 7 similarity-transform DOF) -- one frame can't disambiguate
    DA3's scale too, so depth/translations are left at DA3's own scale."""
    align = gt_c2w0 @ np.linalg.inv(c2ws[0])
    return align @ c2ws


def _written_camera(c2w: np.ndarray, K: np.ndarray, proc_width: int, width: int, height: int) -> Camera:
    """The camera exactly as `write_cam` stores it and `Camera.from_blender` reads it back --
    square pixels, centred principal point, intrinsics rescaled from DA3's process_res to the
    full frame. Building the cloud from *this* rather than DA3's raw K keeps the .ply identical
    to what the viewer draws from the same files."""
    fx = float(K[0, 0]) * width / proc_width
    K_full = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])
    return Camera(K_full, c2w, width, height)


def _subsample(pts: np.ndarray, cols: np.ndarray, max_points: int | None):
    if max_points is None or len(pts) <= max_points:
        return pts, cols
    idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
    return pts[idx], cols[idx]


def run_chunk(model: DepthAnything3, frames: list[Frame], cfg: ModelConfig) -> Prediction:
    ext = ints = None
    if cfg.use_known_poses:
        ext, ints = _known_pose_arrays(frames)
    return model.inference(
        [str(f.rgb_path) for f in frames],
        extrinsics=ext, intrinsics=ints,
        use_ray_pose=cfg.use_ray_pose, ref_view_strategy=cfg.ref_view_strategy,
        process_res=cfg.process_res,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config, Config)

    ds = AssetDataset(cfg.dataset.root)
    variant = cfg.dataset.variant()
    frames = cfg.dataset.select(ds.sequence(variant))  # gt sequence: rgb input (+ known poses if requested)
    if not frames:
        raise SystemExit(f"no frames selected for {variant} in {cfg.dataset.root}")

    depth_dir, cam_dir = output_dirs(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    ply_path = cloud_path(ds.root, ds.layout, cfg.output.method, variant.as_dict())
    if not cfg.output.overwrite and depth_dir.exists() and any(depth_dir.iterdir()):
        raise SystemExit(f"{depth_dir} already has output; set output.overwrite: true to replace")

    print(f"loading {cfg.model.hf_repo} on {cfg.model.device} ...")
    model = DepthAnything3.from_pretrained(cfg.model.hf_repo).to(cfg.model.device)
    model.device = cfg.model.device

    chunks = [frames[i:i + cfg.model.chunk_size] for i in range(0, len(frames), cfg.model.chunk_size)]
    print(f"{len(frames)} frames in {len(chunks)} chunk(s) of up to {cfg.model.chunk_size}")

    cloud_pts, cloud_cols = [], []
    for chunk in chunks:
        pred = run_chunk(model, chunk, cfg.model)
        proc_width = pred.processed_images.shape[2]
        c2ws = np.stack([_w2c_to_c2w(pred.extrinsics[j]) for j in range(len(chunk))])
        c2ws = _align_chunk_to_gt_frame0(c2ws, chunk[0].camera().c2w)
        for j, f in enumerate(chunk):
            width, height = image_size(f.rgb_path)
            depth = cv2.resize(pred.depth[j], (width, height), interpolation=cv2.INTER_LINEAR)
            if cfg.output.min_confidence is not None:
                if pred.conf is None:
                    raise SystemExit("output.min_confidence set but the model produced no confidence map")
                conf = cv2.resize(pred.conf[j], (width, height), interpolation=cv2.INTER_LINEAR)
                depth = np.where(conf >= cfg.output.min_confidence, depth, np.nan).astype(np.float32)
            stem = frame_stem(f.index)
            write_depth(depth_dir / f"{stem}.exr", depth)
            write_cam(cam_dir / f"{stem}.json", c2ws[j], pred.intrinsics[j], width=proc_width)
            if cfg.output.cloud:
                cam = _written_camera(c2ws[j], pred.intrinsics[j], proc_width, width, height)
                pts, cols = cam.unproject(depth, rgb=f.rgb(), stride=cfg.output.cloud_stride)
                cloud_pts.append(pts)
                cloud_cols.append(cols)
        print(f"  wrote frames {chunk[0].index}..{chunk[-1].index}")

    if cfg.output.cloud:
        pts, cols = _subsample(np.concatenate(cloud_pts), np.concatenate(cloud_cols),
                               cfg.output.cloud_max_points)
        write_cloud(ply_path, pts, cols)
        print(f"  wrote {len(pts):,} points -> {ply_path.relative_to(ds.root)}")

    print(f"done: predictions/{cfg.output.method}/ under {ds.root}")


if __name__ == "__main__":
    main()
