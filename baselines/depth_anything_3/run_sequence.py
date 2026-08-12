"""Run Depth Anything 3 on one sequence of an inspection asset and write depth + camera
predictions back into the asset folder (predictions/{method}/...), in the same format the
ground truth is stored in — so inspection_viz's Source toggle picks them up unchanged.

Usage — inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py ../configs/da3_glass_house.yaml

Paths are the same inside the container as on the host, so an absolute config path works too.
Do not run this outside the container: it needs the image's torch/depth-anything-3 install.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines.common

from baselines.common.config import DatasetSelection, OutputSpec, load_config
from baselines.common.writer import (
    cloud_path, frame_stem, output_dirs, write_cam, write_cloud, write_depth,
)
from inspection_viz.dataset import AssetDataset, Frame
from inspection_viz.io import image_size

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction


@dataclasses.dataclass
class ModelConfig:
    hf_repo: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    device: str = "cuda"
    process_res: int = 504
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


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts DA3's world gauge onto the gt one, as far as one frame
    can. DA3 reasons in its own arbitrary gauge when it isn't conditioned on known poses (world
    frame anchored to whichever frame `ref_view_strategy` picked as reference, at an unrecoverable
    scale) -- so raw predictions land nowhere near the gt trajectory even though the *shape* of the
    trajectory is right. This is the relative transform between the gt and predicted pose of frame
    0, so applying it lands frame 0 exactly on its gt pose; it only fixes position/orientation
    (6 of the 7 similarity-transform DOF) -- one frame can't disambiguate DA3's scale too, so
    depth and inter-frame translations stay at DA3's own scale."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def _cloud_from_glb(glb_path: Path, align: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pull xyz+rgb back out of the scene.glb DA3's own exporter wrote, in the gt world frame.

    Asking `inference` for `export_format="glb"` is the one export that runs DA3's real point-cloud
    post-processing -- adaptive confidence threshold, back-projection at process_res, non-finite
    filter, random cap -- all at its own defaults, so we don't reimplement any of it. What it
    writes is a .glb in glTF gauge, though: `export_to_glb` re-centres the scene on the point
    median and rotates it into the first camera's frame. It stashes that exact 4x4 in the scene
    metadata as `hf_alignment`, and trimesh round-trips it through glTF extras losslessly, so
    undoing it recovers DA3's own world gauge; `align` then moves the cloud onto the gt one,
    exactly as it moved the poses. Camera wireframes ride along in the same file as Path3D
    geometry and are skipped here.
    """
    scene = trimesh.load(glb_path, force="scene")
    to_world = align @ np.linalg.inv(np.asarray(scene.metadata["hf_alignment"], dtype=np.float64))
    pts, cols = [], []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.points.PointCloud):
            continue
        node = scene.graph.get(name)[0]  # scene-graph placement, identity in practice
        pts.append(trimesh.transform_points(geom.vertices, to_world @ node))
        cols.append(np.asarray(geom.colors)[:, :3])
    if not pts:
        raise SystemExit(f"{glb_path} has no point cloud -- did every pixel fail the conf filter?")
    return np.concatenate(pts), np.concatenate(cols)


def run_inference(model: DepthAnything3, frames: list[Frame], cfg: ModelConfig,
                  export_dir: Path | None) -> Prediction:
    """One call over the whole sequence -- DA3 reasons jointly over everything it's given, which
    is the only way its poses/depths come out mutually consistent. Long sequences will simply run
    out of GPU memory; there is no chunking fallback, because per-chunk predictions live in
    unrelated world gauges and scales.

    `export_dir` (set when we want a cloud) also gets DA3's own glb export, every parameter left at
    its default: scene.glb + the depth_vis jpgs that come with it. `_cloud_from_glb` turns that
    scene.glb into the .ply the viewer reads, and `main` deletes it right after -- it is a bulky
    intermediate, and nothing downstream reads it."""
    ext = ints = None
    if cfg.use_known_poses:
        ext, ints = _known_pose_arrays(frames)
    return model.inference(
        [str(f.rgb_path) for f in frames],
        extrinsics=ext, intrinsics=ints,
        use_ray_pose=cfg.use_ray_pose, ref_view_strategy=cfg.ref_view_strategy,
        process_res=cfg.process_res,
        export_dir=None if export_dir is None else str(export_dir), export_format="glb",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="path to a run config yaml, e.g. ../configs/da3_glass_house.yaml")
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

    print(f"running {len(frames)} frames in a single inference call")
    export_dir = ply_path.parent / f"{ply_path.stem}_da3_export" if cfg.output.cloud else None
    pred = run_inference(model, frames, cfg.model, export_dir)
    proc_width = pred.processed_images.shape[2]
    c2ws = np.stack([_w2c_to_c2w(pred.extrinsics[j]) for j in range(len(frames))])
    align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
    c2ws = align @ c2ws

    for j, f in enumerate(frames):
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
    print(f"  wrote frames {frames[0].index}..{frames[-1].index}")

    if cfg.output.cloud:
        glb_path = export_dir / "scene.glb"
        pts, cols = _cloud_from_glb(glb_path, align)
        write_cloud(ply_path, pts, cols)
        glb_path.unlink()  # scratch: the .ply carries everything the viewer reads
        print(f"  wrote {len(pts):,} points -> {ply_path.relative_to(ds.root)}")

    print(f"done: predictions/{cfg.output.method}/ under {ds.root}")


if __name__ == "__main__":
    main()
