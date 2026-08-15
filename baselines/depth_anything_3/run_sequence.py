"""Run Depth Anything 3 on the sequences of an inspection asset and write depth + camera
predictions back into the asset folder (predictions/{method}/...), in the same format the
ground truth is stored in — so inspection_viz's Source toggle picks them up unchanged.

Usage — inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/da3.yaml
    python run_sequence.py ../configs/da3_glass_house.yaml      # both halves in one file

The data config picks one sequence of the asset or all of them (see configs/data/); the model
config holds DA3's own knobs and is reusable across assets. Sequences run one after another
through a single loaded model.

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/depth-anything-3 install.
"""

import dataclasses
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

from depth_anything_3.api import DepthAnything3 as DA3Model  # aliased: the Baseline below owns the name
from depth_anything_3.specs import Prediction


@dataclasses.dataclass
class ModelConfig:
    hf_repo: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    device: str = "cuda"
    process_res: int = 504
    use_known_poses: bool = False   # feed gt cam params as conditioning instead of predicting them
    use_ray_pose: bool = False
    ref_view_strategy: str = "saddle_balanced"


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


class DepthAnything3(Baseline):
    """Depth Anything 3, one joint inference call per sequence."""

    name = "da3"
    Config = ModelConfig

    def setup(self) -> None:
        cfg = self.config
        print(f"loading {cfg.hf_repo} on {cfg.device} ...")
        self.model = DA3Model.from_pretrained(cfg.hf_repo).to(cfg.device)
        self.model.device = cfg.device

    def infer(self, frames: list[Frame], export_dir: Path | None) -> Prediction:
        """One call over the whole sequence -- DA3 reasons jointly over everything it's given,
        which is the only way its poses/depths come out mutually consistent. Long sequences will
        simply run out of GPU memory; there is no chunking fallback, because per-chunk predictions
        live in unrelated world gauges and scales.

        `export_dir` (set when we want a cloud) also gets DA3's own glb export, every parameter
        left at its default except `export_depth_vis`, so the only thing written is scene.glb --
        the depth_vis jpgs and the scene.jpg thumbnail copied from them are pure visualisation
        that nothing here reads. `_cloud_from_glb` turns that scene.glb into the .ply the viewer
        reads, and `run` deletes the whole export dir right after -- it is a bulky intermediate.
        """
        cfg = self.config
        ext = ints = None
        if cfg.use_known_poses:
            ext, ints = _known_pose_arrays(frames)
        return self.model.inference(
            [str(f.rgb_path) for f in frames],
            extrinsics=ext, intrinsics=ints,
            use_ray_pose=cfg.use_ray_pose, ref_view_strategy=cfg.ref_view_strategy,
            process_res=cfg.process_res,
            export_dir=None if export_dir is None else str(export_dir), export_format="glb",
            export_kwargs={"glb": {"export_depth_vis": False}},
        )

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        export_dir = out.scratch_dir("da3_export") if out.spec.cloud else None
        pred = self.infer(frames, export_dir)
        proc_width = pred.processed_images.shape[2]
        c2ws = np.stack([_w2c_to_c2w(pred.extrinsics[j]) for j in range(len(frames))])
        align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform per run
        c2ws = align @ c2ws

        for j, f in enumerate(frames):
            width, height = image_size(f.rgb_path)
            depth = cv2.resize(pred.depth[j], (width, height), interpolation=cv2.INTER_LINEAR)
            conf = None if pred.conf is None or out.spec.min_confidence is None else \
                cv2.resize(pred.conf[j], (width, height), interpolation=cv2.INTER_LINEAR)
            out.write_frame(f.index, depth, c2ws[j], pred.intrinsics[j],
                            intrinsics_width=proc_width, conf=conf)

        if out.spec.cloud:
            pts, cols = _cloud_from_glb(export_dir / "scene.glb", align)
            out.write_cloud(pts, cols)
            shutil.rmtree(export_dir)  # scratch: the .ply carries everything the viewer reads
            print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")


if __name__ == "__main__":
    cli_main(DepthAnything3, description=__doc__)
