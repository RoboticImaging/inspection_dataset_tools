"""Run DA3-Streaming on one sequence of an inspection asset and write depth + camera predictions
back into the asset folder (predictions/{method}/...), in the same format the ground truth is
stored in -- so inspection_viz's Source toggle picks them up unchanged.

DA3-Streaming (depth-anything-3/da3_streaming) chunks a sequence, runs each chunk through Depth
Anything 3, and stitches chunks together with weighted alignment and optional SALAD loop closure
-- the same shape as VGGT-Long (see baselines/VGGT-Long/run_sequence.py), just applied to DA3
instead of VGGT/Pi3/MapAnything. Its own CLI (`da3_streaming.py`) resolves weight paths (e.g.
`./weights/model.safetensors`) relative to cwd and isn't packaged to be imported as a library, so
this drives it exactly as documented -- a subprocess, run from inside the da3_streaming submodule,
given its own `--output_dir` (unlike VGGT-Long's CLI, no need to guess where it wrote things) --
then re-exports its outputs (camera_poses.txt for poses, the per-frame depth/conf/intrinsics it
writes to results_output/ whenever Model.save_depth_conf_result is on -- the default -- and its
own fused point cloud) through baselines.common.runner's shared Baseline/SequenceOutput.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/da3_streaming.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/depth-anything-3 install.
"""

import dataclasses
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

# The submodule (its own da3_streaming.py/loop_utils/fastloop/...), as opposed to this directory
# (baselines/da3_streaming), which only holds the container files and this runner.
DA3_STREAMING_ROOT = Path(__file__).resolve().parents[2] / "depth-anything-3" / "da3_streaming"


@dataclasses.dataclass
class ModelConfig:
    # Path to a da3_streaming native config yaml (its own Weights/Model/Loop schema -- see
    # depth-anything-3/da3_streaming/configs/base_config.yaml), resolved relative to
    # DA3_STREAMING_ROOT if not given as absolute. Every da3_streaming-specific knob (chunk_size,
    # overlap, loop closure, alignment method, ...) lives there, not here -- this runner only
    # adds the shared data/output config baselines.common.runner already provides.
    config: str = "configs/base_config.yaml"


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts DA3-Streaming's world gauge onto the gt one, as far as
    one frame can. DA3-Streaming reasons in its own arbitrary gauge (world frame anchored to its
    first chunk's reference frame, at an unrecoverable scale) -- so raw predictions land nowhere
    near the gt trajectory even though the *shape* of the trajectory is right. This is the
    relative transform between the gt and predicted pose of frame 0, so applying it lands frame 0
    exactly on its gt pose; it only fixes position/orientation (6 of the 7 similarity-transform
    DOF) -- one frame can't disambiguate DA3-Streaming's scale too. Same rationale as
    baselines/VGGT-Long/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_da3_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                   proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a 3x3 intrinsic matrix -- measured against the (width, height) DA3 actually
    processed -- back to a frame's original (width, height), so it's expressed in the same pixel
    grid as the depth map written alongside it. da3_streaming's own results_output npz files and
    camera_poses.txt/intrinsic.txt don't do this rescale themselves. Same rationale and math as
    baselines/VGGT-Long/run_sequence.py's rescale_vggt_intrinsics_from_K.
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


def stage_frames(frames: list[Frame], staging_dir: Path) -> None:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{i:08d}.jpg" (`i` = position in
    `frames`, not the frame's own gt index) -- so da3_streaming.py's own `sorted(glob("*.jpg") +
    glob("*.png"))` image loading processes exactly our selected frames, in the same order
    `frames` is already in, and assigns each one that same 0-based `i` as its own internal
    global_idx -- which is exactly what results_output/frame_{i}.npz and camera_poses.txt/
    intrinsic.txt (one line per i) are keyed by, letting us map its output straight back to
    `frames[i]` afterward. Always ".jpg" regardless of the source's real extension: the glob only
    matches "*.jpg"/"*.png" (not e.g. ".jpeg"), but images are decoded by content, not filename,
    so a mislabeled extension here is harmless.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (staging_dir / f"{i:08d}.jpg").symlink_to(f.rgb_path)


def _exists_nonempty(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _download_da3(dest: Path) -> None:
    base_url = "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main"
    subprocess.run(["curl", "-fL", f"{base_url}/config.json", "-o", "config.json"], cwd=dest, check=True)
    subprocess.run(["curl", "-fL", f"{base_url}/model.safetensors", "-o", "model.safetensors"],
                    cwd=dest, check=True)


def _download_salad(dest: Path) -> None:
    url = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    subprocess.run(["curl", "-fL", url, "-o", "dino_salad.ckpt"], cwd=dest, check=True)


@dataclasses.dataclass
class _WeightSpec:
    download: Callable[[Path], None]
    top_level: tuple[str, ...]  # file(s) `download` produces directly under its scratch dir


# Mirrors depth-anything-3/da3_streaming/scripts/download_weights.sh's own commands one-for-one
# (see each _download_* function) -- DA3 bundles two files (config.json + model.safetensors,
# Weights.DA3_CONFIG/DA3) since they're only ever used as a pair.
_WEIGHT_SPECS: dict[str, _WeightSpec] = {
    "DA3": _WeightSpec(_download_da3, ("model.safetensors", "config.json")),
    "SALAD": _WeightSpec(_download_salad, ("dino_salad.ckpt",)),
}


def _needed_weight_keys(model_config_path: Path) -> list[str]:
    """Which of DA3-Streaming's Weights.* checkpoints this config actually needs.

    scripts/download_weights.sh fetches both unconditionally; a single run always needs DA3
    itself, but only needs SALAD (the only loop-closure backend da3_streaming supports -- unlike
    VGGT-Long, there's no DBoW alternative here) if loop closure is actually on
    (Model.loop_enable) -- see loop_utils/loop_detector.py for where Weights.SALAD gets read.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    keys = ["DA3"]
    if cfg.get("Model", {}).get("loop_enable", True):
        keys.append("SALAD")
    return keys


def ensure_weights_downloaded(model_config_path: Path) -> None:
    """Downloads only the checkpoint(s) `model_config_path` actually needs (_needed_weight_keys)
    that aren't already present under weights/ -- same approach as
    baselines/VGGT-Long/run_sequence.py's own ensure_weights_downloaded; see its docstring for why
    each checkpoint downloads into its own scratch dir and is only moved into weights/ once
    confirmed present and non-empty there. Not baked into the Docker image (see the Dockerfile's
    own note), so this is a one-time cost the first time each checkpoint is needed.
    """
    weights_dir = DA3_STREAMING_ROOT / "weights"
    for key in _needed_weight_keys(model_config_path):
        spec = _WEIGHT_SPECS[key]
        targets = [weights_dir / name for name in spec.top_level]
        if all(_exists_nonempty(t) for t in targets):
            continue

        print(f"downloading {key} checkpoint into {weights_dir} ...")
        scratch = Path(tempfile.mkdtemp(dir=DA3_STREAMING_ROOT, prefix=".weights-download-"))
        try:
            spec.download(scratch)
            staged = [scratch / name for name in spec.top_level]
            missing = [s for s in staged if not _exists_nonempty(s)]
            if missing:
                raise SystemExit(
                    f"download for {key} finished but {missing} is/are missing or empty -- "
                    "check its output above for what went wrong, then rerun"
                )
            weights_dir.mkdir(exist_ok=True)
            for s, t in zip(staged, targets):
                s.rename(t)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def avoid_single_chunk_bug(model_config_path: Path, n_frames: int, out_dir: Path) -> Path:
    """Works around a bug in da3_streaming.py's own save_camera_poses: when the whole sequence
    fits in a single chunk (n_frames <= Model.chunk_size), its first-chunk loop still trims the
    last Model.overlap frames off the end, assuming a second chunk will cover them -- but with
    only one chunk, nothing does, so those frames' poses/intrinsics are left as None and it
    crashes at `pose.flatten()`. Chunks past the first always cover their own tail correctly
    (checked: chunk i's trimmed end lines up exactly with chunk i+1's start, since non-final
    chunks are always exactly Model.chunk_size long), so this only ever bites the single-chunk
    case -- every stock config (base_config.yaml/kitti.yaml/tum.yaml) uses chunk_size: 120, which
    is bigger than these baselines' ~99-frame example sequences, so it's easy to hit by accident.

    Returns `model_config_path` unchanged if n_frames > Model.chunk_size (already safe -- this
    never fires on a real, long sequence, which is what da3_streaming is actually for). Otherwise
    writes an adjusted copy -- Model.chunk_size lowered just enough to force at least two chunks,
    Model.overlap capped to stay below it -- to `out_dir` and returns that path instead, printing
    a warning so this doesn't silently diverge from what the config actually asked for.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    model_cfg = cfg.get("Model", {})
    chunk_size, overlap = model_cfg.get("chunk_size"), model_cfg.get("overlap", 0)
    if not chunk_size or n_frames > chunk_size:
        return model_config_path

    new_chunk_size = max(1, n_frames - 1)
    new_overlap = min(overlap, new_chunk_size - 1) if new_chunk_size > 1 else 0
    print(
        f"WARNING: {n_frames} frames <= model.config's Model.chunk_size ({chunk_size}) -- "
        "da3_streaming.py's save_camera_poses crashes in this single-chunk case (see "
        "avoid_single_chunk_bug's docstring in run_sequence.py), so auto-lowering chunk_size to "
        f"{new_chunk_size} and overlap to {new_overlap} for this run only. Set a smaller "
        "Model.chunk_size in your own config if you want control over the exact values used."
    )
    model_cfg["chunk_size"], model_cfg["overlap"] = new_chunk_size, new_overlap
    cfg["Model"] = model_cfg
    out_path = out_dir / "config.yaml"
    out_path.write_text(yaml.safe_dump(cfg))
    return out_path


def run_da3_streaming(image_dir: Path, config_path: Path, output_dir: Path) -> None:
    """Runs da3_streaming.py exactly as documented (its own README: `python da3_streaming.py
    --image_dir ... --config ... --output_dir ...`, from inside the submodule) -- its weight
    paths (e.g. `./weights/model.safetensors`) are cwd-relative and it isn't packaged as an
    importable library, so this subprocess is the only supported entry point.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "da3_streaming.py", "--image_dir", str(image_dir),
         "--config", str(config_path), "--output_dir", str(output_dir)],
        cwd=DA3_STREAMING_ROOT, check=True,
    )


def read_camera_poses(path: Path, n_frames: int) -> np.ndarray:
    """DA3_Streaming.save_camera_poses's own format (same as VGGT-Long's): one line per frame, 16
    space-separated floats (a (4,4) c2w matrix, row-major)."""
    lines = path.read_text().strip().splitlines()
    if len(lines) != n_frames:
        raise SystemExit(f"{path} has {len(lines)} poses, expected {n_frames}")
    return np.array([[float(x) for x in line.split()] for line in lines]).reshape(n_frames, 4, 4)


class DA3Streaming(Baseline):
    """DA3-Streaming, driven through its own chunked/loop-closed CLI (see module docstring)."""

    name = "da3_streaming"
    Config = ModelConfig

    def setup(self) -> None:
        self.model_config_path = Path(self.config.config)
        if not self.model_config_path.is_absolute():
            self.model_config_path = DA3_STREAMING_ROOT / self.model_config_path
        if not self.model_config_path.is_file():
            raise SystemExit(f"model.config {self.model_config_path} not found")
        ensure_weights_downloaded(self.model_config_path)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        output_dir = out.scratch_dir("da3_streaming")
        try:
            with tempfile.TemporaryDirectory(prefix="da3-streaming-") as tmp:
                tmp_path = Path(tmp)
                staging_dir = tmp_path / "frames"
                stage_frames(frames, staging_dir)
                config_path = avoid_single_chunk_bug(self.model_config_path, len(frames), tmp_path)

                print(f"running {len(frames)} frames (chunked, see {self.model_config_path.name}) ...")
                run_da3_streaming(staging_dir, config_path, output_dir)

            c2ws = read_camera_poses(output_dir / "camera_poses.txt", len(frames))
            align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
            c2ws = align @ c2ws

            for i, f in enumerate(frames):
                npz = np.load(output_dir / "results_output" / f"frame_{i}.npz")
                depth_proc, conf_proc = npz["depth"], npz["conf"]
                proc_image_size = (depth_proc.shape[1], depth_proc.shape[0])  # (width, height)

                width, height = image_size(f.rgb_path)
                depth = cv2.resize(depth_proc, (width, height), interpolation=cv2.INTER_LINEAR)
                conf = cv2.resize(conf_proc, (width, height), interpolation=cv2.INTER_LINEAR) \
                    if out.spec.min_confidence is not None else None
                K = rescale_da3_intrinsics_from_K(npz["intrinsics"], (width, height), proc_image_size)
                out.write_frame(f.index, depth, c2ws[i], K, intrinsics_width=width, conf=conf)

            if out.spec.cloud:
                # da3_streaming's own fused, already confidence-filtered/subsampled cloud (its
                # Model.Pointcloud_Save settings) -- reused as-is, same "matches what the model
                # ships as its reference output" rationale as OutputSpec.cloud.
                cloud = trimesh.load(output_dir / "pcd" / "combined_pcd.ply", process=False)
                pts = trimesh.transform_points(np.asarray(cloud.vertices), align)
                cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None
                out.write_cloud(pts, cols)
                print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
        finally:
            # da3_streaming's own output dir (results_output/ + pcd/*.ply, now fully harvested
            # into the predictions/ tree above) is a bulky scratch intermediate nothing downstream
            # reads, same rationale as baselines/depth_anything_3/run_sequence.py's glb cleanup.
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    cli_main(DA3Streaming, description=__doc__)
