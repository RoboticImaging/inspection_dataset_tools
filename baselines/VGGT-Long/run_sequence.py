"""Run VGGT-Long on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

Unlike baselines/vggt/run_sequence.py's single joint inference call, VGGT-Long chunks a sequence,
runs each chunk through a foundation model (VGGT/Pi3/MapAnything, selected in VGGT-Long's own
config), then stitches chunks together with weighted Sim(3) alignment and optional loop closure.
Its own CLI (`vggt_long.py`) resolves weight paths (e.g. `./weights/model.pt`) relative to cwd and
isn't packaged to be imported as a library, so this drives it exactly as documented -- a
subprocess, run from inside the VGGT-Long submodule -- then re-exports its outputs
(camera_poses.txt/intrinsic.txt, the per-chunk depth VGGT-Long computes but normally deletes, and
its own fused point cloud) through baselines.common.runner's shared Baseline/SequenceOutput.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/vggt-long.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/vggt-long install. Weights
aren't baked into the image (see the Dockerfile's own note); the first run that needs a given
checkpoint instead downloads just that one itself (ensure_weights_downloaded, below -- the same
individual downloads ../../VGGT-Long/scripts/download_weights.sh does, but only whichever the
config's chosen model/loop backend actually needs), cached under VGGT-Long/weights/ for every run
after.
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

# The submodule (its own vggt_long.py/loop_utils/base_models/...), as opposed to this directory
# (baselines/VGGT-Long), which only holds the container files and this runner.
VGGT_LONG_ROOT = Path(__file__).resolve().parents[2] / "VGGT-Long"


@dataclasses.dataclass
class ModelConfig:
    # Path to a VGGT-Long native config yaml (its own Weights/Model/Loop schema -- see
    # VGGT-Long/configs/base_config.yaml), resolved relative to the VGGT-Long submodule root if
    # not given as absolute. Every VGGT-Long-specific knob (which foundation model, chunk_size,
    # overlap, loop closure, ...) lives there, not here -- this runner only adds dataset/output.
    config: str = "configs/base_config.yaml"


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts VGGT-Long's world gauge onto the gt one, as far as one
    frame can. VGGT-Long reasons in its own arbitrary gauge (world frame anchored to its first
    chunk's first frame, at an unrecoverable scale -- chunk-to-chunk Sim(3) alignment only
    reconciles chunks against *each other*, not against gt) -- so raw predictions land nowhere
    near the gt trajectory even though the *shape* of the trajectory is right. This is the
    relative transform between the gt and predicted pose of frame 0, so applying it lands frame 0
    exactly on its gt pose; it only fixes position/orientation (6 of the 7 similarity-transform
    DOF) -- one frame can't disambiguate VGGT-Long's scale too. Same rationale as
    baselines/vggt/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_vggt_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                    proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a 3x3 intrinsic matrix -- measured against the (width, height) the underlying
    foundation model actually processed -- back to a frame's original (width, height), so it's
    expressed in the same pixel grid as the depth map written alongside it. VGGT-Long's own
    camera_poses.txt/intrinsic.txt writers don't do this rescale themselves. Same rationale and
    math as baselines/vggt/run_sequence.py's version of this function.
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
    `frames`) -- so vggt_long.py's own `sorted(glob("*.jpg") + glob("*.png"))` image loading
    (VGGT_Long.run) processes exactly our selected frames, in the same order `frames` is already
    in, letting us map its 0..N-1 frame index straight back to `frames[i]` afterward. Always
    ".jpg" regardless of the source's real extension: the glob only matches "*.jpg"/"*.png" (not
    e.g. ".jpeg"), but images are decoded by content, not filename, so a mislabeled extension
    here is harmless.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (staging_dir / f"{i:08d}.jpg").symlink_to(f.rgb_path)


def prepare_vggt_long_config(model_config_path: Path, out_dir: Path) -> Path:
    """A copy of the chosen VGGT-Long config with Model.delete_temp_files forced off.

    VGGT-Long deletes its own per-chunk intermediate .npy files -- which is where its per-frame
    depth actually lives, since neither camera_poses.txt/intrinsic.txt nor the fused point cloud
    carry it -- as its very last step. Forcing this off is the only way to recover that depth
    afterward, since we drive vggt_long.py as a subprocess and never see its in-memory
    predictions ourselves.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    cfg.setdefault("Model", {})["delete_temp_files"] = False
    out_path = out_dir / "config.yaml"
    out_path.write_text(yaml.safe_dump(cfg))
    return out_path


def compute_chunk_indices(n_frames: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Mirrors vggt_long.VGGT_Long.process_long_sequence's own chunk_indices computation exactly
    -- needed to map each chunk_{i}.npy's local frame axis back to global frame indices, since
    vggt_long.py doesn't persist that mapping to disk itself."""
    if n_frames <= chunk_size:
        return [(0, n_frames)]
    step = chunk_size - overlap
    num_chunks = (n_frames - overlap + step - 1) // step
    return [(i * step, min(i * step + chunk_size, n_frames)) for i in range(num_chunks)]


def _exists_nonempty(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


# One entry per VGGT-Long Weights.* key, mirroring scripts/download_weights.sh's own commands for
# that checkpoint one-for-one (see each _download_* function) -- `top_level` is the single
# file/dir that command produces directly under weights/, e.g. huggingface-cli's whole `Pi3/`
# repo checkout, not just the one .safetensors config.py's Weights.Pi3 happens to point at.
def _download_salad(dest: Path) -> None:
    url = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
    subprocess.run(["curl", "-fL", url, "-o", "dino_salad.ckpt"], cwd=dest, check=True)


def _download_dnio(dest: Path) -> None:
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    subprocess.run(["wget", url], cwd=dest, check=True)


def _download_dbow(dest: Path) -> None:
    # download_weights.sh backgrounds this wget (`(wget ...) & wait`), for no apparent reason --
    # bash's argument-less `wait` always returns 0 regardless of the backgrounded job's real exit
    # status, so that form can't actually detect a failed download. Run it in the foreground
    # instead: same command, but check=True below can then trust its exit code.
    url = "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
    subprocess.run(["wget", url], cwd=dest, check=True)
    subprocess.run(["tar", "-xzvf", "ORBvoc.txt.tar.gz"], cwd=dest, check=True)
    (dest / "ORBvoc.txt.tar.gz").unlink()


def _download_vggt(dest: Path) -> None:
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    subprocess.run(["curl", "-fL", url, "-o", "model.pt"], cwd=dest, check=True)


def _download_pi3(dest: Path) -> None:
    subprocess.run(["huggingface-cli", "download", "yyfz233/Pi3",
                     "--local-dir", "Pi3", "--local-dir-use-symlinks", "False"], cwd=dest, check=True)


def _download_map(dest: Path) -> None:
    subprocess.run(["huggingface-cli", "download", "facebook/map-anything",
                     "--local-dir", "map_anything", "--local-dir-use-symlinks", "False"], cwd=dest, check=True)


@dataclasses.dataclass
class _WeightSpec:
    download: Callable[[Path], None]
    top_level: str  # file/dir `download` produces directly under its given scratch dir


_WEIGHT_SPECS: dict[str, _WeightSpec] = {
    "SALAD": _WeightSpec(_download_salad, "dino_salad.ckpt"),
    "DNIO": _WeightSpec(_download_dnio, "dinov2_vitb14_pretrain.pth"),
    "DBoW": _WeightSpec(_download_dbow, "ORBvoc.txt"),
    "VGGT": _WeightSpec(_download_vggt, "model.pt"),
    "Pi3": _WeightSpec(_download_pi3, "Pi3"),
    "Map": _WeightSpec(_download_map, "map_anything"),
}

# Weights.model's own value ('VGGT'/'Pi3'/'Mapanything', per base_models/base_model.py's
# VGGT_Long.__init__) doesn't spell the same as its own Weights.* key ('Map', not 'Mapanything').
_MODEL_TO_WEIGHT_KEY = {"VGGT": "VGGT", "Pi3": "Pi3", "Mapanything": "Map"}


def _needed_weight_keys(model_config_path: Path) -> list[str]:
    """Which of VGGT-Long's Weights.* checkpoints this config actually needs.

    scripts/download_weights.sh fetches all six unconditionally (~6 GiB), but a single run only
    ever touches the one selected foundation model (Weights.model) plus, if loop closure is on
    (Model.loop_enable), whichever one loop-detector backend is selected (Model.useDBoW) -- see
    base_models/base_model.py, LoopModels/LoopModel.py, LoopModels/backbones/dinov2.py and
    LoopModelDBoW/retrieval/retrieval_dbow.py for exactly where each Weights.* key gets read.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    keys = [_MODEL_TO_WEIGHT_KEY[cfg["Weights"]["model"]]]
    model_cfg = cfg.get("Model", {})
    if model_cfg.get("loop_enable", True):
        keys += ["DBoW"] if model_cfg.get("useDBoW", False) else ["SALAD", "DNIO"]
    return keys


def ensure_weights_downloaded(model_config_path: Path) -> None:
    """Downloads only the checkpoint(s) `model_config_path` actually needs (_needed_weight_keys)
    that aren't already present under weights/ -- unlike scripts/download_weights.sh itself,
    which always fetches all six regardless of which model/loop backend is selected.

    Each checkpoint downloads into its own scratch dir and is only moved into weights/ once
    confirmed present and non-empty there -- so a failed/partial download for *one* checkpoint
    can't be mistaken for a completed one on a later run, without forcing a re-download of every
    *other*, already-present checkpoint (weights/ is a shared, incrementally-populated cache
    across configs/runs, not an all-or-nothing unit). Not baked into the Docker image (see the
    Dockerfile's own note), so this is a one-time cost the first time each checkpoint is needed.

    If you hit an auth error here, facebook/map-anything is a gated HF repo --
    `huggingface-cli login` first.
    """
    weights_dir = VGGT_LONG_ROOT / "weights"
    for key in _needed_weight_keys(model_config_path):
        spec = _WEIGHT_SPECS[key]
        target = weights_dir / spec.top_level
        if _exists_nonempty(target):
            continue

        print(f"{target} not found -- downloading {key} checkpoint ...")
        scratch = Path(tempfile.mkdtemp(dir=VGGT_LONG_ROOT, prefix=".weights-download-"))
        try:
            spec.download(scratch)
            staged = scratch / spec.top_level
            if not _exists_nonempty(staged):
                raise SystemExit(
                    f"download for {key} finished but {staged} is missing or empty -- check its "
                    "output above for what went wrong, then rerun"
                )
            weights_dir.mkdir(exist_ok=True)
            staged.rename(target)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def run_vggt_long(image_dir: Path, config_path: Path) -> Path:
    """Runs vggt_long.py exactly as documented (its own README: `python vggt_long.py --image_dir
    ... --config ...`, from inside the submodule) -- its weight paths (e.g. `./weights/model.pt`)
    are cwd-relative and it isn't packaged as an importable library, so this subprocess is the
    only supported entry point. Returns the exp dir vggt_long.py wrote its outputs under (its own
    __main__: `./exps/<image_dir with "/" -> "_">/<timestamp>/`).
    """
    exps_root = VGGT_LONG_ROOT / "exps" / str(image_dir).replace("/", "_")
    before = set(p.name for p in exps_root.iterdir()) if exps_root.is_dir() else set()
    subprocess.run(
        [sys.executable, "vggt_long.py", "--image_dir", str(image_dir), "--config", str(config_path)],
        cwd=VGGT_LONG_ROOT, check=True,
    )
    new = sorted(set(p.name for p in exps_root.iterdir()) - before)
    if not new:
        raise SystemExit(f"vggt_long.py didn't create a new exp dir under {exps_root}")
    return exps_root / new[-1]


def read_camera_poses(path: Path, n_frames: int) -> np.ndarray:
    """VGGT_Long.save_camera_poses's own format: one line per frame, 16 space-separated floats
    (a (4,4) c2w matrix, row-major -- np.ndarray.flatten()'s default order, which is how it was
    written)."""
    lines = path.read_text().strip().splitlines()
    if len(lines) != n_frames:
        raise SystemExit(f"{path} has {len(lines)} poses, expected {n_frames}")
    return np.array([[float(x) for x in line.split()] for line in lines]).reshape(n_frames, 4, 4)


def read_intrinsics(path: Path, n_frames: int) -> np.ndarray:
    """VGGT_Long.save_camera_poses's own format: one line per frame, "fx fy cx cy"."""
    lines = path.read_text().strip().splitlines()
    if len(lines) != n_frames:
        raise SystemExit(f"{path} has {len(lines)} intrinsics, expected {n_frames}")
    Ks = np.tile(np.eye(3), (n_frames, 1, 1))
    for i, line in enumerate(lines):
        fx, fy, cx, cy = (float(x) for x in line.split())
        Ks[i, 0, 0], Ks[i, 1, 1], Ks[i, 0, 2], Ks[i, 1, 2] = fx, fy, cx, cy
    return Ks


def load_chunk_depths(aligned_dir: Path, chunk_indices: list[tuple[int, int]],
                       n_frames: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Per-frame camera-space depth (N, h, w), at the underlying model's own processing
    resolution, plus that resolution as (width, height). Harvested from the per-chunk aligned
    .npy files VGGT-Long normally deletes (see prepare_vggt_long_config) -- overlapping frames
    (shared between consecutive chunks) take the later chunk's prediction, the same resolution
    VGGT-Long's own save_camera_poses uses for the equivalent overlap in poses/intrinsics.
    """
    depths_by_frame = [None] * n_frames
    proc_size = None
    for chunk_idx, (start, end) in enumerate(chunk_indices):
        data = np.load(aligned_dir / f"chunk_{chunk_idx}.npy", allow_pickle=True).item()
        depth = np.asarray(data["depth"])
        if depth.ndim == 4:  # (chunk_len, h, w, 1)
            depth = depth[..., 0]
        if proc_size is None:
            proc_size = (depth.shape[-1], depth.shape[-2])  # (width, height)
        for local_i, global_i in enumerate(range(start, end)):
            depths_by_frame[global_i] = depth[local_i]
    missing = [i for i, d in enumerate(depths_by_frame) if d is None]
    if missing:
        raise SystemExit(f"no depth harvested for frame(s) {missing} -- chunk .npy files missing?")
    return np.stack(depths_by_frame), proc_size


class VGGTLong(Baseline):
    """VGGT-Long, driven through its own chunked/loop-closed CLI (see module docstring)."""

    name = "vggt-long"
    Config = ModelConfig

    def setup(self) -> None:
        self.model_config_path = Path(self.config.config)
        if not self.model_config_path.is_absolute():
            self.model_config_path = VGGT_LONG_ROOT / self.model_config_path
        if not self.model_config_path.is_file():
            raise SystemExit(f"model.config {self.model_config_path} not found")
        ensure_weights_downloaded(self.model_config_path)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        with tempfile.TemporaryDirectory(prefix="vggt-long-") as tmp:
            tmp_path = Path(tmp)
            staging_dir = tmp_path / "frames"
            stage_frames(frames, staging_dir)
            run_config_path = prepare_vggt_long_config(self.model_config_path, tmp_path)

            print(f"running VGGT-Long on {len(frames)} frames (chunked, see {self.model_config_path.name}) ...")
            exp_dir = run_vggt_long(staging_dir, run_config_path)

            try:
                run_config = yaml.safe_load(run_config_path.read_text())
                chunk_indices = compute_chunk_indices(
                    len(frames), run_config["Model"]["chunk_size"], run_config["Model"]["overlap"]
                )

                c2ws = read_camera_poses(exp_dir / "camera_poses.txt", len(frames))
                intrinsics_path = exp_dir / "intrinsic.txt"
                if not intrinsics_path.is_file():
                    raise SystemExit(
                        f"{intrinsics_path} wasn't written -- model.config's Weights.model doesn't "
                        "produce intrinsics itself (only the VGGT adapter does; Pi3/Mapanything "
                        "return intrinsic=None, see VGGT-Long/base_models/base_model.py)"
                    )
                Ks_proc = read_intrinsics(intrinsics_path, len(frames))
                depths_proc, proc_image_size = load_chunk_depths(
                    exp_dir / "_tmp_results_aligned", chunk_indices, len(frames)
                )

                align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
                c2ws = align @ c2ws

                for j, f in enumerate(frames):
                    width, height = image_size(f.rgb_path)
                    depth = cv2.resize(depths_proc[j], (width, height), interpolation=cv2.INTER_LINEAR)
                    K = rescale_vggt_intrinsics_from_K(Ks_proc[j], (width, height), proc_image_size)
                    out.write_frame(f.index, depth, c2ws[j], K, intrinsics_width=width)

                if out.spec.cloud:
                    # VGGT-Long's own fused, already confidence-filtered/subsampled cloud (its
                    # Model.Pointcloud_Save settings) -- reused as-is rather than re-unprojecting
                    # ourselves, same "matches what the model ships as its reference output"
                    # rationale as OutputSpec.cloud.
                    cloud = trimesh.load(exp_dir / "pcd" / "combined_pcd.ply", process=False)
                    pts = trimesh.transform_points(np.asarray(cloud.vertices), align)
                    cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
            finally:
                # VGGT-Long's own exp dir (per-chunk depth kept alive by prepare_vggt_long_config,
                # plus per-chunk .ply files) is a bulky scratch intermediate now fully harvested
                # into the predictions/ tree above -- nothing downstream reads it, same rationale
                # as baselines/depth_anything_3/run_sequence.py's glb cleanup.
                shutil.rmtree(exp_dir.parent, ignore_errors=True)


if __name__ == "__main__":
    cli_main(VGGTLong, description=__doc__)
