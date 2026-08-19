"""Run Pi-Long on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

Pi-Long is DengKaiCQ's own port of VGGT-Long's chunk/loop-closure framework onto Pi3/Pi3X: like
baselines/VGGT-Long/run_sequence.py, this chunks a sequence, runs each chunk through Pi3 (or
Pi3X), then stitches chunks together with weighted Sim(3) alignment and optional loop closure. Its
own CLI (`pi_long.py`) resolves weight paths (e.g. `./weights/pi3.safetensors`) relative to cwd
and isn't packaged to be imported as a library, so this drives it exactly as documented -- a
subprocess, run from inside the Pi-Long submodule -- then re-exports its outputs
(camera_poses.txt, the per-chunk camera-space geometry Pi-Long computes but normally deletes, and
its own fused point cloud) through baselines.common.runner's shared Baseline/SequenceOutput.

Unlike VGGT-Long, Pi-Long never writes an intrinsic.txt of its own (see pi_long.py -- Pi3's own
per-pixel ray directions never get turned into a K anywhere in that codebase, only into world
points), so this recovers per-frame intrinsics itself from the harvested `local_points`, exactly
the way baselines/Pi3/run_sequence.py does for plain Pi3.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/pi-long.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/pi-long install. Weights aren't
baked into the image (see the Dockerfile's own note); the first run that needs a given checkpoint
instead downloads just that one itself (ensure_weights_downloaded, below -- the same individual
downloads ../../Pi-Long/scripts/download_weights.sh does, but only whichever the config's chosen
model/loop backend actually needs), cached under Pi-Long/weights/ for every run after.
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
import torch
import trimesh
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: inspection_viz + baselines

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

# The submodule (its own pi_long.py/loop_utils/pi3/...), as opposed to this directory
# (baselines/Pi-Long), which only holds the container files and this runner.
PI_LONG_ROOT = Path(__file__).resolve().parents[2] / "Pi-Long"
sys.path.insert(0, str(PI_LONG_ROOT))  # Pi-Long's own vendored `pi3` package, for the geometry
                                        # helper below -- not pip-installed anywhere, plain source

from pi3.utils.geometry import recover_intrinsic_from_rays_d  # noqa: E402  (needs the path above)


@dataclasses.dataclass
class ModelConfig:
    # Path to a Pi-Long native config yaml (its own Weights/Model/Loop schema -- see
    # Pi-Long/configs/base_config.yaml), resolved relative to the Pi-Long submodule root if not
    # given as absolute. Every Pi-Long-specific knob (Pi3 vs Pi3X, chunk_size, overlap, loop
    # closure, ...) lives there, not here -- this runner only adds dataset/output.
    config: str = "configs/base_config.yaml"


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts Pi-Long's world gauge onto the gt one, as far as one
    frame can. Pi-Long reasons in its own arbitrary gauge (world frame anchored to its first
    chunk's first frame, at an unrecoverable scale -- chunk-to-chunk Sim(3) alignment only
    reconciles chunks against *each other*, not against gt) -- so raw predictions land nowhere
    near the gt trajectory even though the *shape* of the trajectory is right. This is the
    relative transform between the gt and predicted pose of frame 0, so applying it lands frame 0
    exactly on its gt pose; it only fixes position/orientation (6 of the 7 similarity-transform
    DOF) -- one frame can't disambiguate Pi-Long's scale too. Same rationale as
    baselines/vggt/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_pi_long_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                       proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a 3x3 intrinsic matrix -- recovered at the (width, height) Pi3 actually processed
    inside Pi-Long -- back to a frame's original (width, height), so it's expressed in the same
    pixel grid as the depth map written alongside it. Same rationale and math as
    baselines/vggt/run_sequence.py's version of this function.
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
    `frames`) -- so pi_long.py's own `sorted(glob("*.jpg") + ...)` image loading (Pi_Long.run)
    processes exactly our selected frames, in the same order `frames` is already in, letting us
    map its 0..N-1 frame index straight back to `frames[i]` afterward. Always ".jpg" regardless of
    the source's real extension: images are decoded by content, not filename, so a mislabeled
    extension here is harmless. Same as baselines/VGGT-Long/run_sequence.py's version.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        (staging_dir / f"{i:08d}.jpg").symlink_to(f.rgb_path)


def prepare_pi_long_config(model_config_path: Path, out_dir: Path) -> Path:
    """A copy of the chosen Pi-Long config with Model.delete_temp_files forced off and
    Model.temp_files_location forced to 'disk'.

    Pi-Long deletes its own per-chunk intermediate .npy files -- which is where its per-frame
    depth *and* the local_points intrinsics are recovered from below -- as its very last step;
    forcing this off is the only way to recover them afterward, since we drive pi_long.py as a
    subprocess and never see its in-memory predictions ourselves. temp_files_location is forced to
    'disk' regardless of what the config says: in 'cpu_memory' mode those same intermediates never
    touch disk at all (they live only in the subprocess's own memory, gone the moment it exits),
    so 'disk' is the only setting these harvesting functions can read anything back from.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    cfg.setdefault("Model", {})["delete_temp_files"] = False
    cfg["Model"]["temp_files_location"] = "disk"
    out_path = out_dir / "config.yaml"
    out_path.write_text(yaml.safe_dump(cfg))
    return out_path


def compute_chunk_indices(n_frames: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Mirrors pi_long.Pi_Long.process_long_sequence's own chunk_indices computation exactly
    (identical to VGGT-Long's, which this codebase was ported from) -- needed to map each
    chunk_{i}.npy's local frame axis back to global frame indices, since pi_long.py doesn't
    persist that mapping to disk itself."""
    if n_frames <= chunk_size:
        return [(0, n_frames)]
    step = chunk_size - overlap
    num_chunks = (n_frames - overlap + step - 1) // step
    return [(i * step, min(i * step + chunk_size, n_frames)) for i in range(num_chunks)]


def _exists_nonempty(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


# One entry per Pi-Long Weights.* key this runner might need, mirroring scripts/download_weights.sh's
# own commands for that checkpoint one-for-one -- `produced_name` is the filename `download` leaves
# behind in its given scratch dir. It is deliberately *not* assumed to match the config's own
# Weights.<key> basename: base_config.yaml points Weights.Pi3X at './weights/pi3x.safetensors'
# (lowercase x) while download_weights.sh itself writes "pi3X.safetensors" (capital X) -- a real
# case mismatch in the upstream repo. Downloading under a fixed name here and then renaming to
# whatever the resolved config actually asks for (ensure_weights_downloaded, below) sidesteps that
# bug instead of reproducing it.
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


def _download_pi3(dest: Path) -> None:
    url = "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors"
    subprocess.run(["curl", "-fL", url, "-o", "pi3.safetensors"], cwd=dest, check=True)


def _download_pi3x(dest: Path) -> None:
    url = "https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors"
    subprocess.run(["curl", "-fL", url, "-o", "pi3x.safetensors"], cwd=dest, check=True)


@dataclasses.dataclass
class _WeightSpec:
    download: Callable[[Path], None]
    produced_name: str  # file `download` produces directly under its given scratch dir


_WEIGHT_SPECS: dict[str, _WeightSpec] = {
    "SALAD": _WeightSpec(_download_salad, "dino_salad.ckpt"),
    "DNIO": _WeightSpec(_download_dnio, "dinov2_vitb14_pretrain.pth"),
    "DBoW": _WeightSpec(_download_dbow, "ORBvoc.txt"),
    "Pi3": _WeightSpec(_download_pi3, "pi3.safetensors"),
    "Pi3X": _WeightSpec(_download_pi3x, "pi3x.safetensors"),
}

# Model.model_type's own accepted spellings (pi_long.Pi_Long.__init__ takes 'Pi3'/'pi3' or
# 'Pi3X'/'Pi3x'/'pi3x'/'pi3X') don't all match a Weights.* key spelling -- normalize case first.
_MODEL_TO_WEIGHT_KEY = {"pi3": "Pi3", "pi3x": "Pi3X"}


def _needed_weight_keys(model_config_path: Path) -> list[str]:
    """Which of Pi-Long's Weights.* checkpoints this config actually needs.

    scripts/download_weights.sh fetches all five unconditionally (~4.5-6 GiB), but a single run
    only ever touches the one selected foundation model (Model.model_type) plus, if loop closure
    is on (Model.loop_enable), whichever one loop-detector backend is selected (Model.useDBoW) --
    see pi_long.py's __init__ and get_loop_pairs for exactly where each Weights.* key gets read.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    model_cfg = cfg.get("Model", {})
    model_type = model_cfg.get("model_type", "Pi3").lower()
    if model_type not in _MODEL_TO_WEIGHT_KEY:
        raise SystemExit(f"unsupported Model.model_type {model_cfg.get('model_type')!r} in "
                         f"{model_config_path}")
    keys = [_MODEL_TO_WEIGHT_KEY[model_type]]
    if model_cfg.get("loop_enable", True):
        keys += ["DBoW"] if model_cfg.get("useDBoW", False) else ["SALAD", "DNIO"]
    return keys


def ensure_weights_downloaded(model_config_path: Path) -> None:
    """Downloads only the checkpoint(s) `model_config_path` actually needs (_needed_weight_keys)
    that aren't already present at the exact path its own Weights.<key> entry names -- unlike
    scripts/download_weights.sh itself, which always fetches all five regardless of which
    model/loop backend is selected.

    Each checkpoint downloads into its own scratch dir and is only moved into place once confirmed
    present and non-empty there -- so a failed/partial download for *one* checkpoint can't be
    mistaken for a completed one on a later run, without forcing a re-download of every *other*,
    already-present checkpoint (weights/ is a shared, incrementally-populated cache across
    configs/runs, not an all-or-nothing unit). Not baked into the Docker image (see the
    Dockerfile's own note), so this is a one-time cost the first time each checkpoint is needed.
    """
    cfg = yaml.safe_load(model_config_path.read_text())
    for key in _needed_weight_keys(model_config_path):
        spec = _WEIGHT_SPECS[key]
        target = PI_LONG_ROOT / cfg["Weights"][key]
        if _exists_nonempty(target):
            continue

        print(f"{target} not found -- downloading {key} checkpoint ...")
        scratch = Path(tempfile.mkdtemp(dir=PI_LONG_ROOT, prefix=".weights-download-"))
        try:
            spec.download(scratch)
            staged = scratch / spec.produced_name
            if not _exists_nonempty(staged):
                raise SystemExit(
                    f"download for {key} finished but {staged} is missing or empty -- check its "
                    "output above for what went wrong, then rerun"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(target)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def run_pi_long(image_dir: Path, config_path: Path) -> Path:
    """Runs pi_long.py exactly as documented (its own README: `python pi_long.py --image_dir ...
    --config ...`, from inside the submodule) -- its weight paths (e.g. `./weights/pi3.safetensors`)
    are cwd-relative and it isn't packaged as an importable library, so this subprocess is the
    only supported entry point. Returns the exp dir pi_long.py wrote its outputs under (its own
    __main__: `./exps/<image_dir with "/" -> "_">/<timestamp>/`).
    """
    exps_root = PI_LONG_ROOT / "exps" / str(image_dir).replace("/", "_")
    before = set(p.name for p in exps_root.iterdir()) if exps_root.is_dir() else set()
    subprocess.run(
        [sys.executable, "pi_long.py", "--image_dir", str(image_dir), "--config", str(config_path)],
        cwd=PI_LONG_ROOT, check=True,
    )
    new = sorted(set(p.name for p in exps_root.iterdir()) - before)
    if not new:
        raise SystemExit(f"pi_long.py didn't create a new exp dir under {exps_root}")
    return exps_root / new[-1]


def read_camera_poses(path: Path, n_frames: int) -> np.ndarray:
    """Pi_Long.save_camera_poses's own format: one line per frame, 16 space-separated floats (a
    (4,4) c2w matrix, row-major -- np.ndarray.flatten()'s default order, which is how it was
    written)."""
    lines = path.read_text().strip().splitlines()
    if len(lines) != n_frames:
        raise SystemExit(f"{path} has {len(lines)} poses, expected {n_frames}")
    return np.array([[float(x) for x in line.split()] for line in lines]).reshape(n_frames, 4, 4)


def load_chunk_geometry(unaligned_dir: Path, chunk_indices: list[tuple[int, int]],
                        n_frames: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Per-frame camera-space depth (N, h, w) and 3x3 intrinsics (N, 3, 3), at Pi3's own
    processing resolution, plus that resolution as (width, height). Harvested from the per-chunk
    *unaligned* .npy files Pi-Long normally deletes (see prepare_pi_long_config) -- unlike
    baselines/VGGT-Long's equivalent, this reads the unaligned dir, not the aligned one: Pi-Long's
    own alignment step only ever rewrites a chunk's 'points' (world-space) field, never
    'local_points' (camera-space, gauge-independent), so the unaligned copy already has everything
    needed here and exists for every chunk (the aligned dir has none for chunk 0, which needs no
    alignment).

    Intrinsics aren't in either file: Pi-Long never computes a K itself (see this module's
    docstring), so `local_points` -- Pi3's per-pixel camera-space points, xy already scaled by z --
    doubles exactly as an unnormalized ray direction map, which recover_intrinsic_from_rays_d fits
    a pinhole K from, the same way baselines/Pi3/run_sequence.py does for plain Pi3.

    Overlapping frames (shared between consecutive chunks) take the later chunk's prediction, the
    same convention Pi_Long.save_camera_poses itself uses for the equivalent overlap in poses.
    """
    depths_by_frame = [None] * n_frames
    Ks_by_frame = [None] * n_frames
    proc_size = None
    for chunk_idx, (start, end) in enumerate(chunk_indices):
        data = np.load(unaligned_dir / f"chunk_{chunk_idx}.npy", allow_pickle=True).item()
        local_points = np.asarray(data["local_points"])  # (chunk_len, h, w, 3)
        if proc_size is None:
            proc_size = (local_points.shape[-2], local_points.shape[-3])  # (width, height)
        with torch.no_grad():
            K_chunk = recover_intrinsic_from_rays_d(
                torch.from_numpy(local_points).float(), force_center_principal_point=True
            ).numpy()
        depth_chunk = local_points[..., 2]
        for local_i, global_i in enumerate(range(start, end)):
            depths_by_frame[global_i] = depth_chunk[local_i]
            Ks_by_frame[global_i] = K_chunk[local_i]
    missing = [i for i, d in enumerate(depths_by_frame) if d is None]
    if missing:
        raise SystemExit(f"no geometry harvested for frame(s) {missing} -- chunk .npy files missing?")
    return np.stack(depths_by_frame), np.stack(Ks_by_frame), proc_size


class PiLong(Baseline):
    """Pi-Long, driven through its own chunked/loop-closed CLI (see module docstring)."""

    name = "pi-long"
    Config = ModelConfig

    def setup(self) -> None:
        self.model_config_path = Path(self.config.config)
        if not self.model_config_path.is_absolute():
            self.model_config_path = PI_LONG_ROOT / self.model_config_path
        if not self.model_config_path.is_file():
            raise SystemExit(f"model.config {self.model_config_path} not found")
        ensure_weights_downloaded(self.model_config_path)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        with tempfile.TemporaryDirectory(prefix="pi-long-") as tmp:
            tmp_path = Path(tmp)
            staging_dir = tmp_path / "frames"
            stage_frames(frames, staging_dir)
            run_config_path = prepare_pi_long_config(self.model_config_path, tmp_path)

            print(f"running Pi-Long on {len(frames)} frames (chunked, see {self.model_config_path.name}) ...")
            exp_dir = run_pi_long(staging_dir, run_config_path)

            try:
                run_config = yaml.safe_load(run_config_path.read_text())
                chunk_indices = compute_chunk_indices(
                    len(frames), run_config["Model"]["chunk_size"], run_config["Model"]["overlap"]
                )

                c2ws = read_camera_poses(exp_dir / "camera_poses.txt", len(frames))
                depths_proc, Ks_proc, proc_image_size = load_chunk_geometry(
                    exp_dir / "_tmp_results_unaligned", chunk_indices, len(frames)
                )

                align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
                c2ws = align @ c2ws

                for j, f in enumerate(frames):
                    width, height = image_size(f.rgb_path)
                    depth = cv2.resize(depths_proc[j], (width, height), interpolation=cv2.INTER_LINEAR)
                    K = rescale_pi_long_intrinsics_from_K(Ks_proc[j], (width, height), proc_image_size)
                    out.write_frame(f.index, depth, c2ws[j], K, intrinsics_width=width)

                if out.spec.cloud:
                    # Pi-Long's own fused, already confidence-filtered/subsampled cloud (its
                    # Model.Pointcloud_Save settings) -- reused as-is rather than re-unprojecting
                    # ourselves, same "matches what the model ships as its reference output"
                    # rationale as OutputSpec.cloud.
                    cloud = trimesh.load(exp_dir / "pcd" / "combined_pcd.ply", process=False)
                    pts = trimesh.transform_points(np.asarray(cloud.vertices), align)
                    cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
            finally:
                # Pi-Long's own exp dir (per-chunk geometry kept alive by prepare_pi_long_config,
                # plus per-chunk .ply files) is a bulky scratch intermediate now fully harvested
                # into the predictions/ tree above -- nothing downstream reads it, same rationale
                # as baselines/VGGT-Long/run_sequence.py's own cleanup.
                shutil.rmtree(exp_dir.parent, ignore_errors=True)


if __name__ == "__main__":
    cli_main(PiLong, description=__doc__)
