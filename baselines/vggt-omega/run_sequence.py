"""Run VGGT-Omega on one sequence of an inspection asset and write depth + camera predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

VGGT-Omega (facebookresearch/vggt-omega) is plain feed-forward, like VGGT and Pi3 -- one joint
inference call over the whole given sequence returns per-frame depth, camera pose and confidence
that are already mutually consistent, so this follows the exact same shape as
baselines/vggt/run_sequence.py (not DA3's, which instead calls into the model's own glb exporter).

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/old_house.yaml --model ../configs/vggt-omega.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/vggt-omega install.

Unlike VGGT/Pi3, VGGT-Omega's checkpoints are a *gated* HuggingFace repo (facebook/VGGT-Omega) --
downloading one (ensure_checkpoint_downloaded, below) needs your access request to that repo
approved first, plus either `hf auth login` or $HF_TOKEN set (run_container.sh passes $HF_TOKEN
through if you have it set on the host). Once fetched, it's cached under vggt-omega/checkpoints/
for every run after -- or set model.checkpoint to a path you already downloaded by hand.
"""

import dataclasses
import subprocess
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

from vggt_omega.models.vggt_omega import VGGTOmega as VGGTOmegaModel  # aliased: Baseline below owns the name
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

# The submodule root, for where downloaded checkpoints get cached (matches the `checkpoints/`
# convention its own README/demo_gradio.py already use, and its .gitignore already excludes).
VGGT_OMEGA_ROOT = Path(__file__).resolve().parents[2] / "vggt-omega"


@dataclasses.dataclass
class ModelConfig:
    device: str = "cuda"
    checkpoint: str | None = None       # local .pt path; None = download hf_repo/hf_filename (gated, see module docstring)
    hf_repo: str = "facebook/VGGT-Omega"
    hf_filename: str = "vggt_omega_1b_512.pt"
    enable_alignment: bool = False      # must match the checkpoint: the "-Text-Alignment" one needs this True
    image_resolution: int = 512         # must match the checkpoint too (the alignment one is 256)
    mode: str = "balanced"              # load_and_preprocess_images' mode: 'balanced' or 'max_size'
    patch_size: int = 16


def _exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def ensure_checkpoint_downloaded(hf_repo: str, hf_filename: str) -> Path:
    """The local path for `hf_filename`, downloading it from the gated `hf_repo` first if it
    isn't already cached under vggt-omega/checkpoints/.

    Unlike every other baseline's own weight fetch here, this repo has no public/ungated default:
    facebook/VGGT-Omega requires a manually-approved HuggingFace access request before any
    download (authenticated or not) succeeds. `hf download` is used rather than a plain URL fetch
    (as baselines/vggt does for VGGT's own public checkpoint) because a gated repo needs an
    authenticated request -- it picks up $HF_TOKEN or a cached `hf auth login` automatically. If
    neither is set, or the request isn't approved yet, this fails with huggingface_hub's own
    401/403 error explaining that. (Older huggingface_hub releases called this command
    `huggingface-cli download`; recent ones renamed the CLI to `hf` and turned `huggingface-cli`
    into a stub that just prints a deprecation notice and exits non-zero instead of forwarding --
    since the Dockerfile installs huggingface_hub unpinned, this always gets whatever is current.)
    """
    target = VGGT_OMEGA_ROOT / "checkpoints" / hf_filename
    if _exists_nonempty(target):
        return target

    print(f"{target} not found -- downloading {hf_repo}/{hf_filename} (gated: needs your HF "
         "access request approved, plus `hf auth login` or $HF_TOKEN) ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "download", hf_repo, hf_filename, "--local-dir", str(target.parent)],
        check=True,
    )
    if not _exists_nonempty(target):
        raise SystemExit(f"download finished but {target} is missing or empty -- check the "
                         "output above for what went wrong, then rerun")
    return target


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Accepts (3,4) or (4,4) OpenCV world-to-camera; returns (4,4) camera-to-world."""
    if w2c.shape[0] == 3:
        w2c = np.vstack([w2c, [0, 0, 0, 1]])
    return np.linalg.inv(w2c)


def _gt_frame0_alignment(c2ws: np.ndarray, gt_c2w0: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts VGGT-Omega's world gauge onto the gt one, as far as one
    frame can. VGGT-Omega reasons in its own arbitrary gauge (world frame anchored to camera 0, at
    an unrecoverable scale) -- so raw predictions land nowhere near the gt trajectory even though
    the *shape* of the trajectory is right. This is the relative transform between the gt and
    predicted pose of frame 0, so applying it lands frame 0 exactly on its gt pose; it only fixes
    position/orientation (6 of the 7 similarity-transform DOF) -- one frame can't disambiguate
    VGGT-Omega's scale too, so depth and inter-frame translations stay at its own scale. Same
    rationale as baselines/vggt/run_sequence.py's version of this function."""
    return gt_c2w0 @ np.linalg.inv(c2ws[0])


def rescale_vggt_omega_intrinsics_from_K(K_pred: np.ndarray, orig_image_size: tuple[int, int],
                                          proc_image_size: tuple[int, int]) -> np.ndarray:
    """Rescales a VGGT-Omega 3x3 intrinsic matrix -- measured against the (width, height) it
    actually processed -- back to a frame's original (width, height), so it's expressed in the
    same pixel grid as the depth map written alongside it. Same rationale and math as
    baselines/vggt/run_sequence.py's version of this function.

    load_and_preprocess_images first center-crops any source outside a [0.5, 2.0] aspect ratio,
    then uniformly resizes -- for the landscape video these baselines target that crop never
    fires (same assumption the VGGT version of this function makes), so a uniform scale recovered
    from the actual processed shape is exact.
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


def unproject_depth_map_to_point_map(depth: np.ndarray, extrinsic: np.ndarray,
                                     intrinsic: np.ndarray) -> np.ndarray:
    """World-space points for a whole batch of per-frame planar depth + w2c extrinsic + intrinsic.

    Adapted from vggt-omega's own demo_gradio.py (unproject_depth_map_to_point_map) -- not
    packaged anywhere importable in vggt_omega itself (unlike VGGT's vggt.utils.geometry, which
    baselines/vggt/run_sequence.py imports directly), so reproduced here instead.
    """
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx, fy = intrinsic[:, 0, 0, None, None], intrinsic[:, 1, 1, None, None]
    cx, cy = intrinsic[:, 0, 2, None, None], intrinsic[:, 1, 2, None, None]
    camera_points = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)

    rotation, translation = extrinsic[:, :3, :3], extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)),
                     camera_points - translation[:, None, None, :])


def run_inference(model: VGGTOmegaModel, frames: list[Frame], device: str, image_resolution: int,
                  mode: str, patch_size: int) -> dict:
    """One joint feed-forward call over the whole sequence -- VGGT-Omega reasons across all given
    frames at once, which is the only way its poses/depths come out mutually consistent. Long
    sequences will simply run out of GPU memory; there is no chunking fallback, because per-chunk
    predictions would live in unrelated world gauges and scales (same caveat as VGGT/DA3).

    Mirrors the README's own call into the model (`model(images)`, then encoding_to_camera) --
    unlike baselines/vggt and baselines/Pi3's own run_inference, no explicit autocast wrapping is
    needed here: VGGTOmega.forward already wraps its own compute in torch.autocast internally.
    """
    images = load_and_preprocess_images(
        [str(f.rgb_path) for f in frames], mode=mode, image_resolution=image_resolution, patch_size=patch_size,
    ).to(device)
    with torch.inference_mode():
        predictions = model(images)
    predictions["extrinsic"], predictions["intrinsic"] = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            predictions[key] = val.float().cpu().numpy().squeeze(0)  # drop the batch dim
    return predictions


class VGGTOmega(Baseline):
    """Plain feed-forward VGGT-Omega, one joint inference call per sequence."""

    name = "vggt-omega"
    Config = ModelConfig

    def setup(self) -> None:
        cfg = self.config
        checkpoint_path = Path(cfg.checkpoint) if cfg.checkpoint else \
            ensure_checkpoint_downloaded(cfg.hf_repo, cfg.hf_filename)
        print(f"loading VGGT-Omega on {cfg.device} from {checkpoint_path} ...")
        self.model = VGGTOmegaModel(patch_size=cfg.patch_size, enable_alignment=cfg.enable_alignment)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval().to(cfg.device)

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        predictions = run_inference(self.model, frames, cfg.device, cfg.image_resolution, cfg.mode, cfg.patch_size)

        # VGGT-Omega's own processing (width, height) -- see load_and_preprocess_images -- as
        # opposed to each frame's original size below.
        proc_image_size = (predictions["images"].shape[-1], predictions["images"].shape[-2])
        c2ws = np.stack([_w2c_to_c2w(predictions["extrinsic"][j]) for j in range(len(frames))])
        align = _gt_frame0_alignment(c2ws, frames[0].camera().c2w)  # one rigid transform for the run
        c2ws = align @ c2ws

        if out.spec.cloud:
            # Unprojected once over the whole batch, at VGGT-Omega's own processing resolution --
            # each frame's extrinsic/intrinsic are only meaningful together at that resolution,
            # before the per-frame depth gets resized back to its own original size below.
            world_pts = unproject_depth_map_to_point_map(
                predictions["depth"][..., 0], predictions["extrinsic"], predictions["intrinsic"]
            )  # (S, h, w, 3), VGGT-Omega's own world gauge
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
            K = rescale_vggt_omega_intrinsics_from_K(predictions["intrinsic"][j], (width, height), proc_image_size)
            out.write_frame(f.index, depth, c2ws[j], K, intrinsics_width=width)

        if out.spec.cloud:
            out.write_cloud(world_pts[cloud_mask], cols[cloud_mask])
            print(f"  wrote {cloud_mask.sum():,} points -> {out.cloud_path.relative_to(out.root)}")


if __name__ == "__main__":
    cli_main(VGGTOmega, description=__doc__)
