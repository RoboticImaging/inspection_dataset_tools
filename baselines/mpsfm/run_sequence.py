"""Run MP-SfM on the sequences of an inspection asset and write depth + pose + point-cloud
predictions back into the asset folder (predictions/{method}/...), in the same format the
ground truth is stored in — so inspection_viz's Source toggle picks them up unchanged.

MP-SfM (cvg/mpsfm) is COLMAP-style incremental Structure-from-Motion augmented with monocular
depth/normal priors -- fundamentally different in shape from every other baseline here: it isn't
a feed-forward model, it's given fixed camera intrinsics (we supply the gt ones) and *registers*
images one at a time via classical multi-view geometry, so (a) not every input frame necessarily
ends up registered -- frames MP-SfM couldn't place get no output at all, rather than a bad guess
-- and (b) its depth isn't a network's raw output: it starts from a monocular prior (Metric3Dv2
by default) and *refines* it during registration/bundle adjustment (per-image scale+shift, then a
depth/normal integration pass) -- see `run()` below for where that refined depth actually comes
from, since it isn't on the COLMAP-format reconstruction we read back at all.

Usage — inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/mpsfm.yaml

The data config picks one sequence of the asset or all of them (see configs/data/); the model
config holds MP-SfM's own knobs (its own `--conf` name, `--extract`, `--verbose`) and is reusable
across assets. Sequences run one after another, each its own `reconstruct.py` subprocess (see
`run_reconstruct` below for why).

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's torch/pyceres/pycolmap install.
"""

import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap
import yaml

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from baselines.common.writer import frame_stem
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

# The submodule (its own reconstruct.py/mpsfm package/configs/...), as opposed to this directory
# (baselines/mpsfm/), which only holds the glue that adapts it to inspection_viz's own dataset
# and output conventions.
MPSFM_ROOT = Path(__file__).resolve().parents[2] / "mpsfm"


@dataclasses.dataclass
class ModelConfig:
    # Name of one of mpsfm's own sfm config yamls (mpsfm/configs/, e.g. "sp-lg_m3dv2") -- picks
    # the feature/matcher/depth-prior pipeline reconstruct.py's own --conf flag would.
    conf: str = "sp-lg_m3dv2"
    # reconstruct.py's own --verbose: 0 is clean output, higher levels add debugging output and
    # (>=1) a 3D visualization dumped into --data_dir. Left at 0 by default since this runs one
    # sequence after another unattended.
    verbose: int = 0
    # Which cached priors to force re-extraction of this run: any of "sky"/"features"/"matches"/
    # "depth"/"normals", or [] to reuse whatever's already cached under the run's scratch cache_dir
    # (each sequence gets a fresh one, so in practice this only matters within a single sequence's
    # own retries).
    extract: list[str] = dataclasses.field(default_factory=list)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts MP-SfM's world gauge onto the gt one, as far as one
    frame can. Unlike this repo's feed-forward baselines, MP-SfM's own scale should already be
    correct -- its depth priors are metric (Metric3Dv2), and its bundle adjustment optimizes
    against them -- so this only has position/orientation left to fix, not scale: whichever
    registered frame comes first in the sequence (the "anchor", not necessarily frames[0] itself
    -- MP-SfM may not have registered that one) puts its own reconstructed pose exactly onto its
    gt one, and every other frame rides along on the same transform.
    """
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def stage_frames(frames: list[Frame], staging_dir: Path) -> dict[str, Frame]:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{frame_stem(index)}{suffix}" (original
    extension kept, unlike some other baselines here -- mpsfm's own extractors aren't known to be
    format-agnostic the way PIL/cv2-based loaders elsewhere in this repo are, so this plays it
    safe). Returns {staged filename: Frame}, since pycolmap.Image.name is only ever the bare
    filename -- this is how run() maps a registered image back to `frames[i]` afterward.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for f in frames:
        name = f"{frame_stem(f.index)}{f.rgb_path.suffix}"
        (staging_dir / name).symlink_to(f.rgb_path)
        by_name[name] = f
    return by_name


def write_intrinsics_yaml(frames_by_name: dict[str, Frame], out_path: Path) -> None:
    """mpsfm's own per-scene intrinsics file (--intrinsics_pth): one entry per distinct camera,
    `params: [fx, fy, cx, cy]` (its PINHOLE convention) plus which image filenames use it. Frames
    are grouped by their own (rounded) gt intrinsics rather than assumed shared, so this is still
    correct if a sequence's camera ever changes mid-run -- in the common case (one fixed render
    camera for the whole sequence) every frame lands in the same group and this is one entry.
    """
    groups: dict[tuple, list[str]] = {}
    for name, f in frames_by_name.items():
        K = f.camera().K
        key = (round(float(K[0, 0]), 3), round(float(K[1, 1]), 3),
               round(float(K[0, 2]), 3), round(float(K[1, 2]), 3))
        groups.setdefault(key, []).append(name)

    payload = {i: {"params": list(params), "images": sorted(names)}
               for i, (params, names) in enumerate(groups.items(), start=1)}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def run_reconstruct(data_dir: Path, cfg: ModelConfig) -> Path:
    """Runs reconstruct.py exactly as documented (its own README: `python reconstruct.py --conf
    ... --data_dir ... --images_dir ... --cache_dir ...`, from inside the submodule) -- it isn't
    packaged as an importable library, and needs pyceres/pycolmap (native extensions this
    baseline's image compiles from source), so this subprocess is the only supported entry point.
    Returns the sfm_outputs dir it wrote (reconstruct.py's own hardcoded `<data_dir>/sfm_outputs`),
    in standard COLMAP format.

    Deliberately does NOT pass `--intrinsics_pth`, even though reconstruct.py has the flag: its
    `SimpleParser` (mpsfm/data_proc/simple.py) only ever assigns its local `intrinsics` variable
    inside the `if intrinsics_pth is None:` branch that fills in the *default* path -- passing
    `--intrinsics_pth` explicitly skips that branch and crashes with `NameError: intrinsics is not
    defined`. The workaround is to rely on that same default instead: write our generated
    intrinsics.yaml to `<data_dir>/intrinsics.yaml` (write() below did) and omit the flag so
    `args.intrinsics_pth` stays `None` all the way through. Don't reintroduce `--intrinsics_pth`
    here without first checking whether upstream has fixed the parser.

    Also deliberately does NOT pass `--images_dir`, for the same kind of reason: reconstruct.py
    forwards it straight through as `SimpleParser(rgb_dir=images_dir, ...)` without ever wrapping
    it in `Path(...)` (argparse hands it over as a plain `str`), and `SimpleParser.__init__` only
    does that wrapping itself on the *default* path it builds when `rgb_dir is None` -- pass
    `--images_dir` explicitly and it instead calls `.iterdir()` straight on that string and
    crashes with `AttributeError: 'str' object has no attribute 'iterdir'`. The workaround is the
    same shape as the intrinsics one: rely on that default (`Path(data_dir) / "images"`) instead
    of the flag, so the caller must stage images at `<data_dir>/images` for this to resolve
    anything. Don't reintroduce `--images_dir` here without first checking whether upstream has
    fixed the parser.

    Also deliberately does NOT pass `--cache_dir`, same bug, third occurrence: `SimpleTest.__call__`
    (mpsfm/test/simple.py) only builds `cache_dir = data_dir / "cache_dir"` (a real Path) inside
    its `if cache_dir is None:` branch; passed explicitly, the raw argparse `str` instead flows
    straight through to `extract_mono.main`'s `export_dir.mkdir(...)` and crashes with
    `AttributeError: 'str' object has no attribute 'mkdir'`. Omitting the flag leaves the whole
    run's cache under `<data_dir>/cache_dir` instead of a caller-chosen path -- fine here since
    `data_dir` already lives inside our own throwaway TemporaryDirectory. Don't reintroduce
    `--cache_dir` here without first checking whether upstream has fixed the parser.
    """
    cmd = [
        sys.executable, "reconstruct.py",
        "--data_dir", str(data_dir),
        "--conf", cfg.conf,
        "--verbose", str(cfg.verbose),
    ]
    if cfg.extract:
        cmd += ["--extract", *cfg.extract]
    subprocess.run(cmd, cwd=MPSFM_ROOT, check=True)
    return data_dir / "sfm_outputs"


def c2w_from_image(image: "pycolmap.Image") -> np.ndarray:
    """4x4 camera-to-world matrix from a registered pycolmap Image's own `cam_from_world`
    (world-to-camera) pose."""
    w2c = image.cam_from_world
    R = w2c.rotation.matrix()
    t = np.asarray(w2c.translation)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix from a pycolmap PINHOLE camera's own (fx, fy, cx, cy) params -- the
    model we asked for via write_intrinsics_yaml. Bundle adjustment may have refined these
    slightly from what we fed in; read back what MP-SfM actually converged on, not our input.
    """
    if camera.model.name != "PINHOLE":
        raise SystemExit(
            f"expected a PINHOLE camera, got {camera.model.name} -- this baseline only knows how "
            "to read back (fx, fy, cx, cy) params in that order"
        )
    fx, fy, cx, cy = camera.params
    K = np.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    return K


def read_refined_depth(sfm_outputs: Path, image: "pycolmap.Image", width: int, height: int) -> np.ndarray | None:
    """The depth MP-SfM actually converged on for `image`, at the rgb resolution -- or None if it
    never got one (not every registered image necessarily has an activated depth map; see below).

    This does NOT come off `rec` (the COLMAP-format reconstruction read_reconstruct() returns):
    plain COLMAP has no notion of a per-image dense depth map, so `MpsfmReconstruction.write()`
    (mpsfm/sfm/scene/reconstruction/base.py) writes it separately, to its own `depths.h5` cache
    alongside the `rec/` COLMAP dir, keyed by COLMAP image_id under an `depths/` group -- the
    same file mpsfm's own `cache_depths()` always writes for every registered image whose
    `.depth.data` isn't None, so this is reading back exactly what that method put there, not a
    parallel reimplementation of any of MP-SfM's own refinement.

    Refinement itself happens in two stages inside the reconstruction (not here): each image's
    monocular prior gets a per-image scale+shift fit during registration/bundle adjustment
    (metric alignment against the reconstruction's own triangulated points), then, where enabled,
    a depth/normal integration pass (mpsfm/sfm/scene/image/integration.py) re-solves the depth
    field to be consistent with both the prior and its own predicted normals. `.depth.data` is
    whichever of those two a given image ended up with -- this only ever reads the *result*.

    It's also not at the rgb resolution to start with: mpsfm internally works at a downscaled
    "integration" resolution (its own `camera.init_int_data`, capped by the `normscale` config --
    long side around a few hundred px, not the native image size), so this resizes up the same
    way every other baseline here resizes a network-resolution depth back to native (cv2.resize,
    bilinear) -- it's an upsample of a coarser refined field, not extra resolution MP-SfM itself
    recovered.
    """
    h5_path = sfm_outputs / "depths.h5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        key = f"depths/{image.image_id}"
        if key not in f:
            return None
        depth = np.asarray(f[key], dtype=np.float32)
    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


class MPSfM(Baseline):
    """MP-SfM, driven through its own classical incremental-SfM CLI (see module docstring)."""

    name = "mpsfm"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        with tempfile.TemporaryDirectory(prefix="mpsfm-") as tmp:
            tmp_path = Path(tmp)
            # Staged at <data_dir>/images, not a sibling of data_dir: run_reconstruct() relies on
            # SimpleParser's default `Path(data_dir) / "images"` rather than passing --images_dir
            # (see its docstring) to dodge an upstream bug, so the images have to actually live
            # there for that default to resolve to anything.
            data_dir = tmp_path / "data"
            images_dir = data_dir / "images"
            frames_by_name = stage_frames(frames, images_dir)

            data_dir.mkdir(parents=True, exist_ok=True)
            write_intrinsics_yaml(frames_by_name, data_dir / "intrinsics.yaml")

            print(f"running {len(frames)} frames (conf={self.config.conf}) ...")
            sfm_outputs = run_reconstruct(data_dir, self.config)

            # The COLMAP-format reconstruction lives under sfm_outputs/rec/, not sfm_outputs/
            # itself -- MpsfmReconstruction.write() (mpsfm/sfm/scene/reconstruction/base.py)
            # nests it there alongside the depths.h5 cache read_refined_depth() reads separately.
            rec = pycolmap.Reconstruction(sfm_outputs / "rec")
            registered = {img.name: img for img in rec.images.values()}
            missing = set(frames_by_name) - set(registered)
            if missing:
                print(f"  MP-SfM registered {len(registered)}/{len(frames)} frames -- "
                     f"{len(missing)} unregistered frame(s) get no output")
            if not registered:
                raise SystemExit("MP-SfM registered no frames at all for this sequence")

            # Whichever of our frames MP-SfM registered that comes first in the sequence, not
            # necessarily frames[0] itself (see _gt_frame0_alignment's docstring).
            anchor_name = min(registered, key=lambda n: frames_by_name[n].index)
            anchor_frame = frames_by_name[anchor_name]
            anchor_c2w = c2w_from_image(registered[anchor_name])
            align = _gt_frame0_alignment(anchor_c2w, anchor_frame.camera().c2w)

            no_depth = 0
            for name, image in registered.items():
                f = frames_by_name[name]
                width, height = image_size(f.rgb_path)
                depth = read_refined_depth(sfm_outputs, image, width, height)
                if depth is None:
                    no_depth += 1
                    depth = np.full((height, width), np.nan, dtype=np.float32)
                c2w = align @ c2w_from_image(image)
                K = K_from_camera(rec.cameras[image.camera_id])
                out.write_frame(f.index, depth, c2w, K, intrinsics_width=rec.cameras[image.camera_id].width)
            if no_depth:
                print(f"  {no_depth}/{len(registered)} registered frame(s) had no refined depth "
                     "(never activated) -- written as all-NaN")

            if out.spec.cloud:
                pts = np.array([p.xyz for p in rec.points3D.values()])
                cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
                if len(pts):
                    pts = np.asarray(np_transform_points(pts, align))
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
                else:
                    print("  no triangulated points to write a cloud from")


def np_transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Applies a (4,4) rigid transform to (N,3) points without pulling in trimesh just for this."""
    return points @ transform[:3, :3].T + transform[:3, 3]


if __name__ == "__main__":
    cli_main(MPSfM, description=__doc__)
