"""Run DetectorFreeSfM (zju3dv/DetectorFreeSfM, CVPR 2024) on the sequences of an inspection
asset and write depth + pose + point-cloud predictions back into the asset folder
(predictions/{method}/...), in the same format the ground truth is stored in — so
inspection_viz's Source toggle picks them up unchanged.

Like baselines/colmap and baselines/mpsfm, this is fundamentally different in shape from the
feed-forward baselines here: it self-calibrates and *registers* images one at a time (LoFTR/
ASpanFormer/MatchFormer dense coarse matching, feeding a modified-COLMAP incremental mapper, then
a learned multi-view feature-metric refinement pass on top -- see baselines/DetectorFreeSfM/
Dockerfile's own header), so a frame it couldn't place gets no output at all, rather than a bad
guess, and its whole reconstruction is correct only up to an unknown, unrecoverable global scale
(no metric prior anywhere in this pipeline, unlike mpsfm) -- same `_gt_frame0_alignment` convention
as baselines/colmap/glomap/VGGT-SLAM/VGGT-Long: one rigid (not similarity) transform, from
whichever registered frame comes first in the sequence onto its own gt pose, fixes
rotation+translation for the whole run but leaves scale as reconstructed.

DetectorFreeSfM itself never produces per-pixel depth (its own README: "SfM result ... can be
visualized by `colmap gui`" -- a sparse, refined point cloud + poses, nothing dense). Real depth
here, like baselines/colmap and baselines/glomap, comes from chaining dense MVS (image_undistorter
-> patch_match_stereo -> stereo_fusion) onto the sparse reconstruction afterward, via the same
modified `colmap` CLI binary this baseline's own Dockerfile builds -- see `ModelConfig.dense`'s own
docstring for why this is the LEAST validated part of this file, and how to turn it off if the
old-vintage fork's dense commands don't cooperate.

Driven entirely as a subprocess of DetectorFreeSfM/eval_dataset.py (its own hydra CLI, `+demo=
dfsfm.yaml` plus a handful of overrides -- see `run_dfsfm` below): it isn't packaged as an
importable library either (its own `@hydra.main`-decorated `main()` parses `sys.argv` itself), the
same reason baselines/mpsfm/run_sequence.py subprocesses reconstruct.py instead of importing it.
pycolmap is used only afterward, to read the resulting sparse reconstruction back (poses,
intrinsics) -- the same split every classical-SfM baseline here uses.

REQUIRES a manually-downloaded weight/ dir under the DetectorFreeSfM submodule root (its own
INSTALL.md: a Google Drive weight.tar, extracted to DetectorFreeSfM/weight/ -- confirmed by a real
run to need more than just multiview_matcher.ckpt: the coarse matcher has its own hardcoded weight
path too, e.g. weight/outdoor_ds.ckpt for the default `loftr_official` -- see
baselines/DetectorFreeSfM/build_container.sh's own header for the full list). This baseline cannot
fetch weight/ itself.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/detector-free-sfm.yaml

The data config picks one sequence of the asset or all of them (see configs/data/); the model
config holds DetectorFreeSfM's own knobs (coarse matcher, refinement, dense MVS -- see
`ModelConfig` below) and is reusable across assets. Sequences run one after another, each its own
scratch SfM_dataset/ workspace, mirroring DetectorFreeSfM's own required on-disk layout (its
README: `repo_path/SfM_dataset/<dataset_name>/<scene_name>/images/...`).

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's modified `colmap` CLI, torch, and
pycolmap install.
"""

# Postponed annotation evaluation (PEP 563): lets `int | None` etc. below run on this baseline's
# own Python 3.9 (see baselines/DetectorFreeSfM/Dockerfile's own header for why 3.9, not this
# repo's more usual 3.10+) without needing `typing.Optional` -- see baselines/registry.py's own
# identical comment for the full reasoning.
from __future__ import annotations

import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import trimesh

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from baselines.common.writer import frame_stem
from inspection_viz.colmap import read_array
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size

# The submodule (its own eval_dataset.py/src/hydra_configs/...), as opposed to this directory
# (baselines/DetectorFreeSfM/), which only holds the glue that adapts it to inspection_viz's own
# dataset and output conventions. Same `parents[2]` depth as baselines/mpsfm/run_sequence.py's
# identically-shaped MPSFM_ROOT (this file also lives two directories under the repo root).
DFSFM_ROOT = Path(__file__).resolve().parents[2] / "DetectorFreeSfM"

# Fixed names for the throwaway SfM_dataset/ layout `run_dfsfm` stages each sequence into -- see
# its own docstring for why these don't need to vary run to run (each sequence gets a fresh
# tempdir, so there's never more than one dataset/scene alive at once).
_DATASET_NAME = "run"
_SCENE_NAME = "scene"


def _omega_literal(value) -> str:
    """`value` formatted as an OmegaConf/hydra CLI override literal -- hydra parses the text
    after `=` with its own (YAML-like) grammar, not Python's `str()`, so bool/None/list need
    their own spellings (lowercase true/false, null, [a,b] with no spaces)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_omega_literal(v) for v in value) + "]"
    return str(value)


def _hydra_overrides(prefix: str, cfg) -> list[str]:
    """Every field of dataclass `cfg` as hydra `key=value` CLI overrides, `{prefix}.{field}=value`
    -- field names throughout `NeuralSfMConfig`/`ColmapCfgConfig` below match DetectorFreeSfM's
    own hydra config keys exactly (see hydra_configs/demo/dfsfm.yaml), the same "no name-mapping
    table needed" convention baselines/colmap/run_sequence.py's `_flags` uses for colmap's own CLI
    flags. Unlike `_flags`, `None`-valued fields ARE included (as hydra's own `null`) -- hydra has
    no "omit to keep the compiled-in default" concept the way colmap's argparse CLI does; every
    field here always overrides whatever hydra_configs/demo/dfsfm.yaml set. Recurses into nested
    dataclass fields (`ColmapCfgConfig.reregistration`) rather than trying to format one as a
    single literal -- e.g. `colmap_cfg.reregistration.abs_pose_max_error=12`, matching
    hydra_configs/demo/dfsfm.yaml's own nesting exactly."""
    overrides = []
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        key = f"{prefix}.{f.name}"
        if dataclasses.is_dataclass(val):
            overrides += _hydra_overrides(key, val)
        else:
            overrides.append(f"{key}={_omega_literal(val)}")
    return overrides


def _flags(prefix: str, cfg) -> list[str]:
    """Every field of dataclass `cfg` as plain colmap CLI args, `--{prefix}.{field} value` -- NOT
    hydra overrides (unlike `_hydra_overrides` above): `patch_match`/`stereo_fusion` drive the
    plain `colmap` binary's own argparse-style CLI directly, the same shape
    baselines/colmap/run_sequence.py's own `_flags` uses (duplicated here under the same name --
    this module's own hydra-override helper is named `_hydra_overrides` instead, precisely so the
    two don't collide). bool becomes "1"/"0" (colmap's own CLI convention); `None` is omitted
    entirely (colmap then applies its own compiled-in default)."""
    args = []
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if val is None:
            continue
        if isinstance(val, bool):
            val = "1" if val else "0"
        args += [f"--{prefix}.{f.name}", str(val)]
    return args


# ---------------------------------------------------------------------------------------------
# ModelConfig: DetectorFreeSfM's own `neuralsfm:`/`colmap_cfg:` hydra config groups (subset --
# see each dataclass's own docstring for what's deliberately fixed instead of exposed), plus the
# dense-MVS stage this file chains on afterward (duplicated verbatim from
# baselines/colmap/run_sequence.py's own PatchMatchConfig/StereoFusionConfig -- same repo
# convention baselines/glomap/run_sequence.py's own docstring calls out for the same fields).
# Defaults throughout mirror hydra_configs/demo/dfsfm.yaml's own values.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class NeuralSfMConfig:
    """hydra_configs/demo/dfsfm.yaml's own `neuralsfm:` block. Only the fields meaningful to tune
    for a single self-calibrated (no prior pose/intrinsics, no gt-based eval) run are exposed --
    `triangulation_mode`, `use_prior_intrin`, and `close_eval` stay fixed (False/False/True) in
    `run_dfsfm` below rather than as fields here: flipping the first two needs a prior_pose_dir/
    prior_intrin_dir this baseline never supplies (matching baselines/colmap's own
    self-calibrated-from-scratch convention, not this repo's gt intrinsics), and the third needs a
    gt_pose_dir DetectorFreeSfM would score against internally -- redundant with this repo's own
    evaluator, which scores predictions/ afterward instead."""
    NEUSFM_coarse_matcher: str = "loftr_official"     # 'loftr_official' | 'aspanformer' | 'matchformer'
    NEUSFM_coarse_match_type: str = "coarse_only"
    NEUSFM_coarse_match_thr: float = 0.2
    NEUSFM_coarse_match_round: int | None = None
    NEUSFM_refinement_chunk_size: int = 2000    # lower on GPUs with <12GB VRAM (README's own tip)
    NEUSFM_enable_post_optimization: bool = True   # False -> no colmap_refined/ ever gets written,
                                                    # see run_dfsfm's own docstring
    img_resize: int = 1200
    img_pair_strategy: str = "exhaustive"
    refine_iter_n_times: int = 2


@dataclasses.dataclass
class ReregistrationConfig:
    """hydra_configs/demo/dfsfm.yaml's own `colmap_cfg.reregistration:` block."""
    abs_pose_max_error: float = 12
    abs_pose_min_num_inliers: int = 30
    abs_pose_min_inlier_ratio: float = 0.25
    filter_max_reproj_error: float = 5


@dataclasses.dataclass
class ColmapCfgConfig:
    """hydra_configs/demo/dfsfm.yaml's own `colmap_cfg:` block -- knobs for the modified-COLMAP
    geometry refinement stage itself (src/sfm_runner/sfm_model_geometry_refiner.py), NOT the
    separate dense-MVS stage this file chains on afterward (that's `PatchMatchConfig`/
    `StereoFusionConfig` below, plain vanilla `colmap` CLI flags instead)."""
    ImageReader_camera_mode: str = "auto"
    ImageReader_single_camera: bool = False
    min_model_size: int = 3
    no_refine_intrinsics: bool = False
    use_pba: bool = False
    n_threads: int = 16
    reregistration: ReregistrationConfig = dataclasses.field(default_factory=ReregistrationConfig)


@dataclasses.dataclass
class PatchMatchConfig:
    """patch_match_stereo's --PatchMatchStereo.* flags. NOT a straight copy of
    baselines/colmap/run_sequence.py's own dataclass of the same name (an earlier version of this
    file was, and said so explicitly) -- this fork's own binary genuinely differs from vanilla
    COLMAP here, confirmed directly against its own `colmap patch_match_stereo --help` (COLMAP 3.8,
    commit 03a3bc64) inside the built image, not assumed: it has NO `--PatchMatchStereo.num_threads`
    flag at all (patch_match_stereo is GPU-bound; `gpu_index` is the real per-invocation knob,
    below) -- baselines/colmap's own field list includes one anyway, which would have made every
    dense run here fail immediately with an "unrecognised option" error the first time `dense`
    stayed at its default (true). Also gained `gpu_index`/`depth_min`/`depth_max`/`cache_size`
    (present here, absent from baselines/colmap's own list -- that CLI's own current, much-newer
    `colmap` submodule apparently doesn't expose these, or exposes them differently). Every default
    below is this fork's own --help output verbatim, not colmap.py's assumed ones -- notably
    `geom_consistency` defaults to True here (vanilla COLMAP's own default is False, per
    baselines/colmap/run_sequence.py's own comment) and `sigma_spatial` defaults to -1
    (auto), not 5.0."""
    max_image_size: int = -1
    gpu_index: str = "-1"          # "-1" = auto-select, same convention as ImageReader/mapper's
                                    # own gpu_index-style flags elsewhere in this repo
    depth_min: float = -1          # -1 = auto
    depth_max: float = -1          # -1 = auto
    window_radius: int = 5
    window_step: int = 1
    sigma_spatial: float = -1      # -1 = auto (NOT 5.0 -- see docstring)
    sigma_color: float = 0.2
    num_samples: int = 15
    ncc_sigma: float = 0.6
    min_triangulation_angle: float = 1.0
    incident_angle_sigma: float = 0.9
    num_iterations: int = 5
    geom_consistency: bool = True  # this fork's own default (NOT vanilla COLMAP's False -- see
                                    # docstring); also enables writing .geometric.bin depth maps,
                                    # which `depth_source`'s own default below expects to exist
    geom_consistency_regularizer: float = 0.3
    geom_consistency_max_cost: float = 3.0
    filter: bool = True
    filter_min_ncc: float = 0.1
    filter_min_triangulation_angle: float = 3.0
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    cache_size: int = 32
    allow_missing_files: bool = False
    write_consistency_graph: bool = False


@dataclasses.dataclass
class StereoFusionConfig:
    """stereo_fusion's --StereoFusion.* flags -- confirmed directly against this fork's own
    `colmap stereo_fusion --help` (see PatchMatchConfig's own docstring for how/why). Unlike
    PatchMatchConfig, the field LIST here happens to match baselines/colmap's own dataclass
    exactly except for one addition (`num_threads`, real here, confirmed via --help -- CPU-side
    fusion, unlike patch_match_stereo's GPU-side `gpu_index`); values differ in a few places
    (`max_reproj_error`, `check_num_images`) only in float/int spelling, not substance."""
    mask_path: str | None = None
    num_threads: int = -1
    max_image_size: int = -1
    min_num_pixels: int = 5
    max_num_pixels: int = 10000
    max_traversal_depth: int = 100
    max_reproj_error: float = 2.0
    max_depth_error: float = 0.01
    max_normal_error: float = 10.0
    check_num_images: int = 50
    cache_size: int = 32
    use_cache: bool = False


@dataclasses.dataclass
class ModelConfig:
    neuralsfm: NeuralSfMConfig = dataclasses.field(default_factory=NeuralSfMConfig)
    colmap_cfg: ColmapCfgConfig = dataclasses.field(default_factory=ColmapCfgConfig)
    # Dense MVS chained onto DetectorFreeSfM's own sparse refined reconstruction, via the same
    # modified `colmap` CLI binary this baseline's Dockerfile builds (image_undistorter ->
    # patch_match_stereo -> stereo_fusion, identical shape to baselines/colmap's and
    # baselines/glomap's own dense stage) -- DetectorFreeSfM itself only ever produces a sparse
    # point cloud + poses (see module docstring), so this is the only way this baseline can write
    # real per-pixel depth at all.
    #
    # image_undistorter/patch_match_stereo/stereo_fusion CONFIRMED present on this fork's own
    # binary (COLMAP 3.8, commit 03a3bc64 -- `colmap --help` inside the built image), and
    # PatchMatchConfig/StereoFusionConfig's field lists above are each confirmed directly against
    # that binary's own `--help` output, not copied blind from baselines/colmap's newer,
    # differently-built `colmap` -- see each dataclass's own docstring for exactly what differs
    # and why. `--set model.dense=false` remains available as an escape hatch if a *different*
    # machine's build of this same fork ever disagrees (e.g. a CPU-only patch_match_stereo build,
    # or a future INSTALL.md change) -- every frame this baseline registers still gets a pose
    # either way, just with all-NaN depth and the sparse refined cloud (`points3D`) instead of a
    # dense fused one.
    dense: bool = True
    patch_match: PatchMatchConfig = dataclasses.field(default_factory=PatchMatchConfig)
    stereo_fusion: StereoFusionConfig = dataclasses.field(default_factory=StereoFusionConfig)
    # "geometric" (COLMAP's own --input_type default too) -- NOT "photometric": PatchMatchConfig's
    # own geom_consistency defaults True on this fork (see its own docstring), which is what
    # actually produces .geometric.bin depth maps; .photometric.bin maps are written regardless,
    # but geometric is the one stereo_fusion (and this fork's own default) treats as authoritative.
    depth_source: str = "geometric"   # "photometric" or "geometric"


def stage_frames(frames: list[Frame], staging_dir: Path) -> dict[str, Frame]:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{frame_stem(index)}{suffix}" (original
    extension kept). Returns {staged filename: Frame} -- pycolmap.Image.name is only ever the bare
    filename, so this is how `run()` maps a registered image back to a Frame afterward. Same shape
    as baselines/mpsfm and baselines/colmap's own `stage_frames`."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for f in frames:
        name = f"{frame_stem(f.index)}{f.rgb_path.suffix}"
        (staging_dir / name).symlink_to(f.rgb_path)
        by_name[name] = f
    return by_name


def run_dfsfm(sfm_dataset_dir: Path, cfg: NeuralSfMConfig, colmap_cfg: ColmapCfgConfig) -> Path:
    """Runs DetectorFreeSfM/eval_dataset.py exactly as its own README documents (`python
    eval_dataset.py +demo=dfsfm.yaml`), as a subprocess (see module docstring for why), with
    `dataset_base_dir`/`dataset_name`/`scene_list` overridden to point at this run's own scratch
    `sfm_dataset_dir` (must already contain `<dataset_name>/<scene_name>/images/` -- callers stage
    that via `stage_frames`) instead of the demo config's own placeholder path.

    `use_prior_intrin`/`neuralsfm.triangulation_mode` are hardcoded False here, NOT exposed as
    `NeuralSfMConfig` fields -- see that class's own docstring: this baseline always
    self-calibrates from scratch, matching baselines/colmap's own convention.
    `neuralsfm.close_eval=true` is likewise hardcoded: no gt_pose_dir is ever staged (this repo's
    own evaluator scores predictions/ afterward instead), so DetectorFreeSfM's own internal
    evaluator must stay off or every scene would try to read a "poses" dir that doesn't exist.

    Returns the `colmap_refined` dir DetectorFreeSfM wrote
    (`<sfm_dataset_dir>/<dataset_name>/<scene_name>/<method_name>/colmap_refined`, standard COLMAP
    binary format) -- found by globbing for it rather than recomputing `method_name`
    (src/detectorfree_sfm.py's own string-building logic) here a second time, so this doesn't
    quietly go stale if upstream ever changes that logic. Raises if it's missing (post-optimization
    failed or never ran -- see `NeuralSfMConfig.NEUSFM_enable_post_optimization`'s own docstring:
    disabling it means this never exists, and this file doesn't know how to fall back to
    `colmap_coarse/<best_model_id>` instead) or ambiguous (should never happen: each call gets its
    own fresh `sfm_dataset_dir`, so there is only ever one scene, hence one method_name, here).
    """
    work_dir = sfm_dataset_dir / _DATASET_NAME / _SCENE_NAME
    cmd = [
        sys.executable, "eval_dataset.py", "+demo=dfsfm.yaml",
        f"dataset_base_dir={sfm_dataset_dir}",
        f"dataset_name={_DATASET_NAME}",
        f"scene_list=[{_SCENE_NAME}]",
        "use_prior_intrin=false",
        "verbose=false",
        "visualize=false",
        "neuralsfm.triangulation_mode=false",
        "neuralsfm.close_eval=true",
        *_hydra_overrides("neuralsfm", cfg),
        *_hydra_overrides("colmap_cfg", colmap_cfg),
    ]
    subprocess.run(cmd, cwd=DFSFM_ROOT, check=True)

    matches = sorted(work_dir.glob("*/colmap_refined"))
    if not matches:
        raise SystemExit(f"DetectorFreeSfM produced no colmap_refined/ under {work_dir} -- see "
                         "its own stdout/stderr above for what went wrong (or check "
                         "NEUSFM_enable_post_optimization is still true)")
    if len(matches) > 1:
        raise SystemExit(f"expected exactly one method_name subdir under {work_dir}, found "
                         f"{len(matches)}: {[m.parent.name for m in matches]}")
    return matches[0]


def c2w_from_image(image: "pycolmap.Image") -> np.ndarray:
    """4x4 camera-to-world matrix from a registered pycolmap Image's own `qvec`/`tvec`
    (world-to-camera quaternion + translation) -- pycolmap==0.3.0's own pose representation (this
    baseline's pinned version, per DetectorFreeSfM/requirements.txt). Do NOT copy
    baselines/colmap/run_sequence.py's same-named helper: that file's own (much newer) pycolmap
    exposes `cam_from_world()` instead, an unrelated API shape."""
    R = image.rotation_matrix()
    t = np.asarray(image.tvec)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix for `camera`, via pycolmap's own `calibration_matrix()` -- present in
    pycolmap==0.3.0 too (confirmed against that tag's own reconstruction/camera.cc bindings), same
    method baselines/colmap/run_sequence.py's own K_from_camera uses, for the same reason: don't
    assume any particular camera model, handle whatever's actually in the reconstruction."""
    return camera.calibration_matrix().astype(np.float64)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts DetectorFreeSfM's world gauge onto the gt one, as far
    as one frame can -- its own scale is left exactly as reconstructed (see module docstring for
    why). Whichever registered frame comes first in the sequence (the "anchor") puts its own
    reconstructed pose exactly onto its gt one, and every other frame rides along on the same
    transform. Same convention as baselines/colmap/glomap/VGGT-SLAM/VGGT-Long's own identically-
    named helpers."""
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def read_depth_map(dense_dir: Path, image_name: str, depth_source: str) -> np.ndarray | None:
    """The dense depth map patch_match_stereo wrote for `image_name`, or None if it never got
    one. `0` marks an unestimated pixel in COLMAP's own binary format -- converted to nan here,
    same convention as baselines/colmap/run_sequence.py's identically-named helper (duplicated
    verbatim from it)."""
    path = dense_dir / "stereo" / "depth_maps" / f"{image_name}.{depth_source}.bin"
    if not path.is_file():
        return None
    depth = read_array(path)
    return np.where(depth > 0, depth, np.nan).astype(np.float32)


class DetectorFreeSfM(Baseline):
    """DetectorFreeSfM, driven through its own eval_dataset.py CLI (see module docstring)."""

    name = "detector-free-sfm"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="detector-free-sfm-") as tmp:
            tmp_path = Path(tmp)
            sfm_dataset_dir = tmp_path / "SfM_dataset"
            images_dir = sfm_dataset_dir / _DATASET_NAME / _SCENE_NAME / "images"
            frames_by_name = stage_frames(frames, images_dir)

            print(f"running {len(frames)} frames (coarse_matcher="
                 f"{cfg.neuralsfm.NEUSFM_coarse_matcher!r}) ...")
            colmap_refined_dir = run_dfsfm(sfm_dataset_dir, cfg.neuralsfm, cfg.colmap_cfg)

            rec = pycolmap.Reconstruction(str(colmap_refined_dir))
            registered = {img.name: img for img in rec.images.values()}
            missing = set(frames_by_name) - set(registered)
            if not registered:
                raise SystemExit("DetectorFreeSfM registered no frames at all for this sequence")
            if missing:
                print(f"  registered {len(registered)}/{len(frames)} frames -- "
                     f"{len(missing)} unregistered frame(s) get no output")

            # Whichever of our frames DetectorFreeSfM registered that comes first in the
            # sequence, not necessarily frames[0] itself (see _gt_frame0_alignment's docstring).
            anchor_name = min(registered, key=lambda n: frames_by_name[n].index)
            anchor_frame = frames_by_name[anchor_name]
            anchor_c2w = c2w_from_image(registered[anchor_name])
            align = _gt_frame0_alignment(anchor_c2w, anchor_frame.camera().c2w)

            dense_dir = None
            if cfg.dense:
                dense_dir = colmap_refined_dir.parent / "dense"
                print("  running dense MVS (image_undistorter -> patch_match_stereo -> "
                     "stereo_fusion) on the modified colmap binary ...")
                subprocess.run([
                    "colmap", "image_undistorter",
                    "--image_path", str(images_dir),
                    "--input_path", str(colmap_refined_dir),
                    "--output_path", str(dense_dir),
                ], check=True)
                subprocess.run([
                    "colmap", "patch_match_stereo",
                    "--workspace_path", str(dense_dir),
                    *_flags("PatchMatchStereo", cfg.patch_match),
                ], check=True)
                fused_path = dense_dir / "fused.ply"
                subprocess.run([
                    "colmap", "stereo_fusion",
                    "--workspace_path", str(dense_dir),
                    "--input_type", cfg.depth_source,
                    "--output_path", str(fused_path),
                    *_flags("StereoFusion", cfg.stereo_fusion),
                ], check=True)
                # image_undistorter's own output model -- pixel-aligned with the depth maps
                # (not colmap_refined_dir, read above only for its poses/points3D).
                dense_rec = pycolmap.Reconstruction(str(dense_dir / "sparse"))
                dense_by_name = {img.name: img for img in dense_rec.images.values()}

            no_depth = 0
            for name, image in registered.items():
                f = frames_by_name[name]
                depth = None
                # camera defaults to the sparse/refined model's own -- overridden below by the
                # dense/undistorted model's camera when available, since that's the one
                # pixel-aligned with the depth map (image_undistorter can change the intrinsics/
                # resolution to remove distortion; it never changes the pose -- see below).
                camera = rec.cameras[image.camera_id]
                if cfg.dense and name in dense_by_name:
                    dense_image = dense_by_name[name]
                    camera = dense_rec.cameras[dense_image.camera_id]
                    depth = read_depth_map(dense_dir, dense_image.name, cfg.depth_source)
                if depth is None:
                    no_depth += 1
                    width, height = image_size(f.rgb_path)
                    depth = np.full((height, width), np.nan, dtype=np.float32)
                # Pose comes from the sparse/refined model always, dense or not -- undistortion
                # only changes a camera's intrinsics/resolution, never its (R, t), so this stays
                # correct for the dense/undistorted camera's own pixel grid too.
                c2w = align @ c2w_from_image(image)
                K = K_from_camera(camera)
                out.write_frame(f.index, depth, c2w, K, intrinsics_width=camera.width)
            if no_depth:
                reason = "never estimated" if cfg.dense else "model.dense=false"
                print(f"  {no_depth}/{len(registered)} registered frame(s) had no dense depth "
                     f"map ({reason}) -- written as all-NaN")

            if out.spec.cloud:
                if cfg.dense and dense_dir is not None and (dense_dir / "fused.ply").is_file():
                    cloud = trimesh.load(dense_dir / "fused.ply", process=False)
                    pts = trimesh.transform_points(np.asarray(cloud.vertices), align)
                    cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} dense points -> "
                         f"{out.cloud_path.relative_to(out.root)}")
                else:
                    # No dense stage (model.dense=false) or stereo_fusion produced nothing --
                    # fall back to DetectorFreeSfM's own sparse, refined SfM points, same as
                    # baselines/mpsfm's own sparse-cloud fallback.
                    pts = np.array([p.xyz for p in rec.points3D.values()])
                    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
                    if len(pts):
                        pts = pts @ align[:3, :3].T + align[:3, 3]
                        out.write_cloud(pts, cols)
                        print(f"  wrote {len(pts):,} sparse points -> "
                             f"{out.cloud_path.relative_to(out.root)}")
                    else:
                        print("  no triangulated points to write a cloud from")


if __name__ == "__main__":
    cli_main(DetectorFreeSfM, description=__doc__)
