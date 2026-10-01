"""Run classical COLMAP structure-from-motion + multi-view stereo on the sequences of an
inspection asset and write depth + pose + point-cloud predictions back into the asset folder
(predictions/{method}/...), in the same format the ground truth is stored in -- so
inspection_viz's Source toggle picks them up unchanged.

Like baselines/mpsfm/run_sequence.py, this is fundamentally different in shape from the
feed-forward baselines here: it isn't a network, it self-calibrates and *registers* images one at
a time via classical incremental SfM, so (a) not every input frame necessarily ends up
registered -- a frame COLMAP couldn't place gets no output at all, rather than a bad guess -- and
(b) unlike mpsfm (which starts from a metric monocular depth prior, so only its rotation and
translation need fixing against gt -- see `_gt_frame0_alignment` below), plain COLMAP has no
metric reference at all: its whole reconstruction is correct only up to an unknown, unrecoverable
global scale. That matches VGGT-SLAM/VGGT-Long's own situation (see their `_gt_frame0_alignment`-
shaped helpers), and this file follows the same accepted convention: one rigid (not similarity)
transform, from whichever registered frame comes first in the sequence onto its own gt pose,
fixes rotation+translation for the whole run but leaves scale as COLMAP produced it.

Driven entirely through the `colmap` CLI (built from the `colmap` submodule -- see
baselines/colmap/Dockerfile): feature_extractor -> exhaustive_matcher -> mapper (incremental SfM)
-> model_merger (stitching together any disconnected sub-models the incremental mapper produced)
-> image_undistorter/patch_match_stereo/stereo_fusion (dense MVS). pycolmap is used only
afterward, to read the resulting reconstruction back (poses, intrinsics) -- the same split
mpsfm's own run_sequence.py uses.

This drives the *lower-level* feature_extractor/exhaustive_matcher/mapper commands directly,
not the `automatic_reconstructor` convenience wrapper a first version of this file used --
automatic_reconstructor never exposes most of what those commands can actually tune (confirmed by
reading controllers/automatic_reconstruction.cc: it only ever touches a handful of top-level
fields -- type/num_threads/use_gpu/gpu_index plus its own --quality presets -- on the underlying
SiftExtraction/FeatureMatching/TwoViewGeometry/Mapper options structs). Every `ModelConfig` field
below exists because it is a real, individually-settable flag on one of these commands, confirmed
directly against this exact `colmap` submodule checkout's source (struct default member
initializers, or -- for patch_match_stereo/stereo_fusion -- the actual running binary's own
printed option dump), not general COLMAP familiarity: an earlier, less careful assumption here
(that image_undistorter always outputs a PINHOLE camera model) crashed this baseline's first real
run, right after its ~2-minute dense stage had already finished, so every default in this file has
been checked against the source rather than trusted from memory.

The merge step's "fall back to the bigger reconstruction if a pairwise merge fails" logic comes
from this project's own colmap_merge_loop_smart.sh (see /home/shared/eccv_2026_data) -- lifted as
directional guidance (a non-obvious trick worth keeping), not reproduced literally: that script
(and its sparse/dense sibling scripts) point straight at a whole dataset folder and persist their
workspace on disk, whereas this stages exactly the selected frames (so frame_start/frame_end/
frame_stride apply) into a scratch tempdir and writes only the final predictions, like every
other baseline here.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/colmap.yaml

The data config picks one sequence of the asset or all of them (see configs/data/); the model
config holds COLMAP's own knobs (feature extraction, matching, mapping, dense stereo -- see
`ModelConfig` below, one nested group per underlying colmap command/option-namespace) and is
reusable across assets. Sequences run one after another, each its own scratch workspace.

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's `colmap` CLI binary and pycolmap.
"""

import dataclasses
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import os

import numpy as np
import pycolmap
import trimesh

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from baselines.common.writer import frame_stem
from inspection_viz.colmap import read_array
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size


def _flags(prefix: str | None, cfg, skip: frozenset = frozenset()) -> list[str]:
    """Every field of dataclass `cfg` as CLI args -- `--{prefix}.{field} value` if `prefix` is
    given, else bare `--{field} value` (matching e.g. view_graph_calibrator's own unprefixed
    flags in the glomap baseline). Field names throughout this file are chosen to match colmap's
    own flag suffix exactly, so this needs no per-field name-mapping table. `None`-valued fields
    are omitted entirely (colmap then applies its own compiled-in default -- used for fields
    whose "unset" sentinel isn't a clean literal, e.g. colmap's own -1/INT_MAX "unbounded"
    conventions); bool becomes "1"/"0" (colmap's own convention throughout its CLI); everything
    else is str()'d as-is. `skip` names dataclass fields that exist for this file's own bookkeeping
    but aren't real colmap flags (e.g. a config-only "enabled" toggle)."""
    args = []
    for f in dataclasses.fields(cfg):
        if f.name in skip:
            continue
        val = getattr(cfg, f.name)
        if val is None:
            continue
        if isinstance(val, bool):
            val = "1" if val else "0"
        flag = f"--{prefix}.{f.name}" if prefix else f"--{f.name}"
        args += [flag, str(val)]
    return args


# ---------------------------------------------------------------------------------------------
# ModelConfig: one nested dataclass per colmap CLI command/option-namespace. Every default below
# is colmap's own compiled-in default (verified against this submodule checkout's source, cited
# per group), EXCEPT where this file deliberately picks a different one for this dataset (called
# out explicitly, e.g. ImageReaderConfig.camera_model/single_camera).
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ImageReaderConfig:
    """feature_extractor's --ImageReader.* flags (controllers/image_reader.h:41-88)."""
    camera_model: str = "SIMPLE_PINHOLE"   # colmap's own raw default is SIMPLE_RADIAL; we
                                            # self-calibrate a distortion-free model instead
    single_camera: bool = True             # colmap's own raw default is False; ours is a fixed
                                            # render rig, one shared camera for the whole sequence
    single_camera_per_folder: bool = False
    single_camera_per_image: bool = False
    default_focal_length_factor: float = 1.2   # used only if no camera_params and no EXIF focal
    camera_params: str | None = None       # e.g. "fx,fy,cx,cy" -- None = self-calibrate from scratch
    mask_path: str | None = None
    camera_mask_path: str | None = None
    existing_camera_id: int | None = None  # reuse an existing database camera_id verbatim


@dataclasses.dataclass
class FeatureExtractionConfig:
    """feature_extractor's --FeatureExtraction.* flags (feature/extractor.h:58-94)."""
    use_gpu: bool = True
    num_threads: int = -1          # -1 = auto (all cores)
    gpu_index: str = "-1"          # "-1" = auto-select
    max_image_size: int = -1       # -1 = no downsampling before extraction


@dataclasses.dataclass
class SiftExtractionConfig:
    """feature_extractor's --SiftExtraction.* flags (feature/sift.h:41-101) -- the SIFT
    algorithm's own knobs, on top of FeatureExtractionConfig's type-agnostic ones."""
    max_num_features: int = 8192
    first_octave: int = -1
    num_octaves: int = 4
    octave_resolution: int = 3
    peak_threshold: float = 0.02 / 3        # colmap's own default expression: 0.02 / octave_resolution
    edge_threshold: float = 10.0
    estimate_affine_shape: bool = False
    max_num_orientations: int = 2
    upright: bool = False
    domain_size_pooling: bool = False       # DSP: average descriptors over a scale range
    dsp_min_scale: float = 1.0 / 6.0
    dsp_max_scale: float = 3.0
    dsp_num_scales: int = 10


@dataclasses.dataclass
class FeatureMatchingConfig:
    """exhaustive_matcher's --FeatureMatching.* flags (feature/matcher.h:62-108) -- shared by
    every matcher type, not exhaustive_matcher-specific."""
    num_threads: int = -1
    use_gpu: bool = True
    gpu_index: str = "-1"
    max_num_matches: int = 32768
    guided_matching: bool = False
    skip_geometric_verification: bool = False
    rig_verification: bool = False
    skip_image_pairs_in_same_frame: bool = False


@dataclasses.dataclass
class SiftMatchingConfig:
    """exhaustive_matcher's --SiftMatching.* flags (feature/sift.h:112-138)."""
    max_ratio: float = 0.8               # Lowe's ratio test threshold
    max_distance: float = 0.7
    cross_check: bool = True
    cpu_brute_force_matcher: bool = False
    lightglue_min_score: float = 0.1     # only used if type selects LightGlue matching


@dataclasses.dataclass
class TwoViewGeometryConfig:
    """exhaustive_matcher's --TwoViewGeometry.* flags (estimators/two_view_geometry.h:45-131).
    max_error/confidence/max_num_trials/min_inlier_ratio are colmap's own RANSAC-tuned overrides
    on top of the plain RANSACOptions struct's own (much looser) raw defaults."""
    min_num_inliers: int = 15
    multiple_models: bool = False
    compute_relative_pose: bool = False
    detect_watermark: bool = True
    multiple_ignore_watermark: bool = True
    watermark_detection_max_error: float = 4.0
    filter_stationary_matches: bool = False
    stationary_matches_max_error: float = 4.0
    use_degensac: bool = False
    max_error: float = 4.0               # RANSAC max reprojection/epipolar error, in pixels
    confidence: float = 0.999            # RANSAC confidence
    max_num_trials: int = 10000          # RANSAC max iterations
    min_inlier_ratio: float = 0.25       # RANSAC early-stop inlier ratio
    random_seed: int = -1


@dataclasses.dataclass
class ExhaustiveMatchingConfig:
    """exhaustive_matcher's own single flag, on top of the shared groups above
    (controllers/pairing.h:43-52)."""
    block_size: int = 50


@dataclasses.dataclass
class MapperConfig:
    """`colmap mapper`'s --Mapper.* flags: the top-level IncrementalPipelineOptions
    (controllers/incremental_pipeline.h:47-214) that orchestrate the incremental SfM pipeline as
    a whole (as opposed to IncrementalMapperConfig below, which tunes one image-registration
    step at a time)."""
    min_num_matches: int = 15
    ignore_watermarks: bool = False
    multiple_models: bool = True         # reconstruct every disconnected component (why
                                         # merge_submodels below exists at all)
    max_num_models: int = 50
    max_model_overlap: int = 20
    min_model_size: int = 10
    init_image_id1: int | None = None    # None = auto-pick the initial image pair
    init_image_id2: int | None = None
    init_num_trials: int = 200
    structure_less_registration_fallback: bool = True
    structure_less_registration_only: bool = False
    extract_colors: bool = True
    num_threads: int = -1
    random_seed: int = -1
    min_focal_length_ratio: float = 0.1       # bogus-camera-parameter filter (opening angle ~130deg)
    max_focal_length_ratio: float = 10.0      # (opening angle ~5deg)
    max_extra_param: float = 1.0
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True
    ba_refine_sensor_from_rig: bool = True
    ba_local_function_tolerance: float = 0.0
    ba_local_max_num_iterations: int = -1     # -1 = colmap's own internal default schedule
    ba_global_frames_ratio: float = 1.1       # trigger global BA after this much frame growth
    ba_global_points_ratio: float = 1.1
    ba_global_frames_freq: int = 500
    ba_global_points_freq: int = 250000
    ba_global_function_tolerance: float = 0.0
    ba_global_max_num_iterations: int = -1
    ba_local_max_refinements: int = 2
    ba_local_max_refinement_change: float = 0.001
    ba_global_max_refinements: int = 5
    ba_global_max_refinement_change: float = 0.0005
    ba_use_gpu: bool = False
    ba_gpu_index: str = "-1"
    ba_local_backend: str | None = None       # "CERES" or "CASPAR"
    ba_global_backend: str | None = None
    ba_min_num_residuals_for_cpu_multi_threading: int = 50000
    snapshot_path: str | None = None          # periodic reconstruction checkpoints; unused here
                                              # (short-lived per-sequence scratch workspace)
    snapshot_frames_freq: int = 0             # 0 = disabled
    fix_existing_frames: bool = False
    max_runtime_seconds: int | None = None    # -1 raw default = unbounded


@dataclasses.dataclass
class IncrementalMapperConfig:
    """`colmap mapper`'s --Mapper.* flags that tune one image-registration step at a time
    (sfm/incremental_mapper.h:68-165, `IncrementalMapper::Options`, nested under
    IncrementalPipelineOptions but flattened into the same `Mapper.` CLI prefix). Only the
    fields colmap actually CLI-binds from this struct are here -- confirmed against
    option_manager.cc's own AddMapperOptions() line by line, since several of this struct's own
    fields (abs_pose_refine_focal_length, use_prior_position, image_selection_method, ...) are
    NOT CLI-exposed at all, only settable by editing colmap's own source."""
    init_min_num_inliers: int = 100
    init_max_error: float = 4.0
    init_max_forward_motion: float = 0.95
    init_min_tri_angle: float = 16.0
    init_max_reg_trials: int = 2
    abs_pose_max_error: float = 12.0
    abs_pose_min_num_inliers: int = 30
    abs_pose_min_inlier_ratio: float = 0.25
    filter_max_reproj_error: float = 4.0
    filter_min_tri_angle: float = 1.5
    max_reg_trials: int = 3
    ba_local_num_images: int = 6
    ba_local_min_tri_angle: float = 6.0
    ba_global_ignore_redundant_points3D: bool = False
    ba_global_ignore_redundant_points3D_min_coverage_gain: float = 0.05


@dataclasses.dataclass
class TriangulationConfig:
    """`colmap mapper`'s --Mapper.tri_* flags (sfm/incremental_triangulator.h:45-88,
    `IncrementalTriangulator::Options`). Field names keep the `tri_` prefix so the generic
    `_flags("Mapper", ...)` call produces the right `Mapper.tri_*` flag without a separate
    mapping table."""
    tri_max_transitivity: int = 1
    tri_create_max_angle_error: float = 2.0
    tri_continue_max_angle_error: float = 2.0
    tri_merge_max_reproj_error: float = 4.0
    tri_complete_max_reproj_error: float = 4.0
    tri_complete_max_transitivity: int = 5
    tri_re_max_angle_error: float = 5.0
    tri_re_min_ratio: float = 0.2
    tri_re_max_trials: int = 1
    tri_min_angle: float = 1.5
    tri_ignore_two_view_tracks: bool = True


@dataclasses.dataclass
class PatchMatchConfig:
    """patch_match_stereo's --PatchMatchStereo.* flags. Defaults confirmed from the actual
    running binary's own printed option dump (this session's own terminal log), not source-read
    -- ground truth for what colmap itself resolves these to at runtime."""
    max_image_size: int = 480
    window_radius: int = 5
    window_step: int = 1
    sigma_spatial: float = 5.0
    sigma_color: float = 0.2
    num_samples: int = 15
    ncc_sigma: float = 0.6
    min_triangulation_angle: float = 1.0
    incident_angle_sigma: float = 0.9
    num_iterations: int = 5
    geom_consistency: bool = False    # True also enables writing .geometric.bin depth maps
    geom_consistency_regularizer: float = 0.3
    geom_consistency_max_cost: float = 3.0
    filter: bool = True
    filter_min_ncc: float = 0.1
    filter_min_triangulation_angle: float = 3.0
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    num_threads: int = -1
    write_consistency_graph: bool = False
    allow_missing_files: bool = False


@dataclasses.dataclass
class StereoFusionConfig:
    """stereo_fusion's --StereoFusion.* flags. Defaults confirmed the same way as
    PatchMatchConfig's (the running binary's own printed option dump)."""
    max_image_size: int = 480
    min_num_pixels: int = 5
    max_num_pixels: int = 10000
    max_traversal_depth: int = 100
    max_reproj_error: float = 2.0
    max_depth_error: float = 0.01
    max_normal_error: float = 10.0
    check_num_images: int = 50
    use_cache: bool = False
    cache_size: int = 32
    mask_path: str | None = None


@dataclasses.dataclass
class ModelConfig:
    image_reader: ImageReaderConfig = dataclasses.field(default_factory=ImageReaderConfig)
    feature_extraction: FeatureExtractionConfig = dataclasses.field(default_factory=FeatureExtractionConfig)
    sift_extraction: SiftExtractionConfig = dataclasses.field(default_factory=SiftExtractionConfig)
    feature_matching: FeatureMatchingConfig = dataclasses.field(default_factory=FeatureMatchingConfig)
    sift_matching: SiftMatchingConfig = dataclasses.field(default_factory=SiftMatchingConfig)
    two_view_geometry: TwoViewGeometryConfig = dataclasses.field(default_factory=TwoViewGeometryConfig)
    exhaustive_matching: ExhaustiveMatchingConfig = dataclasses.field(default_factory=ExhaustiveMatchingConfig)
    mapper: MapperConfig = dataclasses.field(default_factory=MapperConfig)
    incremental_mapper: IncrementalMapperConfig = dataclasses.field(default_factory=IncrementalMapperConfig)
    triangulation: TriangulationConfig = dataclasses.field(default_factory=TriangulationConfig)
    patch_match: PatchMatchConfig = dataclasses.field(default_factory=PatchMatchConfig)
    stereo_fusion: StereoFusionConfig = dataclasses.field(default_factory=StereoFusionConfig)
    # stereo_fusion's own --input_type flag -- not under the StereoFusion.* namespace, so it
    # lives here rather than on StereoFusionConfig (which _flags() turns into Namespace.field
    # args uniformly). geom_consistency=False (patch_match's own default above) means only
    # photometric depth maps get written at all, so "photometric" is the only source that will
    # exist unless patch_match.geom_consistency is also flipped on.
    depth_source: str = "photometric"   # "photometric" or "geometric"


def stage_frames(frames: list[Frame], staging_dir: Path) -> dict[str, Frame]:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{frame_stem(index)}{suffix}" (original
    extension kept). Returns {staged filename: Frame} -- pycolmap.Image.name is only ever the bare
    filename, so this is how run() maps a registered image back to a Frame afterward. Same shape
    as baselines/mpsfm/run_sequence.py's stage_frames."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for f in frames:
        name = f"{frame_stem(f.index)}{f.rgb_path.suffix}"
        (staging_dir / name).symlink_to(f.rgb_path)
        by_name[name] = f
    return by_name


def _images_count(log: str, label: str) -> int:
    """The "Images: N" on the line immediately after "{label} ---" -- a python port of
    get_images_count() in colmap_merge_loop_smart.sh (`grep -A1 -- "$label ---" | grep -oP
    'Images:\\s*\\K[0-9]+' | head -1`). Returns 0 if unparseable, matching bash's
    empty-string-as-0 behavior in the `(( images1 >= images2 ))` comparison it feeds."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if f"{label} ---" in line and i + 1 < len(lines):
            m = re.search(r"Images:\s*(\d+)", lines[i + 1])
            return int(m.group(1)) if m else 0
    return 0


def merge_submodels(sparse_dir: Path, merge_dir: Path) -> bool:
    """Merges every disconnected sub-model colmap's mapper left under `sparse_dir` (numbered dirs
    "0", "1", ...) into one reconstruction at `merge_dir` -- a python port of merge_combo() in
    colmap_merge_loop_smart.sh, ported carefully rather than reimplemented from a paraphrase:

    - 0 submodels: nothing to do, returns False.
    - 1 submodel: copied straight to `merge_dir`, no `model_merger` call.
    - >=2: pairwise `colmap model_merger`, folding each next submodel into a running `base_dir`.
      stdout+stderr are captured as one interleaved stream (`stdout=PIPE, stderr=STDOUT`) -- the
      "Images: N" line-adjacency check below depends on the exact interleaving order colmap
      itself writes, which two separately-captured streams concatenated afterward would not
      reproduce. "=> Merge succeeded" adopts the merge as the new `base_dir`/`merge_dir`.
      "=> Merge failed": if nothing has merged successfully yet, fall back to whichever of the
      two inputs has more registered images (parsed from the log); if something already merged,
      keep it and just skip this submodel (base_dir/have_result unchanged) rather than aborting.
      Any other output (neither string present) is its own case -- treated as a failure, but
      deliberately *without* the has-no-prior-success fallback above even when nothing has merged
      yet, matching the reference script's own separate `else` branch exactly.

    Returns True unless there were zero submodels; `merge_dir` may still not exist afterward if
    every pairwise attempt fell into the "unrecognized output" case -- callers must check for
    `merge_dir.is_dir()` too, not just this return value.
    """
    import os
    submodels = sorted((d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()),
                       key=lambda d: int(d.name))
    if not submodels:
        return False

    if len(submodels) == 1:
        if merge_dir.exists():
            shutil.rmtree(merge_dir)
        shutil.copytree(submodels[0], merge_dir)
        return True

    base_dir = submodels[0]
    have_result = False
    tmp_dir = merge_dir.parent / f"{merge_dir.name}_tmp"
    for next_model in submodels[1:]:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["colmap", "model_merger",
             "--input_path1", str(base_dir),
             "--input_path2", str(next_model),
             "--output_path", str(tmp_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        print(os.listdir(merge_dir.parent))

        if "=> Merge succeeded" in log:
            if merge_dir.exists():
                shutil.rmtree(merge_dir)
            tmp_dir.rename(merge_dir)
            base_dir, have_result = merge_dir, True
        elif "=> Merge failed" in log:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not have_result:
                images1 = _images_count(log, "Reconstruction 1")
                images2 = _images_count(log, "Reconstruction 2")
                winner = base_dir if images1 >= images2 else next_model
                if merge_dir.exists():
                    shutil.rmtree(merge_dir)
                shutil.copytree(winner, merge_dir)
                base_dir, have_result = merge_dir, True
            # else: a later merge failed after an earlier success -- keep merge_dir, skip this
            # submodel, base_dir/have_result unchanged.
        else:
            # Unrecognized output -- a failure, but deliberately without the has-no-prior-success
            # fallback above (matches the reference script's own separate branch).
            shutil.rmtree(tmp_dir, ignore_errors=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return merge_dir.exists()


def c2w_from_image(image: "pycolmap.Image") -> np.ndarray:
    """4x4 camera-to-world matrix from a registered pycolmap Image's own `cam_from_world()`
    (world-to-camera) pose. `cam_from_world` is a *method* in this pycolmap version (not a
    property, unlike baselines/mpsfm/run_sequence.py's older-pycolmap-shaped helper of the same
    name) -- matches inspection_viz.colmap's own `image.cam_from_world().matrix()` usage, which
    reads the same reconstruction shape elsewhere in this repo."""
    w2c = image.cam_from_world().matrix()  # (3,4): [R | t]
    R, t = w2c[:, :3], w2c[:, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix for `camera`, via pycolmap's own calibration_matrix() rather than
    manually unpacking `camera.params` for an assumed model. image_undistorter only *removes
    distortion*: a model with no distortion params to begin with (SIMPLE_PINHOLE, our own default
    --ImageReader.camera_model, or PINHOLE) passes straight through unchanged, so the model left
    in the dense reconstruction is NOT reliably PINHOLE the way an earlier version of this
    function assumed (that assumption crashed here, right after the expensive dense stage had
    already finished, on every run using the default SIMPLE_PINHOLE). calibration_matrix()
    handles any camera model, matching inspection_viz.colmap.Reconstruction.camera's own use of
    the same method."""
    return camera.calibration_matrix().astype(np.float64)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts COLMAP's world gauge onto the gt one, as far as one
    frame can -- COLMAP's own scale is left exactly as reconstructed (see module docstring for
    why). Whichever registered frame comes first in the sequence (the "anchor") puts its own
    reconstructed pose exactly onto its gt one, and every other frame rides along on the same
    transform."""
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def read_depth_map(dense_dir: Path, image_name: str, depth_source: str) -> np.ndarray | None:
    """The dense depth map patch_match_stereo wrote for `image_name`, or None if it never got
    one. `0` marks an unestimated pixel in COLMAP's own binary format (see inspection_viz.colmap's
    module docstring) -- converted to nan here, the same convention
    inspection_viz.colmap.Reconstruction.depth already uses when reading this same file shape for
    predictions/legacy colmap workspaces alike."""
    path = dense_dir / "stereo" / "depth_maps" / f"{image_name}.{depth_source}.bin"
    if not path.is_file():
        return None
    depth = read_array(path)
    return np.where(depth > 0, depth, np.nan).astype(np.float32)


class COLMAP(Baseline):
    """Classical COLMAP SfM + MVS, driven through its own CLI (see module docstring)."""

    name = "colmap"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="colmap-") as tmp:
            tmp_path = Path(tmp)
            images_dir = tmp_path / "images"
            ws = tmp_path / "ws"
            frames_by_name = stage_frames(frames, images_dir)
            ws.mkdir(parents=True, exist_ok=True)
            database_path = ws / "database.db"

            print(f"running {len(frames)} frames (camera_model={cfg.image_reader.camera_model}) ...")
            subprocess.run([
                "colmap", "feature_extractor",
                "--image_path", str(images_dir),
                "--database_path", str(database_path),
                *_flags("ImageReader", cfg.image_reader),
                *_flags("FeatureExtraction", cfg.feature_extraction),
                *_flags("SiftExtraction", cfg.sift_extraction),
            ], check=True)

            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(database_path),
                *_flags("FeatureMatching", cfg.feature_matching),
                *_flags("SiftMatching", cfg.sift_matching),
                *_flags("TwoViewGeometry", cfg.two_view_geometry),
                *_flags("ExhaustiveMatching", cfg.exhaustive_matching),
            ], check=True)

            # `colmap mapper` requires --output_path to already exist (it CHECK-fails and aborts
            # otherwise, same as `global_mapper` in the glomap baseline).
            sparse_dir = ws / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "colmap", "mapper",
                "--image_path", str(images_dir),
                "--database_path", str(database_path),
                "--output_path", str(sparse_dir),
                *_flags("Mapper", cfg.mapper),
                *_flags("Mapper", cfg.incremental_mapper),
                *_flags("Mapper", cfg.triangulation),
            ], check=True)

            if not sparse_dir.is_dir() or not any(sparse_dir.iterdir()):
                raise SystemExit("COLMAP registered no images at all for this sequence "
                                 "(mapper produced no sparse/ sub-model)")

            merge_dir = ws / "sparse_merge"
            if not merge_submodels(sparse_dir, merge_dir) or not merge_dir.is_dir():
                raise SystemExit("model_merger never produced a usable merged model for this "
                                 "sequence (see colmap_merge_loop_smart.sh's own logic)")

            dense_dir = ws / "dense"
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(images_dir),
                "--input_path", str(merge_dir),
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

            # image_undistorter's own output model -- pixel-aligned with the depth maps (not the
            # pre-undistortion sparse_merge model read above). Not necessarily PINHOLE (see
            # K_from_camera's docstring) -- distortion-free input models pass through unchanged.
            rec = pycolmap.Reconstruction(dense_dir / "sparse")
            registered = {img.name: img for img in rec.images.values()}
            missing = set(frames_by_name) - set(registered)
            if not registered:
                raise SystemExit("COLMAP registered no frames at all in the final dense model "
                                 "for this sequence")
            if missing:
                print(f"  registered {len(registered)}/{len(frames)} frames -- "
                     f"{len(missing)} unregistered frame(s) get no output")

            # Whichever of our frames COLMAP registered that comes first in the sequence, not
            # necessarily frames[0] itself (see _gt_frame0_alignment's docstring).
            anchor_name = min(registered, key=lambda n: frames_by_name[n].index)
            anchor_frame = frames_by_name[anchor_name]
            anchor_c2w = c2w_from_image(registered[anchor_name])
            align = _gt_frame0_alignment(anchor_c2w, anchor_frame.camera().c2w)

            no_depth = 0
            for name, image in registered.items():
                f = frames_by_name[name]
                camera = rec.cameras[image.camera_id]
                depth = read_depth_map(dense_dir, image.name, cfg.depth_source)
                if depth is None:
                    no_depth += 1
                    width, height = image_size(f.rgb_path)
                    depth = np.full((height, width), np.nan, dtype=np.float32)
                c2w = align @ c2w_from_image(image)
                K = K_from_camera(camera)
                out.write_frame(f.index, depth, c2w, K, intrinsics_width=camera.width)
            if no_depth:
                print(f"  {no_depth}/{len(registered)} registered frame(s) had no dense depth "
                     "map (never estimated) -- written as all-NaN")

            if out.spec.cloud:
                if fused_path.is_file():
                    cloud = trimesh.load(fused_path, process=False)
                    pts = trimesh.transform_points(np.asarray(cloud.vertices), align)
                    cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
                else:
                    print("  stereo_fusion produced no fused cloud")


if __name__ == "__main__":
    cli_main(COLMAP, description=__doc__)
