"""Run GLOMAP (global, non-incremental, structure-from-motion) + classical multi-view stereo on
the sequences of an inspection asset and write depth + pose + point-cloud predictions back into
the asset folder (predictions/{method}/...), in the same format the ground truth is stored in --
so inspection_viz's Source toggle picks them up unchanged.

GLOMAP is NOT a separate tool/submodule here: its pipeline was merged directly into the `colmap`
CLI binary itself (see colmap/CHANGELOG.rst, "Integrated GLOMAP global SfM pipeline into COLMAP
as a first-class alternative" -- "GLOMAP is maintained through the COLMAP repository going
forward"), exposed as `colmap global_mapper` alongside the incremental `colmap mapper` this
repo's own baselines/colmap/run_sequence.py uses. This file is that baseline's sibling: same
architecture (feature_extractor -> exhaustive_matcher -> mapping stage -> merge_submodels ->
dense MVS -> pycolmap read-back), differing only in the mapping stage -- global (this file, via
view_graph_calibrator + global_mapper) instead of incremental (colmap's mapper). Every helper
below (`stage_frames`, `merge_submodels`/`_images_count`, `c2w_from_image`, `K_from_camera`,
`_gt_frame0_alignment`, `read_depth_map`, `_flags`) is duplicated verbatim from
baselines/colmap/run_sequence.py rather than imported -- matching this repo's established
per-baseline self-containment convention (e.g. VGGT-Long/VGGT-SLAM each keep their own local copy
of similar small helpers rather than sharing one).

Like the colmap baseline, this is fundamentally different in shape from the feed-forward
baselines here: it self-calibrates and *registers* images via classical SfM (global, not
incremental, optimization -- every pose solved simultaneously via rotation averaging + global
positioning, rather than one image at a time), so (a) not every input frame necessarily ends up
registered, and (b) GLOMAP has no metric reference at all -- its reconstruction is correct only
up to an unknown, unrecoverable global scale, same situation as VGGT-SLAM/VGGT-Long, so only one
rigid (not similarity) transform is fit against gt (`_gt_frame0_alignment`), leaving scale as
GLOMAP produced it.

Driven entirely through the `colmap` CLI: feature_extractor -> exhaustive_matcher ->
view_graph_calibrator (optional but recommended -- see ViewGraphCalibrationConfig) ->
global_mapper -> model_merger (GlobalMapperConfig.multiple_models defaults true, so a
poorly-connected view graph genuinely can leave multiple disconnected sub-models needing a
merge, exactly like the incremental mapper) -> image_undistorter/patch_match_stereo/stereo_fusion.

This deliberately does NOT use `automatic_reconstructor --mapper global` (a second, higher-level
way colmap can run global SfM): confirmed by reading controllers/automatic_reconstruction.cc that
this shortcut builds fresh, purely-default `ViewGraphCalibrationOptions`/`GlobalPipelineOptions`
and skips `view_graph_calibrator` in any tunable form -- colmap's own docs (doc/cli.rst) call out
`view_graph_calibrator` as specifically mattering when reliable focal-length priors aren't
available, exactly this dataset's self-calibrated-camera situation, so the lower-level sequence
below is more correct here, not merely more literal to the reference script it was lifted from
(/home/shared/eccv_2026_data/glomap_loop.sh) -- and it's the only way to reach the `GlobalMapper.*`
tuning knobs this file exposes at all.

Usage -- inside the container (build_container.sh once, then run_container.sh for a shell):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/glomap.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's `colmap` CLI binary and pycolmap.
"""

import dataclasses
import re
import shutil
import subprocess
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


def _flags(prefix: str | None, cfg, skip: frozenset = frozenset()) -> list[str]:
    """Every field of dataclass `cfg` as CLI args -- `--{prefix}.{field} value` if `prefix` is
    given, else bare `--{field} value` (view_graph_calibrator's own flags have no namespace
    prefix). Field names throughout this file are chosen to match colmap's own flag suffix
    exactly, so this needs no per-field name-mapping table. `None`-valued fields are omitted
    entirely (colmap then applies its own compiled-in default); bool becomes "1"/"0" (colmap's
    own convention throughout its CLI); everything else is str()'d as-is. `skip` names dataclass
    fields that exist for this file's own bookkeeping but aren't real colmap flags (e.g. the
    config-only `enabled` toggle on ViewGraphCalibrationConfig)."""
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
# ModelConfig: one nested dataclass per colmap CLI command/option-namespace. image_reader through
# stereo_fusion are identical in shape/defaults to baselines/colmap/run_sequence.py's own groups
# of the same name (duplicated, not shared -- see module docstring); only view_graph_calibration
# and global_mapper are new here, replacing colmap's mapper/incremental_mapper/triangulation.
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
class ViewGraphCalibrationConfig:
    """view_graph_calibrator's own flags (estimators/view_graph_calibration.h:40-82) -- no
    namespace prefix on the CLI, unlike every other group here. colmap's own docs (doc/cli.rst)
    call this step "optional but recommended" specifically when reliable focal-length priors
    aren't available -- exactly this dataset's self-calibrated SIMPLE_PINHOLE situation -- so
    `enabled` defaults True. It mutates its database in place, so run() works off a copy
    (matching glomap_loop.sh's own `cp database.db database_global.db` before calling it)."""
    enabled: bool = True                            # config-only toggle, not a colmap flag itself
    cross_validate_prior_focal_lengths: bool = True
    min_calibrated_pair_ratio: float = 0.5
    reestimate_relative_pose: bool = True
    min_focal_length_ratio: float = 0.1
    max_focal_length_ratio: float = 10.0
    max_calibration_error: float = 2.0
    relpose_max_error: float = 1.0
    relpose_min_num_inliers: int = 30
    relpose_min_inlier_ratio: float = 0.25


@dataclasses.dataclass
class GlobalMapperConfig:
    """`colmap global_mapper`'s --GlobalMapper.* flags (sfm/global_mapper.h:18-101,
    estimators/global_positioning.h, estimators/rotation_averaging.h). Several fields' *effective*
    default here is an override colmap's own GlobalMapperOptions init-lambda bakes on top of the
    raw BundleAdjustmentOptions/CeresBundleAdjustmentOptions struct defaults (e.g. ba_ceres_use_gpu
    is raw False but effectively True) -- the values below are the lambda-overridden ones actually
    seen through this CLI, confirmed against the source, not the library's raw struct defaults."""
    # General
    min_num_matches: int = 15
    ignore_watermarks: bool = False
    num_threads: int = -1
    random_seed: int = -1
    decompose_relative_pose: bool = True
    multiple_models: bool = True    # reconstruct every connected component (why merge_submodels
                                    # below exists at all -- confirmed a poorly-connected view
                                    # graph genuinely can leave multiple sub-models)
    min_model_size: int = 3
    ba_num_iterations: int = 3
    # Stage skip toggles -- nothing skipped by default
    skip_rotation_averaging: bool = False
    skip_track_establishment: bool = False
    skip_global_positioning: bool = False
    skip_bundle_adjustment: bool = False
    skip_retriangulation: bool = False
    # Track establishment
    track_intra_image_consistency_threshold: float = 10.0
    track_required_tracks_per_view: int | None = None   # raw default INT_MAX ("unbounded")
    track_min_num_views_per_track: int = 3
    keep_max_num_tracks: int | None = None               # raw default INT_MAX ("keep all")
    # Global positioning (estimators/global_positioning.h)
    gp_use_gpu: bool = True
    gp_gpu_index: str = "-1"
    gp_optimize_positions: bool = True
    gp_optimize_points: bool = True
    gp_optimize_scales: bool = True
    gp_loss_function_scale: float = 0.1
    gp_max_num_iterations: int = 100
    # Bundle adjustment
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True
    refine_sensor_from_rig: bool = True
    ba_refine_rig_from_world: bool = True
    ba_refine_points3D: bool = True
    ba_min_track_length: int = 3
    ba_backend: str | None = None       # "CERES" or "CASPAR"
    ba_gpu_index: str = "-1"
    ba_ceres_use_gpu: bool = True
    ba_ceres_loss_function_scale: float = 1.0
    ba_ceres_max_num_iterations: int = 200
    ba_skip_fixed_rotation_stage: bool = False
    ba_skip_joint_optimization_stage: bool = False
    # Retriangulation
    tri_complete_max_reproj_error: float = 15.0
    tri_merge_max_reproj_error: float = 15.0
    tri_min_angle: float = 1.0
    # Rotation averaging (estimators/rotation_averaging.h)
    ra_use_gravity: bool = False   # no GPS/IMU gravity prior available for this synthetic dataset
    ra_use_stratified: bool = True
    ra_max_rotation_error_deg: float = 10.0
    ra_reweighting: str | None = None   # "UNIFORM" or "INLIER_MATCH_COUNT"
    # Thresholds
    max_angular_reproj_error_deg: float = 1.0
    max_normalized_reproj_error: float = 0.01
    min_tri_angle_deg: float = 1.0


@dataclasses.dataclass
class PatchMatchConfig:
    """patch_match_stereo's --PatchMatchStereo.* flags. Defaults confirmed from the actual
    running binary's own printed option dump (this session's own terminal log, from the colmap
    baseline's first successful run), not source-read -- ground truth for what colmap itself
    resolves these to at runtime."""
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
    view_graph_calibration: ViewGraphCalibrationConfig = dataclasses.field(default_factory=ViewGraphCalibrationConfig)
    global_mapper: GlobalMapperConfig = dataclasses.field(default_factory=GlobalMapperConfig)
    patch_match: PatchMatchConfig = dataclasses.field(default_factory=PatchMatchConfig)
    stereo_fusion: StereoFusionConfig = dataclasses.field(default_factory=StereoFusionConfig)
    # stereo_fusion's own --input_type flag -- see baselines/colmap/run_sequence.py's identical
    # field for why it lives here rather than on StereoFusionConfig.
    depth_source: str = "photometric"   # "photometric" or "geometric"


def stage_frames(frames: list[Frame], staging_dir: Path) -> dict[str, Frame]:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{frame_stem(index)}{suffix}" (original
    extension kept). Returns {staged filename: Frame} -- pycolmap.Image.name is only ever the bare
    filename, so this is how run() maps a registered image back to a Frame afterward."""
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
    """Merges every disconnected sub-model global_mapper left under `sparse_dir` (numbered dirs
    "0", "1", ...) into one reconstruction at `merge_dir` -- a python port of merge_combo() in
    colmap_merge_loop_smart.sh (see baselines/colmap/run_sequence.py's identical function for the
    full edge-case rationale). Returns True unless there were zero submodels; `merge_dir` may
    still not exist afterward if every pairwise attempt fell into the "unrecognized output" case.
    """
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
        proc = subprocess.run(
            ["colmap", "model_merger",
             "--input_path1", str(base_dir),
             "--input_path2", str(next_model),
             "--output_path", str(tmp_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout

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
    (world-to-camera) pose -- matches inspection_viz.colmap's own `image.cam_from_world().matrix()`
    usage, which reads the same reconstruction shape elsewhere in this repo."""
    w2c = image.cam_from_world().matrix()  # (3,4): [R | t]
    R, t = w2c[:, :3], w2c[:, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix for `camera`, via pycolmap's own calibration_matrix() rather than
    manually unpacking `camera.params` for an assumed model -- image_undistorter only *removes
    distortion*, so a distortion-free input model (SIMPLE_PINHOLE, our own default) passes
    through unchanged rather than always becoming PINHOLE (see the colmap baseline's identical
    helper for the bug this avoided)."""
    return camera.calibration_matrix().astype(np.float64)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts GLOMAP's world gauge onto the gt one, as far as one
    frame can -- GLOMAP's own scale is left exactly as reconstructed (see module docstring for
    why). Whichever registered frame comes first in the sequence (the "anchor") puts its own
    reconstructed pose exactly onto its gt one, and every other frame rides along on the same
    transform."""
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def read_depth_map(dense_dir: Path, image_name: str, depth_source: str) -> np.ndarray | None:
    """The dense depth map patch_match_stereo wrote for `image_name`, or None if it never got
    one. `0` marks an unestimated pixel in COLMAP's own binary format -- converted to nan here,
    the same convention inspection_viz.colmap.Reconstruction.depth already uses when reading this
    same file shape elsewhere in this repo."""
    path = dense_dir / "stereo" / "depth_maps" / f"{image_name}.{depth_source}.bin"
    if not path.is_file():
        return None
    depth = read_array(path)
    return np.where(depth > 0, depth, np.nan).astype(np.float32)


class GLOMAP(Baseline):
    """GLOMAP global SfM + classical MVS, driven through the `colmap` CLI's own `global_mapper`
    command (see module docstring)."""

    name = "glomap"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="glomap-") as tmp:
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

            # view_graph_calibrator mutates its database in place, so work off a copy -- same as
            # glomap_loop.sh's own `cp database.db database_global.db`, preserving the original
            # matched database untouched.
            if cfg.view_graph_calibration.enabled:
                global_db = ws / "database_global.db"
                shutil.copy(database_path, global_db)
                subprocess.run([
                    "colmap", "view_graph_calibrator",
                    "--database_path", str(global_db),
                    *_flags(None, cfg.view_graph_calibration, skip=frozenset({"enabled"})),
                ], check=True)
            else:
                global_db = database_path

            # global_mapper requires --output_path to already exist (it CHECK-fails and aborts
            # otherwise, same as `mapper` in the colmap baseline).
            sparse_dir = ws / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "colmap", "global_mapper",
                "--image_path", str(images_dir),
                "--database_path", str(global_db),
                "--output_path", str(sparse_dir),
                *_flags("GlobalMapper", cfg.global_mapper),
            ], check=True)

            if not sparse_dir.is_dir() or not any(sparse_dir.iterdir()):
                raise SystemExit("GLOMAP registered no images at all for this sequence "
                                 "(global_mapper produced no sparse/ sub-model)")

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

            rec = pycolmap.Reconstruction(dense_dir / "sparse")
            registered = {img.name: img for img in rec.images.values()}
            missing = set(frames_by_name) - set(registered)
            if not registered:
                raise SystemExit("GLOMAP registered no frames at all in the final dense model "
                                 "for this sequence")
            if missing:
                print(f"  registered {len(registered)}/{len(frames)} frames -- "
                     f"{len(missing)} unregistered frame(s) get no output")

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
    cli_main(GLOMAP, description=__doc__)
