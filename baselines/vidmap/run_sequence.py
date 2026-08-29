"""Run VidMap ("Exploiting Temporal Structure for Video-Based Structure-from-Motion", ECCV 2026)
on the sequences of an inspection asset and write depth + pose + point-cloud predictions back
into the asset folder (predictions/{method}/...), in the same format the ground truth is stored
in -- so inspection_viz's Source toggle picks them up unchanged.

Architecturally this is NOT another baselines/colmap-style CLI-subprocess baseline, unlike
baselines/glomap and baselines/pixel-perfect-sfm (both of which drive the `colmap` binary
directly): vidmap is a pure-Python package with its own CLI entry points and its own Hydra/
OmegaConf-based nested config trees (`vidmap.configuration.FrontendConfig`/`MappingConfig`), not
`colmap`-style flat `--Namespace.field value` flags. There is no `colmap automatic_reconstructor`/
`mapper`/`global_mapper` stage here at all -- vidmap's own frontend (temporal tracks, keyframe
selection, metric depth via Depth-Anything-3, loop closure via MegaLoc retrieval, dense matching
via RoMaV2) and its own global mapper (COLMAP's global mapping pipeline extended with
provenance-aware, depth-consistency-aware weighting) replace that whole stage in one call:
    python -m vidmap.run --input_data <frames> --output <dir> --frontend-conf ... --mapping-conf ...

Two things converge with the `colmap` family despite that very different frontend, both confirmed
directly from vidmap's own source (not the README's prose alone, per this session's established
"verify, don't guess" discipline after two rounds of stale-doc/wrong-assumption bugs on
baselines/colmap and baselines/pixel-perfect-sfm):
  - `$OUTPUT_DIR/rec` is always a *single*, flat pycolmap.Reconstruction (vidmap's mapper solves
    the whole view graph globally, not incrementally, so there is no disconnected-submodel
    concept -- no merge_submodels step is needed here at all, unlike baselines/colmap/glomap).
    Camera model is always PINHOLE. Image names are preserved exactly end-to-end. So the
    read-back half of this baseline (`c2w_from_image`, `K_from_camera`, `_gt_frame0_alignment`)
    is duplicated verbatim from baselines/colmap/run_sequence.py, unchanged -- this repo's
    established per-baseline self-containment convention, not a cross-import.
  - There is no dense-MVS stage (no image_undistorter/patch_match_stereo/stereo_fusion
    equivalent), so `write_cloud` uses the reconstruction's own sparse `points3D` directly, the
    same shape as baselines/mpsfm/run_sequence.py's own cloud-writing code.

Per-frame depth (this baseline's one genuinely novel piece) comes from `--cache-depth-maps`:
vidmap's own DA3-based per-frame metric depth predictions, cached to `$OUTPUT_DIR/
full_depth_maps.h5` (one HDF5 group per image, keyed by exact image name, at DA3's own
processing resolution -- resized back to native here, same cv2.resize pattern
baselines/mpsfm/run_sequence.py's own read_refined_depth already uses), corrected by the
per-image depth-scale factors the mapper solves during bundle adjustment
(`$OUTPUT_DIR/depth_lift_scales.json`, keyed by COLMAP image_id -- a different key space than the
HDF5's own name-keyed groups). Both files are read via `vidmap.depth_artifacts`'s own public
`resolve_depth_paths`/`load_depth_scales` functions, imported directly rather than reimplementing
their schema -- this baseline's own container has `vidmap` installed in the same venv
run_sequence.py itself runs in, so importing the library for this post-processing step (as
opposed to shelling out to its CLI, which is what actually runs the frontend+mapper) is natural.

`ModelConfig` mirrors vidmap's own `FrontendConfig`/`MappingConfig` schema *exactly* field-for-
field (not flattened, not curated down) -- verified directly from
vidmap/vidmap/configuration/config.py, vidmap/vidmap/frontend/options/*.py, and
vidmap/vidmap/mapper/options/*.py, not the README's own doc examples, since Hydra CLI overrides
are literal dotted paths (e.g. `mapping.mapper.gp.first_pass.loss_lc_geometry.weight=0.4`) that
only stay correct if this file's own dataclass nesting matches vidmap's real one exactly -- a
typo here fails loudly at runtime (vidmap's own config dataclasses are pydantic, `extra="forbid"`)
rather than silently no-op-ing. The one deliberate omission is `MapperOptions.replay_cache` and
the handful of `gp`/`ba` fields it gates (`roundtrip_before_ba`, `canonical_checkpoint`,
`skip_zero_observation_points`, `update_point3d_errors`, and the `num_threads`/
`center_init_mode`/`parameter_ordering_strategy`/`camera_center_strategy` fields
`MapperOptions.reject_replay_controls_when_disabled` cross-validates against it) -- these are
coupled, developer-facing reproducibility/replay knobs (raising a validation error if touched
without also enabling `replay_cache`), not tuning parameters, so they're left out of scope here
rather than half-exposed.

Usage -- inside the container (build_container.sh once, then a shell in the image):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/vidmap.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's `vidmap`/`pycolmap`/torch stack.
"""

import dataclasses
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import pycolmap
import yaml
from vidmap.depth_artifacts import load_depth_scales, resolve_depth_paths

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from baselines.common.writer import frame_stem
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size


def _hydra_value(val) -> str:
    """One value formatted for a Hydra/OmegaConf CLI override -- `None`->`null`,
    bool->`true`/`false` (lowercase; NOT the colmap family's `_flags()`'s `"1"`/`"0"` convention),
    list/tuple->`[a,b,c]` (recursing, so a list of lists like `nodes: [[0.0,0.0]]` also works),
    everything else via plain `str()`."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(_hydra_value(v) for v in val) + "]"
    return str(val)


def _hydra_overrides(prefix: str, obj) -> list[str]:
    """Every field of dataclass `obj` as `{prefix}.{field}={value}` Hydra CLI overrides,
    recursing into nested dataclasses. Unlike the colmap family's `_flags()`, every field is
    always emitted -- Hydra's dotlist has no "leave this at the preset's own default" sentinel
    once a full override tree is already being passed (`None` is itself a real, meaningful
    override value here, e.g. `level_indices=null`, not an omit-this-flag marker)."""
    overrides = []
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        key = f"{prefix}.{f.name}"
        if dataclasses.is_dataclass(val):
            overrides += _hydra_overrides(key, val)
        else:
            overrides.append(f"{key}={_hydra_value(val)}")
    return overrides


def stage_frames(frames: list[Frame], staging_dir: Path) -> dict[str, Frame]:
    """Symlinks `frames`' rgb into `staging_dir`, named f"{frame_stem(index)}{suffix}" (original
    extension kept). Returns {staged filename: Frame} -- pycolmap.Image.name is only ever the bare
    filename, so this is how run() maps a registered image back to a Frame afterward. Same shape
    as baselines/colmap/run_sequence.py's stage_frames -- but vidmap's own LocalImageParser only
    recognizes .jpg/.jpeg/.png suffixes (vidmap/datasets/local.py), so the staged suffix must be
    one of those (true for this repo's own rgb assets)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    by_name = {}
    for f in frames:
        name = f"{frame_stem(f.index)}{f.rgb_path.suffix}"
        (staging_dir / name).symlink_to(f.rgb_path)
        by_name[name] = f
    return by_name


def write_intrinsics_yaml(frames_by_name: dict[str, Frame], out_path: Path) -> None:
    """vidmap's own `--intrinsics` sidecar file (vidmap/datasets/local.py's `_add_cameras`):
    `{camera_id: {"params": [fx, fy, cx, cy], "images": [names]}}`, PINHOLE/4-param only. Only
    written (and only passed to vidmap) when `frontend.use_geocalib` is False -- see the module
    docstring / ModelConfig.frontend.use_geocalib's own comment. Same per-camera grouping-by-K
    approach as baselines/mpsfm/run_sequence.py's own write_intrinsics_yaml (that one uses
    mpsfm's own schema, not vidmap's, hence a separate copy rather than a shared import, per this
    repo's per-baseline self-containment convention)."""
    groups: dict[tuple, list[str]] = {}
    for name, f in frames_by_name.items():
        K = f.camera().K
        key = (round(float(K[0, 0]), 3), round(float(K[1, 1]), 3),
               round(float(K[0, 2]), 3), round(float(K[1, 2]), 3))
        groups.setdefault(key, []).append(name)
    payload = {i: {"params": list(params), "images": sorted(names)}
               for i, (params, names) in enumerate(groups.items(), start=1)}
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def c2w_from_image(image: "pycolmap.Image") -> np.ndarray:
    """4x4 camera-to-world matrix from a registered pycolmap Image's own `cam_from_world()`
    (world-to-camera) pose -- see baselines/colmap/run_sequence.py's identical helper."""
    w2c = image.cam_from_world().matrix()  # (3,4): [R | t]
    R, t = w2c[:, :3], w2c[:, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix for `camera`, via pycolmap's own calibration_matrix() -- vidmap's own
    reconstruction is always PINHOLE (vidmap/datasets/local.py's own _add_cameras), so this is
    simpler than it needs to be here, but kept identical to the colmap family's helper of the
    same name for consistency (and in case that assumption is ever wrong)."""
    return camera.calibration_matrix().astype(np.float64)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts vidmap's world gauge onto the gt one, as far as one
    frame can -- vidmap's own scale is left exactly as reconstructed. Even though vidmap solves a
    per-image *relative* depth-scale correction during bundle adjustment (see
    read_cached_depth's own docstring), that only makes its depth internally self-consistent --
    it introduces no external metric reference, so the reconstruction's own global scale is just
    as unrecoverable as plain COLMAP's, matching every other pose-estimating baseline here."""
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def read_cached_depth(depth_file: "h5py.File", image_name: str, scale: float,
                      width: int, height: int) -> Optional[np.ndarray]:
    """The depth vidmap's frontend cached for `image_name` via --cache-depth-maps (one HDF5
    group per image, keyed by exact image name -- vidmap/frontend/depth.py's own
    write_full_depth_cache), or None if this image has no cached group at all. Stored at DA3's
    own processing resolution, not the original image size (the group's own `original_width`/
    `original_height` attrs give the true size) -- resized back to native here, same
    cv2.resize-back-to-native pattern baselines/mpsfm/run_sequence.py's own read_refined_depth
    already uses. `valid=False` pixels (no depth estimated) become NaN. `scale` is the per-image
    correction the mapper's bundle adjustment solved for this image (see
    depth_lift_scales.json's own docstring in the ModelConfig -- keyed by COLMAP image_id, looked
    up by the caller, not here) -- DA3's own raw prediction is metric but noisy; this scale
    factor is what the reconstruction's own poses/points are actually consistent with."""
    if image_name not in depth_file:
        return None
    group = depth_file[image_name]
    depth = np.asarray(group["depth"], dtype=np.float32)
    valid = np.asarray(group["valid"], dtype=bool)
    orig_width = int(group.attrs["original_width"])
    orig_height = int(group.attrs["original_height"])
    if depth.shape != (orig_height, orig_width):
        depth = cv2.resize(depth, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (orig_width, orig_height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    if (orig_width, orig_height) != (width, height):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    depth = depth * scale
    depth[~valid] = np.nan
    return depth.astype(np.float32)


def np_transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Applies a (4,4) rigid transform to (N,3) points -- see baselines/mpsfm/run_sequence.py's
    identical helper."""
    return points @ transform[:3, :3].T + transform[:3, 3]


# ---------------------------------------------------------------------------------------------
# FrontendConfig groups -- verified field-for-field against vidmap/vidmap/configuration/config.py
# and vidmap/vidmap/frontend/options/{matching,keyframes,depth,preparation,tracking}.py.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class RoMaV2Options:
    compile: bool = True


@dataclasses.dataclass
class PreprocessingOptions:
    """Shared by SalientFeatureOptions.preprocessing and DepthEstimationOptions.preprocessing,
    each with different defaults -- see those groups' own default_factory below."""
    grayscale: bool = False
    resize_max: Optional[int] = None
    resize_force: bool = False
    interpolation: str = "cv2_area"


@dataclasses.dataclass
class LowresMatchOptions:
    batch_size: int = 8
    num_workers: int = 16
    resize_to_shape: tuple = (560, 560)
    interpolation: str = "torch_bicubic"


@dataclasses.dataclass
class DetectKeyframesOptions:
    max_normalized_keypoint_drift: float = 0.11
    target_frac: float = 0.4
    certainty_threshold: float = 0.01
    force_gt_keyframes: bool = False       # only meaningful for a dataset with known keyframe
                                           # timestamps to replay against (vidmap's own "gtkf"
                                           # presets) -- not applicable to this repo's own data
    intrinsics_source: str = "geocalib"    # "geocalib" or "ground_truth"


@dataclasses.dataclass
class SalientFeatureOptions:
    preprocessing: PreprocessingOptions = dataclasses.field(
        default_factory=lambda: PreprocessingOptions(resize_max=1200, resize_force=False))
    nms_radius: int = 5
    max_num_keypoints: int = 2000
    sub_pixel: bool = False


@dataclasses.dataclass
class KeyframeOptions:
    matching: LowresMatchOptions = dataclasses.field(default_factory=LowresMatchOptions)
    selection: DetectKeyframesOptions = dataclasses.field(default_factory=DetectKeyframesOptions)
    features: SalientFeatureOptions = dataclasses.field(default_factory=SalientFeatureOptions)


@dataclasses.dataclass
class RoMaImageOptions:
    grayscale: bool = False
    resize_max: Optional[int] = 1200
    resize_force: bool = True
    interpolation: str = "cv2_area"


@dataclasses.dataclass
class SparseTrackOptions:
    num_workers: int = 4
    nms_radius: int = 3
    max_kps: int = 1500
    min_conf: float = 0.05
    window: int = 9
    max_sequential_track_sigma_roma_px: Optional[float] = 8.0
    density_thin_k: float = 0.05
    density_thin_std: float = 0.013
    tvg_max_epipolar_error: float = 4.0
    tvg_max_iterations: int = 50000
    tvg_min_iterations: Optional[int] = 50
    tvg_min_inliers: int = 50
    salient_density_ref_kps: int = 2000
    salient_density_std: float = 0.1
    salient_density_max_scale: float = 3.0
    salient_density_power: float = 1.0
    lt_cov_scale: float = 4.0
    multiflow_hops: Optional[tuple] = (9, 7, 5, 3, 1)


@dataclasses.dataclass
class FrontendTrackOptions:
    images: RoMaImageOptions = dataclasses.field(default_factory=RoMaImageOptions)
    propagation: SparseTrackOptions = dataclasses.field(default_factory=SparseTrackOptions)


@dataclasses.dataclass
class ExtendedMatchOptions:
    retrieval_min_score: float = 0.1
    nquery: int = 10
    tcorr_min_matches: int = 200
    lc_pair_nms: bool = True
    lc_pair_nms_radius: int = 2
    lc_match_thresh: float = 0.05


@dataclasses.dataclass
class Da3VideoOptions:
    type: str = "da3_video"
    window_size: int = 1
    ref_view_strategy: str = "middle"    # "middle", "first", or "saddle_balanced"
    process_res: int = 504


@dataclasses.dataclass
class DepthEstimationOptions:
    backend: Da3VideoOptions = dataclasses.field(default_factory=Da3VideoOptions)
    cache_map: bool = True
    preprocessing: PreprocessingOptions = dataclasses.field(
        default_factory=lambda: PreprocessingOptions(resize_max=None, resize_force=False))
    batch_size: int = 1
    num_workers: int = 4


@dataclasses.dataclass
class CameraPriorEstimationOptions:
    max_images: int = 30


@dataclasses.dataclass
class GeomVerifOptions:
    max_H_inlier_ratio: float = 0.8
    min_num_inliers: int = 15
    ransac_max_num_trials: int = 20_000
    ransac_min_inlier_ratio: float = 0.25
    ransac_max_error: float = 4.0


@dataclasses.dataclass
class LCOptions:
    exclude_lc_matches: bool = False
    exclude_window_lc: bool = False
    min_lc_score: float = 0.0


@dataclasses.dataclass
class PreparationOptions:
    lc: LCOptions = dataclasses.field(default_factory=LCOptions)
    geom_verif: GeomVerifOptions = dataclasses.field(default_factory=GeomVerifOptions)


@dataclasses.dataclass
class FrontendConfig:
    roma: RoMaV2Options = dataclasses.field(default_factory=RoMaV2Options)
    keyframes: KeyframeOptions = dataclasses.field(default_factory=KeyframeOptions)
    tracks: FrontendTrackOptions = dataclasses.field(default_factory=FrontendTrackOptions)
    loop_closure: ExtendedMatchOptions = dataclasses.field(default_factory=ExtendedMatchOptions)
    depth: DepthEstimationOptions = dataclasses.field(default_factory=DepthEstimationOptions)
    camera_priors: CameraPriorEstimationOptions = dataclasses.field(default_factory=CameraPriorEstimationOptions)
    preparation: PreparationOptions = dataclasses.field(default_factory=PreparationOptions)
    # Known vs. self-calibrated intrinsics -- vidmap's own CLI default (True) self-calibrates via
    # GeoCalib; this baseline's own default matches it (consistent with baselines/colmap/glomap's
    # own self-calibrating defaults, not baselines/mpsfm's known-intrinsics one). Setting this
    # False requires run() to also write and pass an --intrinsics sidecar (see
    # write_intrinsics_yaml) -- see baselines/configs/vidmap.yaml's own header for the other
    # fields that move together with this one for a real "calib mode" (view_graph_calibration,
    # keyframes.selection.intrinsics_source, mapping.mapper.ba.annealing/intrinsics.prior_std_factor,
    # mapping.mapper.vgc.filter.enabled).
    use_geocalib: bool = True
    view_graph_calibration: bool = True


# ---------------------------------------------------------------------------------------------
# MappingConfig groups -- verified field-for-field against
# vidmap/vidmap/mapper/options/{mapper,positioning,refinement,view_graph,solver}.py.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class SetupOptions:
    depth_uncertainty_scale: float = 0.05


@dataclasses.dataclass
class DepthConsistencyOptions:
    enabled: bool = True
    depth_outlier_propagation: str = "boundary"   # "boundary" or "forward"
    mark_boundary_depth_outliers: bool = True
    depth_ratio_threshold: float = 1.3


@dataclasses.dataclass
class BANormalOptions:
    iterations: int = 3
    kp_stddev: float = 2.0
    reproj_loss_name: str = "soft_l1"   # "trivial", "soft_l1", "cauchy", or "huber"
    robust_risky_depth: bool = False
    observation_filter_multiplier: float = 4.0
    convergence_filtered_point_ratio: float = 0.001


@dataclasses.dataclass
class BAAnnealingOptions:
    iterations: int = 3
    kp_std: float = 0.5
    depth_magnitude_multiplier: float = 0.25
    reproj_loss_name: str = "soft_l1"
    fix_rotations: bool = False
    robust_risky_depth: bool = True
    depth_cutoff_cauchy_scales: int = 2
    depth_param_multiplier: float = 0.25
    # None falls back to ba.intrinsics.prior_std_factor (vidmap's own
    # BAOptions.resolved_annealing_prior_std_factor property) -- calib/base.yaml sets this null.
    prior_std_factor: Optional[float] = 0.1
    mad_sigma_floor: float = 0.1
    observation_filter_multiplier: float = 4.0
    convergence_filtered_point_ratio: float = 0.001
    final_depth_param_multiplier: float = 0.125


@dataclasses.dataclass
class BATriangulationOptions:
    min_angle: float = 0.001
    ignore_two_view_tracks: bool = False
    create_max_angle_error: float = 3.0
    re_max_angle_error: float = 3.0
    re_max_trials: int = 1_000_000
    re_min_ratio: float = 1.0
    relaxed_max_transitivity: int = 1000


@dataclasses.dataclass
class BADepthOptions:
    param_multiplier: float = 1.0
    magnitude_multiplier: float = 1.0
    cutoff_cauchy_scales: int = 0
    reg_loss_name: str = "cauchy"          # "trivial", "cauchy", or "soft_l1"
    scale_std: float = 1.0
    target_stddev_multiplier: float = 1.0
    scale_reg_loss_name: str = "soft_l1"   # "trivial", "huber", "cauchy", or "soft_l1"
    risky_track_length_threshold: int = 3
    risky_triangulation_angle_deg: float = 1.0


@dataclasses.dataclass
class BAIntrinsicsOptions:
    use_prior: bool = True
    # Weak by default (self-calibrated intrinsics aren't trusted); calib/base.yaml raises this
    # to 1.0 (intrinsics are known, so a strong prior is appropriate).
    prior_std_factor: float = 0.01
    refine_principal_point: bool = False


@dataclasses.dataclass
class SolverBackendOptions:
    """Shared by ba.solver_backend and gp.solver_backend. use_cuda requires Ceres 2.3+ built with
    CUDA and cuDSS support (vidmap's own README, "GPU acceleration") -- deliberately not attempted
    in baselines/vidmap/Dockerfile (plain CPU Ceres, same as baselines/colmap/glomap), so leave
    this False unless you've separately rebuilt the image with a GPU-capable Ceres."""
    linear_solver: str = "sparse_schur"      # "dense_schur", "sparse_schur", or "iterative_schur"
    preconditioner: str = "schur_jacobi"     # "jacobi", "schur_jacobi", "cluster_jacobi", or
                                             # "cluster_tridiagonal"
    use_cuda: bool = False


@dataclasses.dataclass
class BAOptions:
    normal: BANormalOptions = dataclasses.field(default_factory=BANormalOptions)
    annealing: BAAnnealingOptions = dataclasses.field(default_factory=BAAnnealingOptions)
    triangulation: BATriangulationOptions = dataclasses.field(default_factory=BATriangulationOptions)
    depth: BADepthOptions = dataclasses.field(default_factory=BADepthOptions)
    intrinsics: BAIntrinsicsOptions = dataclasses.field(default_factory=BAIntrinsicsOptions)
    solver_backend: SolverBackendOptions = dataclasses.field(default_factory=SolverBackendOptions)
    retriangulation_reproj_multiplier: float = 8.0
    first_iteration_error_multiplier: float = 2.0
    multiply_errors: float = 2.0
    reproj_loss_scale: float = 1.5
    num_threads: Optional[int] = None
    variable_point_track_length_threshold: int = 15
    post_annealing_point_refinement: bool = True


@dataclasses.dataclass
class VGCFilterOptions:
    # calib/base.yaml sets this False (skip view-graph-calibration filtering entirely when
    # intrinsics are already known).
    enabled: bool = True
    num_threads: Optional[int] = None
    min_matches: int = 10
    min_cheirality_points: int = 5
    subsample_size: int = 50
    strong_pair_match_count: int = 50
    min_median_triangulation_angle_deg: float = 16.0
    max_abs_forward_translation_ratio: float = 0.95
    min_strong_kept_pairs: int = 50


@dataclasses.dataclass
class VGCCalibrationOptions:
    unlock_focal: bool = True


@dataclasses.dataclass
class VGCOptions:
    filter: VGCFilterOptions = dataclasses.field(default_factory=VGCFilterOptions)
    calibration: VGCCalibrationOptions = dataclasses.field(default_factory=VGCCalibrationOptions)


@dataclasses.dataclass
class InlierThresholdOptions:
    max_epipolar_error_E: float = 8.0
    max_epipolar_error_F: float = 8.0
    max_epipolar_error_H: float = 8.0
    min_angle_from_epipole: float = 0.001
    max_angle_error: float = 1.0
    min_triangulation_angle: float = 0.001
    min_inlier_num: float = 5.0
    min_inlier_ratio: float = 0.0


@dataclasses.dataclass
class MDRPOptions:
    """Multi-view depth-reprojection filtering."""
    max_workers: Optional[int] = None
    compute_reproj_error_outliers: bool = True
    reproj_outlier_threshold: float = 5.0
    ransac_max_iterations: int = 50000
    ransac_max_epipolar_error: float = 4.0
    depth_stddev_multiplier: float = 1.0


@dataclasses.dataclass
class LossConfig:
    """A ceres robust loss function -- used throughout the gp/track_filter groups below. `name`
    has no real default in vidmap's own schema (it's a required field there); every usage site
    below supplies its own explicit default via default_factory, so this bare default is never
    actually relied on."""
    name: str = "trivial"     # "trivial", "huber", "cauchy", or "soft_l1"
    scale: float = 1.0
    weight: float = 1.0


@dataclasses.dataclass
class GPCommonOptions:
    use_lc_observations: bool = True
    use_metric_depth_constraint: bool = True
    roundtrip_before_ba: bool = False       # replay-only (see module docstring's scope note)
    canonical_checkpoint: bool = False      # replay-only
    bearing_kp_stddev: float = 8.0
    optimize_depth_map_scales: bool = True
    use_log_scale_for_depth_map_scales: bool = False
    smooth_log_linear_transition: bool = False
    log_linear_threshold: float = 0.1
    random_seed: int = 1
    loss_function_type: str = "huber"      # "trivial", "soft_l1", "cauchy", or "huber"
    loss_function_weight: float = 1.0
    loss_function_scale: float = 0.1
    apply_uncalibrated_loss_downweight: bool = False
    num_threads: Optional[int] = None
    parameter_ordering_strategy: str = "grouped"   # or "deterministic_singleton_groups"
    camera_center_strategy: str = "frame"          # "frame" or "image"
    random_init_scale: float = 100.0
    zero_residuals_behind_camera: bool = False


@dataclasses.dataclass
class GPFirstPassOptions:
    scale_prior_stddev: float = 0.05
    max_iterations: int = 100
    function_tolerance_when_second: Optional[float] = 1.0e-4
    gradient_tolerance_when_second: Optional[float] = 1.0e-8
    parameter_tolerance_when_second: Optional[float] = 1.0e-6
    scale_reg_loss_name: str = "trivial"
    scale_reg_weight: float = 1.0
    initialize_warm_start_scales: bool = True
    loss_lc_geometry: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="cauchy", scale=2.0, weight=0.4))
    loss_lc_depth: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="cauchy", scale=2.0, weight=0.4))
    loss_normal_geometry: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="huber"))
    loss_normal_depth: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="trivial"))


@dataclasses.dataclass
class GPSecondPassOptions:
    enabled: bool = True
    scale_prior_stddev: float = 0.05
    max_iterations: int = 100
    use_log_depth_residual: bool = True
    relax_angular_stddevs: float = 1.0
    scale_reg_loss_name: str = "huber"
    scale_reg_weight: float = 1.0
    center_init_mode: str = "native"    # "native" or "python_frame_centers"
    loss_lc_geometry: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="cauchy", scale=4.0, weight=1.0))
    loss_lc_depth: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="cauchy", scale=4.0, weight=1.0))
    loss_normal_geometry: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="huber"))
    loss_normal_depth: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="huber"))


@dataclasses.dataclass
class GPTrackFilterOptions:
    skip_zero_observation_points: bool = False   # replay-only
    update_point3d_errors: bool = False          # replay-only
    min_num_views_per_track: int = 2
    loss_normal_depth_outlier: LossConfig = dataclasses.field(
        default_factory=lambda: LossConfig(name="cauchy", scale=3.0, weight=1.0))
    depth_prior_outlier_max_depth: Optional[float] = None
    depth_prior_outlier_stages: str = "gp1"   # "gp1" or "gp1_gp2"


@dataclasses.dataclass
class GPTemporalAccelerationOptions:
    """Keyframe-coordinate trajectory smoothing -- disabled by default (both *_weight fields are
    0.0). vidmap's own "smooth_trajectory" recommended config turns this on for improved
    robustness on long sequences (experimental) -- see baselines/configs/vidmap.yaml's own header
    for the exact field combination that reproduces it."""
    coordinate: str = "timestamp"   # "timestamp" or "keyframe_index"
    stddev: float = 1.0
    first_pass_weight: float = 0.0
    second_pass_weight: float = 0.0
    first_pass_dead_zone: float = 0.0
    first_pass_huber_width: float = 1.0
    second_pass_dead_zone: float = 0.0
    second_pass_huber_width: float = 1.0


@dataclasses.dataclass
class GPOptions:
    """Global positioning -- vidmap's initial camera-center/point solve, before bundle
    adjustment."""
    common: GPCommonOptions = dataclasses.field(default_factory=GPCommonOptions)
    solver_backend: SolverBackendOptions = dataclasses.field(default_factory=SolverBackendOptions)
    first_pass: GPFirstPassOptions = dataclasses.field(default_factory=GPFirstPassOptions)
    second_pass: GPSecondPassOptions = dataclasses.field(default_factory=GPSecondPassOptions)
    track_filter: GPTrackFilterOptions = dataclasses.field(default_factory=GPTrackFilterOptions)
    temporal_acceleration: GPTemporalAccelerationOptions = dataclasses.field(
        default_factory=GPTemporalAccelerationOptions)


@dataclasses.dataclass
class RAOptions:
    """Rotation averaging."""
    filter_risky_loop_closure_pairs: bool = False
    filter_unregistered_images: bool = True
    random_seed: int = 1                  # pinned for byte-identical rotation averaging
    max_rotation_error_deg: float = 0.0   # 0.0 disables native post-RA edge filtering
    video_tracking_huber_scale: float = 0.1
    video_lc_cauchy_scale: float = 0.05
    num_threads: Optional[int] = None     # None means one thread, -1 means every core


@dataclasses.dataclass
class MapperTrackOptions:
    include_loop_closure_observations: bool = True
    min_num_views_per_track: int = 2
    max_num_views_per_track: int = 100
    two_view_depth_gate: bool = True
    loop_closure_second_pass: bool = True


@dataclasses.dataclass
class MapperOptions:
    setup: SetupOptions = dataclasses.field(default_factory=SetupOptions)
    depth_consistency: DepthConsistencyOptions = dataclasses.field(default_factory=DepthConsistencyOptions)
    ba: BAOptions = dataclasses.field(default_factory=BAOptions)
    vgc: VGCOptions = dataclasses.field(default_factory=VGCOptions)
    inlier_thresholds: InlierThresholdOptions = dataclasses.field(default_factory=InlierThresholdOptions)
    mdrp: MDRPOptions = dataclasses.field(default_factory=MDRPOptions)
    gp: GPOptions = dataclasses.field(default_factory=GPOptions)
    ra: RAOptions = dataclasses.field(default_factory=RAOptions)
    tracks: MapperTrackOptions = dataclasses.field(default_factory=MapperTrackOptions)
    # replay_cache (mode/root/write_stage) deliberately excluded -- see module docstring.


@dataclasses.dataclass
class MappingConfig:
    mapper: MapperOptions = dataclasses.field(default_factory=MapperOptions)


@dataclasses.dataclass
class ModelConfig:
    frontend: FrontendConfig = dataclasses.field(default_factory=FrontendConfig)
    mapping: MappingConfig = dataclasses.field(default_factory=MappingConfig)
    # Which named Hydra preset to start from -- rarely needs changing, since this file's own
    # dataclass defaults already mirror "uncalib/base"'s resolved values and get passed as a full
    # override tree on top regardless (see _hydra_overrides). "calib/base"/"uncalib/gtkf-base"
    # are the other presets vidmap ships; see baselines/configs/vidmap.yaml's own header for what
    # picking "calib/base" actually requires changing elsewhere in this same config too.
    frontend_conf: str = "uncalib/base"
    mapping_conf: str = "uncalib/base"


class VidMap(Baseline):
    """VidMap: video-aware SfM frontend + COLMAP-derived global mapper, driven through its own
    CLI (see module docstring)."""

    name = "vidmap"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="vidmap-") as tmp:
            tmp_path = Path(tmp)
            images_dir = tmp_path / "images"
            out_dir = tmp_path / "output"
            frames_by_name = stage_frames(frames, images_dir)

            cmd = [
                "python", "-m", "vidmap.run",
                "--input_data", str(images_dir),
                "--output", str(out_dir),
                "--frontend-conf", cfg.frontend_conf,
                "--mapping-conf", cfg.mapping_conf,
                "--cache-depth-maps",
            ]
            if not cfg.frontend.use_geocalib:
                intrinsics_path = tmp_path / "intrinsics.yaml"
                write_intrinsics_yaml(frames_by_name, intrinsics_path)
                cmd += ["--intrinsics", str(intrinsics_path)]
            cmd += _hydra_overrides("frontend", cfg.frontend)
            cmd += _hydra_overrides("mapping", cfg.mapping)

            print(f"running {len(frames)} frames (use_geocalib={cfg.frontend.use_geocalib}) ...")
            subprocess.run(cmd, check=True)

            rec_dir = out_dir / "rec"
            if not rec_dir.is_dir():
                raise SystemExit("vidmap produced no reconstruction (rec/) for this sequence")
            rec = pycolmap.Reconstruction(rec_dir)
            registered = {img.name: img for img in rec.images.values() if img.has_pose}
            missing = set(frames_by_name) - set(registered)
            if not registered:
                raise SystemExit("vidmap registered no frames at all for this sequence")
            if missing:
                print(f"  registered {len(registered)}/{len(frames)} frames -- "
                     f"{len(missing)} unregistered frame(s) get no output")

            anchor_name = min(registered, key=lambda n: frames_by_name[n].index)
            anchor_frame = frames_by_name[anchor_name]
            anchor_c2w = c2w_from_image(registered[anchor_name])
            align = _gt_frame0_alignment(anchor_c2w, anchor_frame.camera().c2w)

            try:
                full_depth_path, _sampled_path = resolve_depth_paths(out_dir)
            except FileNotFoundError as exc:
                raise SystemExit(f"vidmap produced no cached depth maps despite "
                                 f"--cache-depth-maps: {exc}")
            scales = load_depth_scales(out_dir)
            if not scales:
                print("  no depth_lift_scales.json found -- writing DA3's raw (unscaled) depth")

            no_depth = 0
            with h5py.File(full_depth_path, "r") as depth_file:
                for name, image in registered.items():
                    f = frames_by_name[name]
                    camera = rec.cameras[image.camera_id]
                    width, height = image_size(f.rgb_path)
                    depth = read_cached_depth(depth_file, name, scales.get(image.image_id, 1.0),
                                              width, height)
                    if depth is None:
                        no_depth += 1
                        depth = np.full((height, width), np.nan, dtype=np.float32)
                    c2w = align @ c2w_from_image(image)
                    K = K_from_camera(camera)
                    out.write_frame(f.index, depth, c2w, K, intrinsics_width=camera.width)
            if no_depth:
                print(f"  {no_depth}/{len(registered)} registered frame(s) had no cached depth "
                     "map -- written as all-NaN")

            if out.spec.cloud:
                pts = np.array([p.xyz for p in rec.points3D.values()])
                cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
                if len(pts):
                    pts = np_transform_points(pts, align)
                    out.write_cloud(pts, cols)
                    print(f"  wrote {len(pts):,} points -> {out.cloud_path.relative_to(out.root)}")
                else:
                    print("  no triangulated points to write a cloud from")


if __name__ == "__main__":
    cli_main(VidMap, description=__doc__)
