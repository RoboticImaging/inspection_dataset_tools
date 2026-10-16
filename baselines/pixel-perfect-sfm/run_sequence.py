"""Run Pixel-Perfect SfM (`pixsfm`, cvg/pixel-perfect-sfm) -- classical COLMAP SfM refined by
directly aligning dense deep features across views -- on the sequences of an inspection asset,
and write depth + pose + point-cloud predictions back into the asset folder
(predictions/{method}/...), in the same format the ground truth is stored in -- so
inspection_viz's Source toggle picks them up unchanged.

Architecturally this is baselines/colmap/run_sequence.py's sibling, not baselines/glomap's: same
pipeline, driven the same way (colmap CLI subprocesses). The front end is shared with that
baseline: feature extraction and matching run as one
`colmap automatic_reconstructor --extraction 1 --matching 1 --sparse 0 --dense 0` call, configured
by the same `AutomaticReconstructorConfig` (data_type, quality, camera_model, single_camera, ...),
so the same yaml values give both baselines identical features and an identical match graph.
Mapping is NOT part of that call (`--sparse 0`): it stays a separate `colmap mapper` stage,
driven by the Mapper/IncrementalMapper/Triangulation groups below, so pixsfm's keypoint
adjustment can slot in between matching and mapping. `stage_frames`, `merge_submodels`/
`_images_count`, `c2w_from_image`, `K_from_camera`, `_gt_frame0_alignment`, `read_depth_map`,
`_flags`, and the AutomaticReconstructor/Mapper/IncrementalMapper/Triangulation/PatchMatch/
StereoFusion config groups are duplicated from baselines/colmap/run_sequence.py, not imported --
this repo's established per-baseline self-containment convention -- EXCEPT the `run()` method's
mapper/patch_match `_flags(...)` call sites, which are NOT verbatim: this baseline's Dockerfile
pins its own `colmap` binary to COLMAP 3.8 (see the Dockerfile's own header), whose CLI predates
several option-group splits/renames (and some options outright) that baselines/colmap's own newer
`colmap` binary has. `_flags()` itself grew `skip`/`rename` params for exactly this; see its own
docstring and the `run()` method's inline comments for the specific COLMAP-3.8 differences,
confirmed against COLMAP 3.8's own src/util/option_manager.cc, not guessed. Don't re-sync `run()`
back to colmap's own file's `_flags()` calls verbatim -- that reintroduces "unrecognised option"
failures against 3.8.
What's new here is two extra refinement steps `pixsfm` inserts
into that same pipeline, both operating through its own `pixsfm.refine_colmap.PixSfM` Python API
(not a CLI) rather than colmap's:

  1. Keypoint adjustment (KA), between matching and mapper: refines the 2D keypoint
     locations of all tentatively-matched features by minimizing their feature-metric
     inconsistency across views, writing a new COLMAP database that `colmap mapper` is then run
     on instead of the raw one -- so the incremental reconstruction itself benefits from more
     accurate keypoints, not just a post-hoc pose cleanup.
  2. Bundle adjustment (BA), between merge_submodels and the dense stage: re-refines the merged
     model's camera poses and 3D points by minimizing the feature-metric error against dense
     features sampled at each point's observations, writing a new COLMAP model that
     image_undistorter is then run on instead of the merged one.

Either step can be disabled independently (`ka.apply`/`ba.apply` in ModelConfig) -- with both off,
this baseline is functionally identical to baselines/colmap's, just carrying pixsfm's own
(otherwise-unused) Python dependency.

Every `dense_features`/`interpolation`/`ka`/`ba` default below was read directly from this exact
pixel-perfect-sfm submodule checkout's own source (`pixsfm/base/main.py`,
`pixsfm/features/extractor.py`, `pixsfm/features/models/s2dnet.py`,
`pixsfm/keypoint_adjustment/main.py`, `pixsfm/bundle_adjustment/main.py`), NOT from the README's
own "Detailed configuration" doc block, which is stale relative to the actual code in several
places (confirmed by diffing them directly): its solver defaults show
`parameter_tolerance: 1.0e-05` where the source's shared `solver_default_conf` actually has `0.0`
(only KA's own optimizer overrides it to 1e-5 -- BA's does not); it's missing
`interpolation.use_float_simd`, `dense_features.load_cache_on_init`, and BA's
`references.keep_observations`/`references.compute_offsets3D` entirely; and its own prose usage
example for `refine_reconstruction` shows argument order `(input, output, image_dir)` where the
actual function signature (pixsfm/refine_colmap.py) is `(output_path, input_path, image_dir)` --
this file calls it by keyword, not position, specifically to never depend on getting that order
right from memory again.

Usage -- inside the container (build_container.sh once, then a shell in the image):
    python run_sequence.py --data ../configs/data/example.yaml --model ../configs/pixel-perfect-sfm.yaml

Paths are the same inside the container as on the host, so absolute config paths work too.
Do not run this outside the container: it needs the image's `colmap` CLI binary, pycolmap,
pyceres, and pixsfm itself (see baselines/pixel-perfect-sfm/Dockerfile).
"""

# Postponed annotation evaluation (PEP 563): lets `X | None` etc. below run on Python < 3.10
# (e.g. this baseline's own pixel-perfect-sfm-eigen-src image, Ubuntu 20.04's default Python 3.8)
# without needing `typing.Optional`/`typing.Union` -- annotations are stored as strings instead
# of evaluated at class-definition time, so `type.__or__` (added in 3.10) is never actually
# called. The main pixel-perfect-sfm image (Ubuntu 22.04, Python 3.10) doesn't need this, but it's
# a no-op there.
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import trimesh
from pixsfm.refine_colmap import PixSfM as PixSfMRefiner

from baselines.common.runner import Baseline, SequenceOutput, cli_main
from baselines.common.writer import frame_stem
from inspection_viz.colmap import read_array
from inspection_viz.dataset import Frame
from inspection_viz.io import image_size


def _flags(prefix: str | None, cfg, skip: frozenset = frozenset(),
           rename: dict[str, str] | None = None) -> list[str]:
    """Every field of dataclass `cfg` as CLI args for a `colmap` subprocess -- see
    baselines/colmap/run_sequence.py's identical helper for the full docstring. Unlike the
    dense_features/interpolation/ka/ba groups below (which go to pixsfm's own Python API as a
    plain nested dict, where None is a real, meaningful value), this is only ever used for the
    colmap-CLI-facing groups, where None means "omit this flag, use colmap's own default".

    `skip`/`rename` exist for this baseline specifically: it runs against COLMAP 3.8 (see the
    Dockerfile's own header), whose CLI predates several option-group splits/renames baselines/
    colmap's own newer `colmap` binary has -- `skip` drops fields 3.8 doesn't accept at all,
    `rename` maps a field to 3.8's differently-named equivalent flag (dataclass field names
    themselves stay as baselines/colmap/run_sequence.py's own, so pixsfm's own Python-API config
    groups elsewhere in this file are unaffected -- only the CLI flag name changes here)."""
    args = []
    for f in dataclasses.fields(cfg):
        if f.name in skip:
            continue
        val = getattr(cfg, f.name)
        if val is None:
            continue
        if isinstance(val, bool):
            val = "1" if val else "0"
        name = (rename or {}).get(f.name, f.name)
        flag = f"--{prefix}.{name}" if prefix else f"--{name}"
        args += [flag, str(val)]
    return args


# ---------------------------------------------------------------------------------------------
# Colmap-side ModelConfig groups: identical to baselines/colmap/run_sequence.py's own (duplicated,
# not imported -- see module docstring). AutomaticReconstructorConfig covers feature extraction +
# matching here (`--sparse 0 --dense 0` are always passed -- mapping and the dense stage are
# driven separately). Comments trimmed to what's not already said there.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class AutomaticReconstructorConfig:
    """`colmap automatic_reconstructor`'s own flags. Field names match its CLI flag names exactly,
    so `_flags(None, cfg)` needs no per-field mapping table. `--extraction 1 --matching 1
    --sparse 0 --dense 0` are always passed (mapping and the dense stage are driven separately,
    below) and are not fields."""
    data_type: str = "video"           # "individual", "video" or "internet". "video" matches
                                       # sequentially (neighbouring frames), not exhaustively
    quality: str = "medium"            # "low", "medium", "high" or "extreme" -- sets feature
                                       # extraction/matching resolution and effort
    camera_model: str = "SIMPLE_PINHOLE"   # self-calibrate a distortion-free model
    single_camera: bool = True         # one shared camera for the whole sequence (a fixed
                                       # render rig)
    camera_params: str | None = None   # e.g. "fx,fy,cx,cy"; None = self-calibrate from scratch
    use_gpu: bool = True
    gpu_index: str = "-1"              # "-1" = auto-select
    num_threads: int = -1              # -1 = auto (all cores)


@dataclasses.dataclass
class MapperConfig:
    min_num_matches: int = 15
    ignore_watermarks: bool = False
    multiple_models: bool = True
    max_num_models: int = 50
    max_model_overlap: int = 20
    min_model_size: int = 10
    init_image_id1: int | None = None
    init_image_id2: int | None = None
    init_num_trials: int = 200
    structure_less_registration_fallback: bool = True
    structure_less_registration_only: bool = False
    extract_colors: bool = True
    num_threads: int = -1
    random_seed: int = -1
    min_focal_length_ratio: float = 0.1
    max_focal_length_ratio: float = 10.0
    max_extra_param: float = 1.0
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True
    ba_refine_sensor_from_rig: bool = True
    ba_local_function_tolerance: float = 0.0
    ba_local_max_num_iterations: int = -1
    ba_global_frames_ratio: float = 1.1
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
    ba_local_backend: str | None = None
    ba_global_backend: str | None = None
    ba_min_num_residuals_for_cpu_multi_threading: int = 50000
    snapshot_path: str | None = None
    snapshot_frames_freq: int = 0
    fix_existing_frames: bool = False
    max_runtime_seconds: int | None = None


@dataclasses.dataclass
class IncrementalMapperConfig:
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
    geom_consistency: bool = False
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


# ---------------------------------------------------------------------------------------------
# pixsfm-side ModelConfig groups: passed to pixsfm.refine_colmap.PixSfM as a plain nested dict
# (via dataclasses.asdict), not CLI flags -- None here is a real value pixsfm itself interprets
# (e.g. level_indices=None means "every level"), never an "omit this" sentinel the way it is for
# the colmap-CLI groups above. Every default is cited against its own source file in the class
# docstring below; see the module docstring for how these diverge from the submodule's README.
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class LossConfig:
    """A ceres robust loss function, as used by KA/BA/references/costmaps below."""
    name: str = "cauchy"        # among {trivial, huber, soft_l1, cauchy, arctan, tukey, ...}
    params: list = dataclasses.field(default_factory=lambda: [0.25])  # loss-specific scale(s)


@dataclasses.dataclass
class SolverConfig:
    """pixsfm/base/main.py's `solver_default_conf` -- the ceres solver options shared (with
    per-group overrides, see KAOptimizerConfig/BAOptimizerConfig below) by every optimization
    problem pixsfm solves. MEMORY/RUNTIME: `num_threads` here is the ceres-internal thread count
    for one single optimization problem, separate from -- and usually far less impactful than --
    the *outer* parallelism knobs (KA's split_in_subproblems/max_kps_per_problem, BA's
    max_tracks_per_problem) that control how many such problems run at once."""
    function_tolerance: float = 0.0
    gradient_tolerance: float = 0.0
    parameter_tolerance: float = 0.0
    minimizer_progress_to_stdout: bool = False
    max_num_iterations: int = 100
    max_linear_solver_iterations: int = 200
    max_num_consecutive_invalid_steps: int = 10
    max_consecutive_nonmonotonic_steps: int = 10
    use_inner_iterations: bool = False
    use_nonmonotonic_steps: bool = False
    update_state_every_iteration: bool = False
    num_threads: int = -1


@dataclasses.dataclass
class InterpolationConfig:
    """pixsfm/base/main.py's `interpolation_default_conf` -- how pixsfm samples the dense
    feature maps at (possibly sub-pixel) 2D locations. Shared by KA and BA (pixsfm's own default
    config points both at one shared `interpolation:` block via OmegaConf variable
    interpolation, `"${..interpolation}"`; this file does the same by construction -- one
    dataclass instance, reused directly by both, rather than two independently-set copies)."""
    nodes: list = dataclasses.field(default_factory=lambda: [[0.0, 0.0]])  # sample offsets (px)
                                                                            # around each keypoint
    mode: str = "BICUBIC"          # or "BILINEAR", "NEAREST", "BICUBIC_GRAD" (interpolant order)
    l2_normalize: bool = True
    ncc_normalize: bool = False    # only meaningful with len(nodes) > 1 (patch-based costs)
    use_float_simd: bool = False


@dataclasses.dataclass
class DenseFeaturesModelConfig:
    """The CNN that extracts dense features (pixsfm/features/extractor.py's `model:` sub-conf).
    Fields below are S2DNet's own (pixsfm/features/models/s2dnet.py) -- the default `name` --
    and are simply ignored if `name` is changed to another model registered in
    pixsfm/features/models/ (e.g. "vggnet" or a plain "sift"/"dsift" descriptor-as-feature
    baseline), which each define their own extra sub-conf instead."""
    name: str = "s2dnet"
    num_layers: int = 1             # how many hypercolumn layers to concatenate (more = slower,
                                    # more memory, but more scale-invariant features)
    checkpointing: int | None = None  # gradient checkpointing period; None = disabled (unused
                                      # anyway -- pixsfm only ever runs this model in inference)
    output_dim: int = 128
    pretrained: str | None = "s2dnet"   # "s2dnet", "imagenet", or None (random init)
    remove_pooling_layers: bool = False
    combine: bool = False


@dataclasses.dataclass
class DenseFeaturesConfig:
    """pixsfm/features/extractor.py's `FeatureExtractor.default_conf` -- how dense features are
    extracted, stored, and cached. MEMORY: the single biggest lever in this whole file. See the
    module docstring / baselines/configs/pixel-perfect-sfm.yaml's own "Memory and runtime" section
    for the concrete combination pixsfm's own `low_memory.yaml` template uses."""
    model: DenseFeaturesModelConfig = dataclasses.field(default_factory=DenseFeaturesModelConfig)
    device: str = "auto"           # "cpu", "cuda", or "auto" (cuda if available)
    dtype: str = "half"            # "half", "float", or "double" -- storage precision
    fast_image_load: bool = False  # approximate (cheaper) resizing for large images
    l2_normalize: bool = True
    max_edge: int = 1600           # downscale images so the longer side is at most this (px)
    patch_size: int = 16           # MEMORY: side length (px) of the sparse patch stored per
                                    # keypoint, if sparse=True -- smaller is strictly cheaper
    pyr_scales: list = dataclasses.field(default_factory=lambda: [1.0])  # extract at multiple
                                                                          # image scales and
                                                                          # concatenate (more
                                                                          # scales = more memory)
    resize: str = "LANCZOS"        # PIL resampling filter name
    sparse: bool = True            # MEMORY: store small patches per keypoint (True) instead of
                                    # the full dense feature map per image (False) -- False is
                                    # far more memory-hungry and rarely needed for KA/BA, which
                                    # only ever look at feature values near known 2D points
    use_cache: bool = False        # MEMORY: write features to an HDF5 file and stream them back
                                    # on demand instead of holding them all in RAM at once
    overwrite_cache: bool = False  # if use_cache, start the cache file fresh instead of reusing
                                    # one left over from a previous (possibly interrupted) run
    load_cache_on_init: bool = False  # if use_cache, whether to eagerly load the whole cache
                                      # into RAM right away (defeats the point of caching if True)
    cache_format: str = "chunked"  # HDF5 layout; "chunked" is what pixsfm's own configs use


@dataclasses.dataclass
class KAOptimizerConfig:
    """pixsfm/keypoint_adjustment/main.py's `KeypointAdjuster.default_conf["optimizer"]`, with
    the `FeatureMetricKeypointAdjuster` (strategy="featuremetric", the default) subclass's own
    extra fields (root_regularize_weight/weight_by_sim/root_edges_only) folded in -- those three
    are simply unused if KAConfig.strategy is changed to "topological_reference" instead."""
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    solver: SolverConfig = dataclasses.field(
        default_factory=lambda: SolverConfig(parameter_tolerance=1e-5, num_threads=1))
    print_summary: bool = False
    bound: float = 4.0             # MEMORY/RUNTIME: max pixel displacement allowed from the
                                    # original keypoint location -- pixel-perfect-sfm's own
                                    # low_memory.yaml tightens this to 2.0 alongside a smaller
                                    # patch_size, since a smaller patch can't support as large a
                                    # displacement anyway
    num_threads: int = -1          # outer parallelism, only used if split_in_subproblems
    root_regularize_weight: float = -1   # add edges to the root node with this weight to limit
                                         # drift; -1 = don't (strategy=featuremetric only)
    weight_by_sim: bool = True          # (strategy=featuremetric only)
    root_edges_only: bool = False       # ignore edges not touching the root node
                                        # (strategy=featuremetric only)


@dataclasses.dataclass
class KAConfig:
    """Keypoint adjustment -- refines 2D keypoints from tentative matches, before `colmap
    mapper` ever runs. See the module docstring for where this sits in the overall pipeline."""
    apply: bool = True
    strategy: str = "featuremetric"   # or "topological_reference": RUNTIME -- linear rather
                                       # than quadratic in track size, "significantly faster...
                                       # but slightly less accurate" (pixsfm's own docstring);
                                       # also what pixsfm's own low_memory.yaml switches to
    level_indices: list | None = None   # which of dense_features.pyr_scales' levels to use;
                                        # None = every level, reversed (coarse-to-fine)
    max_kps_per_problem: int = 50   # MEMORY/RUNTIME: chunk size for split_in_subproblems --
                                    # pixsfm's own low_memory.yaml raises this to 1000 (fewer,
                                    # bigger chunks trade some parallelism for less overhead once
                                    # patches are already small)
    split_in_subproblems: bool = True   # parallelize by chunking tracks into independent problems
    optimizer: KAOptimizerConfig = dataclasses.field(default_factory=KAOptimizerConfig)


@dataclasses.dataclass
class BAOptimizerConfig:
    """pixsfm/bundle_adjustment/main.py's `BundleAdjuster.default_conf["optimizer"]`."""
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    solver: SolverConfig = dataclasses.field(
        default_factory=lambda: SolverConfig(use_inner_iterations=True))
    print_summary: bool = False
    refine_focal_length: bool = True
    refine_principal_point: bool = False
    refine_extra_params: bool = True    # distortion parameters
    refine_extrinsics: bool = True      # camera poses


@dataclasses.dataclass
class ReferencesConfig:
    """How BA computes the fixed per-point3D feature reference it optimizes towards (only used
    by strategy="feature_reference", the default -- see CostmapsConfig for strategy="costmaps"'s
    own equivalent)."""
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    iters: int = 100
    keep_observations: bool = False    # MEMORY: keep every observation's own feature patch
                                       # around after computing the reference, instead of freeing
                                       # it -- pixsfm's own low_memory.yaml explicitly sets this
                                       # False (the default already), i.e. don't turn it on
    compute_offsets3D: bool = False
    num_threads: int = -1


@dataclasses.dataclass
class CostmapsConfig:
    """Only used when BAConfig.strategy == "costmaps" (pixsfm/bundle_adjustment/main.py's
    `CostMapBundleAdjuster.default_conf["costmaps"]`) -- otherwise present but ignored. RUNTIME/
    MEMORY: this strategy "significantly reduces RAM consumption" (pixsfm's own docstring) by
    locally approximating the feature-metric cost with a precomputed cost volume per
    observation instead of holding every dense feature in memory during BA -- the strategy
    pixsfm's own low_memory.yaml switches BA to."""
    loss: LossConfig = dataclasses.field(default_factory=lambda: LossConfig(name="trivial", params=[]))
    as_gradientfield: bool = True
    compute_cross_derivative: bool = False
    num_threads: int = -1


@dataclasses.dataclass
class BAConfig:
    """Bundle adjustment -- refines camera poses and 3D points of the merged sparse model,
    before the dense stage runs. See the module docstring for where this sits in the pipeline."""
    apply: bool = True
    strategy: str = "feature_reference"   # or "costmaps" (MEMORY: see CostmapsConfig), or the
                                          # less commonly used "patch_warp"/"geometric" (each has
                                          # its own extra sub-conf, not modeled here -- edit
                                          # run_sequence.py directly if you need those)
    level_indices: list | None = None
    max_tracks_per_problem: int = 10   # MEMORY/RUNTIME: chunk size for reference/costmap
                                       # computation -- pixsfm's own low_memory.yaml raises this
                                       # to 100 (same trade-off as KA's max_kps_per_problem)
    optimizer: BAOptimizerConfig = dataclasses.field(default_factory=BAOptimizerConfig)
    references: ReferencesConfig = dataclasses.field(default_factory=ReferencesConfig)
    costmaps: CostmapsConfig = dataclasses.field(default_factory=CostmapsConfig)


@dataclasses.dataclass
class ModelConfig:
    # Colmap-side (same groups as baselines/colmap/run_sequence.py's own ModelConfig, except that
    # the mapper stage's knobs stay individually exposed here -- see module docstring).
    automatic_reconstructor: AutomaticReconstructorConfig = dataclasses.field(default_factory=AutomaticReconstructorConfig)
    mapper: MapperConfig = dataclasses.field(default_factory=MapperConfig)
    incremental_mapper: IncrementalMapperConfig = dataclasses.field(default_factory=IncrementalMapperConfig)
    triangulation: TriangulationConfig = dataclasses.field(default_factory=TriangulationConfig)
    patch_match: PatchMatchConfig = dataclasses.field(default_factory=PatchMatchConfig)
    stereo_fusion: StereoFusionConfig = dataclasses.field(default_factory=StereoFusionConfig)
    depth_source: str = "photometric"

    # pixsfm-side.
    dense_features: DenseFeaturesConfig = dataclasses.field(default_factory=DenseFeaturesConfig)
    interpolation: InterpolationConfig = dataclasses.field(default_factory=InterpolationConfig)
    ka: KAConfig = dataclasses.field(default_factory=KAConfig)
    ba: BAConfig = dataclasses.field(default_factory=BAConfig)


def _pixsfm_conf(cfg: ModelConfig) -> dict:
    """The plain nested dict pixsfm.refine_colmap.PixSfM(conf=...) expects -- matching its own
    default_conf shape exactly (dense_features/interpolation/KA/BA as siblings, uppercase KA/BA
    keys). dataclasses.asdict() recurses through every nested group, so this needs no per-field
    handling the way the colmap-CLI `_flags()` helper does."""
    return {
        "dense_features": dataclasses.asdict(cfg.dense_features),
        "interpolation": dataclasses.asdict(cfg.interpolation),
        "KA": dataclasses.asdict(cfg.ka),
        "BA": dataclasses.asdict(cfg.ba),
    }


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
    """The "Images: N" on the line immediately after "{label} ---" -- see
    baselines/colmap/run_sequence.py's identical function for the full rationale."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if f"{label} ---" in line and i + 1 < len(lines):
            m = re.search(r"Images:\s*(\d+)", lines[i + 1])
            return int(m.group(1)) if m else 0
    return 0


def merge_submodels(sparse_dir: Path, merge_dir: Path) -> bool:
    """Merges every disconnected sub-model colmap's mapper left under `sparse_dir` into one
    reconstruction at `merge_dir` -- see baselines/colmap/run_sequence.py's identical function
    for the full edge-case rationale (ported from colmap_merge_loop_smart.sh)."""
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
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return merge_dir.exists()


def c2w_from_image(image: "pycolmap.Image") -> np.ndarray:
    """4x4 camera-to-world matrix from a registered pycolmap Image's own world-to-camera pose --
    see baselines/colmap/run_sequence.py's identical helper, which uses that newer pycolmap's
    `image.cam_from_world().matrix()` instead. pycolmap 0.4.0 (pinned here, see this baseline's
    own Dockerfile) predates the Rigid3d/`cam_from_world()` API entirely -- real error seen:
    "AttributeError: 'pycolmap.Image' object has no attribute 'cam_from_world'".
    `image.projection_matrix()` is 0.4.0's own equivalent, confirmed identical in shape/semantics
    (its own docstring: "Compose the projection matrix from world to image space", returning the
    same (3, 4) [R | t] world-to-camera matrix `cam_from_world().matrix()` would)."""
    w2c = image.projection_matrix()  # (3,4): [R | t]
    R, t = w2c[:, :3], w2c[:, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def K_from_camera(camera: "pycolmap.Camera") -> np.ndarray:
    """3x3 intrinsic matrix for `camera`, via pycolmap's own calibration_matrix() -- see
    baselines/colmap/run_sequence.py's identical helper for why this doesn't assume PINHOLE."""
    return camera.calibration_matrix().astype(np.float64)


def _gt_frame0_alignment(anchor_c2w: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """The (4,4) rigid transform that puts the reconstruction's world gauge onto the gt one, as
    far as one frame can -- scale is left exactly as reconstructed (see
    baselines/colmap/run_sequence.py's identical helper for the full rationale: pixsfm refines
    poses/points against dense features, but introduces no new metric reference, so this
    baseline's scale is exactly as unrecoverable as plain COLMAP's)."""
    return gt_c2w @ np.linalg.inv(anchor_c2w)


def read_depth_map(dense_dir: Path, image_name: str, depth_source: str) -> np.ndarray | None:
    """The dense depth map patch_match_stereo wrote for `image_name`, or None if it never got
    one -- see baselines/colmap/run_sequence.py's identical helper."""
    path = dense_dir / "stereo" / "depth_maps" / f"{image_name}.{depth_source}.bin"
    if not path.is_file():
        return None
    depth = read_array(path)
    return np.where(depth > 0, depth, np.nan).astype(np.float32)


class PixelPerfectSfM(Baseline):
    """Pixel-Perfect SfM: classical COLMAP SfM + dense-feature keypoint/bundle refinement (see
    module docstring)."""

    name = "pixel-perfect-sfm"
    Config = ModelConfig

    def run(self, frames: list[Frame], out: SequenceOutput) -> None:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="pixsfm-") as tmp:
            tmp_path = Path(tmp)
            images_dir = tmp_path / "images"
            ws = tmp_path / "ws"
            frames_by_name = stage_frames(frames, images_dir)
            ws.mkdir(parents=True, exist_ok=True)
            database_path = ws / "database.db"

            ar = cfg.automatic_reconstructor
            print(f"running {len(frames)} frames (automatic_reconstructor: "
                  f"data_type={ar.data_type}, quality={ar.quality}, "
                  f"camera_model={ar.camera_model}, "
                  f"KA={'on' if cfg.ka.apply else 'off'}, BA={'on' if cfg.ba.apply else 'off'}) ...")
            # Feature extraction + matching only (--sparse 0 --dense 0): same call, same options
            # as baselines/colmap's, minus its mapping stage. Writes ws/database.db (features,
            # matches, verified two-view geometries). Mapping below stays a separate
            # `colmap mapper` so keypoint adjustment can run on the database first.
            subprocess.run([
                "colmap", "automatic_reconstructor",
                "--image_path", str(images_dir),
                "--workspace_path", str(ws),
                "--extraction", "1",
                "--matching", "1",
                "--sparse", "0",
                "--dense", "0",
                *_flags(None, ar),
            ], check=True)
            if not database_path.is_file():
                raise SystemExit("automatic_reconstructor produced no database.db "
                                 "(feature extraction/matching did not run)")

            # One PixSfM instance drives both refinement steps below, so its dense feature
            # extractor (and cache, if dense_features.use_cache) is shared between them instead
            # of loading the model / re-extracting features twice.
            refiner = PixSfMRefiner(conf=_pixsfm_conf(cfg))
            cache_path = ws / "dense_features.h5" if cfg.dense_features.use_cache else None

            # Keypoint adjustment: refine 2D keypoints from tentative matches, before mapping --
            # write to a new database rather than overwriting the matched one in place, keeping
            # it available for debugging a bad refinement.
            mapper_database_path = database_path
            if cfg.ka.apply:
                ka_database_path = ws / "database_ka.db"
                print("  running keypoint adjustment ...")
                refiner.refine_keypoints_from_db(
                    output_path=ka_database_path,
                    database_path=database_path,
                    image_dir=images_dir,
                    cache_path=cache_path,
                )
                mapper_database_path = ka_database_path

            # `colmap mapper` requires --output_path to already exist (it CHECK-fails and aborts
            # otherwise, same as `global_mapper` in the glomap baseline).
            sparse_dir = ws / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            # COLMAP 3.8 predates the Rig/Frame refactor: no per-rig BA controls
            # (ba_refine_sensor_from_rig), no BA-backend selection (ba_use_gpu/ba_gpu_index/
            # ba_local_backend/ba_global_backend -- 3.8 only has a global-BA-specific
            # ba_global_use_pba/ba_global_pba_gpu_index, not a clean equivalent to rename to), no
            # structure-less registration, and no max_runtime_seconds -- all skipped below.
            # images->frames terminology renames (this repo's own config field names already
            # match the newer colmap; only the *CLI flag name* reverts to 3.8's here) and
            # local_ba_min_tri_angle's swapped word order are exact equivalents, so renamed rather
            # than dropped. Confirmed against COLMAP 3.8's own option_manager.cc, not guessed.
            #
            # cfg.mapper.ba_local_max_num_iterations/ba_global_max_num_iterations default to -1,
            # meaning "no cap, let Ceres's own tolerances decide" in the newer colmap this config
            # group was written against. COLMAP 3.8's IncrementalMapperController::Options::Check()
            # hard-requires both >= 0 -- real error seen: "[incremental_mapper.cc:288] Check
            # failed: ba_local_max_num_iterations >= 0 (-1 vs. 0)". A genuinely huge stand-in
            # (tried: 1000) turned out wrong in practice, not just in principle: cfg.mapper's own
            # ba_local_function_tolerance/ba_global_function_tolerance are BOTH 0.0 (this repo's
            # own config, matching the newer colmap's intent that "-1 iterations" is the real
            # stopping criterion), which disables Ceres's usual early-stop-on-plateau behavior --
            # so a huge cap doesn't mean "unlikely to ever be reached", it means "always reached".
            # Watched real `colmap mapper` output: global BA's cost plateaus by ~iteration 150
            # every time, then burns through the rest of a 1000-iteration cap doing nothing (each
            # such call taking 20-30+ seconds, repeated for every incremental registration round).
            # Using COLMAP 3.8's own tuned defaults instead (30-40 local / 75-100 global, per its
            # own src/util/option_manager.cc) with headroom, not the raw incremental_mapper.h
            # struct defaults (25/50) or an arbitrarily large number.
            _NO_MAX_LOCAL_BA_ITERATIONS_3_8 = 100
            _NO_MAX_GLOBAL_BA_ITERATIONS_3_8 = 200
            mapper_ba_local_iters = (cfg.mapper.ba_local_max_num_iterations
                                      if cfg.mapper.ba_local_max_num_iterations >= 0
                                      else _NO_MAX_LOCAL_BA_ITERATIONS_3_8)
            mapper_ba_global_iters = (cfg.mapper.ba_global_max_num_iterations
                                       if cfg.mapper.ba_global_max_num_iterations >= 0
                                       else _NO_MAX_GLOBAL_BA_ITERATIONS_3_8)
            subprocess.run([
                "colmap", "mapper",
                "--image_path", str(images_dir),
                "--database_path", str(mapper_database_path),
                "--output_path", str(sparse_dir),
                "--Mapper.ba_local_max_num_iterations", str(mapper_ba_local_iters),
                "--Mapper.ba_global_max_num_iterations", str(mapper_ba_global_iters),
                *_flags("Mapper", cfg.mapper, skip=frozenset({
                    "structure_less_registration_fallback", "structure_less_registration_only",
                    "ba_refine_sensor_from_rig", "ba_use_gpu", "ba_gpu_index",
                    "ba_local_backend", "ba_global_backend",
                    "ba_min_num_residuals_for_cpu_multi_threading", "max_runtime_seconds",
                    # One GLOBAL --random_seed in 3.8, not Mapper.random_seed -- see the matching
                    # call's own skip above for the real error and why dropping this is a no-op
                    # here (already at the class default).
                    "random_seed",
                    # Handled above instead of passed through as-is (-1) -- see this block's own
                    # comment.
                    "ba_local_max_num_iterations", "ba_global_max_num_iterations",
                }), rename={
                    "ba_global_frames_ratio": "ba_global_images_ratio",
                    "ba_global_frames_freq": "ba_global_images_freq",
                    "snapshot_frames_freq": "snapshot_images_freq",
                    "fix_existing_frames": "fix_existing_images",
                }),
                *_flags("Mapper", cfg.incremental_mapper, skip=frozenset({
                    "ba_global_ignore_redundant_points3D",
                    "ba_global_ignore_redundant_points3D_min_coverage_gain",
                }), rename={"ba_local_min_tri_angle": "local_ba_min_tri_angle"}),
                *_flags("Mapper", cfg.triangulation),
            ], check=True)

            if not sparse_dir.is_dir() or not any(sparse_dir.iterdir()):
                raise SystemExit("COLMAP registered no images at all for this sequence "
                                 "(mapper produced no sparse/ sub-model)")

            merge_dir = ws / "sparse_merge"
            if not merge_submodels(sparse_dir, merge_dir) or not merge_dir.is_dir():
                raise SystemExit("model_merger never produced a usable merged model for this "
                                 "sequence (see colmap_merge_loop_smart.sh's own logic)")

            # Bundle adjustment: refine the merged model's poses and 3D points against dense
            # features, before the dense stage -- write to a new model dir rather than
            # overwriting merge_dir in place, for the same debugging reason as KA above.
            dense_input_dir = merge_dir
            if cfg.ba.apply:
                ba_model_dir = ws / "sparse_ba"
                print("  running bundle adjustment ...")
                refiner.refine_reconstruction(
                    output_path=ba_model_dir,
                    input_path=merge_dir,
                    image_dir=images_dir,
                    cache_path=cache_path,
                )
                dense_input_dir = ba_model_dir

            dense_dir = ws / "dense"
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", str(images_dir),
                "--input_path", str(dense_input_dir),
                "--output_path", str(dense_dir),
            ], check=True)
            # geom_consistency_regularizer/filter_min_triangulation_angle/
            # filter_geom_consistency_max_cost/num_threads postdate COLMAP 3.8's
            # PatchMatchStereo -- confirmed against its own option_manager.cc, not guessed.
            # StereoFusion below needs no such skip: every StereoFusionConfig field already
            # matches a 3.8-era StereoFusion.* option.
            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", str(dense_dir),
                *_flags("PatchMatchStereo", cfg.patch_match, skip=frozenset({
                    "geom_consistency_regularizer", "filter_min_triangulation_angle",
                    "filter_geom_consistency_max_cost", "num_threads",
                })),
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
                raise SystemExit("COLMAP registered no frames at all in the final dense model "
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
    cli_main(PixelPerfectSfM, description=__doc__)
