"""Dataset layout config. If the folder structure changes, edit (or pass) a Layout."""

# Postponed annotation evaluation (PEP 563): lets `str | tuple[str, ...]` etc. below run on
# Python < 3.10 (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu 20.04's
# default Python 3.8) without needing `typing.Optional`/`typing.Union` -- annotations are stored
# as strings instead of evaluated at class-definition time, so `type.__or__` (added in 3.10) is
# never actually called.
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    # Dir templates relative to the asset root. Placeholders: {traj} {illumination} {soil} {reflectivity}.
    # Soil and reflectivity are varied one at a time (never both), so each gets its own template;
    # every template is tried and the resulting dirs are merged.
    rgb_dir: tuple = (
        "{illumination}/soil={soil},traj={traj}",
        "{illumination}/ref={reflectivity},traj={traj}",
    )
    depth_dir: str = "depths/traj={traj}"
    cam_dir: str = "cam_params/traj={traj}"
    # Evaluation-region masks, beside depths/ with the same structure and frame numbering. Pixel
    # 0 = ignore, anything else = inside the region. window_masks only exist for scenes that have
    # windows (the dir is simply absent otherwise).
    mask_dir: str = "masks/traj={traj}"
    window_mask_dir: str = "window_masks/traj={traj}"
    # Baseline predictions live alongside gt data, keyed by {method} (e.g. "da3") *and* the full
    # variant. Note these mirror rgb_dir's placeholders, not depth_dir's: gt depth is a render of
    # fixed scene geometry, so one gt sequence serves every illumination/soil of a trajectory --
    # but a baseline infers depth *from rgb*, so a different illumination or soil level is a
    # genuinely different prediction and needs a dir of its own (otherwise runs clobber).
    # Same frame-id / extension conventions as depth_dir/cam_dir, so they index the same way.
    pred_depth_dir: tuple = (
        "predictions/{method}/{illumination}/depths/soil={soil},traj={traj}",
        "predictions/{method}/{illumination}/depths/ref={reflectivity},traj={traj}",
    )
    pred_cam_dir: tuple = (
        "predictions/{method}/{illumination}/cam_params/soil={soil},traj={traj}",
        "predictions/{method}/{illumination}/cam_params/ref={reflectivity},traj={traj}",
    )
    # One fused world-space point cloud per prediction. Alone among these templates the leaf is a
    # *file*, not a dir -- there's exactly one cloud per variant, not one per frame.
    pred_cloud_file: tuple = (
        "predictions/{method}/{illumination}/clouds/soil={soil},traj={traj}.ply",
        "predictions/{method}/{illumination}/clouds/ref={reflectivity},traj={traj}.ply",
    )
    # The config a baseline ran with, written once per method (see baselines/common/runner.py).
    # Not a variant template: it records the whole run, every variant of it included.
    pred_config_file: str = "predictions/{method}/config.yaml"
    # Marker left when a baseline hits a GPU OOM predicting this variant -- same axes as
    # pred_cloud_file (a leaf file, not a dir), so a later scan can tell "never run" apart from
    # "ran and ran out of memory" without re-running anything. Contains the exception text.
    pred_oom_file: tuple = (
        "predictions/{method}/{illumination}/oom/soil={soil},traj={traj}.txt",
        "predictions/{method}/{illumination}/oom/ref={reflectivity},traj={traj}.txt",
    )
    # A colmap run's folder for one variant, one {mode} (e.g. "no_gt") at a time -- same
    # illumination/soil/reflectivity axes as pred_depth_dir and for the same reason: colmap runs
    # SfM from rgb, so a different illumination or soil level is a genuinely different
    # reconstruction. Poses, depth and the fused cloud all come from *inside* this one folder
    # (see inspection_viz.colmap), unlike a regular method's separate depth/cam/cloud templates.
    colmap_dir: tuple = (
        "colmap/{mode}/{illumination}/soil={soil},traj={traj}",
        "colmap/{mode}/{illumination}/ref={reflectivity},traj={traj}",
    )
    mesh_file: str = "{name}_{kind}.glb"  # gt meshes at the asset root, keyed by {kind}
    rgb_exts: tuple = (".jpg", ".jpeg", ".png")
    depth_exts: tuple = (".exr",)
    cam_exts: tuple = (".json",)
    mask_exts: tuple = (".png", ".npy", ".tif", ".tiff")
    depth_scale: float = 1.0  # multiplier to convert stored depth to meters
    frame_id: str = r"(\d+)"  # last match in the file stem is the frame index


def fill_template(template: str | tuple[str, ...], **binds) -> str:
    """Render a dir/file template with `binds`, the inverse of matching it with `segment_patterns`.

    A tuple is a set of mutually exclusive alternatives (soil-varying vs. reflectivity-varying);
    the first one whose every placeholder is bound to a non-None value wins.
    """
    alts = (template,) if isinstance(template, str) else template
    for t in alts:
        if all(binds.get(n) is not None for n in re.findall(r"\{(\w+)\}", t)):
            return t.format(**binds)
    bound = sorted(k for k, v in binds.items() if v is not None)
    raise KeyError(f"no alternative of {template!r} is satisfied by bindings {bound}")


def segment_patterns(template: str) -> list[re.Pattern]:
    """One regex per path segment; {name} becomes a named group."""
    pats = []
    for seg in template.split("/"):
        parts = re.split(r"\{(\w+)\}", seg)
        rx = "".join(
            re.escape(p) if i % 2 == 0 else f"(?P<{p}>[^/,=]+)"
            for i, p in enumerate(parts)
        )
        pats.append(re.compile(f"^{rx}$"))
    return pats
