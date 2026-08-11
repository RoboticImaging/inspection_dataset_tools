"""Dataset layout config. If the folder structure changes, edit (or pass) a Layout."""

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
    mesh_file: str = "{name}_{kind}.glb"  # gt meshes at the asset root, keyed by {kind}
    rgb_exts: tuple = (".jpg", ".jpeg", ".png")
    depth_exts: tuple = (".exr",)
    cam_exts: tuple = (".json",)
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
