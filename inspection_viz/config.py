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
    mesh_file: str = "{name}_{kind}.glb"  # gt meshes at the asset root, keyed by {kind}
    rgb_exts: tuple = (".jpg", ".jpeg", ".png")
    depth_exts: tuple = (".exr",)
    cam_exts: tuple = (".json",)
    depth_scale: float = 1.0  # multiplier to convert stored depth to meters
    frame_id: str = r"(\d+)"  # last match in the file stem is the frame index


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
