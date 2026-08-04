import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import pytest

from inspection_viz import Layout

WH = (8, 6)


def write_rgb(path: Path, wh=WH):
    img = np.random.default_rng(0).integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def write_depth_exr(path: Path, value=5.0, wh=WH):
    d = np.full((wh[1], wh[0]), value, np.float32)
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {"Z": d}).write(str(path))


def write_cam(path: Path, loc=(0.0, 0.0, 5.0), rot=(1.0, 0.0, 0.0, 0.0)):
    path.write_text(json.dumps({
        "loc": list(loc), "rot": list(rot),
        "focal_length": 50.0, "sensor_width": 36.0,
        "clip_start": 0.1, "clip_end": 1000.0,
    }))


def make_asset(root: Path, layout: Layout = Layout(),
               trajs=("trajA",), illums=("sunny", "night"), soils=("low", "high"),
               n_frames=3, rgb_pad=8, depth_pad=8, cam_pad=8,
               rgb_ext=".jpg", stem_prefix="", mesh_kinds=()) -> Path:
    """Build a synthetic asset tree following the given layout's dir templates."""
    def stem(i, pad):
        return f"{stem_prefix}{i:0{pad}d}"

    root.mkdir(parents=True, exist_ok=True)
    for kind in mesh_kinds:
        (root / layout.mesh_file.format(name=root.name, kind=kind)).write_bytes(b"glTF")

    rgb_template = layout.rgb_dir[0] if isinstance(layout.rgb_dir, tuple) else layout.rgb_dir
    for traj, illum, soil in itertools.product(trajs, illums, soils):
        binds = {"traj": traj, "illumination": illum, "soil": soil}
        rgb_dir = root / rgb_template.format(**binds)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            write_rgb(rgb_dir / (stem(i, rgb_pad) + rgb_ext))
    for traj in trajs:
        binds = {"traj": traj, "illumination": illums[0], "soil": soils[0]}
        depth_dir = root / layout.depth_dir.format(**binds)
        cam_dir = root / layout.cam_dir.format(**binds)
        depth_dir.mkdir(parents=True, exist_ok=True)
        cam_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            write_depth_exr(depth_dir / (stem(i, depth_pad) + ".exr"))
            write_cam(cam_dir / (stem(i, cam_pad) + ".json"))
    return root


def make_predictions(root: Path, layout: Layout, method: str, binds: dict, n_frames=3, pad=8):
    """Write synthetic predictions/{method}/... files for one variant, following the layout."""
    kw = {"method": method, **binds}
    depth_dir = root / layout.pred_depth_dir.format(**kw)
    cam_dir = root / layout.pred_cam_dir.format(**kw)
    depth_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        write_depth_exr(depth_dir / f"{i:0{pad}d}.exr", value=1.0)
        write_cam(cam_dir / f"{i:0{pad}d}.json")


@pytest.fixture
def asset(tmp_path):
    return make_asset(tmp_path / "asset")
