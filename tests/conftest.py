import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import pytest
import trimesh

from inspection_viz import Layout
from inspection_viz.config import fill_template

WH = (8, 6)


def write_rgb(path: Path, wh=WH):
    img = np.random.default_rng(0).integers(0, 255, (wh[1], wh[0], 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def write_depth_exr(path: Path, value=5.0, wh=WH):
    """`value` is a constant depth, or a (h, w) map when a test needs holes in it."""
    d = np.ascontiguousarray(np.broadcast_to(np.asarray(value, np.float32), (wh[1], wh[0])))
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage},
                 {"Z": d}).write(str(path))


def write_cloud_ply(path: Path, n=64):
    """A small colored world-space point cloud, as a baseline's fused .ply."""
    rng = np.random.default_rng(0)
    cloud = trimesh.PointCloud(rng.normal(size=(n, 3)).astype(np.float32),
                               colors=rng.integers(0, 255, (n, 3), dtype=np.uint8))
    path.write_bytes(cloud.export(file_type="ply"))


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


def make_predictions(root: Path, layout: Layout, method: str, binds: dict,
                     n_frames=3, pad=8, cloud=False):
    """Write synthetic predictions/{method}/... files for one variant, following the layout."""
    kw = {"method": method, **binds}
    depth_dir = root / fill_template(layout.pred_depth_dir, **kw)
    cam_dir = root / fill_template(layout.pred_cam_dir, **kw)
    depth_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        write_depth_exr(depth_dir / f"{i:0{pad}d}.exr", value=1.0)
        write_cam(cam_dir / f"{i:0{pad}d}.json")
    if cloud:
        ply = root / fill_template(layout.pred_cloud_file, **kw)
        ply.parent.mkdir(parents=True, exist_ok=True)
        write_cloud_ply(ply)


def write_array(path: Path, arr: np.ndarray):
    """Inverse of `inspection_viz.colmap.read_array`, for building depth-map fixtures in the
    same binary format colmap's dense stereo writes."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    with open(path, "wb") as f:
        f.write(f"{w}&{h}&{c}&".encode("ascii"))
        f.write(np.asfortranarray(np.transpose(arr, (1, 0, 2))).tobytes(order="F"))


def make_colmap(root: Path, layout: Layout, mode: str, binds: dict, n_frames=3, wh=WH,
                depth_value=5.0, cloud=True, unregistered=()) -> Path:
    """Write a synthetic colmap `<mode>` run for one variant: a real (tiny) pycolmap sparse
    model under dense/sparse, a depth .bin per registered frame, and a fused .ply -- the same
    shape `inspection_viz.colmap.Reconstruction` reads out of a real colmap run. `unregistered`
    drops those frame indices from the model, the way colmap itself drops frames it couldn't
    register."""
    import pycolmap

    dense = root / fill_template(layout.colmap_dir, mode=mode, **binds) / "dense"
    depth_dir = dense / "stereo" / "depth_maps"
    depth_dir.mkdir(parents=True, exist_ok=True)

    w, h = wh
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(
        pycolmap.Camera.create_from_model_id(1, pycolmap.CameraModelId.SIMPLE_PINHOLE, 50.0, w, h))
    for i in range(n_frames):
        if i in unregistered:
            continue
        name = f"{i:08d}.jpg"
        image = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(image)
        w2c = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, 5.0]))
        rec.image(image.image_id).frame.set_cam_from_world(1, w2c)
        rec.register_frame(rec.image(image.image_id).frame_id)
        write_array(depth_dir / f"{name}.photometric.bin", np.full((h, w), depth_value, np.float32))
    (dense / "sparse").mkdir(parents=True, exist_ok=True)
    rec.write(dense / "sparse")
    if cloud:
        write_cloud_ply(dense / "fused.ply")
    return dense.parent


@pytest.fixture
def asset(tmp_path):
    return make_asset(tmp_path / "asset")
