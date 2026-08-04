from pathlib import Path

import numpy as np

from inspection_viz.camera import Camera
from inspection_viz.config import Layout
from inspection_viz.dataset import Variant
from inspection_viz import io
from baselines.common.writer import frame_stem, output_dirs, write_cam, write_depth


def test_output_dirs_match_layout_templates(tmp_path):
    layout = Layout()
    depth_dir, cam_dir = output_dirs(tmp_path, layout, "da3", Variant("trajA").as_dict())
    assert depth_dir == tmp_path / "predictions/da3/depths/traj=trajA"
    assert cam_dir == tmp_path / "predictions/da3/cam_params/traj=trajA"


def test_depth_round_trip(tmp_path):
    depth = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    path = tmp_path / "predictions/da3/depths/traj=trajA" / f"{frame_stem(3)}.exr"
    write_depth(path, depth)
    assert path.name == "00000003.exr"
    got = io.read_depth(path)
    assert np.allclose(got[:1], depth[:1])
    assert np.isnan(got[1, 1])


def test_cam_round_trips_through_camera_from_blender(tmp_path):
    # an arbitrary OpenCV camera-to-world pose + pinhole intrinsics
    width, height = 640, 480
    fx = 500.0
    K = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]])
    theta = 0.3
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    t = np.array([1.5, -2.0, 10.0])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, t

    path = tmp_path / "cam.json"
    write_cam(path, c2w, K, width=width, clip_end=100.0)
    cam = Camera.from_blender(io.read_cam(path), width, height)

    assert np.allclose(cam.c2w[:3, :3], R, atol=1e-6)
    assert np.allclose(cam.position, t, atol=1e-6)
    assert np.isclose(cam.K[0, 0], fx, atol=1e-3)
    assert cam.far == 100.0
