from pathlib import Path

import numpy as np
import pytest
import trimesh

from inspection_viz.camera import Camera
from inspection_viz.config import Layout
from inspection_viz.dataset import Variant
from inspection_viz import io
from baselines.common.writer import (
    _atomic_write, _write_verified, cloud_path, frame_stem, output_dirs, write_cam, write_cloud,
    write_depth,
)


def test_output_dirs_match_layout_templates(tmp_path):
    layout = Layout()
    v = Variant("trajA", "sunny", soil="low").as_dict()
    depth_dir, cam_dir = output_dirs(tmp_path, layout, "da3", v)
    assert depth_dir == tmp_path / "predictions/da3/sunny/depths/soil=low,traj=trajA"
    assert cam_dir == tmp_path / "predictions/da3/sunny/cam_params/soil=low,traj=trajA"
    assert cloud_path(tmp_path, layout, "da3", v) == \
        tmp_path / "predictions/da3/sunny/clouds/soil=low,traj=trajA.ply"


def test_output_dirs_pick_the_reflectivity_alternative(tmp_path):
    layout = Layout()
    v = Variant("proxy_surface_0.0", "overcast", reflectivity="0.0").as_dict()
    depth_dir, _ = output_dirs(tmp_path, layout, "da3", v)
    assert depth_dir == tmp_path / "predictions/da3/overcast/depths/ref=0.0,traj=proxy_surface_0.0"
    assert cloud_path(tmp_path, layout, "da3", v) == \
        tmp_path / "predictions/da3/overcast/clouds/ref=0.0,traj=proxy_surface_0.0.ply"


def test_output_dirs_reject_an_underspecified_variant(tmp_path):
    # neither alternative is satisfiable without soil or reflectivity -- better to fail loudly
    # than to silently drop the axis and let two runs share a dir
    with pytest.raises(KeyError):
        output_dirs(tmp_path, Layout(), "da3", Variant("trajA", "sunny").as_dict())


def test_cloud_round_trips(tmp_path):
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = tmp_path / "clouds" / "soil=low,traj=trajA.ply"
    write_cloud(path, pts, cols)
    got = trimesh.load(path)
    assert np.allclose(np.asarray(got.vertices), pts, atol=1e-6)
    assert np.array_equal(np.asarray(got.colors)[:, :3], cols)


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


# --- atomic / verified writes (a process killed mid-write, or a storage hiccup, must never ------
# --- leave a truncated file at the real path -- see baselines.common.writer._atomic_write) ------


def test_atomic_write_leaves_the_old_file_untouched_if_write_fn_raises(tmp_path):
    path = tmp_path / "out.bin"
    path.write_bytes(b"old contents")

    def bad_write(tmp):
        tmp.write_bytes(b"partial")
        raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError):
        _atomic_write(path, bad_write)

    assert path.read_bytes() == b"old contents"  # never replaced with the partial write
    assert list(path.parent.iterdir()) == [path]  # no stray temp file left behind either


def test_write_verified_retries_until_verify_passes(tmp_path):
    path = tmp_path / "out.bin"
    verify_calls = []

    def write_fn(tmp):
        tmp.write_bytes(b"data")

    def verify_fn(written):
        verify_calls.append(written)
        if len(verify_calls) < 2:
            raise ValueError("not yet")

    _write_verified(path, write_fn, verify_fn, attempts=3)

    assert len(verify_calls) == 2
    assert path.read_bytes() == b"data"


def test_write_verified_gives_up_and_cleans_up_after_exhausting_attempts(tmp_path):
    path = tmp_path / "out.bin"
    write_calls = []

    def write_fn(tmp):
        write_calls.append(tmp)
        tmp.write_bytes(b"data")

    def always_bad_verify(written):
        raise ValueError("never valid")

    with pytest.raises(IOError, match="didn't read back valid"):
        _write_verified(path, write_fn, always_bad_verify, attempts=3)

    assert len(write_calls) == 3  # tried the full write, not just a re-check, each attempt
    assert not path.exists()  # a known-bad file must not be left looking like valid output


def test_write_depth_retries_after_a_verification_failure(tmp_path, monkeypatch):
    real_read_depth = io.read_depth
    calls = []

    def flaky_read_depth(path):
        calls.append(path)
        if len(calls) == 1:
            raise IOError("simulated corrupt read")
        return real_read_depth(path)

    monkeypatch.setattr(io, "read_depth", flaky_read_depth)

    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "depths" / "00000000.exr"
    write_depth(path, depth, attempts=3)

    assert len(calls) == 2  # failed verifying once, then wrote+verified again and it stuck
    assert np.allclose(io.read_depth(path), depth)


def test_write_depth_raises_and_cleans_up_after_persistent_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "read_depth", lambda path: (_ for _ in ()).throw(IOError("corrupt")))

    depth = np.zeros((2, 2), dtype=np.float32)
    path = tmp_path / "depths" / "00000000.exr"
    with pytest.raises(IOError, match="didn't read back valid"):
        write_depth(path, depth, attempts=3)

    assert not path.exists()


def test_write_depth_verify_false_skips_the_readback(tmp_path):
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "depths" / "00000000.exr"
    write_depth(path, depth, verify=False)  # must not raise, and must still write a valid file
    assert np.allclose(io.read_depth(path), depth)
