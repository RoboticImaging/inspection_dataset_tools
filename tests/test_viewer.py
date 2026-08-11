import socket
import time

import pytest
import trimesh

from inspection_viz import Layout, Viewer
from tests.conftest import make_asset, make_predictions


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for(pred, timeout=10.0) -> bool:
    """Poll `pred` until it holds; the viewer loads clouds and meshes off-thread."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def viewer():
    v = Viewer(port=free_port())
    yield v
    v.server.stop()


def nodes(v) -> int:
    return len(v.server.scene._handle_from_node_name)


def test_starts_empty(viewer):
    assert viewer.ds is None and viewer.seq is None
    assert viewer._path.value == ""
    assert viewer._traj.options == ("-",)


def test_load_and_switch_asset(tmp_path, viewer):
    a = make_asset(tmp_path / "a", illums=("sunny", "night"))
    b = make_asset(tmp_path / "b", illums=("dusk",), soils=("mud",), n_frames=5)
    trimesh.creation.box().export(b / "b_accuracy.glb")

    viewer.load_asset(a)
    assert viewer._path.value == str(a)
    assert viewer._illum.options == ("night", "sunny")
    assert viewer._geometry.options == ("None", "Point cloud")
    assert len(viewer.seq) == 3

    viewer.load_asset(b)
    assert viewer._illum.options == ("dusk",) and viewer._axis_value.options == ("mud",)
    assert viewer._geometry.options == ("None", "Point cloud", "mesh: accuracy")
    assert len(viewer.seq) == 5


def test_switching_does_not_accumulate(tmp_path, viewer):
    a, b = make_asset(tmp_path / "a"), make_asset(tmp_path / "b", n_frames=4)
    viewer.load_asset(a)
    viewer.load_asset(b)
    baseline = nodes(viewer)
    for _ in range(5):
        viewer.load_asset(a)
        viewer.load_asset(b)
    assert nodes(viewer) == baseline


def test_bad_path_is_reported_and_recoverable(tmp_path, viewer):
    a = make_asset(tmp_path / "a")
    viewer.load_asset(a)
    viewer.load_asset(tmp_path / "nope")
    assert viewer.ds is None and viewer.seq is None
    assert "cannot load asset" in viewer._status.content
    viewer.load_asset(a)
    assert len(viewer.seq) == 3


def _asset_with_predictions(root, cloud: bool):
    """An asset whose *default* variant (one traj/illumination/soil) has da3 predictions."""
    a = make_asset(root, illums=("sunny",), soils=("low",))
    make_predictions(a, Layout(), "da3",
                     {"traj": "trajA", "illumination": "sunny", "soil": "low"}, cloud=cloud)
    return a


def test_prediction_cloud_replaces_per_frame_depth(tmp_path, viewer):
    viewer.load_asset(_asset_with_predictions(tmp_path / "a", cloud=True))
    viewer._source.value = "da3"
    viewer._stride.value = 1
    assert "fused cloud" in viewer._status.content
    assert wait_for(lambda: viewer._fused is not None)
    assert viewer._fused.visible and len(viewer._fused.points) == 64
    assert viewer._clouds == {}  # per-frame unprojection is skipped entirely

    viewer._stride.value = 2  # thins the fused cloud (by stride², like the per-frame ones)
    assert len(viewer._fused.points) == 16

    viewer._geometry.value = "None"
    assert not viewer._fused.visible

    viewer._source.value = "gt"  # gt has no fused cloud, so it goes back to per-frame clouds
    viewer._geometry.value = "Point cloud"
    assert viewer._fused is None and viewer._clouds


def test_prediction_without_cloud_falls_back_to_depth(tmp_path, viewer):
    viewer.load_asset(_asset_with_predictions(tmp_path / "a", cloud=False))
    viewer._source.value = "da3"
    assert viewer._fused is None and viewer._cloud_file is None
    assert viewer._clouds and all(h is not None for h in viewer._clouds.values())


def test_stale_mesh_bake_is_dropped(tmp_path, viewer):
    a = make_asset(tmp_path / "a")
    trimesh.creation.box().export(a / "a_gt.glb")
    b = make_asset(tmp_path / "b")
    viewer.load_asset(a)
    gen = viewer._gen
    viewer._geometry.value = "mesh: gt"
    viewer.load_asset(b)  # bake (if still running) now belongs to a dead generation
    assert viewer._gen != gen
    assert viewer._meshes == {}
