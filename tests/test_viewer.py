import socket

import pytest
import trimesh

from inspection_viz import Viewer
from tests.conftest import make_asset


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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
    assert viewer._illum.options == ("dusk",) and viewer._soil.options == ("mud",)
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
