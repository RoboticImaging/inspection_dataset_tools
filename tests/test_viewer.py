import socket
import time

import numpy as np
import pytest
import trimesh

from inspection_viz import Layout, Viewer
from inspection_viz.config import fill_template
from tests.conftest import make_asset, make_predictions, write_cam


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


def _asset_with_a_rescaled_prediction(root):
    """gt steps along +x at unit spacing; da3's own poses are the same trajectory at 1/3 scale,
    shifted -- so a correct fit recovers scale=3 and lines every frame back up with gt."""
    layout = Layout()
    binds = {"traj": "trajA", "illumination": "sunny", "soil": "low"}
    a = make_asset(root, trajs=("trajA",), illums=("sunny",), soils=("low",), n_frames=4)
    cam_dir = a / layout.cam_dir.format(**binds)
    for i in range(4):
        write_cam(cam_dir / f"{i:08d}.json", loc=(float(i), 0.0, 5.0))
    make_predictions(a, layout, "da3", binds, n_frames=4)
    pred_cam_dir = a / fill_template(layout.pred_cam_dir, method="da3", **binds)
    for i in range(4):
        write_cam(pred_cam_dir / f"{i:08d}.json", loc=(i / 3.0 + 10.0, 0.0, 5.0))
    return a


def test_align_to_gt_fits_a_prediction_into_the_gt_gauge(tmp_path, viewer):
    viewer.load_asset(_asset_with_a_rescaled_prediction(tmp_path / "a"))
    viewer._source.value = "da3"

    assert viewer._aligned
    assert viewer._transform.scale == pytest.approx(3.0, rel=1e-3)
    assert "aligned to gt" in viewer._status.content

    gt0 = viewer.ds.sequence(viewer._variant(), "gt")[0].camera().position
    aligned0 = viewer._transform.c2w(viewer.seq[0].camera().c2w)[:3, 3]
    assert np.allclose(aligned0, gt0, atol=1e-4)
    # the frustum on screen is built from the aligned pose, not the raw predicted one
    assert np.allclose(viewer._frustums[viewer._frame.value].position, aligned0, atol=1e-4)


def test_align_toggle_moves_geometry_without_reloading_the_sequence(tmp_path, viewer):
    viewer.load_asset(_asset_with_a_rescaled_prediction(tmp_path / "a"))
    viewer._source.value = "da3"
    seq_before = viewer.seq
    aligned_pos = viewer._frustums[viewer._frame.value].position.copy()

    viewer._align.value = False
    assert not viewer._aligned and "aligned to gt" not in viewer._status.content
    assert viewer.seq is seq_before  # no reload -- same frames, just re-placed
    raw_pos = viewer._frustums[viewer._frame.value].position.copy()
    assert not np.allclose(raw_pos, aligned_pos)
    assert np.allclose(raw_pos, viewer.seq[viewer._frame.value].camera().position)

    viewer._align.value = True
    assert viewer.seq is seq_before
    assert np.allclose(viewer._frustums[viewer._frame.value].position, aligned_pos, atol=1e-4)


def test_align_off_shows_gt_unmoved(tmp_path, viewer):
    viewer.load_asset(_asset_with_a_rescaled_prediction(tmp_path / "a"))
    assert viewer._source.value == "gt"
    assert not viewer._aligned  # nothing to align gt to
    assert "aligned to gt" not in viewer._status.content
    assert viewer._transform.scale == 1.0


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
