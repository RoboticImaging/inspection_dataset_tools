import shutil
from pathlib import Path

import numpy as np
import pytest

from inspection_viz import AssetDataset, Layout, Variant
from tests.conftest import make_asset, make_predictions

GLASS_HOUSE = Path("/home/ngoncharov/repos/inspection_datset_visualizer/glass_house")
V = Variant("trajA", "sunny", "low")


def test_discovers_variants_and_frames(asset):
    ds = AssetDataset(asset)
    assert ds.trajectories == ["trajA"]
    assert ds.illuminations == ["night", "sunny"]
    assert ds.soils == ["high", "low"]
    assert len(ds.variants) == 4
    seq = ds.sequence(V)
    assert len(seq) == 3
    assert [f.index for f in seq] == [0, 1, 2]
    for f in seq:
        assert f.rgb().shape == (6, 8, 3)
        assert f.depth().shape == (6, 8)
        assert f.camera() is not None


def test_png_instead_of_jpg(tmp_path):
    ds = AssetDataset(make_asset(tmp_path / "a", rgb_ext=".png"))
    assert len(ds.sequence(V)) == 3


def test_zero_padding_changed(tmp_path):
    # rgb padded to 2 digits, depth to 5, cams to 8 — frames must still align by integer id
    ds = AssetDataset(make_asset(tmp_path / "a", rgb_pad=2, depth_pad=5, cam_pad=8))
    seq = ds.sequence(V)
    assert len(seq) == 3
    assert all(f.depth_path is not None and f.cam_path is not None for f in seq)


def test_prefixed_filenames(tmp_path):
    ds = AssetDataset(make_asset(tmp_path / "a", stem_prefix="frame_v2_"))
    seq = ds.sequence(V)
    assert [f.index for f in seq] == [0, 1, 2]
    assert all(f.depth_path is not None for f in seq)


def test_restructured_layout(tmp_path):
    # the data guy rearranged everything: only the Layout templates need editing
    layout = Layout(
        rgb_dir="renders/{traj}/{illumination}/soil_{soil}",
        depth_dir="geometry/{traj}/depth",
        cam_dir="geometry/{traj}/cameras",
    )
    ds = AssetDataset(make_asset(tmp_path / "a", layout=layout), layout=layout)
    assert ds.illuminations == ["night", "sunny"]
    seq = ds.sequence(V)
    assert len(seq) == 3
    assert all(f.depth_path is not None and f.cam_path is not None for f in seq)


def test_no_soil_axis(tmp_path):
    layout = Layout(rgb_dir="{illumination}/traj={traj}")
    ds = AssetDataset(make_asset(tmp_path / "a", layout=layout, soils=("x",)), layout=layout)
    assert ds.soils == []
    assert len(ds.sequence(Variant("trajA", "sunny"))) == 3


def test_missing_depth_and_cam_tolerated(asset):
    (asset / "depths/traj=trajA/00000001.exr").unlink()
    shutil.rmtree(asset / "cam_params")
    seq = AssetDataset(asset).sequence(V)
    assert len(seq) == 3
    assert seq[1].depth() is None and seq[0].depth() is not None
    assert all(f.camera() is None for f in seq)


def test_junk_files_ignored(asset):
    (asset / "sunny,junk").mkdir()
    (asset / "notes.txt").write_text("hi")
    (asset / "sunny/soil=low,traj=trajA/.DS_Store").write_text("")
    (asset / "sunny/soil=low,traj=trajA/thumbs.db").write_text("")
    (asset / "depths/traj=trajA/preview.txt").write_text("")
    ds = AssetDataset(asset)
    assert len(ds.variants) == 4
    assert len(ds.sequence(V)) == 3


def test_sparse_variants(tmp_path):
    root = make_asset(tmp_path / "a")
    shutil.rmtree(root / "night/soil=high,traj=trajA")
    ds = AssetDataset(root)
    assert len(ds.variants) == 3
    missing = Variant("trajA", "night", "high")
    assert not ds.has(missing)
    with pytest.raises(KeyError):
        ds.sequence(missing)


def test_camera_intrinsics_and_pose(asset):
    cam = AssetDataset(asset).sequence(V)[0].camera()
    assert np.isclose(cam.K[0, 0], 8 * 50.0 / 36.0)
    assert np.allclose(cam.K[:2, 2], [4.0, 3.0])
    # identity blender rot at (0,0,5): looks straight down in OpenCV convention
    assert np.allclose(cam.c2w[:3, :3], np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(cam.position, [0, 0, 5])
    assert cam.far == 1000.0


def test_unproject_plane(asset):
    frame = AssetDataset(asset).sequence(V)[0]
    cam, depth = frame.camera(), frame.depth()
    pts, colors = cam.unproject(depth, rgb=frame.rgb(), stride=2)
    assert pts.shape == (3 * 4, 3) and colors.shape == pts.shape
    assert np.allclose(pts[:, 2], 0.0, atol=1e-5)  # depth 5 from z=5 looking down -> ground plane
    far = depth.copy()
    far[0, :] = 2000.0  # beyond clip_end
    pts_far, _ = cam.unproject(far, stride=2)
    assert len(pts_far) < len(pts)


def test_mesh_discovery(tmp_path):
    ds = AssetDataset(make_asset(tmp_path / "house", mesh_kinds=("accuracy", "completeness")))
    assert set(ds.meshes) == {"accuracy", "completeness"}
    assert all(p.is_file() for p in ds.meshes.values())
    assert AssetDataset(make_asset(tmp_path / "bare")).meshes == {}


def test_mesh_custom_template(tmp_path):
    layout = Layout(mesh_file="gt_{kind}.glb")
    root = tmp_path / "a"
    make_asset(root, layout=layout)
    (root / "gt_full.glb").write_bytes(b"glTF")
    assert set(AssetDataset(root, layout=layout).meshes) == {"full"}


def test_depth_scale(tmp_path):
    layout = Layout(depth_scale=0.01)
    ds = AssetDataset(make_asset(tmp_path / "a"), layout=layout)
    assert np.allclose(ds.sequence(V)[0].depth(), 0.05)


def test_no_methods_without_predictions(asset):
    ds = AssetDataset(asset)
    assert ds.methods == []
    with pytest.raises(KeyError):
        ds.sequence(V, source="da3")


def test_predictions_discovered_and_selectable(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "a", layout=layout)
    make_predictions(root, layout, "da3", V.as_dict())
    ds = AssetDataset(root, layout=layout)
    assert ds.methods == ["da3"]

    seq = ds.sequence(V, source="da3")
    assert len(seq) == 3
    for f in seq:
        assert f.rgb().shape == (6, 8, 3)   # rgb still comes from gt
        assert f.depth() is not None
        assert f.camera() is not None

    # a variant with no gt data still 404s the same way regardless of source
    with pytest.raises(KeyError):
        ds.sequence(Variant("nope"), source="da3")
    # predictions are keyed by traj only (mirroring depth_dir), so they apply across
    # illumination/soil the same way gt depth does...
    assert len(ds.sequence(Variant("trajA", "night", "low"), source="da3")) == 3
    # ...but an unknown method name 404s
    with pytest.raises(KeyError):
        ds.sequence(V, source="nope")


def test_predictions_dont_leak_into_gt(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "a", layout=layout)
    make_predictions(root, layout, "da3", V.as_dict())
    ds = AssetDataset(root, layout=layout)
    assert len(ds.sequence(V)) == 3  # gt sequence unaffected by predictions/ existing


@pytest.mark.skipif(not GLASS_HOUSE.is_dir(), reason="real dataset not available")
def test_real_glass_house_matches_colmap_gt():
    ds = AssetDataset(GLASS_HOUSE)
    assert "trendspek" in ds.trajectories
    assert set(ds.soils) == {"high", "low", "medium", "verylow"}
    assert set(ds.meshes) == {"accuracy", "completeness"}
    seq = ds.sequence(Variant("trendspek", ds.illuminations[0], "low"))
    assert len(seq) == 96
    cam = seq[0].camera()

    lines = [l for l in (GLASS_HOUSE / "colmap_style_gt/traj=trendspek/images.txt")
             .read_text().splitlines() if l and not l.startswith("#")]
    qw, qx, qy, qz, tx, ty, tz = map(float, lines[0].split()[1:8])
    from inspection_viz.camera import quat_to_rot
    R_w2c = quat_to_rot([qw, qx, qy, qz])
    assert np.allclose(cam.c2w[:3, :3], R_w2c.T, atol=1e-4)
    assert np.allclose(cam.position, -R_w2c.T @ [tx, ty, tz], atol=1e-3)

    cam_line = [l for l in (GLASS_HOUSE / "colmap_style_gt/traj=trendspek/cameras.txt")
                .read_text().splitlines() if not l.startswith("#")][0].split()
    assert np.isclose(cam.K[0, 0], float(cam_line[4]))
    assert (cam.width, cam.height) == (int(cam_line[2]), int(cam_line[3]))
