import numpy as np
import trimesh

from inspection_viz.mesh import load_mesh_for_vis


def test_downsample_colors_and_cache(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    sphere.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(sphere.vertices), 1))
    p = tmp_path / "asset_gt.glb"
    sphere.export(p)

    cache = tmp_path / "cache"
    m = load_mesh_for_vis(p, max_faces=300, cache_dir=cache)
    assert len(m.faces) <= 300
    colors = np.asarray(m.visual.vertex_colors)
    assert colors[:, 0].mean() > 200 and colors[:, 1].mean() < 60

    assert len(list(cache.glob("*.ply"))) == 1
    m2 = load_mesh_for_vis(p, max_faces=300, cache_dir=cache)
    assert len(m2.faces) == len(m.faces)


def test_extreme_budget_does_not_crash(tmp_path):
    scene = trimesh.Scene()
    for i in range(3):
        s = trimesh.creation.icosphere(subdivisions=3)
        s.apply_translation([i * 3, 0, 0])
        scene.add_geometry(s)
    p = tmp_path / "a_gt.glb"
    scene.export(p)
    m = load_mesh_for_vis(p, max_faces=10, cache_dir=tmp_path / "c")
    assert 0 < len(m.faces) < 3 * 1280


def test_small_mesh_not_decimated(tmp_path):
    box = trimesh.creation.box()
    p = tmp_path / "a_gt.glb"
    box.export(p)
    m = load_mesh_for_vis(p, max_faces=1000, cache_dir=tmp_path / "c")
    assert len(m.faces) == len(box.faces)
