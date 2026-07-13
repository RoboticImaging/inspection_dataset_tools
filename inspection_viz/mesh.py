"""GT mesh loading for vis: bake colors to vertices, decimate per unique part
(instance-aware, so instanced scenes never get merged at full resolution), cache to disk."""

from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.color import uv_to_color

CACHE_DIR = Path("~/.cache/inspection_viz").expanduser()
GRAY = np.array([180, 180, 180, 255], np.uint8)


def _part_colors(g: trimesh.Trimesh) -> np.ndarray:
    vis = g.visual
    mat = getattr(vis, "material", None)
    img = getattr(mat, "baseColorTexture", None)
    uv = getattr(vis, "uv", None)
    if img is not None and uv is not None:
        c = np.asarray(uv_to_color(uv, img))
    elif isinstance(vis, trimesh.visual.ColorVisuals):
        c = np.asarray(vis.vertex_colors)
    else:
        c = np.asarray(getattr(mat, "main_color", GRAY))
    if c.ndim != 2 or len(c) != len(g.vertices):
        c = np.tile(np.resize(c, 4), (len(g.vertices), 1))
    return c.astype(np.uint8)


def _take_faces(v, f, colors, target: int):
    """Keep an even subset of faces; for disconnected-card geometry (foliage) that
    edge collapse cannot reduce further."""
    f = f[np.linspace(0, len(f) - 1, min(target, len(f))).astype(int)]
    used, inv = np.unique(f.ravel(), return_inverse=True)
    return v[used], inv.reshape(-1, 3).astype(f.dtype), colors[used]


def _decimate(v, f, colors, target: int):
    target = max(target, 16)
    if target >= len(f) or len(f) < 200:
        return v, f, colors
    import fast_simplification

    v32, f32 = v.astype(np.float32), f.astype(np.int32)
    v2, f2, collapses = fast_simplification.simplify(v32, f32, target_count=target, return_collapses=True)
    if len(f2) == 0:  # collapse went degenerate on an extreme budget
        return _take_faces(v, f, colors, target)
    _, _, mapping = fast_simplification.replay_simplification(v32, f32, collapses)
    c2 = np.full((len(v2), 4), 255, np.uint8)
    ok = (mapping >= 0) & (mapping < len(v2))
    c2[mapping[ok]] = colors[ok]
    if len(f2) > target * 1.2:
        v2, f2, c2 = _take_faces(v2, f2, c2, target)
    return v2, f2, c2


def bake(loaded, max_faces: int) -> trimesh.Trimesh:
    if not isinstance(loaded, trimesh.Scene):
        v, f, c = _decimate(loaded.vertices, loaded.faces, _part_colors(loaded), max_faces)
        return trimesh.Trimesh(v, f, vertex_colors=c, process=False)

    placements = [loaded.graph[n] for n in loaded.graph.nodes_geometry]  # (transform, geom name)
    uses = Counter(gname for _, gname in placements)
    total = sum(len(loaded.geometry[g].faces) * n for g, n in uses.items())
    scale = min(1.0, max_faces / max(total, 1))
    parts = {}
    for gname in uses:
        g = loaded.geometry[gname]
        parts[gname] = _decimate(g.vertices, g.faces, _part_colors(g), int(len(g.faces) * scale))

    V, F, C, off = [], [], [], 0
    for T, gname in placements:
        v, f, c = parts[gname]
        V.append(trimesh.transform_points(v, T))
        F.append(f + off)
        C.append(c)
        off += len(v)
    return trimesh.Trimesh(np.vstack(V), np.vstack(F), vertex_colors=np.vstack(C), process=False)


def load_mesh_for_vis(path: Path, max_faces: int = 1_000_000, cache_dir: Path = CACHE_DIR) -> trimesh.Trimesh:
    st = path.stat()
    cache = cache_dir / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}-f{max_faces}.ply"
    if cache.is_file():
        return trimesh.load(cache)
    mesh = bake(trimesh.load(path), max_faces)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(cache)
    return mesh
