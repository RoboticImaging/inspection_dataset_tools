"""Point-cloud helpers: gt clouds sampled off the gt meshes, and nearest-neighbour distances.

The ground truth for surface metrics is a mesh (`{asset}_{kind}.glb`), so it has to be turned
into points before it can be compared with a predicted cloud. Sampling is area-weighted and
seeded, and the result is cached on disk keyed by (mesh, size, mtime, n): the accuracy meshes run
to hundreds of MB and loading one costs far more than every metric in this repo put together.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh

CACHE_DIR = Path("~/.cache/inspection_eval").expanduser()
# glTF is Y-up, the dataset's world (and its cam jsons) is Z-up: +90 deg about x, matching the
# viewer's GLTF_TO_Z_UP.
GLTF_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def sample_mesh(path: Path, n_points: int, cache_dir: Path = CACHE_DIR,
                seed: int = 0) -> np.ndarray:
    """(n_points, 3) world-space points sampled uniformly over the surface of a gt mesh."""
    st = path.stat()
    cache = cache_dir / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}-n{n_points}-s{seed}.npy"
    if cache.is_file():
        return np.load(cache)

    mesh = trimesh.load(path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    pts = finite(np.asarray(pts, dtype=np.float32) @ GLTF_TO_Z_UP.T)
    if len(pts) == 0:
        raise ValueError(f"{path} sampled to no finite point")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Every variant of the same asset shares this exact cache path (same mesh, same default
    # seed), so with --workers > 1 two processes racing to sample it for the first time is the
    # common case, not an edge one. Written to a temp file in the same directory, then swapped
    # into place with os.replace -- atomic on POSIX for a same-filesystem rename -- so a
    # concurrent reader always sees either the old (missing) state or the fully-written file,
    # never a torn np.save() in progress.
    fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix=f"{cache.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, pts)
        os.replace(tmp_name, cache)
    except BaseException:
        os.unlink(tmp_name)
        raise
    return pts


def finite(points: np.ndarray) -> np.ndarray:
    """The rows of an (N,3) cloud that are finite in every coordinate.

    A fused .ply is free to carry nan or inf vertices — a KD-tree built over them is not, and a
    single one of them turns every distance statistic downstream into nan.
    """
    pts = np.asarray(points)
    if pts.size == 0:
        return pts
    return pts[np.all(np.isfinite(pts), axis=1)]


def nearest_distances(query: np.ndarray, reference: np.ndarray,
                      chunk: int = 200_000) -> np.ndarray:
    """Distance from every point of `query` to the closest point of `reference`."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(reference, dtype=np.float64))
    out = np.empty(len(query), dtype=np.float64)
    for i in range(0, len(query), chunk):
        out[i:i + chunk], _ = tree.query(np.asarray(query[i:i + chunk], dtype=np.float64),
                                         workers=-1)
    return out


def bounds(points: np.ndarray, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of `points`, grown by `margin` * the bbox diagonal on every side."""
    points = finite(points)
    if len(points) == 0:
        raise ValueError("cannot take the bounds of a cloud with no finite point")
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = margin * float(np.linalg.norm(hi - lo))
    return lo - pad, hi + pad


def crop(points: np.ndarray, lo, hi) -> np.ndarray:
    keep = np.all((points >= np.asarray(lo)) & (points <= np.asarray(hi)), axis=1)
    return points[keep]


def subsample(points: np.ndarray, max_points: int | None, seed: int = 0) -> np.ndarray:
    """Cap a cloud's size by taking a random subset (KD-tree queries scale with both sizes)."""
    if max_points is None or len(points) <= max_points:
        return points
    idx = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
    return points[np.sort(idx)]
