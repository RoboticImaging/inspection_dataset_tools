"""Point-cloud helpers: gt clouds sampled off the gt meshes, and nearest-neighbour distances.

The ground truth for surface metrics is a mesh (`{asset}_{kind}.glb`), so it has to be turned
into points before it can be compared with a predicted cloud. Sampling is area-weighted and
seeded, and the result is cached on disk keyed by (mesh, size, mtime, n): the accuracy meshes run
to hundreds of MB and loading one costs far more than every metric in this repo put together.

Clouds are kept in float32 on the eval device — they are the largest arrays in the package by
three orders of magnitude, and `nearest_distances` is the one place where that matters.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

from .device import numpy as to_numpy
from .device import tensor

CACHE_DIR = Path("~/.cache/inspection_eval").expanduser()
POINT_DTYPE = torch.float32
PAIR_BUDGET = 1 << 24  # candidate (query, reference) pairs held at once, ~200 MB of coordinates
CELL_TARGET = 4.0  # reference points per non-empty cell the search grid is cut for
SHELL_YIELD = 0.5  # a shell that leaves more than this share of its queries unsettled has failed
BRUTE_BUDGET = 1 << 31  # pairs worth measuring outright rather than searching for (~0.3 s)
CELL_BUDGET = 1 << 22  # (query, cell) lookups held at once, whatever the cells turn out to hold
# glTF is Y-up, the dataset's world (and its cam jsons) is Z-up: +90 deg about x, matching the
# viewer's GLTF_TO_Z_UP.
GLTF_TO_Z_UP = tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def sample_mesh(path: Path, n_points: int, cache_dir: Path = CACHE_DIR,
                seed: int = 0) -> torch.Tensor:
    """(n_points, 3) world-space points sampled uniformly over the surface of a gt mesh."""
    st = path.stat()
    cache = cache_dir / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}-n{n_points}-s{seed}.npy"
    if cache.is_file():
        return tensor(np.load(cache), POINT_DTYPE)

    mesh = trimesh.load(path, force="mesh", process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    pts = finite(tensor(pts, POINT_DTYPE) @ tensor(GLTF_TO_Z_UP, POINT_DTYPE).T)
    if len(pts) == 0:
        raise ValueError(f"{path} sampled to no finite point")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, to_numpy(pts))  # the cache stays host-side .npy, and device-independent
    return pts


def finite(points) -> torch.Tensor:
    """The rows of an (N,3) cloud that are finite in every coordinate.

    A fused .ply is free to carry nan or inf vertices — a distance search over them is not, and a
    single one of them turns every distance statistic downstream into nan.
    """
    pts = tensor(points)
    if pts.numel() == 0:
        return pts
    return pts[torch.isfinite(pts).all(dim=1)]


def nearest_distances(query, reference, chunk: int = 1 << 20) -> torch.Tensor:
    """Distance from every point of `query` to the closest point of `reference`.

    The device's answer to `scipy.spatial.cKDTree`. A tree is the wrong shape for hardware that
    wants the same work done to a million points at once — descending one is a data-dependent
    walk per query — so the reference is bucketed into a uniform grid instead (`_Grid`) and every
    query measures against the cells around its own, all of them at once, in widening shells
    until the answer is provably in hand. Comparing every pair is simpler still and is what the
    small-input path does, but it is quadratic: on the default million-point clouds it is two
    orders of magnitude slower than the tree this replaced, where the grid is level with it.

    The result is exact, not an approximation. What it is not is uniformly fast: where the two
    clouds overlap it matches the tree, but a query with nothing near it — gt surface a method
    never reconstructed, so its neighbour is cells and cells away — settles no shell, and those
    queries end up measured against the whole reference (`_brute_squared`). A prediction that
    covers little of the asset can therefore take tens of seconds on the completeness direction
    where the tree took one. Fewer points (`cloud.gt_points`, `cloud.max_points`) is the knob.
    """
    q = tensor(query, POINT_DTYPE)
    r = tensor(reference, POINT_DTYPE)
    if len(r) == 0:
        raise ValueError("cannot take distances to a cloud with no point")

    out = torch.empty(len(q), dtype=POINT_DTYPE, device=q.device)
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = _nearest_squared(q[i:i + chunk], r)
    return out.clamp_min(0).sqrt()  # rounding can put an exact match a hair below zero


def _nearest_squared(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Squared distance from each of `q` to the nearest of `r`, one shell of cells at a time."""
    if len(q) * len(r) <= PAIR_BUDGET:
        return _brute_squared(q, r)  # too little to pay for bucketing

    grid = _Grid.build(r, _cell_size(r))
    best = torch.full((len(q),), float("inf"), dtype=q.dtype, device=q.device)
    todo = torch.arange(len(q), device=q.device)
    ring = 0
    while True:
        best[todo] = grid.shell(q[todo], best[todo], ring)
        # Having searched every cell within `ring` of its own, a query is settled once its
        # neighbour is nearer than `ring` cells: nothing in the cells left out can beat that.
        # At ring 0 it is nothing at all — a query sits anywhere in its own cell, its edge
        # included, so a point in that cell says nothing about what the next one holds.
        left = todo[best[todo] > (ring * grid.h) ** 2]
        if left.numel() == 0:
            break
        # The grid pays for itself as long as each shell settles most of what is left. It stops
        # doing that when the queries left are in genuinely empty country — a gt surface the
        # method never reconstructed, so the nearest prediction is cells and cells away — and
        # then wider shells are just wider passes over nothing. Measuring what is left against
        # the whole reference is quadratic, but it is bounded, exact, and pair for pair quicker
        # than gathering cells, which past that point is the better bargain.
        if left.numel() * len(r) <= BRUTE_BUDGET or (ring and left.numel() > SHELL_YIELD
                                                     * todo.numel()):
            best[left] = _brute_squared(q[left], r)
            break
        todo = left
        ring += 1
    return best


def _brute_squared(q: torch.Tensor, r: torch.Tensor, budget: int = PAIR_BUDGET) -> torch.Tensor:
    """Every pair against every pair. Exact while that fits in one tile, and a matmul past it.

    The differences themselves are what a distance is, and taking them costs three loads a pair.
    The `|q|^2 - 2qr + |r|^2` expansion is a matmul instead — several times quicker — but it
    subtracts large numbers to find small ones, and in float32 the pair it picks as nearest can
    be wrong by a few tenths of a millimetre per meter the cloud sits from its own centre. Which
    is nothing against the metres of empty space that send a query down this path at all, and
    everything against a cloud sitting on the surface it is being scored against. So: the
    expansion only where the work is too big to do properly, and only to *choose* the neighbour,
    whose distance is then measured exactly.
    """
    if len(q) * len(r) <= budget:
        return ((q[:, None, :] - r[None, :, :]) ** 2).sum(dim=-1).amin(dim=1)

    origin = r.mean(dim=0)  # free, and an order of magnitude off the cancellation below
    q, r = q - origin, r - origin
    r_sq = (r * r).sum(dim=1)
    pick = torch.empty(len(q), dtype=torch.long, device=q.device)
    step = max(1, budget // len(r))
    for i in range(0, len(q), step):
        block = q[i:i + step]
        d_sq = (block * block).sum(dim=1, keepdim=True) + r_sq - 2.0 * (block @ r.T)
        pick[i:i + step] = d_sq.argmin(dim=1)
    return ((q - r[pick]) ** 2).sum(dim=1)


def _cell_size(points: torch.Tensor, target: float = CELL_TARGET, steps: int = 10) -> float:
    """A cell side leaving a handful of points in each non-empty cell: fine enough that a query
    measures against few candidates, coarse enough that it finds any at all.

    Halved towards it rather than solved for, because the occupancy of a cloud is not the
    occupancy of the box around it: these clouds are surfaces, so the cells that hold anything
    grow with the square of the resolution while the box's count grows with the cube — and a
    cloud that is neither, which every real one is somewhere, lands between the two. Counted over
    the whole cloud, for the same reason: a sample of it would answer for its own density, not
    the density the search will actually meet.
    """
    lo, hi = points.amin(dim=0), points.amax(dim=0)
    diag = float(torch.linalg.norm(hi - lo)) or 1.0  # a degenerate cloud gets any positive h
    h = diag / max((len(points) / target) ** (1 / 3), 1.0)  # what a solid cloud would want
    for _ in range(steps):
        res = _resolution(lo, hi, h)
        occupied = torch.unique(_cell_keys(_cell_coords(points, lo, h), res)).numel()
        if len(points) <= target * occupied:
            break
        h /= 2
    return h


def _resolution(lo: torch.Tensor, hi: torch.Tensor, h: float) -> torch.Tensor:
    return ((hi - lo) / h).floor().long() + 1


def _cell_coords(points: torch.Tensor, lo: torch.Tensor, h: float) -> torch.Tensor:
    """(N,3) integer cell each point falls in. Not clamped: a query outside the reference's grid
    keeps the coordinates that say so, and `_Grid.nearest` reads them as the empty cells they
    are, rather than being folded onto the border and matched against the wrong neighbours."""
    return ((points - lo) / h).floor().long()


def _cell_keys(coords: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """The (N,3) cells as one number each, so a cell is a lookup in a sorted array."""
    return (coords[:, 0] * res[1] + coords[:, 1]) * res[2] + coords[:, 2]


def _shell_offsets(ring: int, device: torch.device) -> torch.Tensor:
    """The cell offsets exactly `ring` away — the surface of the cube, not its inside, which an
    earlier pass has already searched. Built once per ring and cached: it depends on nothing
    else, and a shell of a few hundred offsets is a few hundred python steps per pass."""
    if ring not in _SHELLS:
        span = torch.arange(-ring, ring + 1)
        cube = torch.cartesian_prod(span, span, span)
        _SHELLS[ring] = cube[cube.abs().amax(dim=1) == ring]
    return _SHELLS[ring].to(device)


_SHELLS: dict[int, torch.Tensor] = {}


@dataclass(frozen=True)
class _Grid:
    """A cloud bucketed into cubes of side `h`.

    Stored the way the device wants it: the points ordered by cell, the key of every *non-empty*
    cell in a sorted array, and a cell found by binary-searching that array — so empty space
    costs nothing at all, however finely the grid is cut, which is what lets `_cell_size` cut it
    for the surface the points actually lie on instead of for the box around them.
    """

    points: torch.Tensor
    lo: torch.Tensor
    res: torch.Tensor
    h: float
    keys: torch.Tensor  # sorted, one entry per non-empty cell
    start: torch.Tensor  # where that cell's points begin in `order`
    count: torch.Tensor
    order: torch.Tensor  # point indices, grouped by cell

    @classmethod
    def build(cls, points: torch.Tensor, h: float) -> "_Grid":
        lo, hi = points.amin(dim=0), points.amax(dim=0)
        res = _resolution(lo, hi, h)
        order = _cell_keys(_cell_coords(points, lo, h), res).argsort()
        keys, count = torch.unique_consecutive(_cell_keys(_cell_coords(points[order], lo, h),
                                                          res), return_counts=True)
        return cls(points, lo, res, h, keys, count.cumsum(dim=0) - count, count, order)

    def shell(self, q: torch.Tensor, best: torch.Tensor, ring: int) -> torch.Tensor:
        """`best`, lowered wherever the cells exactly `ring` away from a query's own hold
        something nearer — the query's own cell at ring 0, the 26 around it at ring 1, and so on.

        A whole shell goes in one pass: every (query, cell offset) is looked up at once, because
        the one thing this loop must not do is ask the device a question per offset. Reading a
        single number back costs a millisecond and a half here, which is more than a shell's
        worth of arithmetic — so the sizes below are worked out from what the grid already knows,
        and the only value fetched is the length of the candidate list itself.
        """
        offsets = _shell_offsets(ring, q.device)
        step = max(1, CELL_BUDGET // len(offsets))
        for i in range(0, len(q), step):
            block = slice(i, i + step)
            cells = _cell_coords(q[block], self.lo, self.h)[:, None, :] + offsets
            self._fold(q[block], cells, best[block])
        return best

    def _fold(self, q: torch.Tensor, cells: torch.Tensor, best: torch.Tensor) -> None:
        """Measure each query against the points of its (M,K,3) cells, keeping the minimum.

        A cell outside the grid, or empty, contributes nothing but a zero to the count — which
        is what makes the whole shell one flat list of lookups. How much each of them is worth
        is another matter: a cloud with a clump in it can put a hundred thousand points in one
        cell and four in the next, so the list is cut where its running total crosses the budget
        rather than at a stride guessed from the average.
        """
        key = _cell_keys(cells.reshape(-1, 3), self.res)
        slot = torch.searchsorted(self.keys, key).clamp(max=len(self.keys) - 1)
        hit = ((cells >= 0) & (cells < self.res)).all(dim=-1).reshape(-1) & (self.keys[slot] == key)
        count = torch.where(hit, self.count[slot], 0)
        total = int(count.sum())
        if total == 0:
            return
        budget = max(PAIR_BUDGET, len(self.points))  # one cell's worth is never split
        cuts = torch.searchsorted(count.cumsum(dim=0),
                                  torch.arange(0, total, budget, device=q.device)).tolist()
        query_of = torch.arange(len(key), device=q.device) // cells.shape[1]
        for a, b in zip(cuts, cuts[1:] + [len(count)]):
            self._measure(q, count[a:b], self.start[slot][a:b], query_of[a:b], best)

    def _measure(self, q: torch.Tensor, count: torch.Tensor, start: torch.Tensor,
                 query_of: torch.Tensor, best: torch.Tensor) -> None:
        """Fold one batch of cell lookups into `best`, in place.

        The candidates are flattened into a single (query, reference) list — `count[i]` of them
        for lookup i — and reduced back onto the queries with a scatter, which is the shape of
        every variable-length gather here.
        """
        total = int(count.sum())
        if total == 0:
            return
        first = (count.cumsum(dim=0) - count).repeat_interleave(count)
        within = torch.arange(total, device=q.device) - first  # rank inside the cell
        ref = self.points[self.order[start.repeat_interleave(count) + within]]
        query_of = query_of.repeat_interleave(count)
        d_sq = ((q[query_of] - ref) ** 2).sum(dim=1)
        best.scatter_reduce_(0, query_of, d_sq, reduce="amin")


def bounds(points, margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounds of `points`, grown by `margin` * the bbox diagonal on every side."""
    points = finite(points)
    if len(points) == 0:
        raise ValueError("cannot take the bounds of a cloud with no finite point")
    lo, hi = points.amin(dim=0), points.amax(dim=0)
    pad = margin * float(torch.linalg.norm(hi - lo))
    return lo - pad, hi + pad


def crop(points, lo, hi) -> torch.Tensor:
    points = tensor(points)
    keep = ((points >= tensor(lo, points.dtype)) & (points <= tensor(hi, points.dtype))).all(dim=1)
    return points[keep]


def subsample(points, max_points: int | None, seed: int = 0) -> torch.Tensor:
    """Cap a cloud's size by taking a random subset (the distance search scales with both
    sizes)."""
    if max_points is None or len(points) <= max_points:
        return points
    generator = torch.Generator(device=points.device).manual_seed(seed)
    idx = torch.randperm(len(points), generator=generator, device=points.device)[:max_points]
    return points[idx.sort().values]
