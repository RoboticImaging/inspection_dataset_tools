# Postponed annotation evaluation (PEP 563): lets `AssetDataset | str | Path | None` etc. below
# run on Python < 3.10 (e.g. this repo's pixel-perfect-sfm-eigen-src baseline image, Ubuntu
# 20.04's default Python 3.8) without needing `typing.Optional`/`typing.Union` -- annotations are
# stored as strings instead of evaluated at class-definition time, so `type.__or__` (added in
# 3.10) is never actually called.
from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import viser

from .align import Sim3, fit_cameras
from .dataset import AssetDataset, Sequence, Variant
from .io import read_cloud
from .mesh import load_mesh_for_vis

FRUSTUM_IMG_W = 160
PANEL_IMG_W = 480
MESH_MAX_FACES = 1_000_000
GLTF_TO_Z_UP = (2**-0.5, 2**-0.5, 0.0, 0.0)  # wxyz, +90 deg about x


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    h = max(1, round(img.shape[0] * width / img.shape[1]))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def colorize_depth(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    t = np.clip((np.nan_to_num(depth) - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _finite_cams(seq: Sequence) -> dict:
    """index -> Camera, skipping frames with no pose and the odd nan one (see EvalCase's own
    `_usable_cam` -- same reasoning: a nan here would poison the Sim3 fit for every frame)."""
    out = {}
    for f in seq:
        cam = f.camera()
        if cam is not None and np.all(np.isfinite(cam.c2w)):
            out[f.index] = cam
    return out


class Viewer:
    def __init__(self, dataset: AssetDataset | str | Path | None = None,
                 port: int = 8080, mesh_max_faces: int = MESH_MAX_FACES):
        self.ds = None
        self.mesh_max_faces = mesh_max_faces
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.seq = None
        self._frustums = {}   # frame idx -> handle
        self._clouds = {}     # frame idx -> handle | None
        self._meshes = {}     # mesh kind -> handle | None (None = bake in flight)
        self._cloud_file = None    # fused .ply of the current variant+source, if the baseline wrote one
        self._transform = Sim3()   # source -> gt gauge (see _fit_transform); identity until loaded
        self._aligned = False      # whether _transform is a real fit, vs. the identity fallback
        self._fused = None         # its handle (see _ensure_fused)
        self._fused_pts = None     # its parsed (points, colors), kept across stride changes
        self._fused_pending = None  # path of an in-flight fused-cloud read
        self._panel_cache = {}  # frame idx -> (small rgb, small depth vis)
        self._depth_range = None
        self._lock = threading.RLock()
        self._counts = ""
        self._gen = 0         # bumped per asset load; stale workers drop their result
        self._suspend = False  # ignore gui callbacks while rebuilding dropdowns
        self._build_gui()
        self.load_asset(dataset) if dataset is not None else self._load_sequence()
        self.server.on_client_connect(self._frame_client)

    def _frame_client(self, client: viser.ClientHandle):
        if not self.seq:
            return
        cams = [f.camera() for f in (self.seq[0], self.seq[len(self.seq) // 2])]
        positions = [self._transform.points(c.position[None])[0] for c in cams if c]
        center = np.mean(positions, axis=0) if positions else np.zeros(3)
        client.camera.position = center + np.array([40.0, 40.0, 30.0])
        client.camera.look_at = center - np.array([0.0, 0.0, 10.0])

    # ------------------------------------------------------------------ gui
    def _build_gui(self):
        g = self.server.gui
        with g.add_folder("Asset"):
            self._path = g.add_text("Path", "", hint="asset folder, e.g. /data/glass_house")
            self._load_btn = g.add_button("Load")
        with g.add_folder("Data"):
            self._traj = g.add_dropdown("Trajectory", ["-"])
            self._illum = g.add_dropdown("Illumination", ["-"])
            # Soil and reflectivity are never both varied at once (the asset only ever
            # sweeps one at a time for a given traj+illumination), so a single "vary" axis
            # picks which one is in play, and one dropdown supplies its value.
            self._axis = g.add_button_group("Vary", ("soil",))
            self._axis_value = g.add_dropdown("Soil", ["-"])
            self._source = g.add_dropdown("Source", ["gt"])
            self._align = g.add_checkbox(
                "Align to gt", True,
                hint="fit a similarity transform (rotation + scale + translation) to whatever "
                     "camera poses this source shares with gt, so switching sources doesn't "
                     "need a re-navigate -- off shows the source exactly as it was written")
        with g.add_folder("View"):
            self._mode = g.add_dropdown("Mode", ("Timeline", "All frames"), initial_value="All frames")
            self._frame = g.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self._play = g.add_checkbox("Play", False)
            self._fps = g.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
            self._accumulate = g.add_checkbox("Accumulate", False)
            self._show_frustums = g.add_checkbox("Frustums", True)
            self._geometry = g.add_dropdown("Geometry", ("None", "Point cloud"), initial_value="Point cloud")
            self._stride = g.add_slider("Point stride", min=1, max=16, step=1, initial_value=4)
            self._frame_step = g.add_slider("Frame step (all)", min=1, max=32, step=1, initial_value=8)
            self._point_size = g.add_slider("Point size", min=0.005, max=0.2, step=0.005, initial_value=0.03)
            self._frustum_scale = g.add_slider("Frustum scale", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        self._panel = g.add_image(np.zeros((2, 2, 3), np.uint8), label="RGB")
        self._panel_depth = g.add_image(np.zeros((2, 2, 3), np.uint8), label="Depth")
        self._status = g.add_markdown("")

        self._load_btn.on_click(lambda _: self.load_asset(self._path.value))
        for h in (self._traj, self._illum):
            h.on_update(lambda _: self._suspend or self._on_axis_change())
        self._axis.on_click(lambda _: self._suspend or self._on_axis_change())
        self._axis_value.on_update(lambda _: self._suspend or self._load_sequence())
        self._source.on_update(lambda _: self._suspend or self._load_sequence())
        self._align.on_update(lambda _: self._suspend or self._realign())
        for h in (self._mode, self._show_frustums, self._geometry, self._frame_step, self._accumulate):
            h.on_update(lambda _: self._suspend or self._refresh())
        self._frame.on_update(lambda _: self._refresh())
        self._stride.on_update(lambda _: self._reset_clouds())
        self._point_size.on_update(
            lambda _: self._apply({**self._clouds, "fused": self._fused}, "point_size", self._point_size.value))
        self._frustum_scale.on_update(lambda _: self._apply(self._frustums, "scale", self._frustum_scale.value))

    def _variant(self) -> Variant:
        pick = lambda h: None if h.value == "-" else h.value
        axis, val = self._axis.value, pick(self._axis_value)
        return Variant(pick(self._traj), pick(self._illum),
                       soil=val if axis == "soil" else None,
                       reflectivity=val if axis == "reflectivity" else None)

    def _on_axis_change(self):
        """Traj/illumination/vary-toggle changed: re-derive the axis dropdowns, then reload."""
        self._suspend = True
        try:
            self._sync_axes()
        finally:
            self._suspend = False
        self._load_sequence()

    def _sync_axes(self):
        """Refresh the Vary toggle and its value dropdown for the current traj+illumination.

        Soil and reflectivity are only ever swept one at a time for a given traj+illumination
        (never both), so at most one is "on offer" at once — this discovers which, exposes the
        toggle only when both exist, and repopulates the value list to match.
        """
        pick = lambda h: None if h.value == "-" else h.value
        axes = self.ds.axes(pick(self._traj), pick(self._illum)) if self.ds else {}
        keys = list(axes) or ["-"]
        self._axis.options = tuple(keys)
        if self._axis.value not in keys:
            self._axis.value = keys[0]
        self._axis.visible = len(keys) > 1
        axis = self._axis.value
        self._axis_value.label = axis.capitalize() if axis in axes else "Value"
        self._axis_value.options = axes.get(axis) or ["-"]  # dropdown resets .value if stale
        self._axis_value.visible = axis in axes

    @staticmethod
    def _apply(handles: dict, attr: str, value):
        for h in handles.values():
            if h is not None:
                setattr(h, attr, value)

    # ---------------------------------------------------------------- scene
    def load_asset(self, source: AssetDataset | str | Path | None):
        """Swap the asset. Drops every scene object and cache of the previous one."""
        with self._lock:
            self._gen += 1  # in-flight mesh bakes belong to the old asset now
            self._clear_scene(meshes=True)
            self.ds = self.seq = None
            try:
                self.ds = source if isinstance(source, AssetDataset) else AssetDataset(str(source).strip())
            except Exception as e:
                self._status.content = f"cannot load asset: {e}"
                return
            ds = self.ds
            self._suspend = True
            try:
                self._path.value = str(ds.root)
                self._traj.options = ds.trajectories or ["-"]
                self._illum.options = ds.illuminations or ["-"]
                self._sync_axes()
                src = self._source.value
                self._source.options = ("gt", *ds.methods)
                self._source.value = src if src in self._source.options else "gt"
                geo = self._geometry.value
                self._geometry.options = ("None", "Point cloud", *(f"mesh: {k}" for k in ds.meshes))
                self._geometry.value = geo if geo in self._geometry.options else "Point cloud"
            finally:
                self._suspend = False
            self._load_sequence()
            for client in self.server.get_clients().values():
                self._frame_client(client)

    def _clear_scene(self, meshes: bool = False):
        handles = [*self._frustums.values(), *self._clouds.values(), self._fused]
        if meshes:
            handles += list(self._meshes.values())
            self._meshes = {}
        for h in handles:
            if h is not None:
                h.remove()
        self._frustums, self._clouds, self._panel_cache = {}, {}, {}
        self._cloud_file = self._fused = self._fused_pts = None  # an in-flight read owns _fused_pending
        self._depth_range = None

    def _load_sequence(self):
        with self._lock:
            self._clear_scene()
            if self.ds is None:
                self.seq = None
                self._status.content = "no asset loaded — enter a path above"
                return
            v, source = self._variant(), self._source.value
            error = None
            try:
                self.seq = self.ds.sequence(v, source) if self.ds.has(v) else None
            except Exception as e:  # a bad colmap folder (or pycolmap missing) must not wedge the gui
                self.seq, error = None, e
            n = len(self.seq) if self.seq else 0
            self._frame.max = max(1, n - 1)
            self._frame.value = min(self._frame.value, max(0, n - 1))
            if not self.seq:
                self._status.content = (f"cannot load {source}: {error}" if error is not None
                                        else f"no data for {v.as_dict()} (source={source})")
                return
            self._update_transform(v, source)
            n_depth = sum(f.depth_path is not None for f in self.seq)
            n_cam = sum(f.cam_path is not None for f in self.seq)
            self._cloud_file = self.ds.cloud_path(v, source) if source != "gt" else None
            self._counts = f"{n} frames · depth {n_depth}/{n} · cams {n_cam}/{n}"
            if self._cloud_file is not None:
                self._counts += " · fused cloud"
            self._refresh()

    def _fit_transform(self, v: Variant, source: str) -> tuple[Sim3, bool]:
        """The Sim3 that puts `source`'s cameras (already loaded as `self.seq`) into gt's gauge,
        fitted to whatever frames the two share -- so switching sources on screen doesn't need a
        re-navigate. Identity, and reported as unaligned, when that's off, this already is gt, or
        too few poses overlap to fix a rotation (see `fit_cameras`)."""
        if source == "gt" or not self._align.value:
            return Sim3(), False
        try:
            gt_cams = _finite_cams(self.ds.sequence(v, "gt"))
        except Exception:
            return Sim3(), False
        src_cams = _finite_cams(self.seq)
        return fit_cameras(src_cams, gt_cams), len(src_cams.keys() & gt_cams.keys()) >= 3

    def _update_transform(self, v: Variant, source: str):
        self._transform, self._aligned = self._fit_transform(v, source)

    def _realign(self):
        """"Align to gt" toggled: re-fit (or drop) the transform and rebuild the scene objects
        whose position it's baked into, without re-reading rgb/depth/the fused .ply from disk."""
        with self._lock:
            if not self.seq:
                return
            self._update_transform(self._variant(), self._source.value)
            for h in (*self._frustums.values(), *self._clouds.values(), self._fused):
                if h is not None:
                    h.remove()
            self._frustums, self._clouds, self._fused = {}, {}, None
            self._refresh()

    def _reset_clouds(self):
        with self._lock:
            for h in (*self._clouds.values(), self._fused):
                if h is not None:
                    h.remove()
            self._clouds, self._fused = {}, None
            self._refresh()

    def _refresh(self):
        """Sync scene objects and visibility with current mode/frame/toggles."""
        with self._lock:
            if not self.seq:
                return
            n = len(self.seq)
            cur = min(self._frame.value, n - 1)
            all_mode = self._mode.value == "All frames"
            show_clouds = self._geometry.value == "Point cloud"
            # A baseline that fused its own cloud gets shown as that cloud; only fall back to
            # unprojecting its per-frame depth maps when it didn't write one (gt never does).
            fused = show_clouds and self._cloud_file is not None
            per_frame = show_clouds and not fused
            if fused:
                self._ensure_fused()
            if self._fused is not None:
                self._fused.visible = fused
            mesh_kind = self._geometry.value.removeprefix("mesh: ") if self._geometry.value.startswith("mesh: ") else None
            if mesh_kind:
                self._ensure_mesh(mesh_kind)
            for k, h in self._meshes.items():
                if h is not None:
                    h.visible = k == mesh_kind
            targets = set(range(0, n, self._frame_step.value)) if all_mode else {cur}
            for i in sorted(targets):
                if self._show_frustums.value:
                    self._ensure_frustum(i)
                if per_frame:
                    self._ensure_cloud(i)
            # nothing to keep once the fused cloud is on screen: these are its inputs, redundant
            keep = set() if fused else \
                targets | (set(self._clouds) if not all_mode and self._accumulate.value else set())
            for i, h in self._frustums.items():
                h.visible = self._show_frustums.value and i in targets
            for i, h in self._clouds.items():
                if h is not None:
                    h.visible = per_frame and i in keep
            # prune only after the new content is live, so playback never blinks
            for i in [k for k in self._clouds if k not in keep]:
                h = self._clouds.pop(i)
                if h is not None:
                    h.remove()
            self._update_panel(cur)
            live = [h for h in self._clouds.values() if h is not None and h.visible]
            if fused and self._fused is not None:
                live.append(self._fused)
            n_pts = sum(h.points.shape[0] for h in live)
            self._status.content = f"{self._counts} · {n_pts:,} pts{self._align_note()}"

    def _align_note(self) -> str:
        if self._source.value == "gt" or not self._align.value:
            return ""
        return (f" · aligned to gt (×{self._transform.scale:.3g})" if self._aligned
               else " · not aligned (< 3 poses shared with gt)")

    def _ensure_frustum(self, i: int):
        if i in self._frustums:
            return
        cam = self.seq[i].camera()
        if cam is None:
            return
        cam = replace(cam, c2w=self._transform.c2w(cam.c2w))  # source gauge -> gt gauge, if aligned
        self._frustums[i] = self.server.scene.add_camera_frustum(
            f"/frustums/{i:04d}", fov=cam.fov_y, aspect=cam.aspect,
            scale=self._frustum_scale.value, image=_resize(self.seq[i].rgb(), FRUSTUM_IMG_W),
            wxyz=cam.wxyz, position=cam.position, visible=False,
        )

    def _ensure_cloud(self, i: int):
        if i in self._clouds:
            return
        frame = self.seq[i]
        cam, depth = frame.camera(), frame.depth()
        if cam is None or depth is None:
            self._clouds[i] = None
            return
        pts, colors = cam.unproject(depth, rgb=frame.rgb(), stride=self._stride.value)
        pts = self._transform.points(pts)  # unproject in the camera's own gauge, then align it
        self._clouds[i] = self.server.scene.add_point_cloud(
            f"/clouds/{i:04d}", points=pts.astype(np.float32),
            colors=colors if colors is not None else np.full_like(pts, 180, dtype=np.uint8),
            point_size=self._point_size.value, point_shape="rounded", visible=False,
        )

    def _ensure_fused(self):
        """Bring up the fused .ply the current baseline wrote for this variant.

        It is one world-space cloud for the whole sequence -- the baseline's own fusion, with
        whatever confidence filtering and capping it applies -- so unlike the per-frame clouds it
        ignores the frame slider and Accumulate. Read off-thread: these files run to millions of
        points, and the read must not block the gui.
        """
        path = self._cloud_file
        if self._fused is not None or self._fused_pending == path:
            return
        if self._fused_pts is not None:
            self._add_fused(*self._fused_pts)  # only the stride changed; reuse the parsed file
            return
        self._fused_pending = path
        self._status.content = f"loading fused cloud {path.name} ..."
        gen = self._gen

        def work():
            err = None
            try:
                pts, colors = read_cloud(path)
            except Exception as e:
                err = e
            with self._lock:
                if self._fused_pending == path:
                    self._fused_pending = None
                if gen != self._gen or self._cloud_file != path:  # asset/variant moved on; drop it
                    return
                if err is not None:
                    self._status.content = f"cannot read {path.name}: {err}"
                    return
                self._fused_pts = (pts, colors)
                self._refresh()  # takes the cached branch above and adds the handle

        threading.Thread(target=work, daemon=True).start()

    def _add_fused(self, pts: np.ndarray, colors: np.ndarray | None):
        s = self._stride.value ** 2  # per-frame stride thins a 2d grid; square it to thin as much
        # `_fused_pts` (this method's caller) caches the .ply exactly as read -- unaligned -- so
        # toggling "Align to gt" only needs to re-run this transform, not re-read the file.
        pts = self._transform.points(pts[::s]).astype(np.float32)
        self._fused = self.server.scene.add_point_cloud(
            "/clouds/fused", points=pts,
            colors=colors[::s] if colors is not None else np.full_like(pts, 180, dtype=np.uint8),
            point_size=self._point_size.value, point_shape="rounded", visible=False,
        )

    def _ensure_mesh(self, kind: str):
        if kind in self._meshes:
            return
        self._meshes[kind] = None  # marks "loading" so we spawn only one worker
        self._status.content = f"baking mesh '{kind}' in the background (cached after first run)..."
        gen, path = self._gen, self.ds.meshes[kind]

        def work():
            try:
                mesh = load_mesh_for_vis(path, max_faces=self.mesh_max_faces)
            except Exception as e:
                with self._lock:
                    if gen == self._gen:
                        self._meshes.pop(kind, None)
                        self._status.content = f"mesh '{kind}' failed: {e}"
                return
            with self._lock:
                if gen != self._gen:  # asset changed while baking; drop it
                    return
                self._meshes[kind] = self.server.scene.add_mesh_trimesh(
                    f"/meshes/{kind}", mesh, wxyz=GLTF_TO_Z_UP, visible=False,
                )
                self._refresh()

        threading.Thread(target=work, daemon=True).start()

    def _update_panel(self, i: int):
        if i not in self._panel_cache:
            rgb = _resize(self.seq[i].rgb(), PANEL_IMG_W)
            depth = self.seq[i].depth()
            if depth is None:
                vis = np.zeros_like(rgb)
            else:
                if self._depth_range is None:
                    self._depth_range = np.percentile(depth[np.isfinite(depth)], [2, 98])
                vis = colorize_depth(_resize(depth, PANEL_IMG_W), *self._depth_range)
            self._panel_cache[i] = (rgb, vis)
        self._panel.image, self._panel_depth.image = self._panel_cache[i]

    # ----------------------------------------------------------------- run
    def run(self):
        print(f"viewer running: http://localhost:{self.server.get_port()}")
        while True:
            if self._play.value and self.seq and len(self.seq) > 1:
                self._frame.value = (self._frame.value + 1) % len(self.seq)
                time.sleep(1 / self._fps.value)
            else:
                time.sleep(0.05)
