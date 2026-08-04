import threading
import time
from pathlib import Path

import cv2
import numpy as np
import viser

from .dataset import AssetDataset, Variant
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
        center = np.mean([c.position for c in cams if c], axis=0) if any(cams) else np.zeros(3)
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
        with g.add_folder("View"):
            self._mode = g.add_dropdown("Mode", ("Timeline", "All frames"))
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
        for h in (self._mode, self._show_frustums, self._geometry, self._frame_step, self._accumulate):
            h.on_update(lambda _: self._suspend or self._refresh())
        self._frame.on_update(lambda _: self._refresh())
        self._stride.on_update(lambda _: self._reset_clouds())
        self._point_size.on_update(lambda _: self._apply(self._clouds, "point_size", self._point_size.value))
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
                geo = self._geometry.value
                self._geometry.options = ("None", "Point cloud", *(f"mesh: {k}" for k in ds.meshes))
                self._geometry.value = geo if geo in self._geometry.options else "Point cloud"
            finally:
                self._suspend = False
            self._load_sequence()
            for client in self.server.get_clients().values():
                self._frame_client(client)

    def _clear_scene(self, meshes: bool = False):
        handles = [*self._frustums.values(), *self._clouds.values()]
        if meshes:
            handles += list(self._meshes.values())
            self._meshes = {}
        for h in handles:
            if h is not None:
                h.remove()
        self._frustums, self._clouds, self._panel_cache = {}, {}, {}
        self._depth_range = None

    def _load_sequence(self):
        with self._lock:
            self._clear_scene()
            if self.ds is None:
                self.seq = None
                self._status.content = "no asset loaded — enter a path above"
                return
            v = self._variant()
            self.seq = self.ds.sequence(v) if self.ds.has(v) else None
            n = len(self.seq) if self.seq else 0
            self._frame.max = max(1, n - 1)
            self._frame.value = min(self._frame.value, max(0, n - 1))
            if not self.seq:
                self._status.content = f"no data for {v.as_dict()}"
                return
            n_depth = sum(f.depth_path is not None for f in self.seq)
            n_cam = sum(f.cam_path is not None for f in self.seq)
            self._counts = f"{n} frames · depth {n_depth}/{n} · cams {n_cam}/{n}"
            self._refresh()

    def _reset_clouds(self):
        with self._lock:
            for h in self._clouds.values():
                if h is not None:
                    h.remove()
            self._clouds = {}
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
                if show_clouds:
                    self._ensure_cloud(i)
            keep = targets | (set(self._clouds) if not all_mode and self._accumulate.value else set())
            for i, h in self._frustums.items():
                h.visible = self._show_frustums.value and i in targets
            for i, h in self._clouds.items():
                if h is not None:
                    h.visible = show_clouds and i in keep
            # prune only after the new content is live, so playback never blinks
            for i in [k for k in self._clouds if k not in keep]:
                h = self._clouds.pop(i)
                if h is not None:
                    h.remove()
            self._update_panel(cur)
            n_pts = sum(h.points.shape[0] for h in self._clouds.values() if h is not None)
            self._status.content = f"{self._counts} · {n_pts:,} pts"

    def _ensure_frustum(self, i: int):
        if i in self._frustums:
            return
        cam = self.seq[i].camera()
        if cam is None:
            return
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
        self._clouds[i] = self.server.scene.add_point_cloud(
            f"/clouds/{i:04d}", points=pts.astype(np.float32),
            colors=colors if colors is not None else np.full_like(pts, 180, dtype=np.uint8),
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
