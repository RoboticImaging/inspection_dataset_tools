import threading
import time

import cv2
import numpy as np
import viser

from .dataset import AssetDataset, Variant

FRUSTUM_IMG_W = 160
PANEL_IMG_W = 480
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
    def __init__(self, dataset: AssetDataset, port: int = 8080):
        self.ds = dataset
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.seq = None
        self._frustums = {}   # frame idx -> handle
        self._clouds = {}     # frame idx -> handle | None
        self._meshes = {}     # mesh kind -> handle, persists across variants
        self._panel_cache = {}  # frame idx -> (small rgb, small depth vis)
        self._depth_range = None
        self._lock = threading.RLock()
        self._counts = ""
        self._build_gui()
        self._load_sequence()
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
        g, ds = self.server.gui, self.ds
        with g.add_folder("Data"):
            self._traj = g.add_dropdown("Trajectory", ds.trajectories or ["-"])
            self._illum = g.add_dropdown("Illumination", ds.illuminations or ["-"])
            self._soil = g.add_dropdown("Soil", ds.soils or ["-"])
        with g.add_folder("View"):
            self._mode = g.add_dropdown("Mode", ("Timeline", "All frames"))
            self._frame = g.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self._play = g.add_checkbox("Play", False)
            self._fps = g.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
            self._accumulate = g.add_checkbox("Accumulate", False)
            self._show_frustums = g.add_checkbox("Frustums", True)
            geo_opts = ("None", "Point cloud", *(f"mesh: {k}" for k in ds.meshes))
            self._geometry = g.add_dropdown("Geometry", geo_opts, initial_value="Point cloud")
            self._stride = g.add_slider("Point stride", min=1, max=16, step=1, initial_value=4)
            self._frame_step = g.add_slider("Frame step (all)", min=1, max=32, step=1, initial_value=8)
            self._point_size = g.add_slider("Point size", min=0.005, max=0.2, step=0.005, initial_value=0.03)
            self._frustum_scale = g.add_slider("Frustum scale", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        self._panel = g.add_image(np.zeros((2, 2, 3), np.uint8), label="RGB")
        self._panel_depth = g.add_image(np.zeros((2, 2, 3), np.uint8), label="Depth")
        self._status = g.add_markdown("")

        for h in (self._traj, self._illum, self._soil):
            h.on_update(lambda _: self._load_sequence())
        for h in (self._mode, self._show_frustums, self._geometry, self._frame_step, self._accumulate):
            h.on_update(lambda _: self._refresh())
        self._frame.on_update(lambda _: self._refresh())
        self._stride.on_update(lambda _: self._reset_clouds())
        self._point_size.on_update(lambda _: self._apply(self._clouds, "point_size", self._point_size.value))
        self._frustum_scale.on_update(lambda _: self._apply(self._frustums, "scale", self._frustum_scale.value))

    def _variant(self) -> Variant:
        pick = lambda h: None if h.value == "-" else h.value
        return Variant(pick(self._traj), pick(self._illum), pick(self._soil))

    @staticmethod
    def _apply(handles: dict, attr: str, value):
        for h in handles.values():
            if h is not None:
                setattr(h, attr, value)

    # ---------------------------------------------------------------- scene
    def _load_sequence(self):
        with self._lock:
            for h in [*self._frustums.values(), *self._clouds.values()]:
                if h is not None:
                    h.remove()
            self._frustums, self._clouds, self._panel_cache = {}, {}, {}
            self._depth_range = None
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
        self._status.content = f"loading mesh '{kind}'..."
        self._meshes[kind] = self.server.scene.add_glb(
            f"/meshes/{kind}", self.ds.meshes[kind].read_bytes(),
            wxyz=GLTF_TO_Z_UP, visible=False,
        )

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
