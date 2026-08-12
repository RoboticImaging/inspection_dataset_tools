# inspection_dataset_tools

Dataloader + viser visualizer for inspection assets (rgb / depth / camera per frame,
organized by trajectory × illumination × soil).

## Run visualizer

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python vis.py [--port 8080]
```

The asset path is optional — with no argument the viewer starts empty and you type a path
into the "Asset" field and hit Load. Switching assets drops all scene objects and caches of
the previous one, so you can browse many assets in one session.

Open the printed URL. GUI: pick trajectory / illumination / soil; Timeline mode with a
frame slider + Play (rgb + colorized depth video panels) and an Accumulate toggle for a
cumulative point-cloud trail, or "All frames" mode, subsampled by the "Frame step" slider
to keep the browser alive. The Geometry dropdown switches between the unprojected point
cloud and the GT meshes (`{asset}_{kind}.glb` at the asset root). Meshes are merged,
texture-baked to vertex colors, and decimated to `--mesh-faces` (default 1M) for the
browser; the result is cached in `~/.cache/inspection_viz/`, so only the first load is slow.

The **Source** dropdown switches the depth/cam data driving the point cloud and depth panel
between `gt` and any baseline that has written predictions for the loaded asset (see
Baselines below) — rgb is always ground truth either way. If the selected baseline also
wrote a fused cloud for the variant (`predictions/<method>/<illumination>/clouds/<variant>.ply`),
"Point cloud" shows that file — the model's own fusion — instead of unprojecting its
per-frame depth maps; being one cloud for the whole sequence, it ignores the frame slider
and Accumulate. The status line says `fused cloud` when one is in play, and the Point stride
slider thins it. Baselines that wrote no `.ply` fall back to per-frame unprojection.

## Baselines

Each submodule (e.g. `depth-anything-3/`) is a baseline model; its Docker image, run script,
and per-sequence config live in [`baselines/`](baselines/), which writes depth + camera
predictions back into `<asset>/predictions/<method>/...`, in the same file formats as ground
truth. See [`baselines/README.md`](baselines/README.md) for how to run one and how to add a
new one. Once predictions exist, they show up in the viewer's Source dropdown automatically.

## Evaluation

```bash
venv/bin/python eval.py /data/old_house/predictions/da3     # or .../predictions/* for all methods
```

Scores a method's predictions against ground truth on **every parameter combination** in that
folder, and prints/writes one results table per metric (pose, depth, point cloud). Predictions
are first put in the gt gauge by a Sim(3) fitted to the camera centers, since a method that
estimates its own poses works up to an unknown similarity transform. See
[`inspection_eval/README.md`](inspection_eval/README.md) for the metric definitions, the
`--set metric.option=value` knobs, and how to add a metric.

## When the data folder structure changes

Everything about the layout lives in `inspection_viz/config.py` (`Layout`). Edit the dir
templates / extensions there, or pass a custom one:

```python
AssetDataset(path, Layout(rgb_dir="renders/{traj}/{illumination}/soil_{soil}"))
```

File stems are matched by integer frame id, so zero-padding and stem prefixes don't matter.
New depth formats: add a reader to `DEPTH_READERS` in `inspection_viz/io.py`.

## Conventions (verified against `colmap_style_gt`)

- cam json: `loc` + `rot` = WXYZ camera-to-world quaternion, Blender camera (-z forward);
  `fx = width * focal_length / sensor_width`
- depth exr: planar z-depth in meters (cross-frame alignment error 0.10m planar vs 0.16m ray)
- `Camera.c2w` is stored in OpenCV convention (+z forward)

## Tests

```bash
venv/bin/python -m pytest tests/
```

Cover: png/jpg swap, zero-padding changes, stem prefixes, restructured folder trees,
missing depth/cams, junk files, sparse variant combos, unprojection math, baseline
prediction discovery/selection, the baseline output writer's round-trip through
`Camera.from_blender`, the evaluation metrics against predictions whose error is known by
construction, and a check of real-data poses against the COLMAP ground truth
(skipped if `glass_house` is absent).
