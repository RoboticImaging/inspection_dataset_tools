# inspection_dataset_visualizer

Dataloader + viser visualizer for inspection assets (rgb / depth / camera per frame,
organized by trajectory × illumination × soil).

## Run

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python main.py /path/to/glass_house [--port 8080]
```

Open the printed URL. GUI: pick trajectory / illumination / soil; Timeline mode with a
frame slider + Play (rgb + colorized depth video panels) and an Accumulate toggle for a
cumulative point-cloud trail, or "All frames" mode, subsampled by the "Frame step" slider
to keep the browser alive. The Geometry dropdown switches between the unprojected point
cloud and the GT meshes (`{asset}_{kind}.glb` at the asset root). Meshes are merged,
texture-baked to vertex colors, and decimated to `--mesh-faces` (default 1M) for the
browser; the result is cached in `~/.cache/inspection_viz/`, so only the first load is slow.

## When the folder structure changes

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
missing depth/cams, junk files, sparse variant combos, unprojection math, and a check of
real-data poses against the COLMAP ground truth (skipped if `glass_house` is absent).
