# Baselines

Each submodule under the repo root (e.g. `depth-anything-3/`) is a baseline. Its runner code
lives here, in `baselines/<name>/`, and is kept separate from the submodule itself so the
submodule stays an unmodified upstream checkout.

```
baselines/
  common/            # shared code: config loading, output writer — used by every baseline
  configs/            # example run configs (yaml)
  depth_anything_3/   # one folder per baseline
    Dockerfile
    run_container.sh
    run_sequence.py
```

## Running Depth Anything 3

```bash
baselines/depth_anything_3/run_container.sh baselines/configs/da3_glass_house.yaml
```

This builds the Docker image (only takes a while the first time), then runs
`run_sequence.py` inside it. Edit a copy of `baselines/configs/da3_glass_house.yaml` to point
`dataset.root` at your asset and pick the trajectory/illumination/soil (or reflectivity) — see
the comments in that file for every toggle (frame range/stride, model checkpoint, known-pose
conditioning, confidence filtering, ...).

Output lands at `<dataset.root>/predictions/<output.method>/{depths,cam_params}/traj=.../*`,
in the same file formats as ground truth. Open the asset in the viewer
(`venv/bin/python vis.py`) and the new **Source** dropdown will offer `gt` and `da3` (or
whatever `output.method` was set to) for that sequence.

## Adding another baseline

1. `git submodule add <url> <name>` at the repo root.
2. Copy the `depth_anything_3/` folder as a starting point: `Dockerfile` (install the submodule
   + `inspection_viz`'s light deps — see comments), `run_container.sh` (usually just the image
   name changes), `run_sequence.py`.
3. In `run_sequence.py`, use `baselines.common.config.{DatasetSelection,OutputSpec,load_config}`
   for the dataset-selection/output-toggle boilerplate, and your own dataclass for
   model-specific knobs. Load frames with `AssetDataset(...).sequence(variant)` (always `"gt"` —
   baselines consume rgb, not another baseline's predictions) and
   `baselines.common.config.DatasetSelection.select()` to apply the frame range/stride.
4. Write predictions with `baselines.common.writer.{output_dirs,write_depth,write_cam}` — see
   that module's docstring for the camera-json convention (predicted poses/intrinsics are mapped
   losslessly into the same schema ground-truth cams use, so `Camera.from_blender` and the
   viewer need no changes for a new method).
5. Add an example config under `baselines/configs/`.

## Conventions

- Depth is planar z-depth in meters (camera-space Z, not ray distance) — same as gt. If a
  model's native output is ray depth, convert before writing.
- Predicted intrinsics are written assuming square pixels (one focal length, `cx`/`cy` at the
  image center) since that's what the gt cam-json schema (and `Camera.from_blender`) represents;
  fine for the near-square-pixel models these baselines target.
- Depth maps are resized (bilinear) to the original rgb resolution before writing, so they align
  pixel-for-pixel with rgb/gt depth for unprojection — expect some smoothing at edges if the
  model's native processing resolution was much lower.
