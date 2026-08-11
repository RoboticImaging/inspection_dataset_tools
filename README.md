# Run visualizer
`python vis.py <optional: asset path>`

Alternatively, specify asset path in the web interface

# Baselines

Each submodule under the repo root (e.g. `depth-anything-3/`) is a baseline. Its runner code
lives here, in `baselines/<name>/`, and is kept separate from the submodule itself so the
submodule stays an unmodified upstream checkout.

```
baselines/
  common/               # shared code, used by every baseline
    config.py           #   yaml -> dataclass loading, dataset selection, output spec
    writer.py           #   depth/cam/cloud writers, in the gt file formats
  configs/              # example run configs (yaml)
  depth_anything_3/     # one folder per baseline
    Dockerfile
    build_container.sh  #   build the image
    run_container.sh    #   shell into the image
    run_sequence.py     #   the actual run, from inside the container
```

## Running Depth Anything 3

```bash
baselines/depth_anything_3/build_container.sh          # once (and after Dockerfile changes)
baselines/depth_anything_3/run_container.sh            # drops you into a shell in the image
# then, inside the container (cwd is the baseline folder):
python run_sequence.py ../configs/da3_glass_house.yaml
```

The container is just the environment: the host's paths (`/home`, `/mnt`, `/media`) are
bind-mounted onto **themselves**, so the repo, config yamls and dataset roots are all at their
normal absolute paths inside — nothing to translate, and code edits need no rebuild. It runs as
your uid/gid with your supplementary groups, so predictions written from inside are owned by you
and deletable from outside. Extra host trees: `MOUNTS='/data /nfs' run_container.sh`. GPUs
default to all; pick some with `GPUS='"device=1"' run_container.sh`.

Edit a copy of `baselines/configs/da3_glass_house.yaml` to point `dataset.root` at your asset and
pick the trajectory/illumination/soil (or reflectivity) — see the comments in that file for every
toggle (frame range/stride, model checkpoint, known-pose conditioning, confidence filtering, ...).
The whole selected sequence goes through DA3 in a single inference call (that's what makes its
poses and depths mutually consistent), so a long one can simply run out of GPU memory — there's
no chunking fallback; shorten the frame range or raise the stride.

Output lands under `<dataset.root>/predictions/<output.method>/`, in the same file formats as
ground truth, keyed by the *full* variant (a baseline infers depth from rgb, so each
illumination/soil is a distinct prediction):

```
predictions/<method>/<illumination>/depths/soil=<soil>,traj=<traj>/*.exr
predictions/<method>/<illumination>/cam_params/soil=<soil>,traj=<traj>/*.json
predictions/<method>/<illumination>/clouds/soil=<soil>,traj=<traj>.ply    # if output.cloud
```

(`ref=<reflectivity>` instead of `soil=...` on assets that vary that axis.) Open the asset in the
viewer (`venv/bin/python vis.py`) and the **Source** dropdown will offer `gt` and `da3` (or
whatever `output.method` was set to) for that sequence.

## Adding another baseline

1. `git submodule add <url> <name>` at the repo root.
2. Copy the `depth_anything_3/` folder as a starting point: `Dockerfile` (install the submodule
   + `inspection_viz`'s light deps — see comments), `build_container.sh` / `run_container.sh`
   (usually just the image name changes), `run_sequence.py`.
3. In `run_sequence.py`, use `baselines.common.config.{DatasetSelection,OutputSpec,load_config}`
   for the dataset-selection/output-toggle boilerplate, and your own dataclass for
   model-specific knobs. Load frames with `AssetDataset(...).sequence(variant)` (defaults to the
   `"gt"` source — baselines consume rgb, not another baseline's predictions) and
   `DatasetSelection.select()` to apply the frame range/stride.
4. Write predictions with `baselines.common.writer.{output_dirs,write_depth,write_cam}` (plus
   `cloud_path`/`write_cloud` for a fused .ply) — see that module's docstring for the camera-json
   convention (predicted poses/intrinsics are mapped losslessly into the same schema ground-truth
   cams use, so `Camera.from_blender` and the viewer need no changes for a new method).
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
- A model that predicts its own poses works in an arbitrary world gauge, so predictions are
  rigidly aligned to the gt pose of the first selected frame. That fixes position/orientation
  only — depth and inter-frame translations stay at the model's own scale, which one frame can't
  disambiguate. (Not an issue with `use_known_poses: true`.)
