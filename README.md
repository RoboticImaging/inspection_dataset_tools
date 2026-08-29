# Run visualizer
`python vis.py <optional: asset path>`

Alternatively, specify asset path in the web interface

# Baselines

Each submodule under the repo root (e.g. `depth-anything-3/`) is a baseline. Its runner code
lives here, in `baselines/<name>/`, and is kept separate from the submodule itself so the
submodule stays an unmodified upstream checkout.

```
baselines/
  registry.py           # every baseline, by name -> the class that runs it (one line each)
  run.py                # run any registered baseline: --baseline <name> --data ... --model ...
  common/               # shared code, used by every baseline
    config.py           #   yaml -> dataclass loading: data config + baseline config
    runner.py           #   the run driver: sequence selection, skipping, writing, run record
    writer.py           #   depth/cam/cloud writers, in the gt file formats
  configs/
    da3.yaml            #   baseline configs: one per baseline, reusable across assets
    data/old_house.yaml #   data configs: one per asset, reusable across baselines
  depth_anything_3/     # one folder per baseline
    Dockerfile
    build_container.sh  #   build the image
    run_container.sh    #   shell into the image
    run_sequence.py     #   the Baseline subclass + its own entry point
```

### Registered baselines

| name | what it runs |
|---|---|
| `da3` | Depth Anything 3, one joint inference call per sequence |
| `da3_streaming` | DA3-Streaming, driven through its own chunked/loop-closed CLI |
| `vggt` | Plain feed-forward VGGT, one joint inference call per sequence |
| `vggt-omega` | Plain feed-forward VGGT-Omega, one joint inference call per sequence |
| `pi3` | Plain feed-forward Pi3, one joint inference call per sequence |
| `pi3x` | Pi3X, driven through its own chunked VO pipeline |
| `vggt-slam` | VGGT-SLAM, driven through its own submap/loop-closure pipeline |
| `vggt-long` | VGGT-Long, driven through its own chunked/loop-closed CLI |
| `mpsfm` | MP-SfM, driven through its own classical incremental-SfM CLI |
| `pi-long` | Pi-Long, driven through its own chunked/loop-closed CLI |
| `colmap` | Classical COLMAP SfM + MVS, driven through its own CLI |
| `glomap` | GLOMAP global SfM + classical MVS, driven through the `colmap` CLI's own `global_mapper` command |
| `pixel-perfect-sfm` | Classical COLMAP SfM + dense-feature keypoint/bundle refinement |
| `vidmap` | Video-aware SfM frontend + COLMAP-derived global mapper, driven through its own CLI |

`baselines/registry.py` is the source of truth — the registry key doesn't always match the
baseline's directory name or image tag (e.g. `da3` lives in `baselines/depth_anything_3/`;
`da3_streaming`'s image is `inspection-baselines/da3-streaming`, hyphen not underscore). Each
baseline's own `run_container.sh`/`build_container.sh` resolve their own image tag.

## Configs

A run takes two files, so neither has to be rewritten to change the other half:

- **data config** (`configs/data/<asset>.yaml`) — `asset:`, one key per variant axis
  (`trajectory`, `illumination`, `soil`, `reflectivity`), and `frame_start` / `frame_end` /
  `frame_stride`. Reusable by every baseline.
  - each axis takes one value (`sunny`), a list (`[sunny, overcast]`), or `null` = every value
    that asset has on that axis, discovered from the asset itself.
- **baseline config** (`configs/<baseline>.yaml`) — `model:` (that baseline's own knobs) and
  `output:` (`method`, `overwrite`, `min_confidence`, `cloud`). Reusable across assets.
  - `output.method` is the `predictions/<method>/` folder name; `null` = the baseline's
    registered name. Set it to keep two configurations of one baseline apart.

Rules that hold for every baseline:

- Selected sequences run one after another through a single loaded model.
- A sequence that already has predictions is skipped unless `output.overwrite: true` — re-run the
  same command to resume an interrupted run or to extend it to more variants.
- Every runner's `--set section.path=value` (repeatable, e.g. `--set model.ka.strategy=basic`,
  `--set output.overwrite=true`) patches one field of the loaded data/model/output config without
  writing a new yaml file — the same mechanism `eval.py`'s own `--set metric.option=value` (see
  `# Evaluation` below) uses, shared by every baseline runner via
  `baselines.common.config.add_config_args`.
- After each run, `<asset>/predictions/<method>/config.yaml` records both configs, the timestamp
  and the variants written. It is rewritten per run: it describes what is on disk now.
- Both halves may still be given as a single file (`dataset:` + `model:` + `output:`) passed as
  the one positional argument, e.g. `configs/da3_glass_house.yaml`.

## Running a baseline (example: Depth Anything 3)

1. Build the image once (and after `Dockerfile` changes):
   `baselines/depth_anything_3/build_container.sh`
2. Write the data config. **One sequence** — name every axis the asset varies:
   ```yaml
   # baselines/configs/data/old_house.yaml
   asset: /data/old_house
   trajectory: trendspek
   illumination: sunny
   soil: verylow          # or `reflectivity:`, whichever axis this asset varies
   frame_stride: 1
   ```
   **All sequences** — drop the axis keys (or set them `null`); a list runs a subset:
   ```yaml
   asset: /data/old_house
   illumination: [sunny, overcast]   # and every trajectory and soil level, since neither is named
   ```
3. Pick or edit the baseline config (`baselines/configs/da3.yaml`: checkpoint, `process_res`,
   `use_known_poses`, `min_confidence`, `cloud`, `overwrite`).
4. Run it — same command for one sequence or all, only the data config differs:
   ```bash
   baselines/depth_anything_3/run_container.sh python run_sequence.py \
       --data ../configs/data/old_house.yaml --model ../configs/da3.yaml
   ```

Notes:

- `run_container.sh <cmd>` runs one command and exits; with no command it opens a shell whose cwd
  is the baseline folder, where `python run_sequence.py --data ... --model ...` is the same run.
- `python ../run.py --baseline da3 --data ... --model ...` does the same through the registry —
  use it in a script driving several baselines.
- The container is only the environment: host paths (`/home`, `/mnt`, `/media`) are bind-mounted
  onto **themselves**, so absolute paths work unchanged and code edits need no rebuild. It runs as
  your uid/gid, so predictions are owned by you. Extra trees: `MOUNTS='/data /nfs'`; pick GPUs
  with `GPUS='"device=1"'`.
- DA3 runs each sequence in a single inference call (that is what makes its poses and depths
  mutually consistent), so a long one can run out of GPU memory — there is no chunking fallback;
  raise `frame_stride` or narrow `frame_start`/`frame_end`.

Output lands under `<asset>/predictions/<method>/`, in the gt file formats, keyed by the *full*
variant (a baseline infers depth from rgb, so each illumination/soil is a distinct prediction):

```
predictions/<method>/config.yaml                                         # what produced these
predictions/<method>/<illumination>/depths/soil=<soil>,traj=<traj>/*.exr
predictions/<method>/<illumination>/cam_params/soil=<soil>,traj=<traj>/*.json
predictions/<method>/<illumination>/clouds/soil=<soil>,traj=<traj>.ply    # if output.cloud
```

(`ref=<reflectivity>` instead of `soil=...` on assets that vary that axis.) Open the asset in the
viewer (`venv/bin/python vis.py`) and the **Source** dropdown offers `gt` and every method.

## Adding another baseline

1. `git submodule add <url> <name>` at the repo root.
2. Copy `baselines/depth_anything_3/` as a starting point: `Dockerfile` (install the submodule +
   `inspection_viz`'s light deps — see comments), `build_container.sh` / `run_container.sh`
   (usually only the image name changes), `run_sequence.py`.
3. In `run_sequence.py`, subclass `baselines.common.runner.Baseline`:
   - `name` — the default `predictions/<name>/` folder and the viewer's Source label.
   - `Config` — your dataclass of model knobs, loaded from the baseline config's `model:`.
   - `setup()` — load the model. Called once per run, before any sequence.
   - `run(frames, out)` — predict one sequence and write it. `frames` are gt frames:
     `frame.rgb_path` is the input, `frame.camera()` the gt pose if you condition on it.
   - `__main__` is one line: `cli_main(YourClass)`.
4. Write through the `SequenceOutput` (`out`) handed to `run()` — it owns the folder layout and
   applies `output.min_confidence`, so the viewer and eval need no changes:
   - `out.write_frame(index, depth, c2w, K, intrinsics_width, conf=None)` per frame — planar
     z-depth in meters at rgb resolution, OpenCV camera-to-world pose.
   - `out.write_cloud(points, colors)` for a fused `.ply`, and `out.scratch_dir(name)` if the
     model's own exporter needs a working dir.
   - `out.spec` is the `OutputSpec`, `out.variant` the sequence being run.
5. Register it: one line in `baselines/registry.py`, `"<name>": "<module>:<class>"` (imported
   lazily, so its dependencies are only needed when it runs).
6. Add a baseline config under `baselines/configs/`. Data configs are shared — do not add one.

Everything else — resolving the data config, looping over sequences, skipping ones already done,
writing `predictions/<method>/config.yaml` — is `run_baseline`'s job and identical for every
baseline.

`baselines.common.config.{DatasetSelection,OutputSpec,load_config}` still behave as they did for
runners written before the config split (`DatasetSelection` is an alias of `DataConfig`, keeping
`root` and `variant()`), but a new baseline should take the `Baseline` route above.

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
  disambiguate. (Not an issue with `use_known_poses: true`.) The evaluator solves the scale
  itself, so this is a convenience for the viewer, not something metrics depend on.

# Evaluation

Scores predictions against ground truth, one row per (asset, method, variant). Nothing to
configure: point it at what the baselines wrote. Details, metric definitions and how to add a
metric: [inspection_eval/README.md](inspection_eval/README.md).

```bash
venv/bin/python eval.py /data/old_house/predictions/da3     # every variant in that folder
venv/bin/python eval.py /data/old_house/predictions/*       # every method, one table each
venv/bin/python eval.py /data/old_house                     # same: an asset root means all its methods
```

- Prints a markdown table per metric and writes `.csv` + `.md` to
  `eval_results/<assets>-<methods>-<timestamp>/` (`--out`, `--no-timestamp`, `--format`).
- Metrics: `pose` (needs gt + predicted cams), `depth` (gt + predicted depth), `cloud` (gt mesh +
  the method's fused `.ply`, or its depth unprojected). A metric with nothing to work on is
  skipped for that variant and reported at the end, not fatal.
- Predictions are Sim(3)-aligned to the gt camera centers before any metric runs, so an unknown
  world gauge and scale are not penalised — `--align se3` for methods that claim metric scale,
  `--align none` for runs conditioned on gt poses.
- Useful flags: `--metrics pose,depth`, `--traj/--illumination/--soil/--reflectivity` to score a
  single variant, `--set metric.option=value`, `--list-metrics`, `--allow-missing-frames`.

# Analysis tools

Root-level scripts that read (never write) prediction/eval output — none needs a baseline's own
container, just the host venv.

- **`check_completion_status.py`** — scans an `eccv_2026_data`-style root and reports, per
  registered baseline and asset, whether every sequence variant has a completed prediction
  (judged by the fused cloud existing) as `COMPLETE`/`PARTIAL`/`NOT STARTED`/`OOM`, colorized.
  ```bash
  python check_completion_status.py [--eccv-root PATH] [--baseline NAME ...] [--example] [--verbose]
  ```
- **`collate_results.py`** — collates every csv under an `eval.py --save-hierarchical` output tree
  into one combined csv per metric category (cloud/depth/pose), for further analysis in
  pandas/a spreadsheet — raw input, not a human-facing summary.
- **`latex_results_table.py`** — prints ready-to-paste LaTeX rows for per-scene results tables from
  the same `--save-hierarchical` tree.
  ```bash
  python latex_results_table.py RESULTS_ROOT vggt-omega da3 pi-long
  ```
- **`plot_variant_metrics.py`** — plots pose/cloud/depth metrics against the soiling/reflectivity/
  illumination axes `eval.py`'s variants sweep, comparing every method on the same axes.
  ```bash
  python plot_variant_metrics.py glass_house2 --out plots/glass_house2
  ```
