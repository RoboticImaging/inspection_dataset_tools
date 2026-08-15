# Evaluation

Scores a baseline's predictions against the dataset's ground truth. Point it at a prediction
folder — the same `<asset>/predictions/<method>/` tree the baselines write — and it evaluates
**every parameter combination inside it**, one row per variant:

```bash
venv/bin/python eval.py /data/old_house/predictions/da3
venv/bin/python eval.py /data/old_house/predictions/*        # every method, side by side
```

It prints a markdown table per metric and writes them to `--out` (default `eval_results/`) as
`.csv` + `.md` (`--format csv,md,tex` for LaTeX too):

```
## pose

| Asset     | Method    | Traj      | Illumination | Soil    | RRA@5 % | ... | ATE RMSE (m) | Align scale |
|-----------|-----------|-----------|--------------|---------|---------|-----|--------------|-------------|
| old_house | da3       | trendspek | sunny        | verylow | 100.0   | ... | 0.1023       | 1.2303      |
| old_house | vggt-slam | trendspek | sunny        | verylow | 100.0   | ... | 1.2871       | 23.7177     |
```

Useful flags: `--metrics pose,depth` to run a subset, `--traj/--illumination/--soil/--reflectivity`
to evaluate one variant, `--align`, `--set` (below), `--device`, `--list-metrics` to print every
metric with its options and defaults.

## Device

Every array is a torch tensor and every metric runs on the gpu when there is one. The device is
picked once, in `device.py`: cuda if available, else cpu, overridden by `--device cuda:1`, by
`$INSPECTION_EVAL_DEVICE`, or by `inspection_eval.device.use()` in code. Nothing else in the
package knows what a device is — data arrives from `inspection_viz` as numpy (EXR maps, `.ply`
clouds, sampled meshes, cam jsons), crosses over in `device.tensor`, and leaves as plain floats
in a results row.

Poses and depth are computed in float64 and match the numbers the numpy version gave to the last
bit. Clouds are float32 — they are three orders of magnitude larger than anything else here —
which moves a point-to-surface distance by a micron or so, and with it the odd point across an
F-score threshold: expect the cloud table to agree to about four digits rather than exactly.

Reading a single number back from the device costs about a millisecond and a half, which is more
than most of these metrics' arithmetic. If you add a metric, keep `float(...)`, `.item()` and
`int(tensor)` out of per-frame and per-cell loops — build the row at the end.

## Alignment: the gauge problem

A method that estimates its own poses reconstructs the scene up to a similarity transform — the
world origin, orientation and **scale** are its own. So before any metric runs, a Sim(3) is
fitted to the camera centers (Umeyama) and applied to *everything* that method produced: poses,
depth, clouds. `--align se3` fits rotation and translation only (for methods that claim metric
scale), `--align none` takes predictions at face value (for runs conditioned on gt poses).

The fitted scale is reported in the tables (`Align scale`, `Depth scale`) — it is worth a glance:
a scale far from 1 means the prediction was in its own units, and every absolute number in the
row depends on having corrected it.

## Metrics

| Metric | Needs | Reports |
|--------|-------|---------|
| `pose` | gt + predicted cams | RRA/RTA % at 5/15/30°, AUC@30, ATE RMSE + median, absolute rotation error |
| `depth` | gt + predicted depth | RMSE, MAE, relative MAE, scale-invariant MSE, accuracy/completeness at ratio thresholds, valid-pixel ratios |
| `cloud` | gt mesh + predicted cloud | accuracy, completeness, Chamfer, precision/recall/F-score at 5/10/25 cm |

Details, definitions and references are in each module's docstring
(`metrics/poses.py`, `metrics/depth.py`, `metrics/cloud.py`). Three things worth knowing up front:

- **RRA/RTA are relative** (per frame pair) and so are unaffected by alignment; ATE is absolute
  and only means anything after it.
- **Depth accuracy vs. completeness** differ only in what they divide by — valid *predicted*
  pixels vs. valid *gt* pixels — so a method that predicts little but predicts it well scores
  high on the first and low on the second. GT pixels at the render's clip distance are
  background, not surface, and count as invalid.
- **The two gt meshes are used for different directions**: `{asset}_accuracy.glb` (the full
  scene) for accuracy/precision, `{asset}_completeness.glb` (the asset of interest) for
  completeness/recall, so a method is not punished for skipping scenery nobody asked it to
  reconstruct. Assets with one mesh use it for both. The predicted cloud is a method's fused
  `.ply` when it wrote one, otherwise its per-frame depth unprojected with its own poses, and it
  is cropped to the gt bounding box first.

## Options

Every metric's knobs are a dataclass with defaults; override any of them from the command line:

```bash
venv/bin/python eval.py .../predictions/da3 \
  --set depth.scale=median \          # per-frame median ratio instead of the pose scale
  --set depth.max_frames=20 \         # a quick pass over a long sequence
  --set cloud.thresholds=0.1,0.5 \    # F-score distances, in meters
  --set cloud.gt_points=200000 \      # points sampled off the gt mesh
  --set cloud.source=unprojected      # ignore the method's fused .ply
```

`--list-metrics` prints them all. GT meshes are sampled once and cached in
`~/.cache/inspection_eval/` — the first `cloud` run on a large asset pays for loading a
several-hundred-MB `.glb`, later ones don't.

## Adding a metric

Write a function of an `EvalCase`, declare what it needs and what it can be tuned with, and
import the module in `metrics/__init__.py`. Nothing else changes: the runner picks it up, the
CLI exposes its options, and it gets a table whose columns are whatever keys it returned.

```python
# inspection_eval/metrics/normals.py
from dataclasses import dataclass

from ..case import CAP_DEPTH, EvalCase
from .base import register


@dataclass
class NormalOptions:
    max_frames: int | None = None


@register("normals", needs=(CAP_DEPTH,), options=NormalOptions)
def normal_metrics(case: EvalCase, o: NormalOptions) -> dict:
    ...
    return {"Mean angular error (deg)": ...}
```

`EvalCase` (see `case.py`) is the whole interface a metric needs: `gt_cams` / `pred_cams`
(aligned), `depth_pair(index)` (size-matched, invalid pixels `nan`), `pred_cloud()` /
`gt_cloud(kind)`, and `transform` if the metric cares about the gauge. Everything is lazy and
cached, so asking for poses never loads a depth map.

A metric that needs data no case has yet (gt normals, rendered novel views, ...) needs a new
capability: add a `CAP_*` constant in `case.py`, have `EvalCase.capabilities` decide when it is
available, and list it in the metric's `needs` — cases without it are then skipped and reported
instead of failing.

## Tests

```bash
venv/bin/python -m pytest tests/test_eval.py
```

Each metric is checked against data whose answer is known by construction — a prediction that is
the ground truth expressed in a different gauge must come out perfect after alignment, and a
prediction that is wrong by a known amount must report that amount.
