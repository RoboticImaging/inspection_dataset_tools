"""Evaluation: the alignment math, each metric on data whose answer is known by construction,
and the runner's discovery of variants inside a prediction folder."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from conftest import WH, make_asset, make_colmap, make_predictions, write_cam, write_depth_exr
from inspection_eval import (Sim3, Target, evaluate, frame_gaps, target_from_path,
                             targets_from_paths, umeyama)
from inspection_eval.case import EvalCase, match_size
from inspection_eval.metrics import REGISTRY, get
from inspection_eval.metrics.cloud import CloudOptions
from inspection_eval.metrics.depth import DepthOptions
from inspection_eval.metrics.poses import PoseOptions
from inspection_eval.report import Table
from inspection_viz import Layout
from inspection_viz.config import fill_template
from inspection_viz.dataset import AssetDataset, Variant

try:
    import pycolmap  # noqa: F401  (only used to decide whether to skip the colmap tests below)
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False
needs_pycolmap = pytest.mark.skipif(not HAS_PYCOLMAP, reason="pycolmap not installed")

BINDS = {"traj": "trajA", "illumination": "sunny", "soil": "low"}
VARIANT = Variant("trajA", "sunny", "low")


# --- alignment ----------------------------------------------------------------------------


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    truth = Sim3(2.5, trimesh.transformations.random_rotation_matrix(rng.random(3))[:3, :3],
                 np.array([1.0, -2.0, 3.0]))
    fitted = umeyama(src, truth.points(src))

    assert fitted.scale == pytest.approx(truth.scale)
    assert np.allclose(fitted.R, truth.R, atol=1e-9)
    assert np.allclose(fitted.points(src), truth.points(src), atol=1e-9)


def test_umeyama_without_scale_keeps_unit_scale():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(10, 3))
    fit = umeyama(src, 3.0 * src, with_scale=False)
    assert fit.scale == 1.0


def test_umeyama_needs_three_points():
    with pytest.raises(ValueError):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_match_size_downsamples_the_larger_map():
    a, b = match_size(np.ones((8, 6), np.float32), np.zeros((4, 3), np.float32))
    assert a.shape == b.shape == (4, 3)


# --- fixtures for a full case -------------------------------------------------------------


def _poses(n: int):
    """A gt trajectory: cameras stepping along +x, all with the same orientation."""
    return [((float(i), 0.0, 5.0), (1.0, 0.0, 0.0, 0.0)) for i in range(n)]


def case_asset(tmp_path: Path, method="da3", scale=1.0, shift=(0.0, 0.0, 0.0),
               depth_gt=5.0, depth_pred=5.0, n_frames=4, cloud=False) -> AssetDataset:
    """An asset whose prediction is the gt trajectory in a different gauge: positions scaled by
    `scale` and shifted, depth scaled to match. A perfect prediction, told in another language —
    so every metric should come out perfect once the case has aligned it."""
    layout = Layout()
    root = make_asset(tmp_path / "asset", trajs=("trajA",), illums=("sunny",), soils=("low",),
                      n_frames=n_frames)
    cam_dir = root / layout.cam_dir.format(**BINDS)
    depth_dir = root / layout.depth_dir.format(**BINDS)
    for i, (loc, rot) in enumerate(_poses(n_frames)):
        write_cam(cam_dir / f"{i:08d}.json", loc=loc, rot=rot)
        write_depth_exr(depth_dir / f"{i:08d}.exr", value=depth_gt)

    make_predictions(root, layout, method, BINDS, n_frames=n_frames, cloud=cloud)
    pred_cam_dir = root / fill_template(layout.pred_cam_dir, method=method, **BINDS)
    pred_depth_dir = root / fill_template(layout.pred_depth_dir, method=method, **BINDS)
    for i, (loc, rot) in enumerate(_poses(n_frames)):
        moved = tuple(scale * c + s for c, s in zip(loc, shift))
        write_cam(pred_cam_dir / f"{i:08d}.json", loc=moved, rot=rot)
        write_depth_exr(pred_depth_dir / f"{i:08d}.exr", value=depth_pred)
    return AssetDataset(root, layout)


def write_gt_mesh(root: Path, kind: str, extent=(4.0, 4.0, 0.1), center=(1.5, 0.0, 0.0)):
    """A flat gt slab, described in the dataset's Z-up world and written out Y-up as glTF is —
    so a case that forgets to rotate the mesh back gets a slab standing on its edge."""
    from inspection_eval.geometry import GLTF_TO_Z_UP

    box = trimesh.creation.box(extents=extent)
    box.apply_translation(center)
    to_gltf = np.eye(4)
    to_gltf[:3, :3] = GLTF_TO_Z_UP.T
    box.apply_transform(to_gltf)
    box.export(root / f"{root.name}_{kind}.glb")


# --- pose metric --------------------------------------------------------------------------


def test_pose_metrics_are_perfect_on_a_rescaled_copy(tmp_path):
    ds = case_asset(tmp_path, scale=3.0, shift=(10.0, 0.0, 0.0))
    case = EvalCase(ds, "da3", VARIANT)
    row = get("pose").fn(case, PoseOptions())

    assert case.transform.scale == pytest.approx(1 / 3.0, rel=1e-4)
    assert row["RRA@5 %"] == 100.0 and row["RTA@5 %"] == 100.0
    assert row["AUC@30"] == pytest.approx(1.0)
    assert row["ATE RMSE (m)"] == pytest.approx(0.0, abs=1e-6)
    assert row["Cam pairs"] == 3


def test_pose_metrics_penalise_a_wrong_pose(tmp_path):
    ds = case_asset(tmp_path, n_frames=5)
    layout = ds.layout
    bad = ds.root / fill_template(layout.pred_cam_dir, method="da3", **BINDS) / "00000002.json"
    bad.write_text(json.dumps({**json.loads(bad.read_text()), "loc": [2.0, 4.0, 5.0]}))

    row = get("pose").fn(EvalCase(ds, "da3", VARIANT), PoseOptions())
    assert row["RTA@5 %"] < 100.0
    assert row["ATE RMSE (m)"] > 0.1


def test_pose_pairs_all_uses_every_combination(tmp_path):
    ds = case_asset(tmp_path, n_frames=5)
    row = get("pose").fn(EvalCase(ds, "da3", VARIANT), PoseOptions(pairs="all"))
    assert row["Cam pairs"] == 10


def test_align_none_leaves_the_prediction_where_it_was(tmp_path):
    ds = case_asset(tmp_path, scale=3.0)
    row = get("pose").fn(EvalCase(ds, "da3", VARIANT, align_mode="none"), PoseOptions())
    assert row["Align scale"] == 1.0
    assert row["ATE RMSE (m)"] > 1.0  # the unresolved gauge now shows up as absolute error


# --- depth metric -------------------------------------------------------------------------


def test_depth_metrics_are_perfect_when_the_scale_matches_the_poses(tmp_path):
    # prediction lives at 1/2 scale: cameras and depth are both half of gt, consistently
    ds = case_asset(tmp_path, scale=0.5, depth_gt=6.0, depth_pred=3.0)
    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions())

    assert row["Depth scale"] == pytest.approx(2.0, rel=1e-4)
    assert row["RMSE (m)"] == pytest.approx(0.0, abs=1e-4)
    assert row["Accuracy @1.005"] == pytest.approx(1.0)
    assert row["Completeness @1.005"] == pytest.approx(1.0)
    assert row["Valid px ratio (gt)"] == 1.0


def test_depth_errors_show_up(tmp_path):
    ds = case_asset(tmp_path, depth_gt=5.0, depth_pred=6.0)
    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions())
    assert row["MAE (m)"] == pytest.approx(1.0, rel=1e-3)
    assert row["Relative MAE"] == pytest.approx(0.2, rel=1e-3)
    assert row["Accuracy @1.05"] == 0.0  # ratio 1.2 is past every threshold


def test_depth_median_scaling_forgives_a_global_scale(tmp_path):
    ds = case_asset(tmp_path, depth_gt=5.0, depth_pred=10.0)
    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions(scale="median"))
    assert row["RMSE (m)"] == pytest.approx(0.0, abs=1e-4)


def test_depth_max_frames_subsamples(tmp_path):
    ds = case_asset(tmp_path, n_frames=4)
    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions(max_frames=2))
    assert row["Frames"] == 2


# --- cloud metric -------------------------------------------------------------------------


def test_cloud_metrics_on_a_prediction_that_sits_on_the_gt_surface(tmp_path):
    ds = case_asset(tmp_path, n_frames=3, cloud=True)
    for kind in ("accuracy", "completeness"):
        write_gt_mesh(ds.root, kind)
    ds = AssetDataset(ds.root, ds.layout)  # re-scan, the meshes are new

    # a cloud lying exactly on the top face of the slab
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-0.5, 3.5, 500), rng.uniform(-2, 2, 500),
                           np.full(500, 0.05)])
    ply = ds.cloud_path(VARIANT, "da3")
    ply.write_bytes(trimesh.PointCloud(pts.astype(np.float32)).export(file_type="ply"))

    case = EvalCase(ds, "da3", VARIANT, align_mode="none", cache_dir=tmp_path / "cache")
    row = get("cloud").fn(case, CloudOptions(gt_points=20_000, thresholds=(0.1,)))

    assert row["Accuracy (m)"] < 0.05
    assert row["Precision @0.1m"] == pytest.approx(1.0)
    assert row["Cloud source"] == "fused"
    assert row["Pred points"] == 500


def test_cloud_metrics_punish_a_cloud_off_the_surface(tmp_path):
    ds = case_asset(tmp_path, n_frames=3, cloud=True)
    write_gt_mesh(ds.root, "accuracy")
    ds = AssetDataset(ds.root, ds.layout)

    pts = np.column_stack([np.linspace(0, 3, 200), np.zeros(200), np.full(200, 0.25)])
    ply = ds.cloud_path(VARIANT, "da3")
    ply.write_bytes(trimesh.PointCloud(pts.astype(np.float32)).export(file_type="ply"))

    row = get("cloud").fn(EvalCase(ds, "da3", VARIANT, "none", tmp_path / "cache"),
                          CloudOptions(gt_points=20_000, thresholds=(0.1,)))
    assert row["Accuracy (m)"] == pytest.approx(0.2, abs=0.02)  # 0.25m over a slab 0.05m thick
    assert row["Precision @0.1m"] == 0.0
    assert row["Recall @0.1m"] == 0.0


def test_cloud_outside_the_gt_region_scores_zero_rather_than_failing(tmp_path):
    ds = case_asset(tmp_path, n_frames=3, cloud=True)
    write_gt_mesh(ds.root, "accuracy")
    ds = AssetDataset(ds.root, ds.layout)

    far_away = np.full((50, 3), 100.0, dtype=np.float32)
    ds.cloud_path(VARIANT, "da3").write_bytes(
        trimesh.PointCloud(far_away).export(file_type="ply"))

    row = get("cloud").fn(EvalCase(ds, "da3", VARIANT, "none", tmp_path / "cache"),
                          CloudOptions(gt_points=5_000, thresholds=(0.1,)))
    assert row["Pred points"] == 0 and row["Cropped %"] == 100.0
    assert row["F-score @0.1m"] == 0.0 and np.isnan(row["Accuracy (m)"])


def test_cloud_falls_back_to_unprojecting_depth(tmp_path):
    ds = case_asset(tmp_path, n_frames=3, cloud=False, depth_gt=5.0, depth_pred=5.0)
    write_gt_mesh(ds.root, "accuracy", extent=(20.0, 20.0, 0.2), center=(0.0, 0.0, 0.0))
    ds = AssetDataset(ds.root, ds.layout)

    case = EvalCase(ds, "da3", VARIANT, align_mode="none", cache_dir=tmp_path / "cache")
    assert case.fused_cloud_path is None
    row = get("cloud").fn(case, CloudOptions(gt_points=20_000, thresholds=(0.5,),
                                             unproject_stride=1))
    assert row["Cloud source"] == "unprojected"
    assert row["Pred points"] == 3 * WH[0] * WH[1]  # cameras look down at the slab


# --- runner and reporting -----------------------------------------------------------------


def test_target_from_path_rejects_a_non_prediction_folder(tmp_path):
    (tmp_path / "somewhere").mkdir()
    with pytest.raises(ValueError):
        target_from_path(tmp_path / "somewhere")


def test_asset_or_predictions_dir_stands_for_every_method(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "asset")
    for method in ("da3", "vggt"):
        make_predictions(root, layout, method, BINDS)

    for path in (root, root / "predictions"):
        assert [t.method for t in targets_from_paths([path])] == ["da3", "vggt"]
    # a duplicate (the asset and one of its methods) is only evaluated once
    assert len(targets_from_paths([root, root / "predictions" / "da3"])) == 2


def test_target_from_path_recognizes_a_colmap_mode_folder(tmp_path):
    (tmp_path / "asset" / "colmap" / "no_gt").mkdir(parents=True)
    target = target_from_path(tmp_path / "asset" / "colmap" / "no_gt")
    assert target.method == "colmap/no_gt"
    assert target.dataset.root == tmp_path / "asset"


def test_asset_or_colmap_dir_stands_for_every_mode(tmp_path):
    for mode in ("gt_cams", "no_gt"):
        (tmp_path / "asset" / "colmap" / mode).mkdir(parents=True)
    for path in (tmp_path / "asset", tmp_path / "asset" / "colmap"):
        methods = sorted(t.method for t in targets_from_paths([path]))
        assert methods == ["colmap/gt_cams", "colmap/no_gt"]


def test_asset_root_expands_predictions_and_colmap_together(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "asset")
    make_predictions(root, layout, "da3", BINDS)
    (root / "colmap" / "no_gt").mkdir(parents=True)
    assert sorted(t.method for t in targets_from_paths([root])) == ["colmap/no_gt", "da3"]


@needs_pycolmap
def test_colmap_case_scores_depth_and_cloud(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "asset", trajs=("trajA",), illums=("sunny",), soils=("low",),
                      n_frames=3)  # gt depth is 5.0, written by make_asset
    make_colmap(root, layout, "no_gt", BINDS, n_frames=3, depth_value=3.0)
    ds = AssetDataset(root, layout)

    case = EvalCase(ds, "colmap/no_gt", VARIANT, align_mode="none")
    assert "depth" in case.capabilities
    assert case.fused_cloud_path is not None  # the fixture's fused.ply, discovered like any cloud
    row = get("depth").fn(case, DepthOptions(scale="none"))
    assert row["MAE (m)"] == pytest.approx(2.0, rel=1e-3)  # |5.0 gt - 3.0 pred|

    results = evaluate(targets_from_paths([root / "colmap" / "no_gt"]), ["depth"])
    assert len(results.tables["depth"]) == 1
    assert results.tables["depth"].rows[0]["Method"] == "colmap/no_gt"


def test_runner_scores_every_variant_of_a_method(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "asset", illums=("sunny", "night"), soils=("low", "high"))
    for illum in ("sunny", "night"):
        for soil in ("low", "high"):
            make_predictions(root, layout, "da3",
                             {"traj": "trajA", "illumination": illum, "soil": soil})

    target = target_from_path(root / "predictions" / "da3")
    assert target.method == "da3"

    results = evaluate([target], ["pose", "depth"])
    assert len(results.tables["pose"]) == 4
    assert len(results.tables["depth"]) == 4
    assert {r["Illumination"] for r in results.tables["pose"].rows} == {"sunny", "night"}
    assert results.tables["pose"].columns[:3] == ["Asset", "Method", "Traj"]


def test_runner_filters_and_skips_what_it_cannot_score(tmp_path):
    layout = Layout()
    root = make_asset(tmp_path / "asset", illums=("sunny", "night"), soils=("low",))
    for illum in ("sunny", "night"):
        make_predictions(root, layout, "da3",
                         {"traj": "trajA", "illumination": illum, "soil": "low"})

    results = evaluate([target_from_path(root / "predictions" / "da3")],
                       filters={"illumination": "sunny"})
    assert len(results.tables["pose"]) == 1
    # no gt mesh in this asset, so the cloud metric is skipped rather than failing the run
    assert len(results.tables["cloud"]) == 0
    assert [s.metric for s in results.skips] == ["cloud"]
    assert "no cloud" in results.skips[0].reason


def test_runner_reports_a_failing_metric_without_losing_the_others(tmp_path):
    ds = case_asset(tmp_path)
    results = evaluate([target_from_path(ds.root / "predictions" / "da3")],
                       overrides={"depth": {"scale": "nonsense"}})
    assert len(results.tables["pose"]) == 1
    assert len(results.tables["depth"]) == 0
    assert any(s.metric == "depth" and "ValueError" in s.reason for s in results.skips)


def test_option_overrides_are_typed(tmp_path):
    from inspection_eval.metrics import apply_overrides

    opts = apply_overrides(DepthOptions(), {"thresholds": "1.5,2", "max_frames": "3",
                                            "scale": "median"})
    assert opts.thresholds == (1.5, 2) and opts.max_frames == 3 and opts.scale == "median"
    assert apply_overrides(DepthOptions(), {"max_frames": "none"}).max_frames is None
    with pytest.raises(KeyError):
        apply_overrides(DepthOptions(), {"nope": "1"})


def test_every_registered_metric_declares_its_needs_and_options():
    for metric in REGISTRY.values():
        assert metric.options is not None and metric.default_options() is not None
        assert all(need in ("cameras", "depth", "cloud") for need in metric.needs)


def test_table_renders_and_writes(tmp_path):
    table = Table("pose")
    table.add({"Method": "da3", "AUC@30": 0.5})
    table.add({"Method": "vggt", "AUC@30": 0.25, "Extra %": 1.0})  # a new column appears late

    assert table.columns == ["Method", "AUC@30", "Extra %"]
    assert "| da3" in table.to_markdown()
    assert table.to_csv().splitlines()[0] == "Method,AUC@30,Extra %"
    assert r"Extra \%" in table.to_latex()

    written = table.write(tmp_path / "out", ("csv", "md", "tex"))
    assert {p.suffix for p in written} == {".csv", ".md", ".tex"}


# --- GPU OOM: columns survive even when nothing was ever scored ---------------------------
#
# A metric's columns are normally learned from the keys a real row returns (see report.py); an
# OOM case never calls `fn`, so a run where *every* case is OOM would otherwise never learn its
# metrics' columns at all. `_columns` (declared alongside `fn` in each metric module) is a pure
# function of options, so the runner can seed a fully-"OOM" row without ever touching data.


def test_cloud_columns_match_what_the_metric_actually_returns(tmp_path):
    ds = case_asset(tmp_path, cloud=True)
    for kind in ("accuracy", "completeness"):
        write_gt_mesh(ds.root, kind)
    ds = AssetDataset(ds.root, ds.layout)  # re-scan, the meshes are new

    o = CloudOptions(gt_points=20_000, thresholds=(0.1, 0.25))
    case = EvalCase(ds, "da3", VARIANT, align_mode="none", cache_dir=tmp_path / "cache")
    row = get("cloud").fn(case, o)
    assert set(row.keys()) == set(get("cloud").columns(o))


def test_depth_columns_match_what_the_metric_actually_returns(tmp_path):
    ds = case_asset(tmp_path)
    o = DepthOptions()
    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), o)
    assert set(row.keys()) == set(get("depth").columns(o))


def test_pose_columns_match_what_the_metric_actually_returns(tmp_path):
    ds = case_asset(tmp_path, n_frames=5)
    o = PoseOptions()
    row = get("pose").fn(EvalCase(ds, "da3", VARIANT), o)
    assert set(row.keys()) == set(get("pose").columns(o))


def oom_target(tmp_path: Path, method="da3") -> Target:
    """An asset whose only variant never produced a single prediction -- just the GPU-OOM marker
    a baseline leaves behind (`baselines.common.writer.write_oom`) -- so `evaluate` has to report
    it as OOM without ever calling a metric's `fn`, the corner this whole section guards."""
    from baselines.common.writer import oom_path, write_oom

    root = make_asset(tmp_path / "asset", trajs=("trajA",), illums=("sunny",), soils=("low",),
                      mesh_kinds=("accuracy", "completeness"))
    layout = Layout()
    write_oom(oom_path(root, layout, method, BINDS), "CUDA out of memory")
    return Target(AssetDataset(root, layout), method)


def test_an_all_oom_run_still_gets_every_column_of_the_cloud_metric(tmp_path):
    results = evaluate([oom_target(tmp_path)], ["cloud"])
    table = results.tables["cloud"]
    expected = get("cloud").columns(CloudOptions())

    assert len(table) == 1
    assert set(expected) <= set(table.columns)
    assert all(table.rows[0][c] == "OOM" for c in expected)


def test_an_all_oom_run_still_gets_every_column_of_the_depth_metric(tmp_path):
    results = evaluate([oom_target(tmp_path)], ["depth"])
    table = results.tables["depth"]
    expected = get("depth").columns(DepthOptions())

    assert len(table) == 1
    assert set(expected) <= set(table.columns)
    assert all(table.rows[0][c] == "OOM" for c in expected)


def test_an_all_oom_run_still_gets_every_column_of_the_pose_metric(tmp_path):
    results = evaluate([oom_target(tmp_path)], ["pose"])
    table = results.tables["pose"]
    expected = get("pose").columns(PoseOptions())

    assert len(table) == 1
    assert set(expected) <= set(table.columns)
    assert all(table.rows[0][c] == "OOM" for c in expected)


def test_an_all_oom_runs_written_csv_keeps_every_column_too(tmp_path):
    from inspection_eval.report import read_tables

    results = evaluate([oom_target(tmp_path)], ["depth"])
    results.write(tmp_path / "out", ("csv",))

    back = read_tables(tmp_path / "out")["depth"]
    expected = get("depth").columns(DepthOptions())
    assert set(expected) <= set(back.columns)
    assert all(back.rows[0][c] == "OOM" for c in expected)


# --- holes in the data --------------------------------------------------------------------
#
# Every metric aggregates over data that is allowed to be incomplete, and the failure mode these
# guard is silent: one nan pixel, one stationary camera pair or one nan vertex in a .ply used to
# turn a whole row of the results table into "nan" with nothing saying why.


def test_stats_reduce_over_the_defined_entries_only():
    from inspection_eval.metrics import stats

    values = [1.0, np.nan, 3.0, np.inf]
    assert stats.mean(values) == pytest.approx(2.0)
    assert stats.median(values) == pytest.approx(2.0)
    assert stats.rms([3.0, np.nan, 4.0]) == pytest.approx(np.sqrt(12.5))
    assert stats.fraction_below(values, 2.0) == pytest.approx(0.5)  # 1 of the 2 defined


def test_stats_on_nothing_defined_return_the_callers_default():
    from inspection_eval.metrics import stats

    with np.errstate(all="raise"):  # np.nanmean would warn here; these must not
        assert np.isnan(stats.mean([np.nan, np.nan]))
        assert np.isnan(stats.median([]))
        assert stats.fraction_below([np.nan], 1.0, default=0.0) == 0.0


def test_depth_holes_cost_the_valid_ratio_not_the_error(tmp_path):
    """Half the predicted map is nan and a sixth is a nonsensical zero; the error over what is
    left must still be the exact answer, and the ratios must show what was missing."""
    ds = case_asset(tmp_path, depth_gt=5.0, depth_pred=6.0, n_frames=2)
    pred = np.full((WH[1], WH[0]), 6.0, np.float32)
    pred[:3] = np.nan  # of 6 rows: 3 nan, 1 zero (unusable: the metrics divide by depth), 2 good
    pred[3] = 0.0
    pred_dir = ds.root / fill_template(ds.layout.pred_depth_dir, method="da3", **BINDS)
    for path in sorted(pred_dir.glob("*.exr")):
        write_depth_exr(path, value=pred)

    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions())

    assert row["MAE (m)"] == pytest.approx(1.0, rel=1e-3)  # not nan, and not diluted
    assert row["Relative MAE"] == pytest.approx(0.2, rel=1e-3)
    assert np.isfinite(row["Scale invariant MSE"])
    assert row["Valid px ratio (est)"] == pytest.approx(2 / 6)
    assert row["Valid px ratio (gt)"] == pytest.approx(1.0)


def test_depth_median_scaling_survives_a_frame_it_cannot_scale(tmp_path):
    """An all-nan predicted frame has no median ratio to fit; it must not nan the sequence."""
    ds = case_asset(tmp_path, depth_gt=5.0, depth_pred=5.0, n_frames=3)
    pred_dir = ds.root / fill_template(ds.layout.pred_depth_dir, method="da3", **BINDS)
    write_depth_exr(sorted(pred_dir.glob("*.exr"))[0],
                    value=np.full((WH[1], WH[0]), np.nan, np.float32))

    row = get("depth").fn(EvalCase(ds, "da3", VARIANT), DepthOptions(scale="median"))
    assert row["RMSE (m)"] == pytest.approx(0.0, abs=1e-4)
    assert row["Frames"] == 3 and row["Valid px ratio (est)"] == pytest.approx(2 / 3)


def test_pose_metrics_drop_a_pair_with_no_baseline_rather_than_failing_it(tmp_path):
    """Two frames at the same place have no relative translation direction. Scoring that pair as
    a miss would punish a method for a stationary stretch it reproduced perfectly."""
    ds = case_asset(tmp_path, n_frames=4)
    for template in (ds.layout.cam_dir, ds.layout.pred_cam_dir):  # in both trajectories
        cam_dir = ds.root / fill_template(template, method="da3", **BINDS)
        write_cam(cam_dir / "00000001.json", loc=(0.0, 0.0, 5.0))  # same place as frame 0

    row = get("pose").fn(EvalCase(ds, "da3", VARIANT, align_mode="none"), PoseOptions())

    assert row["Cam pairs"] == 3 and row["Undefined pairs"] == 1
    assert row["RTA@5 %"] == pytest.approx(100.0)  # the 2 scorable pairs, both exact
    assert row["RRA@5 %"] == pytest.approx(100.0)
    assert np.isfinite(row["AUC@30"]) and row["ATE RMSE (m)"] == pytest.approx(0.0, abs=1e-6)


def test_a_camera_with_a_nan_pose_is_dropped_like_a_missing_one(tmp_path):
    ds = case_asset(tmp_path, n_frames=4)
    pred_cam_dir = ds.root / fill_template(ds.layout.pred_cam_dir, method="da3", **BINDS)
    write_cam(pred_cam_dir / "00000002.json", loc=(float("nan"), 0.0, 5.0))

    case = EvalCase(ds, "da3", VARIANT)
    assert 2 not in case.raw_pred_cams and case.cam_frames == [0, 1, 3]

    row = get("pose").fn(case, PoseOptions())
    assert row["Frames"] == 3
    assert np.isfinite(row["Align scale"]) and np.isfinite(row["ATE RMSE (m)"])


def test_cloud_metrics_ignore_nan_vertices_in_a_fused_ply(tmp_path):
    """A nan vertex poisons a KD-tree query; it has to go before any distance is taken."""
    ds = case_asset(tmp_path, n_frames=3, cloud=True)
    write_gt_mesh(ds.root, "accuracy")
    ds = AssetDataset(ds.root, ds.layout)

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-0.5, 3.5, 300), rng.uniform(-2, 2, 300),
                           np.full(300, 0.05)]).astype(np.float32)
    pts[::10] = np.nan  # 30 of the 300 are unusable
    ds.cloud_path(VARIANT, "da3").write_bytes(
        trimesh.PointCloud(pts).export(file_type="ply"))

    row = get("cloud").fn(EvalCase(ds, "da3", VARIANT, "none", tmp_path / "cache"),
                          CloudOptions(gt_points=20_000, thresholds=(0.1,)))

    assert row["Pred points"] == 270
    assert np.isfinite(row["Accuracy (m)"]) and row["Accuracy (m)"] < 0.05
    assert row["Precision @0.1m"] == pytest.approx(1.0)


# --- missing frames (frame_gaps, and eval.py's --allow-missing-frames) ---------------------
#
# EvalCase already tolerates an rgb frame with no matching depth/cam on either side -- it just
# drops out of `depth_frames`/`cam_frames` -- so the thing under test here is the *preflight*:
# by default eval.py should refuse to run quietly on an incomplete case, and only proceed once
# the flag says that is fine.


def test_frame_gaps_reports_a_missing_depth_and_cam_file(tmp_path):
    ds = case_asset(tmp_path, n_frames=4)
    gt_depth_dir = ds.root / ds.layout.depth_dir.format(**BINDS)
    (gt_depth_dir / "00000002.exr").unlink()  # frame 2's gt depth just isn't there
    pred_cam_dir = ds.root / fill_template(ds.layout.pred_cam_dir, method="da3", **BINDS)
    (pred_cam_dir / "00000001.json").unlink()  # frame 1's predicted pose just isn't there

    gaps = frame_gaps([target_from_path(ds.root / "predictions" / "da3")])

    by_field = {(g.source, g.field): g.indices for g in gaps}
    assert by_field[("gt", "depth")] == [2]
    assert by_field[("da3", "cam")] == [1]


def test_frame_gaps_is_quiet_on_a_complete_case(tmp_path):
    ds = case_asset(tmp_path, n_frames=4)
    assert frame_gaps([target_from_path(ds.root / "predictions" / "da3")]) == []


def test_frame_gaps_does_not_flag_a_field_a_source_never_had(tmp_path):
    # a method that predicted no depth at all for a variant isn't "missing" it frame by frame --
    # that would be indistinguishable from every real gap, so it must not be reported as one.
    ds = case_asset(tmp_path, n_frames=4)
    pred_depth_dir = ds.root / fill_template(ds.layout.pred_depth_dir, method="da3", **BINDS)
    for f in pred_depth_dir.glob("*.exr"):
        f.unlink()

    gaps = frame_gaps([target_from_path(ds.root / "predictions" / "da3")])
    assert not any(g.source == "da3" and g.field == "depth" for g in gaps)


def test_frame_gaps_checks_gt_once_across_methods_that_share_it(tmp_path):
    ds = case_asset(tmp_path, n_frames=4)
    make_predictions(ds.root, ds.layout, "vggt", BINDS, n_frames=4)
    (ds.root / ds.layout.depth_dir.format(**BINDS) / "00000002.exr").unlink()

    gaps = frame_gaps(targets_from_paths([ds.root / "predictions"]))
    assert sum(1 for g in gaps if g.source == "gt") == 1


def test_eval_cli_refuses_a_case_with_missing_frames_by_default(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=4)
    pred_depth_dir = ds.root / fill_template(ds.layout.pred_depth_dir, method="da3", **BINDS)
    (pred_depth_dir / "00000003.exr").unlink()

    with pytest.raises(SystemExit) as exc:
        eval_cli.main([str(ds.root / "predictions" / "da3"), "--out", str(tmp_path / "out")])

    message = str(exc.value)
    assert "--allow-missing-frames" in message
    assert "da3" in message and "depth" in message and "3" in message
    assert not (tmp_path / "out").exists()


def test_eval_cli_allow_missing_frames_runs_anyway(tmp_path):
    import csv

    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=4)
    pred_depth_dir = ds.root / fill_template(ds.layout.pred_depth_dir, method="da3", **BINDS)
    (pred_depth_dir / "00000003.exr").unlink()

    rc = eval_cli.main([str(ds.root / "predictions" / "da3"), "--allow-missing-frames",
                        "--out", str(tmp_path / "out"), "--no-timestamp", "-q"])

    assert rc == 0
    rows = list(csv.DictReader((tmp_path / "out" / "depth.csv").open()))
    assert rows[0]["Frames"] == "3"  # frame 3 dropped, the other 3 still scored


# --- progress and streaming ------------------------------------------------------------------


def multi_variant_asset(tmp_path: Path, illums=("sunny", "night")) -> Path:
    """An asset with one case per illumination, so a run has several cases to march through."""
    layout = Layout()
    root = make_asset(tmp_path / "asset", illums=illums, soils=("low",), n_frames=2)
    for illum in illums:
        make_predictions(root, layout, "da3",
                         {"traj": "trajA", "illumination": illum, "soil": "low"}, n_frames=2)
    return root / "predictions" / "da3"


def test_evaluate_counts_every_case_before_it_starts(tmp_path):
    events = []
    evaluate(targets_from_paths([multi_variant_asset(tmp_path)]), ["pose"],
             on_event=lambda kind, message, **info: events.append((kind, info)))

    assert ("plan", {"total": 2, "metrics": ["pose"], "done": 0}) in events
    assert [i["index"] for k, i in events if k == "case"] == [1, 2]
    assert ("done", {"index": 2, "total": 2}) in events


def test_evaluate_hands_over_the_results_after_every_case(tmp_path):
    rows = []
    results = evaluate(targets_from_paths([multi_variant_asset(tmp_path)]), ["pose"],
                       after_case=lambda partial: rows.append(len(partial.tables["pose"])))

    assert rows == [1, 2]  # grows as it goes, and is the same object that comes back
    assert len(results.tables["pose"]) == 2


def test_progress_bar_counts_and_fills():
    import eval as eval_cli

    filled, empty = eval_cli.BLOCKS
    assert eval_cli.progress_bar(1, 4, width=4) == f"[1/4] {filled}{empty * 3}"
    assert eval_cli.progress_bar(4, 4, width=4) == f"[4/4] {filled * 4}"
    assert eval_cli.progress_bar(3, 12, width=4).startswith("[ 3/12] ")  # count right-aligned


def test_progress_prints_one_counter_line_per_case(capsys):
    import eval as eval_cli

    progress = eval_cli.Progress(quiet=False)
    progress("plan", "2 case(s) x 1 metric(s)", total=2, metrics=["pose"])
    progress("case", "asset / da3 / v1", index=1, total=2)
    progress("metric", "  pose: ok")
    progress("case", "asset / da3 / v2", index=2, total=2)
    progress("done", "2 case(s) scored", index=2, total=2)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "2 case(s) x 1 metric(s)"
    assert lines[1].startswith("[1/2] ") and lines[1].endswith(" asset / da3 / v1")
    assert lines[2] == "  pose: ok"
    assert lines[3].startswith("[2/2] ")


def test_progress_redraws_one_line_when_quiet_on_a_terminal():
    import io

    import eval as eval_cli

    stream = io.StringIO()
    stream.isatty = lambda: True
    progress = eval_cli.Progress(quiet=True, stream=stream)
    progress("plan", "2 case(s)", total=2, metrics=["pose"])
    progress("case", "asset / da3 / v1", index=1, total=2)
    progress("metric", "  pose: ok")  # swallowed: the bar owns the line
    progress("done", "2 case(s) scored", index=2, total=2)

    out = stream.getvalue()
    assert out.count("\r") == 2 and out.count("\n") == 1 and out.endswith("\n")
    assert "pose: ok" not in out and "2 case(s)" not in out


def test_progress_stays_silent_when_quiet_and_piped(capsys):
    import eval as eval_cli

    progress = eval_cli.Progress(quiet=True)  # capsys' stdout is not a tty
    progress("plan", "2 case(s)", total=2, metrics=["pose"])
    progress("case", "asset / da3 / v1", index=1, total=2)
    progress("done", "2 case(s) scored", index=2, total=2)

    assert capsys.readouterr().out == ""


def test_eval_cli_writes_each_case_as_it_is_scored(tmp_path):
    """The second case must find the first one's row already on disk."""
    import csv

    import eval as eval_cli
    from inspection_eval.metrics import REGISTRY, register

    out = tmp_path / "out"
    seen = []

    @register("spy")
    def spy(case, options):
        seen.append(sorted(p.read_text() for p in out.rglob("spy.csv")))
        return {"Case": len(seen)}

    try:
        rc = eval_cli.main([str(multi_variant_asset(tmp_path)), "--out", str(out),
                            "--metrics", "spy", "--format", "csv", "-q"])
    finally:
        REGISTRY.pop("spy")

    assert rc == 0
    assert seen[0] == []  # nothing written yet on the first case
    first = seen[1][0].strip().splitlines()  # the second case sees the first one's csv
    assert len(first) == 2 and first[0].endswith("Case") and first[1].endswith(",1")
    rows = list(csv.DictReader(next(out.rglob("spy.csv")).open()))
    assert [r["Case"] for r in rows] == ["1", "2"]


def test_eval_cli_no_stream_writes_only_at_the_end(tmp_path):
    import eval as eval_cli
    from inspection_eval.metrics import REGISTRY, register

    out = tmp_path / "out"
    seen = []

    @register("spy_end")
    def spy_end(case, options):
        seen.append(list(out.rglob("spy_end.csv")))
        return {"Case": len(seen)}

    try:
        rc = eval_cli.main([str(multi_variant_asset(tmp_path)), "--out", str(out), "--no-stream",
                            "--metrics", "spy_end", "--format", "csv", "-q"])
    finally:
        REGISTRY.pop("spy_end")

    assert rc == 0
    assert seen == [[], []]  # nothing on disk while the run is going
    assert len(list(out.rglob("spy_end.csv"))) == 1


# --- reusing a named folder ------------------------------------------------------------------


def test_eval_cli_writes_into_the_folder_named_by_name(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)
    rc = eval_cli.main([str(ds.root / "predictions" / "da3"), "--out", str(tmp_path / "out"),
                        "--name", "run7", "-q"])

    assert rc == 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["run7"]
    assert (tmp_path / "out" / "run7" / "depth.csv").is_file()


def test_read_tables_types_the_numbers_back(tmp_path):
    from inspection_eval.report import Table, read_tables

    table = Table("pose")
    table.add({"Asset": "old_house", "ATE RMSE (m)": 0.1023, "Frames": 12})
    table.write(tmp_path, ("csv",))

    back = read_tables(tmp_path)["pose"]
    assert back.columns == ["Asset", "ATE RMSE (m)", "Frames"]
    assert back.rows == [{"Asset": "old_house", "ATE RMSE (m)": 0.1023, "Frames": 12}]
    assert back.to_markdown() == table.to_markdown()  # so a resumed run renders like one run


def test_evaluate_resumes_past_the_cases_a_previous_run_finished(tmp_path):
    from inspection_eval.report import read_tables

    targets = targets_from_paths([multi_variant_asset(tmp_path)])
    first = evaluate(targets, ["pose"], filters={"illumination": "sunny"})
    first.write(tmp_path / "out", ("csv",))
    assert len(first.tables["pose"]) == 1

    scored = []
    second = evaluate(targets, ["pose"], resume=read_tables(tmp_path / "out"),
                      on_event=lambda kind, message, **info: scored.append(kind))

    assert len(second.tables["pose"]) == 2  # the kept row plus the one it was missing
    assert scored.count("resumed") == 1 and scored.count("metric") == 1
    assert {r["Illumination"] for r in second.tables["pose"].rows} == {"sunny", "night"}


def test_eval_cli_continues_a_named_folder_from_where_it_stopped(tmp_path):
    import csv

    import eval as eval_cli

    predictions = multi_variant_asset(tmp_path)
    out = tmp_path / "out" / "run7"
    argv = [str(predictions), "--out", str(tmp_path / "out"), "--name", "run7",
            "--metrics", "pose", "--format", "csv", "-q"]

    assert eval_cli.main(argv + ["--illumination", "sunny"]) == 0  # a run cut short
    assert len(list(csv.DictReader((out / "pose.csv").open()))) == 1

    assert eval_cli.main(argv + ["--resume"]) == 0  # the rest of it

    rows = list(csv.DictReader((out / "pose.csv").open()))
    assert [r["Illumination"] for r in rows] == ["sunny", "night"]  # kept row first, then the new


def test_eval_cli_overwrite_starts_the_named_folder_again(tmp_path):
    import csv

    import eval as eval_cli

    predictions = multi_variant_asset(tmp_path)
    out = tmp_path / "out" / "run7"
    argv = [str(predictions), "--out", str(tmp_path / "out"), "--name", "run7",
            "--metrics", "pose", "--format", "csv", "-q"]

    assert eval_cli.main(argv + ["--illumination", "sunny"]) == 0
    (out / "notes.md").write_text("mine\n")  # not a table: left alone

    assert eval_cli.main(argv + ["--overwrite", "--illumination", "night"]) == 0

    rows = list(csv.DictReader((out / "pose.csv").open()))
    assert [r["Illumination"] for r in rows] == ["night"]  # the sunny row is gone, not kept
    assert (out / "notes.md").is_file()


def test_eval_cli_asks_what_to_do_with_an_existing_folder(tmp_path, monkeypatch):
    import csv

    import eval as eval_cli

    predictions = multi_variant_asset(tmp_path)
    out = tmp_path / "out" / "run7"
    argv = [str(predictions), "--out", str(tmp_path / "out"), "--name", "run7",
            "--metrics", "pose", "--format", "csv", "-q"]
    assert eval_cli.main(argv + ["--illumination", "sunny"]) == 0
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    monkeypatch.setattr("builtins.input", lambda *a: "q")  # quit changes nothing
    with pytest.raises(SystemExit):
        eval_cli.main(argv)
    assert len(list(csv.DictReader((out / "pose.csv").open()))) == 1

    monkeypatch.setattr("builtins.input", lambda *a: "")  # default: continue
    assert eval_cli.main(argv) == 0
    assert len(list(csv.DictReader((out / "pose.csv").open()))) == 2

    monkeypatch.setattr("builtins.input", lambda *a: "o")  # overwrite
    assert eval_cli.main(argv + ["--illumination", "night"]) == 0
    assert len(list(csv.DictReader((out / "pose.csv").open()))) == 1


def test_eval_cli_keeps_results_when_asked_nothing(tmp_path, capsys):
    """No terminal to answer the question: continue, because that destroys nothing."""
    import csv

    import eval as eval_cli

    predictions = multi_variant_asset(tmp_path)
    out = tmp_path / "out" / "run7"
    argv = [str(predictions), "--out", str(tmp_path / "out"), "--name", "run7",
            "--metrics", "pose", "--format", "csv", "-q"]
    assert eval_cli.main(argv + ["--illumination", "sunny"]) == 0

    assert eval_cli.main(argv) == 0  # capsys' stdin is not a tty

    assert len(list(csv.DictReader((out / "pose.csv").open()))) == 2
    assert "continuing from there" in capsys.readouterr().out


# --- eval.py's output folder naming ---------------------------------------------------------


def test_run_label_flattens_a_colmap_methods_slash(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)  # any dataset root; run_label never reads its contents
    (ds.root / "colmap" / "no_gt").mkdir(parents=True)
    targets = targets_from_paths([ds.root / "colmap" / "no_gt"])
    assert eval_cli.run_label(targets) == "asset-colmap-no_gt"  # not nested folders on disk


def test_run_label_names_a_single_asset_and_method(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)
    targets = targets_from_paths([ds.root / "predictions" / "da3"])
    assert eval_cli.run_label(targets) == "asset-da3"


def test_run_label_lists_every_method_of_a_wildcard_run(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)
    make_predictions(ds.root, ds.layout, "vggt", BINDS, n_frames=2)
    targets = targets_from_paths([ds.root / "predictions"])
    assert eval_cli.run_label(targets) == "asset-da3+vggt"


def test_run_label_falls_back_to_a_count_past_the_limit(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)  # already writes method "da3"
    for method in ("m1", "m2", "m3"):
        make_predictions(ds.root, ds.layout, method, BINDS, n_frames=2)
    targets = targets_from_paths([ds.root / "predictions"])
    assert eval_cli.run_label(targets) == "asset-4-methods"


def test_eval_cli_writes_into_a_named_timestamped_subfolder(tmp_path):
    import eval as eval_cli

    ds = case_asset(tmp_path, n_frames=2)
    rc = eval_cli.main([str(ds.root / "predictions" / "da3"), "--out", str(tmp_path / "out"),
                        "-q"])
    assert rc == 0

    subdirs = list((tmp_path / "out").iterdir())
    assert len(subdirs) == 1
    assert subdirs[0].name.startswith("asset-da3-")
    assert (subdirs[0] / "depth.csv").is_file()
