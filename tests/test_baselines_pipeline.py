"""Config loading + the generic run driver, exercised with a fake baseline (no model needed)."""

import dataclasses

import numpy as np
import pytest
import yaml

from baselines.common.config import (
    DataConfig, DatasetSelection, OutputSpec, RunConfig, load_run_config,
)
from baselines import registry
from baselines.common.runner import Baseline, cli_main, run_baseline
from inspection_eval import targets_from_paths
from inspection_viz import AssetDataset, Layout, Variant
from inspection_viz.config import fill_template
from tests.conftest import make_predictions


@dataclasses.dataclass
class FakeModelConfig:
    value: float = 1.0


class ConstantDepth(Baseline):
    """Writes a constant depth map and the gt camera for every frame it is given."""

    name = "fake"
    Config = FakeModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.sequences = []
        self.setup_calls = 0

    def setup(self):
        self.setup_calls += 1

    def run(self, frames, out):
        self.sequences.append(out.variant)
        for f in frames:
            cam = f.camera()
            out.write_frame(f.index, np.full((6, 8), self.config.value, np.float32),
                            cam.c2w, cam.K, intrinsics_width=8)
        if out.spec.cloud:
            out.write_cloud(np.zeros((4, 3), np.float32))


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload))
    return path


# --- data config ----------------------------------------------------------------------------


def test_asset_and_root_are_the_same_field():
    assert DataConfig(asset="/a").root == "/a"
    assert DataConfig(root="/a").asset == "/a"  # pre-`asset` configs still load
    assert DatasetSelection is DataConfig
    with pytest.raises(ValueError):
        DataConfig(asset="/a", root="/b")


def test_null_axis_selects_every_value(asset):
    ds = AssetDataset(asset)  # trajA x {sunny, night} x {low, high}
    assert len(DataConfig(asset=str(asset)).variants(ds)) == 4
    assert DataConfig(asset=str(asset), illumination="sunny").variants(ds) == [
        Variant("trajA", "sunny", "high"), Variant("trajA", "sunny", "low")]
    assert len(DataConfig(asset=str(asset), illumination=["sunny", "night"],
                          soil="low").variants(ds)) == 2
    assert DataConfig(asset=str(asset), illumination="dusk").variants(ds) == []


def test_a_concrete_sequence_still_resolves_through_variant(asset):
    cfg = DataConfig(asset=str(asset), trajectory="trajA", illumination="sunny", soil="low")
    assert cfg.variant() == Variant("trajA", "sunny", "low")  # what pre-split runners call
    assert cfg.variants(AssetDataset(asset)) == [cfg.variant()]
    with pytest.raises(ValueError):
        DataConfig(illumination=["sunny", "night"]).variant()


def test_frame_range_and_stride(asset):
    ds = AssetDataset(asset)
    seq = ds.sequence(Variant("trajA", "sunny", "low"))
    assert [f.index for f in DataConfig(frame_stride=2).select(seq)] == [0, 2]
    assert [f.index for f in DataConfig(frame_start=1, frame_end=2).select(seq)] == [1]


# --- config loading -------------------------------------------------------------------------


def test_split_configs(tmp_path):
    data = write_yaml(tmp_path / "d.yaml", {"asset": "/a", "illumination": ["sunny", "night"]})
    model = write_yaml(tmp_path / "m.yaml",
                       {"model": {"value": 3.0}, "output": {"method": "x", "cloud": False}})
    cfg = load_run_config(FakeModelConfig, data=data, model=model)
    assert cfg.data.asset == "/a" and cfg.data.illumination == ["sunny", "night"]
    assert cfg.model.value == 3.0
    assert cfg.output.method == "x" and cfg.output.cloud is False


def test_combined_config_still_loads(tmp_path):
    combined = write_yaml(tmp_path / "c.yaml", {
        "dataset": {"root": "/a", "trajectory": "trajA", "frame_stride": 2},
        "model": {"value": 2.0},
        "output": {"method": "da3", "overwrite": True},
    })
    cfg = load_run_config(FakeModelConfig, config=combined)
    assert cfg.data.asset == "/a" and cfg.data.frame_stride == 2
    assert cfg.model.value == 2.0 and cfg.output.overwrite is True


def test_data_flag_overrides_the_combined_half(tmp_path):
    combined = write_yaml(tmp_path / "c.yaml",
                          {"dataset": {"root": "/a"}, "output": {"method": "da3"}})
    data = write_yaml(tmp_path / "d.yaml", {"dataset": {"asset": "/b"}})  # still wrapped: fine
    cfg = load_run_config(FakeModelConfig, config=combined, data=data)
    assert cfg.data.asset == "/b" and cfg.output.method == "da3"


def test_dataset_section_in_a_baseline_config_is_rejected(tmp_path):
    model = write_yaml(tmp_path / "m.yaml", {"dataset": {"asset": "/a"}, "model": {}})
    with pytest.raises(ValueError, match="data config"):
        load_run_config(FakeModelConfig, model=model)


def test_defaults_when_no_baseline_config_is_given(tmp_path):
    data = write_yaml(tmp_path / "d.yaml", {"asset": "/a"})
    cfg = load_run_config(FakeModelConfig, data=data)
    assert cfg.model.value == 1.0 and cfg.output == OutputSpec()


# --- running --------------------------------------------------------------------------------


def run(asset, baseline=None, **data_kwargs):
    baseline = baseline or ConstantDepth(FakeModelConfig())
    output = data_kwargs.pop("output", OutputSpec(cloud=False))
    cfg = RunConfig(DataConfig(asset=str(asset), **data_kwargs), baseline.config, output)
    return baseline, run_baseline(baseline, cfg, say=lambda *_: None)


def test_runs_every_selected_sequence_with_one_setup(asset):
    baseline, done = run(asset)
    assert len(done) == 4 and baseline.setup_calls == 1  # model loaded once for the whole run
    depth_dir = asset / fill_template(Layout().pred_depth_dir, method="fake",
                                      traj="trajA", illumination="sunny", soil="low")
    assert len(list(depth_dir.iterdir())) == 3
    assert (asset / "predictions/fake/config.yaml").is_file()
    # the run record sits inside predictions/<method>/ without being mistaken for data there
    ds = AssetDataset(asset)
    assert ds.methods == ["fake"] and len(ds.prediction_variants("fake")) == 4
    assert [t.method for t in targets_from_paths([asset])] == ["fake"]


def test_run_config_records_the_run(asset):
    _, done = run(asset, illumination="sunny", frame_stride=2,
                  output=OutputSpec(method="fake_v2", cloud=False))
    recorded = yaml.safe_load((asset / "predictions/fake_v2/config.yaml").read_text())
    assert recorded["baseline"] == "fake" and recorded["method"] == "fake_v2"
    assert recorded["data"]["illumination"] == "sunny" and recorded["data"]["frame_stride"] == 2
    assert "root" not in recorded["data"]  # the dump uses the current name only
    assert len(recorded["variants"]) == len(done) == 2
    assert recorded["output"]["method"] == "fake_v2"
    assert recorded["model"] == {"value": 1.0}


def test_existing_sequences_are_skipped_unless_overwrite(asset):
    layout = Layout()
    make_predictions(asset, layout, "fake",
                     {"traj": "trajA", "illumination": "sunny", "soil": "low"}, n_frames=3)
    _, done = run(asset)
    assert Variant("trajA", "sunny", "low") not in done and len(done) == 3

    baseline, done = run(asset, output=OutputSpec(overwrite=True, cloud=False))
    assert len(done) == 4


def test_nothing_to_run_is_an_error(asset):
    run(asset)
    with pytest.raises(SystemExit, match="already have predictions"):
        run(asset)
    with pytest.raises(SystemExit, match="no sequences"):
        run(asset, illumination="dusk")


def test_cli_main_runs_from_the_two_config_files(asset, tmp_path):
    data = write_yaml(tmp_path / "d.yaml",
                      {"asset": str(asset), "illumination": "sunny", "soil": "low"})
    model = write_yaml(tmp_path / "m.yaml", {"model": {"value": 2.0}, "output": {"cloud": False}})
    cli_main(ConstantDepth, argv=["--data", str(data), "--model", str(model)])
    recorded = yaml.safe_load((asset / "predictions/fake/config.yaml").read_text())
    assert recorded["model"] == {"value": 2.0} and len(recorded["variants"]) == 1


def test_cli_main_needs_a_data_config(asset):
    with pytest.raises(SystemExit, match="nothing to run"):
        cli_main(ConstantDepth, argv=[])


def test_registry_rejects_an_unknown_baseline():
    assert "da3" in registry.available()
    with pytest.raises(SystemExit, match="unknown baseline"):
        registry.get("no_such_baseline")


def test_min_confidence_masks_low_confidence_pixels(asset):
    from inspection_viz import io

    class Confident(ConstantDepth):
        def run(self, frames, out):
            for f in frames:
                cam = f.camera()
                conf = np.zeros((6, 8), np.float32)
                conf[0] = 1.0
                out.write_frame(f.index, np.ones((6, 8), np.float32), cam.c2w, cam.K,
                                intrinsics_width=8, conf=conf)

    baseline = Confident(FakeModelConfig())
    run(asset, baseline=baseline, illumination="sunny", soil="low",
        output=OutputSpec(min_confidence=0.5, cloud=False))
    depth_dir = asset / fill_template(Layout().pred_depth_dir, method="fake",
                                      traj="trajA", illumination="sunny", soil="low")
    depth = io.read_depth(sorted(depth_dir.iterdir())[0])
    assert np.all(depth[0] == 1.0) and np.all(np.isnan(depth[1:]))


def test_a_baseline_that_wants_no_confidence_map_still_errors_loudly(asset):
    with pytest.raises(SystemExit, match="min_confidence"):
        run(asset, illumination="sunny", soil="low",
            output=OutputSpec(min_confidence=0.5, cloud=False))
