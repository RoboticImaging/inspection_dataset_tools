from pathlib import Path

import pytest

from conftest import make_asset
from inspection_eval.report import Table


def _write(out_root: Path, asset: str, method: str, metric: str, rows: list[dict],
          suffix: str = "") -> None:
    table = Table(metric)
    for row in rows:
        table.add(row)
    table.write(out_root / asset / method, ("csv",), suffix=suffix)


def _cells(row: str) -> list[str]:
    """`row` split back into [label, office, bridge, ship, crane, old_house, glass_house, unfin]
    -- robust to however many spaces a blank cell renders as, unlike hand-counting them."""
    return row.removesuffix(r" \\").split(" & ")


# --- _latest_metric_csvs -----------------------------------------------------------------------


def test_latest_metric_csvs_prefers_the_newest_timestamp(tmp_path):
    import latex_results_table as lrt

    method_dir = tmp_path / "asset" / "da3"
    _write(tmp_path, "asset", "da3", "pose", [{"Asset": "asset"}], suffix="_20260101-000000")
    _write(tmp_path, "asset", "da3", "pose", [{"Asset": "asset"}], suffix="_20260301-000000")
    _write(tmp_path, "asset", "da3", "pose", [{"Asset": "asset"}], suffix="_20260201-000000")

    found = lrt._latest_metric_csvs(method_dir)
    assert found["pose"].name == "pose_20260301-000000.csv"


def test_latest_metric_csvs_prefers_a_bare_file_over_any_timestamp(tmp_path):
    import latex_results_table as lrt

    method_dir = tmp_path / "asset" / "da3"
    _write(tmp_path, "asset", "da3", "pose", [{"Asset": "asset"}], suffix="_20990101-000000")
    _write(tmp_path, "asset", "da3", "pose", [{"Asset": "asset"}])

    found = lrt._latest_metric_csvs(method_dir)
    assert found["pose"].name == "pose.csv"


# --- _format_cell (completeness-agnostic: it only ever sees an already-complete combo) ---------


def test_format_cell_averages_every_scored_variant(tmp_path):
    import latex_results_table as lrt

    _write(tmp_path, "asset", "da3", "pose", [{"AUC@30": 0.8}, {"AUC@30": 0.6}])
    _write(tmp_path, "asset", "da3", "cloud", [{"Chamfer (m)": 0.10}, {"Chamfer (m)": 0.20}])

    cell = lrt._format_cell(tmp_path / "asset" / "da3", lrt.TABLE_SPECS["cloud"], decimals=3)
    assert cell == "0.700 / 0.150"


def test_format_cell_is_oom_when_every_row_is_oom(tmp_path):
    import latex_results_table as lrt

    _write(tmp_path, "asset", "da3", "pose", [{"AUC@30": "OOM"}, {"AUC@30": "OOM"}])
    _write(tmp_path, "asset", "da3", "cloud", [{"Chamfer (m)": "OOM"}, {"Chamfer (m)": "OOM"}])

    cell = lrt._format_cell(tmp_path / "asset" / "da3", lrt.TABLE_SPECS["cloud"], decimals=3)
    assert cell == "OOM"


def test_format_cell_brackets_a_partial_oom_average(tmp_path):
    """A resumed hierarchical run can legitimately leave one scene's variants a mix of real
    numbers (kept from an earlier, successful attempt) and OOM (this run's own failed one) --
    see eval.py's --resume/--save-hierarchical support."""
    import latex_results_table as lrt

    _write(tmp_path, "asset", "da3", "pose",
          [{"AUC@30": 0.8}, {"AUC@30": "OOM"}, {"AUC@30": 0.6}])
    _write(tmp_path, "asset", "da3", "cloud",
          [{"Chamfer (m)": 0.1}, {"Chamfer (m)": "OOM"}, {"Chamfer (m)": 0.3}])

    cell = lrt._format_cell(tmp_path / "asset" / "da3", lrt.TABLE_SPECS["cloud"], decimals=3)
    assert cell == "[0.700 / 0.200]"  # averaged over the two non-OOM rows only


def test_format_cell_is_blank_when_the_scene_was_never_evaluated(tmp_path):
    import latex_results_table as lrt

    cell = lrt._format_cell(tmp_path / "asset" / "da3", lrt.TABLE_SPECS["cloud"], decimals=3)
    assert cell == ""


def test_format_cell_is_blank_when_the_metric_csv_has_no_rows(tmp_path):
    import latex_results_table as lrt

    _write(tmp_path, "asset", "da3", "pose", [])
    _write(tmp_path, "asset", "da3", "cloud", [])

    cell = lrt._format_cell(tmp_path / "asset" / "da3", lrt.TABLE_SPECS["cloud"], decimals=3)
    assert cell == ""


# --- completeness: has eval.py actually reached every variant the scene has? -------------------


def test_asset_variants_reads_the_true_variant_count(tmp_path):
    import latex_results_table as lrt

    make_asset(tmp_path / "data" / "bridge", illums=("sunny", "night"), soils=("low",))

    variants = lrt._asset_variants(tmp_path / "data", "bridge")
    assert variants is not None and len(variants) == 2


def test_asset_variants_is_none_for_an_asset_not_present_at_data_root(tmp_path):
    import latex_results_table as lrt

    assert lrt._asset_variants(tmp_path / "data", "nope") is None


def test_is_complete_true_once_every_variant_has_a_row(tmp_path):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny", "night"), soils=("low",))
    _write(results_root, "bridge", "da3", "pose",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "AUC@30": 0.5},
           {"Traj": "trajA", "Illumination": "night", "Soil": "low", "AUC@30": 0.6}])

    assert lrt._is_complete(data_root, results_root / "bridge" / "da3")


def test_is_complete_false_when_a_variant_has_no_row_yet(tmp_path):
    """The crane/pi-long shape: only some of the scene's variants have been scored so far."""
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny", "night"), soils=("low",))
    _write(results_root, "bridge", "da3", "pose",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "AUC@30": 0.5}])

    assert not lrt._is_complete(data_root, results_root / "bridge" / "da3")


def test_is_complete_false_when_the_asset_cant_be_loaded(tmp_path):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    _write(results_root, "bridge", "da3", "pose", [{"Traj": "trajA", "AUC@30": 0.5}])

    assert not lrt._is_complete(data_root, results_root / "bridge" / "da3")


def test_is_complete_counts_an_oom_row_as_reached(tmp_path):
    """An all-OOM combo gets a row for every variant (see inspection_eval.runner.evaluate's OOM
    handling), so it reads as complete even though nothing was actually scored."""
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny", "night"), soils=("low",))
    _write(results_root, "bridge", "da3", "pose",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "AUC@30": "OOM"},
           {"Traj": "trajA", "Illumination": "night", "Soil": "low", "AUC@30": "OOM"}])

    assert lrt._is_complete(data_root, results_root / "bridge" / "da3")


# --- build_row --------------------------------------------------------------------------------


def test_build_row_covers_every_scene_column_in_order(tmp_path):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny",), soils=("low",))
    make_asset(data_root / "crane", illums=("sunny",), soils=("low",))
    row_bind = {"Traj": "trajA", "Illumination": "sunny", "Soil": "low"}
    _write(results_root, "bridge", "vggt-omega", "pose", [{**row_bind, "AUC@30": 0.5}])
    _write(results_root, "bridge", "vggt-omega", "cloud", [{**row_bind, "Chamfer (m)": 0.25}])
    _write(results_root, "crane", "vggt-omega", "pose", [{**row_bind, "AUC@30": "OOM"}])
    _write(results_root, "crane", "vggt-omega", "cloud", [{**row_bind, "Chamfer (m)": "OOM"}])
    # every other scene column has no ground truth and no results at all -> blank

    row = lrt.build_row(results_root, data_root, "vggt-omega", "cloud", decimals=3)

    assert _cells(row) == [r"VGGT-$\Omega$", "", "0.500 / 0.250", "", "OOM", "", "", ""]


def test_build_row_skips_a_combo_that_isnt_completely_evaluated_yet(tmp_path):
    """Regression test for crane/pi-long: 1 of 2 variants scored, with real (not OOM) numbers --
    the cell must stay blank rather than average just the row that exists."""
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny", "night"), soils=("low",))
    _write(results_root, "bridge", "pi-long", "pose",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "AUC@30": 0.9}])
    _write(results_root, "bridge", "pi-long", "cloud",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "Chamfer (m)": 0.1}])

    row = lrt.build_row(results_root, data_root, "pi-long", "cloud", decimals=3)

    assert _cells(row) == ["Pi-Long", "", "", "", "", "", "", ""]


def test_build_row_still_brackets_a_partial_oom_once_the_combo_is_otherwise_complete(tmp_path):
    """Contrast with the test above: every variant *has* a row (so the combo passes the
    completeness gate), just a mix of real and OOM -- that still averages+brackets as before."""
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny", "night"), soils=("low",))
    _write(results_root, "bridge", "pi-long", "pose",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "AUC@30": 0.8},
           {"Traj": "trajA", "Illumination": "night", "Soil": "low", "AUC@30": "OOM"}])
    _write(results_root, "bridge", "pi-long", "cloud",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low", "Chamfer (m)": 0.2},
           {"Traj": "trajA", "Illumination": "night", "Soil": "low", "Chamfer (m)": "OOM"}])

    row = lrt.build_row(results_root, data_root, "pi-long", "cloud", decimals=3)

    assert _cells(row)[2] == "[0.800 / 0.200]"


def test_build_row_flattens_a_colmap_methods_slash_for_the_folder_lookup(tmp_path):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny",), soils=("low",))
    row_bind = {"Traj": "trajA", "Illumination": "sunny", "Soil": "low"}
    _write(results_root, "bridge", "colmap-no_gt", "pose", [{**row_bind, "AUC@30": 0.9}])
    _write(results_root, "bridge", "colmap-no_gt", "cloud", [{**row_bind, "Chamfer (m)": 0.05}])

    row = lrt.build_row(results_root, data_root, "colmap/no_gt", "cloud", decimals=3)
    cells = _cells(row)
    assert cells[0] == "COLMAP"
    assert cells[2] == "0.900 / 0.050"


def test_build_row_falls_back_to_the_raw_key_for_an_unknown_method(tmp_path, capsys):
    import latex_results_table as lrt

    row = lrt.build_row(tmp_path / "results", tmp_path / "data", "some_new_baseline", "cloud",
                        decimals=3)
    assert _cells(row)[0] == "some_new_baseline"
    assert "no known LaTeX label" in capsys.readouterr().err


def test_build_row_depth_table_reads_smse_and_absrel(tmp_path):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny",), soils=("low",))
    _write(results_root, "bridge", "da3", "depth",
          [{"Traj": "trajA", "Illumination": "sunny", "Soil": "low",
            "Scale invariant MSE": 0.02, "Relative MAE": 0.15}])

    row = lrt.build_row(results_root, data_root, "da3", "depth", decimals=3)
    assert _cells(row)[2] == "0.020 / 0.150"


# --- build_table --------------------------------------------------------------------------------


def test_build_table_reproduces_the_templates_sections(tmp_path):
    import latex_results_table as lrt

    text = lrt.build_table(tmp_path / "results", tmp_path / "data", "cloud", decimals=3)
    lines = text.splitlines()

    assert lines[0] == r"\hline"
    assert lines[1] == r"\multicolumn{8}{c}{\cellcolor{catgray}\textbf{Structure from Motion}} \\"
    assert lines[-1] == r"\hline"
    assert sum(line.startswith(r"\multicolumn") for line in lines) == 3
    # every method appears exactly once, in SECTIONS' order
    labels = [lrt._label(m) for _, methods in lrt.SECTIONS for m in methods]
    seen = [line.split(" & ")[0] for line in lines if " & " in line]
    assert seen == labels


# --- CLI --------------------------------------------------------------------------------------


def test_main_both_tables_prints_a_header_and_a_row_per_method_per_table(tmp_path, capsys):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny",), soils=("low",))
    row_bind = {"Traj": "trajA", "Illumination": "sunny", "Soil": "low"}
    for method, auc, chamfer, smse, absrel in (("da3", 0.5, 0.1, 0.02, 0.1),
                                                ("vggt", 0.6, 0.2, 0.03, 0.2)):
        _write(results_root, "bridge", method, "pose", [{**row_bind, "AUC@30": auc}])
        _write(results_root, "bridge", method, "cloud", [{**row_bind, "Chamfer (m)": chamfer}])
        _write(results_root, "bridge", method, "depth",
              [{**row_bind, "Scale invariant MSE": smse, "Relative MAE": absrel}])

    rc = lrt.main([str(results_root), "da3", "vggt", "--data-root", str(data_root)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "% --- point cloud table ---" in out
    assert "% --- depth table ---" in out
    cloud_idx = out.index("point cloud table")
    depth_idx = out.index("depth table")
    da3_cloud = out.index("DA3 &")
    vggt_cloud = out.index(r"VGGT &")
    assert cloud_idx < da3_cloud < vggt_cloud < depth_idx


def test_main_single_table_omits_the_header(tmp_path, capsys):
    import latex_results_table as lrt

    data_root, results_root = tmp_path / "data", tmp_path / "results"
    make_asset(data_root / "bridge", illums=("sunny",), soils=("low",))
    row_bind = {"Traj": "trajA", "Illumination": "sunny", "Soil": "low"}
    _write(results_root, "bridge", "da3", "pose", [{**row_bind, "AUC@30": 0.5}])
    _write(results_root, "bridge", "da3", "cloud", [{**row_bind, "Chamfer (m)": 0.1}])

    rc = lrt.main([str(results_root), "da3", "--table", "cloud", "--data-root", str(data_root)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "% ---" not in out
    assert out.strip().startswith("DA3 & ")


def test_main_full_table_prints_every_section_and_ignores_method_args(tmp_path, capsys):
    import latex_results_table as lrt

    rc = lrt.main([str(tmp_path / "results"), "--full-table", "--table", "cloud",
                  "--data-root", str(tmp_path / "data")])
    out = capsys.readouterr().out

    assert rc == 0
    for title, methods in lrt.SECTIONS:
        assert title in out
        for method in methods:
            assert lrt._label(method) in out


def test_parse_args_errors_when_full_table_and_methods_are_both_given(tmp_path):
    import latex_results_table as lrt

    with pytest.raises(SystemExit):
        lrt.parse_args([str(tmp_path), "da3", "--full-table"])


def test_parse_args_errors_when_neither_methods_nor_full_table_are_given(tmp_path):
    import latex_results_table as lrt

    with pytest.raises(SystemExit):
        lrt.parse_args([str(tmp_path)])
