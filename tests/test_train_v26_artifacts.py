# -*- coding: utf-8 -*-

from spatial_encoder.v26.train_v26 import run_export_v26


def test_run_export_v26_returns_written_path(tmp_path):
    output = run_export_v26(base_dir=tmp_path)

    assert output["output_path"].endswith("export_bundle.json")
