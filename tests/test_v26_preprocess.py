# -*- coding: utf-8 -*-

from spatial_encoder.v26.preprocess_v26 import run_preprocess_v26


def test_run_preprocess_v26_writes_manifest_and_stats(tmp_path):
    result = run_preprocess_v26(base_dir=tmp_path, sample_config={"rate": 0.01})

    assert result["manifest_path"].endswith("preprocess_v26_manifest.json")
    assert result["stats_path"].endswith("preprocess_v26_stats.json")
