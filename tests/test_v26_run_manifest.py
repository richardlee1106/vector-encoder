# -*- coding: utf-8 -*-

from spatial_encoder.v26.run_manifest import build_run_manifest


def test_build_run_manifest_includes_core_fields():
    manifest = build_run_manifest(
        experiment="v26-unified-spatial-encoder",
        stage="preprocess",
        config={"h3_resolution": 9},
        data_source={"mode": "postgis"},
    )

    assert manifest["experiment"] == "v26-unified-spatial-encoder"
    assert manifest["stage"] == "preprocess"
    assert isinstance(manifest["run_id"], str)
    assert isinstance(manifest["timestamp"], str)
    assert "environment" in manifest
