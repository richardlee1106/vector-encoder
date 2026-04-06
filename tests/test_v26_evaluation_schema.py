# -*- coding: utf-8 -*-

from spatial_encoder.v26.evaluation_schema import build_v26_evaluation_schema


def test_build_v26_evaluation_schema_contains_core_sections():
    schema = build_v26_evaluation_schema()

    assert "metrics" in schema
    assert "spatial_retrieval" in schema["metrics"]
    assert schema["metrics"]["spatial_retrieval"]["status"] == "pending"
