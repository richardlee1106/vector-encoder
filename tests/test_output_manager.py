# -*- coding: utf-8 -*-

import json

from spatial_encoder.v26.output_manager import ensure_v26_output_dirs, save_v26_json


def test_ensure_v26_output_dirs_creates_expected_keys(tmp_path):
    paths = ensure_v26_output_dirs(tmp_path)

    assert {"root", "reports", "exports", "metrics"} <= set(paths.keys())
    assert paths["reports"].exists()
    assert paths["exports"].exists()
    assert paths["metrics"].exists()


def test_save_v26_json_writes_utf8_file(tmp_path):
    output_path = save_v26_json("demo", {"标题": "空间智能"}, tmp_path)

    assert output_path.exists()
    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content["标题"] == "空间智能"


def test_save_v26_json_supports_metrics_dir(tmp_path):
    output_path = save_v26_json(
        "quick_validate_v26_metrics",
        {"unique_cells": 12},
        tmp_path,
        artifact_type="metrics",
    )

    assert output_path.exists()
    assert output_path.parent.name == "metrics"
