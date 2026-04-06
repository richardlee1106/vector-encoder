# -*- coding: utf-8 -*-

import json

from spatial_encoder.v26.export_v26 import save_v26_export_bundle


def test_save_v26_export_bundle_writes_manifest_to_exports_dir(tmp_path):
    output_path = save_v26_export_bundle(base_dir=tmp_path, embedding_dim=64)

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["version"] == "v26"
    assert "llm_context" in payload["manifest"]["supports"]
