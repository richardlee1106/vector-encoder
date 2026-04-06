# -*- coding: utf-8 -*-

from spatial_encoder.v26.export_contract import build_export_manifest


def test_build_export_manifest_marks_rag_and_llm_consumption():
    manifest = build_export_manifest(
        version="v26",
        embedding_dim=64,
        supports=["spatial_search", "hybrid_retrieval", "llm_context"],
    )

    assert manifest["version"] == "v26"
    assert "hybrid_retrieval" in manifest["supports"]
    assert manifest["embedding_dim"] == 64


def test_build_export_manifest_declares_agent_modalities():
    manifest = build_export_manifest(
        version="v26",
        embedding_dim=64,
        supports=["spatial_search", "hybrid_retrieval", "llm_context"],
    )

    assert manifest["spatial_tokenization"]["type"] == "h3"
    assert manifest["agent_types"] == ["point", "line", "polygon"]
    assert manifest["consumption_modes"]["llm_context"]["enabled"] is True
