# -*- coding: utf-8 -*-
"""
V2.6 导出契约定义。
"""

from __future__ import annotations


def build_export_manifest(
    version: str, embedding_dim: int, supports: list[str]
) -> dict:
    """构造导出 manifest。"""
    support_set = set(supports)

    return {
        "version": version,
        "embedding_dim": embedding_dim,
        "supports": supports,
        "spatial_tokenization": {
            "type": "h3",
            "resolution": "configurable",
        },
        "agent_types": ["point", "line", "polygon"],
        "relation_types": [
            "adjacency",
            "road_topology",
            "cooccurrence",
            "functional_similarity",
            "od_flow",
        ],
        "consumption_modes": {
            "spatial_search": {
                "enabled": "spatial_search" in support_set,
                "target": "vector_retrieval",
            },
            "hybrid_retrieval": {
                "enabled": "hybrid_retrieval" in support_set,
                "target": "text_space_rag",
            },
            "llm_context": {
                "enabled": "llm_context" in support_set,
                "target": "prompt_context_features",
            },
        },
        "record_schema": {
            "agent_record_fields": [
                "agent_id",
                "agent_type",
                "cells",
                "primary_cell",
                "metadata",
            ],
            "cell_record_fields": ["cell", "weight"],
        },
    }
