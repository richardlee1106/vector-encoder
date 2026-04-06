# -*- coding: utf-8 -*-
"""
V2.6 导出工具。
"""

from __future__ import annotations

import json
from pathlib import Path

from spatial_encoder.v26.export_contract import build_export_manifest
from spatial_encoder.v26.output_manager import ensure_v26_output_dirs


def build_v26_export_bundle(embedding_dim: int = 64) -> dict:
    """构造面向下游消费的最小导出结构。"""
    return {
        "bundle_type": "llm_consumable_spatial_embedding",
        "manifest": build_export_manifest(
            version="v26",
            embedding_dim=embedding_dim,
            supports=["spatial_search", "hybrid_retrieval", "llm_context"],
        ),
    }


def save_v26_export_bundle(
    base_dir: str | Path | None = None, embedding_dim: int = 64
) -> Path:
    """将导出 bundle 写入 V2.6 exports 目录。"""
    bundle = build_v26_export_bundle(embedding_dim)
    paths = ensure_v26_output_dirs(base_dir)
    output_path = paths["exports"] / "export_bundle.json"
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path
