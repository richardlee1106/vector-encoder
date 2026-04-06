# -*- coding: utf-8 -*-
"""
V2.6 统一空间编码实验入口。
"""

from __future__ import annotations

from spatial_encoder.v26.config_v26 import V26Config
from spatial_encoder.v26.data_sources import resolve_sample_source
from spatial_encoder.v26.evaluation_schema import build_v26_evaluation_schema
from spatial_encoder.v26.export_v26 import (
    build_v26_export_bundle,
    save_v26_export_bundle,
)
from spatial_encoder.v26.output_manager import save_v26_json
from spatial_encoder.v26.run_manifest import build_run_manifest


def _config_payload(cfg: V26Config) -> dict:
    return {
        "experiment": cfg.experiment_name,
        "h3_resolution": cfg.h3_resolution,
        "neighborhood_rings": cfg.neighborhood_rings,
        "embedding_dim": cfg.embedding_dim,
        "hidden_dim": cfg.hidden_dim,
        "relation_types": list(cfg.relation_types),
        "direction_head": cfg.enable_direction_head,
        "region_head": cfg.enable_region_head,
    }


def run_train_v26(base_dir=None) -> dict:
    """返回最小训练快照，便于后续接入真实训练流程。"""
    cfg = V26Config()
    data_source = resolve_sample_source()
    manifest = build_run_manifest(
        experiment=cfg.experiment_name,
        stage="train",
        config=_config_payload(cfg),
        data_source=data_source,
    )
    manifest_path = save_v26_json("train_v26_manifest", manifest, base_dir)

    return {
        "experiment": cfg.experiment_name,
        "h3_resolution": cfg.h3_resolution,
        "embedding_dim": cfg.embedding_dim,
        "manifest_path": str(manifest_path),
    }


def run_validate_v26(base_dir=None) -> dict:
    """返回最小验证快照。"""
    cfg = V26Config()
    payload = _config_payload(cfg)
    payload["evaluation_schema"] = build_v26_evaluation_schema()
    output_path = save_v26_json("validate_v26", payload, base_dir)
    payload["output_path"] = str(output_path)
    return payload


def run_export_v26(base_dir=None) -> dict:
    """返回最小导出结构。"""
    cfg = V26Config()
    bundle = build_v26_export_bundle(cfg.embedding_dim)
    output_path = save_v26_export_bundle(
        base_dir=base_dir, embedding_dim=cfg.embedding_dim
    )
    bundle["output_path"] = str(output_path)
    return bundle


if __name__ == "__main__":
    print(run_train_v26())
