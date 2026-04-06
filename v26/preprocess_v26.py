# -*- coding: utf-8 -*-
"""
V2.6 预处理骨架。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from spatial_encoder.v26.config_v26 import V26Config
from spatial_encoder.v26.data_sources import resolve_sample_source
from spatial_encoder.v26.output_manager import save_v26_json
from spatial_encoder.v26.run_manifest import build_run_manifest


@dataclass
class PreprocessConfig:
    batch_size: int = 20000
    h3_resolution: int = 9
    max_cells_per_agent: int = 200

    def as_dict(self) -> dict:
        return asdict(self)


def run_preprocess_v26(
    base_dir=None,
    sample_config: dict | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> dict:
    """预处理骨架：写出清单与空统计，后续补真实预处理。"""
    cfg = V26Config()
    prep_cfg = preprocess_config or PreprocessConfig(h3_resolution=cfg.h3_resolution)
    data_source = resolve_sample_source()

    manifest = build_run_manifest(
        experiment=cfg.experiment_name,
        stage="preprocess",
        config={
            "h3_resolution": cfg.h3_resolution,
            "embedding_dim": cfg.embedding_dim,
            "preprocess": prep_cfg.as_dict(),
        },
        data_source=data_source,
        sample_config=sample_config,
        notes="预处理骨架，尚未执行实际数据投影。",
    )

    stats = {
        "status": "pending",
        "agent_counts": {"point": 0, "line": 0, "polygon": 0},
        "cell_count": 0,
        "sample_config": sample_config or {},
    }

    manifest_path = save_v26_json("preprocess_v26_manifest", manifest, base_dir)
    stats_path = save_v26_json(
        "preprocess_v26_stats",
        stats,
        base_dir,
        artifact_type="metrics",
    )

    return {
        "status": "pending",
        "manifest_path": str(manifest_path),
        "stats_path": str(stats_path),
    }


if __name__ == "__main__":
    print(run_preprocess_v26())
