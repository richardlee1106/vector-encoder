# -*- coding: utf-8 -*-
"""
V2.6 统一空间编码实验配置
"""

from dataclasses import dataclass, field


@dataclass
class V26Config:
    """V2.6 配置对象。"""

    experiment_name: str = "v26-unified-spatial-encoder"
    h3_resolution: int = 9
    neighborhood_rings: int = 2
    embedding_dim: int = 64
    hidden_dim: int = 128
    relation_types: tuple[str, ...] = field(
        default_factory=lambda: (
            "adjacency",
            "road_topology",
            "cooccurrence",
            "functional_similarity",
        )
    )
    enable_direction_head: bool = True
    enable_region_head: bool = True
    export_for_spatial_search: bool = True
    export_for_hybrid_retrieval: bool = True
    export_for_llm_context: bool = True
