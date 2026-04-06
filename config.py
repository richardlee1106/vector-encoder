# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 配置模块

核心能力：学习空间拓扑关系（位置、距离、邻接）
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SpatialEncoderConfig:
    """空间编码器配置"""

    # ===== 模型配置 =====
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1

    # ===== 图构建 =====
    knn_k: int = 10

    # ===== 训练配置 =====
    num_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    num_distance_pairs: int = 5000  # 距离采样对数

    # ===== 损失权重 =====
    coord_recon_weight: float = 1.0      # 坐标重构
    distance_preserve_weight: float = 2.0  # 距离保持
    neighbor_consistency_weight: float = 1.0  # 邻居一致性

    # ===== 数据配置 =====
    num_categories: int = 50
    num_landuses: int = 15
    num_aoi_types: int = 30
    num_road_classes: int = 20

    # ===== 聚类配置 =====
    n_clusters: int = 15

    # ===== 设备 =====
    device: str = "cuda"


@dataclass
class FeatureSchema:
    """特征Schema"""

    # 离散特征
    category_col: int = 0
    landuse_col: int = 1
    aoi_type_col: int = 2
    road_class_col: int = 3

    # 数值特征
    numerical_cols: Tuple[int, ...] = (4, 5, 6)  # density, entropy, road_dist

    # 坐标
    coord_cols: Tuple[int, int] = (0, 1)  # lng, lat


# 默认配置
DEFAULT_CONFIG = SpatialEncoderConfig()
DEFAULT_SCHEMA = FeatureSchema()
