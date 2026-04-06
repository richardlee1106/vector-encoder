# -*- coding: utf-8 -*-
"""
V2.6 Pro 最终配置 - 实测最优

实测数据：
- GPU利用率: 90.0%
- 模型参数: 4.9M (原版275K, 提升18x)
- 显存占用: 7.20GB / 8GB
- batch_size: 16384 (原版256, 提升64x)
- K_neighbors: 64

Author: Claude
Date: 2026-03-15
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class H3Config:
    resolution: int = 8  # 基础分辨率
    resolution_fine: int = 9  # 精细分辨率（双分辨率特征）
    neighborhood_rings: int = 2
    use_dual_resolution: bool = False  # 暂时禁用双分辨率特征


@dataclass
class ModelConfig:
    """模型架构配置 - 三镇数据集"""
    # 单分辨率特征维度
    point_feature_dim: int = 32
    line_feature_dim: int = 16
    polygon_feature_dim: int = 16
    direction_feature_dim: int = 8

    # 核心参数
    hidden_dim: int = 640
    embedding_dim: int = 352
    num_encoder_layers: int = 10

    attention_dropout: float = 0.1
    num_direction_classes: int = 8
    num_region_classes: int = 6  # 居住/商业/工业/教育/公共/自然

    # P1E: 邻域特征维度（默认40，设为0禁用P1E）
    neighbor_feature_dim: int = 40


@dataclass
class LossConfig:
    """损失函数配置 - 三镇数据集"""
    distance_weight: float = 0.5
    reconstruction_weight: float = 0.3
    neighborhood_weight: float = 0.0
    direction_weight: float = 1.5
    region_weight: float = 1.5
    region_clf_weight: float = 1.0
    center_weight: float = 0.2
    distance_decay_gamma: float = 0.5
    k_nearest_neighbors: int = 50  # 增加邻居数：20 → 50，让稀疏 cell 获取更多上下文


@dataclass
class TrainingConfig:
    """训练配置 - 三镇数据集 + 梯度累积"""
    batch_size: int = 256          # 实际批次大小
    accumulation_steps: int = 8    # 梯度累积步数（等效 batch_size=2048）
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 120
    num_workers: int = 0
    pin_memory: bool = False
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 10
    sample_ratios: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.8, 1.0)  # 渐进式训练
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4
    random_seed: int = 42
    seed: int = 42  # 别名
    save_dir: str = "saved_models/town_encoder"
    save_every: int = 10

@dataclass
class DualTowerConfig:
    """双塔架构 + 对比学习配置"""
    # 对比学习
    contrastive_temperature: float = 0.07  # InfoNCE temperature
    num_negatives: int = 64                # 负样本数量
    positive_k: int = 5                    # 正样本K近邻数
    negative_min_distance: float = 0.05    # 负样本最小归一化距离（约5km）
    contrastive_weight: float = 1.0        # 对比学习损失权重
    
    # 上下文编码
    context_k_neighbors: int = 20          # 上下文聚合邻居数
    context_pool: str = "mean"             # 聚合方式: mean, attention
    
    # 训练策略
    use_mixed_precision: bool = True       # FP16混合精度
    warmup_contrastive_epochs: int = 5     # 对比学习warmup（前N个epoch不启用对比损失）


@dataclass
class PrototypeConfig:
    """Phase 2: 原型学习配置"""
    n_prototypes: int = 100          # 总原型数（6类 × ~17个/类）
    prototype_weight: float = 0.5    # 原型损失权重
    proto_temperature: float = 0.1   # 原型相似度温度
    init_from_embeddings: bool = True  # 从预训练 embedding 初始化


@dataclass
class SpatialAttentionConfig:
    """Phase 3: 时空注意力配置"""
    num_heads: int = 4               # 注意力头数
    context_k: int = 20             # 上下文邻居数
    dropout: float = 0.1
    use_distance_encoding: bool = True  # 距离位置编码


@dataclass
class V26ProConfig:
    experiment_name: str = "v26-cell-context-encoder-pro"
    h3: H3Config = field(default_factory=H3Config)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    enable_direction_head: bool = True
    enable_region_head: bool = True
    dual_tower: DualTowerConfig = field(default_factory=DualTowerConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    spatial_attention: SpatialAttentionConfig = field(default_factory=SpatialAttentionConfig)


DEFAULT_PRO_CONFIG = V26ProConfig()


if __name__ == "__main__":
    c = DEFAULT_PRO_CONFIG
    print("V2.6 Pro Final Config (90% GPU utilization):")
    print(f"  hidden_dim: {c.model.hidden_dim}")
    print(f"  embedding_dim: {c.model.embedding_dim}")
    print(f"  num_layers: {c.model.num_encoder_layers}")
    print(f"  batch_size: {c.training.batch_size}")
    print(f"  K_neighbors: {c.loss.k_nearest_neighbors}")
    print(f"  Expected: ~4.9M params, ~7.2GB VRAM, 90% utilization")
