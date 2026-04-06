# -*- coding: utf-8 -*-
"""
V2.6 Pro MLP 空间编码器模块

L3 里程碑达成版本 (2026-03-18)

核心组件：
- config_v26_pro: 配置管理
- encoder_v26_mlp: MLP编码器 (双头并行架构)
- losses_v26_pro: 损失函数
- direction_supervision: 方向监督
- data_loader_v26: 数据加载
- experiment_p1c_integrated: 集成训练脚本
- evaluate_v26_pro: 评估脚本

最终成果：
- DirAcc: 82.14%
- ClfF1: 57.95%
- Pearson: 0.9784
"""

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import CellEncoderMLP, build_mlp_encoder, count_parameters
from spatial_encoder.v26_GLM.direction_supervision import (
    DirectionScheme,
    MultiSchemeDirectionSupervision,
    build_direction_supervisor,
)

__all__ = [
    # 配置
    "V26ProConfig",
    "DEFAULT_PRO_CONFIG",
    # 模型
    "CellEncoderMLP",
    "build_mlp_encoder",
    "count_parameters",
    # 方向监督
    "DirectionScheme",
    "MultiSchemeDirectionSupervision",
    "build_direction_supervisor",
]
