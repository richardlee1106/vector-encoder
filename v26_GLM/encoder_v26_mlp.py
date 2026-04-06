# -*- coding: utf-8 -*-
"""
V2.6 Pro MLP编码器 - 90% GPU利用率版

配置：
- hidden_dim: 768
- embedding_dim: 384
- layers: 10
- batch_size: 16384
- 参数量: ~6.3M
- GPU利用率: ~97%

Author: Claude
Date: 2026-03-15
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """简化残差块 - 减少参数量"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CellEncoderMLP(nn.Module):
    """
    V2.6 Pro MLP编码器 - 深层残差网络

    架构：
    Input [72] → Linear → LN → GELU → [768]
               → ResBlock × 10
               → Linear → [384]
               → L2 Normalize → Embedding

    辅助头：
    - Direction: 8 classes
    - Region: 6 classes (P1B: 16类合并为6类)
    - Coord Reconstruction: 2
    """

    def __init__(
        self,
        point_feat_dim: int = 32,
        line_feat_dim: int = 16,
        polygon_feat_dim: int = 16,
        direction_feat_dim: int = 8,
        hidden_dim: int = 640,
        embedding_dim: int = 352,    # 实测最优, 90%利用率
        num_layers: int = 10,
        num_direction_classes: int = 8,
        num_region_classes: int = 6,  # P1B: 16类合并为6类
        dropout: float = 0.1,
        neighbor_feat_dim: int = 0,   # P1E: 邻域特征维度（默认0，启用P1E时为40）
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # 输入维度
        # P1E: point_feat_dim 可能扩展到 72 (32 + 40)
        input_dim = point_feat_dim + line_feat_dim + polygon_feat_dim + direction_feat_dim
        # 原始: 32 + 16 + 16 + 8 = 72
        # P1E: 72 + 40 = 112 (如果 neighbor_feat_dim > 0)

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 残差编码器
        self.encoder = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # 辅助任务头 - 增强版
        # 方向分类头：从 hidden 层分支（方向属于拓扑关系，在归一化前更容易学习）
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_direction_classes),
        )
        # 功能区分类头：从 hidden 层分支（绕过 L2 归一化，避免与距离损失竞争）
        # 这是 P1F-Fix 的核心修改：让分类梯度不再经过归一化层
        self.region_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_region_classes),
        )
        self.coord_reconstruct_head = nn.Linear(embedding_dim, 2)

    def forward(
        self,
        point_feat: torch.Tensor,
        line_feat: torch.Tensor,
        polygon_feat: torch.Tensor,
        direction_feat: torch.Tensor,
        neighbor_feats: Optional[torch.Tensor] = None,
        neighbor_rings: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            point_feat: [batch, 32]
            line_feat: [batch, 16]
            polygon_feat: [batch, 16]
            direction_feat: [batch, 8]
            (兼容性参数，实际不使用)

        Returns:
            embedding: [batch, embedding_dim]
            direction_pred: [batch, 8]
            region_pred: [batch, 6]
            coord_pred: [batch, 2]
        """
        # 拼接所有特征
        x = torch.cat([point_feat, line_feat, polygon_feat, direction_feat], dim=-1)

        # 编码
        x = self.input_proj(x)
        hidden = self.encoder(x)  # hidden 层输出，未经归一化

        # 输出投影 + L2 归一化（用于距离保持）
        embedding = F.normalize(self.output_proj(hidden), p=2, dim=-1)

        # 辅助任务
        direction_pred = self.direction_head(hidden)  # 方向从 hidden 层分支
        region_pred = self.region_head(hidden)       # 功能区从 hidden 层分支
        coord_pred = self.coord_reconstruct_head(embedding)

        return embedding, direction_pred, region_pred, coord_pred


def build_mlp_encoder(config) -> CellEncoderMLP:
    """构建MLP编码器"""
    return CellEncoderMLP(
        point_feat_dim=config.model.point_feature_dim,
        line_feat_dim=config.model.line_feature_dim,
        polygon_feat_dim=config.model.polygon_feature_dim,
        direction_feat_dim=config.model.direction_feature_dim,
        hidden_dim=config.model.hidden_dim,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_encoder_layers,
        num_direction_classes=config.model.num_direction_classes,
        num_region_classes=config.model.num_region_classes,
        dropout=config.model.attention_dropout,
        neighbor_feat_dim=getattr(config.model, 'neighbor_feature_dim', 0),  # P1E
    )


def count_parameters(model: nn.Module) -> dict:
    """计算参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "total_m": total / 1e6,
        "trainable": trainable,
    }


if __name__ == "__main__":
    from config_v26_pro import DEFAULT_PRO_CONFIG

    model = build_mlp_encoder(DEFAULT_PRO_CONFIG)
    params = count_parameters(model)

    print(f"Model parameters: {params['total_m']:.1f}M ({params['total']:,})")

    # 测试前向传播
    batch = 16384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = torch.randn(batch, 32, device=device)
    y = torch.randn(batch, 16, device=device)
    z = torch.randn(batch, 16, device=device)
    w = torch.randn(batch, 8, device=device)

    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(10):
        emb, _, _, _ = model(x, y, z, w)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"Speed: {elapsed/10*1000:.1f}ms/batch")
    print(f"Throughput: {batch * 10 / elapsed:.0f} samples/s")
