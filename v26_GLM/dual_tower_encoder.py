# -*- coding: utf-8 -*-
"""
双塔编码器架构 - 生产版

核心思想：
- 共享 POI 编码器（与 CellEncoderMLP 相同架构）
- 上下文聚合器：平均池化邻居 embedding 后与自身拼接
- 完全兼容现有 forward 签名：(point, line, polygon, direction) → (emb, dir, reg, coord)
- 当提供 neighbor_features 时启用上下文增强路径

参数预算：
- 共享编码器: ~4.9M（与 CellEncoderMLP 相同）
- 上下文聚合器: ~0.5M（Linear(704,352) + Linear(352,352)）
- 总计: ~5.4M（+0.5M，显存增加 <0.3GB）

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_encoder.v26_GLM.encoder_v26_mlp import ResidualBlock


class DualTowerEncoder(nn.Module):
    """
    双塔空间编码器

    架构：
    Input [72] → input_proj → [640] → ResBlock × N → hidden [640]
        ├── output_proj → L2 norm → embedding [352]  (距离保持)
        ├── direction_head → [8]                       (方向分类)
        ├── region_head → [6]                          (功能区分类)
        └── coord_head → [2]                           (坐标重构)

    上下文增强路径（可选）：
    poi_emb [352] + mean(neighbor_embs) [352] → concat [704]
        → context_aggregator → L2 norm → enhanced_emb [352]
    """

    def __init__(
        self,
        point_feat_dim: int = 32,
        line_feat_dim: int = 16,
        polygon_feat_dim: int = 16,
        direction_feat_dim: int = 8,
        hidden_dim: int = 640,
        embedding_dim: int = 352,
        num_layers: int = 10,
        num_direction_classes: int = 8,
        num_region_classes: int = 6,
        dropout: float = 0.1,
        context_k: int = 20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.context_k = context_k

        # 输入维度: 32 + 16 + 16 + 8 = 72
        input_dim = point_feat_dim + line_feat_dim + polygon_feat_dim + direction_feat_dim

        # ============ 共享 POI 编码器（与 CellEncoderMLP 相同） ============
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.encoder = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影 → embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # ============ 辅助任务头（从 hidden 层分支） ============
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_direction_classes),
        )

        self.region_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_region_classes),
        )

        self.coord_reconstruct_head = nn.Linear(embedding_dim, 2)

        # ============ 上下文聚合器（双塔核心） ============
        self.context_aggregator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def get_hidden(self, features: torch.Tensor) -> torch.Tensor:
        """
        获取 hidden 层输出（未经归一化），用于对比学习损失。

        Args:
            features: [B, 72] 拼接后的 POI 特征

        Returns:
            hidden: [B, 640]
        """
        x = self.input_proj(features)
        hidden = self.encoder(x)
        return hidden

    def encode_poi(self, features: torch.Tensor) -> torch.Tensor:
        """
        编码单个 POI（无上下文），返回 L2 归一化的 embedding。

        Args:
            features: [B, 72] 拼接后的 POI 特征

        Returns:
            embedding: [B, 352] L2 归一化
        """
        hidden = self.get_hidden(features)
        embedding = F.normalize(self.output_proj(hidden), p=2, dim=-1)
        return embedding

    def encode_with_context(
        self,
        poi_features: torch.Tensor,
        neighbor_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        上下文增强编码：POI + K 个邻居的平均 embedding。

        Args:
            poi_features: [B, 72] 查询 POI 特征
            neighbor_features: [B, K, 72] K 个邻居的原始特征

        Returns:
            enhanced_emb: [B, 352] 上下文增强的 L2 归一化 embedding
        """
        batch_size = poi_features.size(0)
        k = neighbor_features.size(1)

        # 编码查询 POI
        poi_emb = self.encode_poi(poi_features)  # [B, 352]

        # 编码邻居（共享编码器，批量处理）
        neighbor_flat = neighbor_features.reshape(-1, poi_features.size(-1))  # [B*K, 72]
        neighbor_embs = self.encode_poi(neighbor_flat)  # [B*K, 352]
        neighbor_embs = neighbor_embs.reshape(batch_size, k, self.embedding_dim)  # [B, K, 352]

        # 上下文聚合（平均池化）
        context_emb = neighbor_embs.mean(dim=1)  # [B, 352]

        # 融合 POI + 上下文
        combined = torch.cat([poi_emb, context_emb], dim=-1)  # [B, 704]
        enhanced_emb = self.context_aggregator(combined)  # [B, 352]

        # L2 归一化
        enhanced_emb = F.normalize(enhanced_emb, p=2, dim=-1)

        return enhanced_emb

    def forward(
        self,
        point_feat: torch.Tensor,
        line_feat: torch.Tensor,
        polygon_feat: torch.Tensor,
        direction_feat: torch.Tensor,
        neighbor_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 — 兼容 CellEncoderMLP 的 4 输出签名。

        Args:
            point_feat: [B, 32]
            line_feat: [B, 16]
            polygon_feat: [B, 16]
            direction_feat: [B, 8]
            neighbor_features: [B, K, 72] 邻居原始特征（可选）

        Returns:
            embedding: [B, 352] L2 归一化
            direction_pred: [B, 8] 方向分类 logits
            region_pred: [B, 6] 功能区分类 logits
            coord_pred: [B, 2] 坐标重构
        """
        # 拼接所有特征
        features = torch.cat([point_feat, line_feat, polygon_feat, direction_feat], dim=-1)

        # 共享编码器 → hidden
        hidden = self.get_hidden(features)

        # 辅助任务头（始终从 hidden 分支，与 CellEncoderMLP 一致）
        direction_pred = self.direction_head(hidden)
        region_pred = self.region_head(hidden)

        # Embedding 路径
        if neighbor_features is not None:
            # 上下文增强路径
            embedding = self.encode_with_context(features, neighbor_features)
        else:
            # 标准路径（与 CellEncoderMLP 相同）
            embedding = F.normalize(self.output_proj(hidden), p=2, dim=-1)

        # 坐标重构（从 embedding）
        coord_pred = self.coord_reconstruct_head(embedding)

        return embedding, direction_pred, region_pred, coord_pred


def build_dual_tower_encoder(config) -> DualTowerEncoder:
    """根据配置构建双塔编码器"""
    return DualTowerEncoder(
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
        context_k=getattr(config.dual_tower, 'context_k_neighbors', 20),
    )


if __name__ == "__main__":
    from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
    from spatial_encoder.v26_GLM.encoder_v26_mlp import count_parameters

    model = build_dual_tower_encoder(DEFAULT_PRO_CONFIG)
    params = count_parameters(model)

    print(f"DualTowerEncoder parameters: {params['total_m']:.1f}M ({params['total']:,})")

    # 测试前向传播（无上下文）
    batch = 4096
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    p = torch.randn(batch, 32, device=device)
    l = torch.randn(batch, 16, device=device)
    g = torch.randn(batch, 16, device=device)
    d = torch.randn(batch, 8, device=device)

    emb, dir_pred, reg_pred, coord_pred = model(p, l, g, d)
    print(f"无上下文: emb={emb.shape}, dir={dir_pred.shape}, reg={reg_pred.shape}, coord={coord_pred.shape}")

    # 测试前向传播（有上下文）
    k = 20
    neighbor_feats = torch.randn(batch, k, 72, device=device)
    emb2, dir_pred2, reg_pred2, coord_pred2 = model(p, l, g, d, neighbor_features=neighbor_feats)
    print(f"有上下文: emb={emb2.shape}, dir={dir_pred2.shape}, reg={reg_pred2.shape}, coord={coord_pred2.shape}")

    # 验证 L2 归一化
    norms = torch.norm(emb, dim=-1)
    print(f"Embedding L2 norms: mean={norms.mean():.4f}, std={norms.std():.6f}")

    # 测试 get_hidden
    features = torch.cat([p, l, g, d], dim=-1)
    hidden = model.get_hidden(features)
    print(f"Hidden: {hidden.shape}")
