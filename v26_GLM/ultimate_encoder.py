# -*- coding: utf-8 -*-
"""
终极空间编码器 - Phase 3

架构：双塔 + 原型 + 时空注意力

组合优势：
- 时空注意力：捕获上下文（替代 DualTowerEncoder 的平均池化）
- 原型学习：加速检索 + 类间分离（提升 Region F1）
- 多任务损失：距离 + 方向 + 区域 + 对比 + 原型

长期路线：
  L3 (当前) → L4 空间推理 → L5 矢量理解/面生成 → L6 空间智能体

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_encoder.v26_GLM.encoder_v26_mlp import ResidualBlock
from spatial_encoder.v26_GLM.spatial_attention_encoder import SpatialAttentionEncoder
from spatial_encoder.v26_GLM.prototype_learning import PrototypeLearning
from spatial_encoder.v26_GLM.contrastive_losses import DualTowerMultiTaskLoss


class UltimateSpatialEncoder(nn.Module):
    """
    终极空间编码器：双塔 + 原型 + 时空注意力

    架构：
    Input [72] → input_proj → [640] → ResBlock × N → hidden [640]
        ├── output_proj → L2 norm → poi_emb [352]
        ├── direction_head → [8]
        ├── region_head → [6]
        └── coord_head → [2]

    上下文增强（时空注意力替代平均池化）：
    poi_emb [352] + SpatialAttention(poi_emb, neighbor_embs, dists) → enhanced_emb [352]

    原型学习（训练时附加损失，推理时加速检索）：
    enhanced_emb → PrototypeLearning → proto_loss

    参数预算：
    - 共享编码器: ~4.9M
    - 时空注意力: ~0.75M
    - 原型模块:   ~0.14KB（可忽略）
    - 总计: ~5.65M（+0.75M vs DualTowerEncoder）
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
        # 时空注意力参数
        attn_num_heads: int = 4,
        attn_context_k: int = 20,
        use_distance_encoding: bool = True,
        # 原型参数
        n_prototypes: int = 100,
        proto_temperature: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        input_dim = point_feat_dim + line_feat_dim + polygon_feat_dim + direction_feat_dim

        # ============ 共享 POI 编码器 ============
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # ============ 辅助任务头 ============
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
        self.coord_head = nn.Linear(embedding_dim, 2)

        # ============ 时空注意力（替代平均池化） ============
        self.spatial_attention = SpatialAttentionEncoder(
            embedding_dim=embedding_dim,
            num_heads=attn_num_heads,
            dropout=dropout,
            use_distance_encoding=use_distance_encoding,
        )

        # ============ 原型学习 ============
        self.prototype_learning = PrototypeLearning(
            n_prototypes=n_prototypes,
            embedding_dim=embedding_dim,
            n_classes=num_region_classes,
            temperature=proto_temperature,
        )

    def encode_poi(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码单个 POI，返回 (L2 归一化 embedding, hidden 状态)。

        Args:
            features: [B, 72]

        Returns:
            emb:    [B, 352] L2 归一化
            hidden: [B, 640] 用于辅助任务头
        """
        x = self.input_proj(features)
        hidden = self.encoder(x)
        emb = F.normalize(self.output_proj(hidden), p=2, dim=-1)
        return emb, hidden

    def forward(
        self,
        point_feat: torch.Tensor,
        line_feat: torch.Tensor,
        polygon_feat: torch.Tensor,
        direction_feat: torch.Tensor,
        neighbor_features: Optional[torch.Tensor] = None,
        neighbor_distances: Optional[torch.Tensor] = None,
        region_labels: Optional[torch.Tensor] = None,
        cell_context: Optional[torch.Tensor] = None,
        cell_distances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        Args:
            point_feat:         [B, 32]
            line_feat:          [B, 16]
            polygon_feat:       [B, 16]
            direction_feat:     [B, 8]
            neighbor_features:  [B, K, 72]  可选，启用时空注意力
            neighbor_distances: [B, K]      可选，距离位置编码
            region_labels:      [B]         可选，启用原型损失
            cell_context:       [B, 352]    可选，所属 Cell 的宏观 embedding（层次化多尺度）
            cell_distances:     [B, 1]      可选，POI 到 Cell 中心距离

        Returns:
            emb:        [B, 352] 最终 embedding（有邻居时为注意力增强版）
            dir_pred:   [B, 8]
            reg_pred:   [B, 6]
            coord_pred: [B, 2]
            proto_loss: 标量 or None
        """
        features = torch.cat([point_feat, line_feat, polygon_feat, direction_feat], dim=-1)

        # 编码当前 POI
        poi_emb, hidden = self.encode_poi(features)

        # 辅助任务头（从 hidden 分支）
        dir_pred = self.direction_head(hidden)
        reg_pred = self.region_head(hidden)
        coord_pred = self.coord_head(poi_emb)

        # 时空注意力上下文增强（含可选的 cell_context 宏观邻居）
        if neighbor_features is not None:
            B, K, feat_dim = neighbor_features.shape
            # 批量编码邻居
            neighbor_flat = neighbor_features.reshape(-1, feat_dim)
            neighbor_embs, _ = self.encode_poi(neighbor_flat)
            neighbor_embs = neighbor_embs.reshape(B, K, self.embedding_dim)

            emb, _ = self.spatial_attention(
                poi_emb, neighbor_embs, neighbor_distances,
                cell_context=cell_context, cell_distances=cell_distances,
            )
        elif cell_context is not None:
            # 无微观邻居但有宏观 Cell 上下文：用 cell_context 作为单邻居
            cell_emb = cell_context.unsqueeze(1)  # [B, 1, D]
            emb, _ = self.spatial_attention(
                poi_emb, cell_emb, cell_distances,
            )
        else:
            emb = poi_emb

        # 原型损失（训练时）
        proto_loss = None
        if region_labels is not None:
            proto_loss, _ = self.prototype_learning(emb, region_labels)

        return emb, dir_pred, reg_pred, coord_pred, proto_loss

    def forward_simple(
        self,
        point_feat: torch.Tensor,
        line_feat: torch.Tensor,
        polygon_feat: torch.Tensor,
        direction_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        兼容 CellEncoderMLP / DualTowerEncoder 的 4 输出签名（无邻居）。
        用于评估和推理。
        """
        emb, dir_pred, reg_pred, coord_pred, _ = self.forward(
            point_feat, line_feat, polygon_feat, direction_feat
        )
        return emb, dir_pred, reg_pred, coord_pred


# ============ 终极损失函数 ============

class UltimateLoss(nn.Module):
    """
    终极多任务损失 = DualTowerMultiTaskLoss + 原型损失

    权重推荐：
    - distance_weight    = 0.5
    - contrastive_weight = 1.0
    - direction_weight   = 1.5
    - region_weight      = 1.5
    - prototype_weight   = 0.5
    """

    def __init__(
        self,
        k_nearest: int = 50,
        distance_weight: float = 0.5,
        reconstruction_weight: float = 0.3,
        direction_weight: float = 1.5,
        region_weight: float = 3.0,
        contrastive_weight: float = 1.0,
        supcon_weight: float = 1.5,
        prototype_weight: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.prototype_weight = prototype_weight

        self.base_loss = DualTowerMultiTaskLoss(
            k_nearest=k_nearest,
            distance_weight=distance_weight,
            reconstruction_weight=reconstruction_weight,
            direction_weight=direction_weight,
            region_weight=region_weight,
            contrastive_weight=contrastive_weight,
            supcon_weight=supcon_weight,
            temperature=temperature,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_direction: torch.Tensor,
        pred_region: torch.Tensor,
        true_coords: torch.Tensor,
        direction_labels: torch.Tensor,
        region_labels: torch.Tensor,
        positive_indices: Optional[torch.Tensor] = None,
        negative_indices: Optional[torch.Tensor] = None,
        proto_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:

        total_loss, loss_dict = self.base_loss(
            embeddings=embeddings,
            pred_coords=pred_coords,
            pred_direction=pred_direction,
            pred_region=pred_region,
            true_coords=true_coords,
            direction_labels=direction_labels,
            region_labels=region_labels,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
        )

        if proto_loss is not None:
            total_loss = total_loss + self.prototype_weight * proto_loss
            loss_dict["prototype"] = proto_loss.item()
        else:
            loss_dict["prototype"] = 0.0

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ============ 工厂函数 ============

def build_ultimate_encoder(config) -> UltimateSpatialEncoder:
    """从 V26ProConfig 构建终极编码器。"""
    m = config.model
    sa = config.spatial_attention
    pt = config.prototype

    return UltimateSpatialEncoder(
        point_feat_dim=m.point_feature_dim,
        line_feat_dim=m.line_feature_dim,
        polygon_feat_dim=m.polygon_feature_dim,
        direction_feat_dim=m.direction_feature_dim,
        hidden_dim=m.hidden_dim,
        embedding_dim=m.embedding_dim,
        num_layers=m.num_encoder_layers,
        num_direction_classes=m.num_direction_classes,
        num_region_classes=m.num_region_classes,
        dropout=m.attention_dropout,
        attn_num_heads=sa.num_heads,
        attn_context_k=sa.context_k,
        use_distance_encoding=sa.use_distance_encoding,
        n_prototypes=pt.n_prototypes,
        proto_temperature=pt.proto_temperature,
    )


if __name__ == "__main__":
    from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
    from spatial_encoder.v26_GLM.encoder_v26_mlp import count_parameters

    model = build_ultimate_encoder(DEFAULT_PRO_CONFIG)
    params = count_parameters(model)
    print(f"UltimateSpatialEncoder: {params['total_m']:.2f}M parameters")

    B, K = 16, 20
    point_feat = torch.randn(B, 32)
    line_feat = torch.randn(B, 16)
    polygon_feat = torch.randn(B, 16)
    direction_feat = torch.randn(B, 8)
    neighbor_features = torch.randn(B, K, 72)
    neighbor_distances = torch.rand(B, K) * 0.1
    region_labels = torch.randint(0, 7, (B,))

    emb, dir_pred, reg_pred, coord_pred, proto_loss = model(
        point_feat, line_feat, polygon_feat, direction_feat,
        neighbor_features, neighbor_distances, region_labels,
    )

    print(f"emb: {emb.shape}, dir: {dir_pred.shape}, reg: {reg_pred.shape}")
    print(f"proto_loss: {proto_loss.item():.4f}")
    print(f"L2 norm: {emb.norm(dim=-1).mean():.4f}")
