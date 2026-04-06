# -*- coding: utf-8 -*-
"""
时空注意力编码器 - Phase 3

核心组件：
1. DistancePositionEncoding - 对数尺度距离位置编码（Sinusoidal）
2. SpatialAttentionEncoder  - 多头注意力聚合邻居信息（替代 GNN）

设计原则：
- 无需显式构建图结构（预计算 K 近邻即可）
- 显存友好：+0.6M 参数，+0.4GB 显存
- 可解释性：注意力权重反映邻居重要性

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistancePositionEncoding(nn.Module):
    """
    距离位置编码（对数尺度 Sinusoidal）

    将距离编码为高维向量，让注意力感知空间关系。
    使用对数尺度处理距离大范围变化（10m ~ 10km）。

    编码公式：
        log_d = log(d + 1)
        enc[2i]   = sin(log_d / 10000^(2i/D))
        enc[2i+1] = cos(log_d / 10000^(2i/D))
    """

    def __init__(self, d_model: int = 352):
        super().__init__()
        self.d_model = d_model

        # 预计算频率（固定，不可学习）
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [N, K] 归一化距离值

        Returns:
            encoding: [N, K, D]
        """
        N, K = distances.shape
        device = distances.device

        log_dist = torch.log(distances.clamp(min=1e-6) + 1.0)  # [N, K]
        position = log_dist.unsqueeze(-1)                        # [N, K, 1]

        encoding = torch.zeros(N, K, self.d_model, device=device)
        encoding[:, :, 0::2] = torch.sin(position * self.div_term)
        encoding[:, :, 1::2] = torch.cos(position * self.div_term)

        return encoding


class SpatialAttentionEncoder(nn.Module):
    """
    时空注意力编码器

    架构：
        poi_emb [D]
            ↓
        query = poi_emb.unsqueeze(1)                    [N, 1, D]
        key/value = neighbor_embs + dist_encoding       [N, K, D]
            ↓
        context, attn_weights = MHA(query, key, value)  [N, 1, D]
            ↓
        enhanced = LayerNorm(poi_emb + context.squeeze) [N, D]
            ↓
        output = FFN(enhanced)                          [N, D]
            ↓
        L2 normalize                                    [N, D]

    参数预算：
    - MHA(352, 4):  ~0.5M
    - FFN(352→352): ~0.25M
    - 总计: ~0.75M
    """

    def __init__(
        self,
        embedding_dim: int = 352,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_distance_encoding: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_distance_encoding = use_distance_encoding

        if use_distance_encoding:
            self.dist_encoding = DistancePositionEncoding(embedding_dim)

        # 多头注意力（batch_first=True）
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Post-attention LayerNorm（Pre-LN 风格更稳定）
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        poi_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        neighbor_distances: Optional[torch.Tensor] = None,
        cell_context: Optional[torch.Tensor] = None,
        cell_distances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            poi_emb:            [N, D] 查询 POI 的 embedding（L2 归一化）
            neighbor_embs:      [N, K, D] K 个邻居的 embedding（L2 归一化）
            neighbor_distances: [N, K] 邻居距离（可选，用于位置编码）
            cell_context:       [N, D] 所属 Cell 的宏观 embedding（可选，层次化多尺度）
            cell_distances:     [N, 1] POI 到 Cell 中心的距离（可选）

        Returns:
            enhanced_emb:  [N, D] 上下文增强的 L2 归一化 embedding
            attn_weights:  [N, K] 注意力权重（可解释性，不含 cell_context 位置）
        """
        N, D = poi_emb.shape
        K = neighbor_embs.size(1)

        # 加入距离位置编码
        if self.use_distance_encoding and neighbor_distances is not None:
            dist_enc = self.dist_encoding(neighbor_distances)  # [N, K, D]
            kv = neighbor_embs + dist_enc
        else:
            kv = neighbor_embs

        # 将 cell_context 作为额外的宏观邻居拼入 key/value 序列
        if cell_context is not None:
            cell_emb = cell_context.unsqueeze(1)  # [N, 1, D]
            if self.use_distance_encoding and cell_distances is not None:
                cell_dist_enc = self.dist_encoding(cell_distances)  # [N, 1, D]
                cell_emb = cell_emb + cell_dist_enc
            kv = torch.cat([kv, cell_emb], dim=1)  # [N, K+1, D]

        # Pre-LN: 归一化 query
        query = self.norm1(poi_emb).unsqueeze(1)  # [N, 1, D]

        # 多头注意力
        context, attn_weights_full = self.attention(
            query=query,
            key=kv,
            value=kv,
            need_weights=True,
            average_attn_weights=True,
        )  # context: [N, 1, D], attn_weights_full: [N, 1, K(+1)]

        context = context.squeeze(1)                    # [N, D]
        attn_weights = attn_weights_full.squeeze(1)[:, :K]  # [N, K] 只返回邻居部分

        # 残差连接 1
        x = poi_emb + context

        # FFN + 残差连接 2
        x = x + self.ffn(self.norm2(x))

        # L2 归一化输出
        enhanced_emb = F.normalize(x, p=2, dim=-1)

        return enhanced_emb, attn_weights


# ============ 邻居预计算工具 ============

def precompute_neighbors(
    coords: np.ndarray,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    离线预计算 K 近邻索引和距离。

    Args:
        coords: [N, 2] 坐标数组（归一化后）
        k:      邻居数量

    Returns:
        neighbor_indices:   [N, K] 邻居索引
        neighbor_distances: [N, K] 邻居距离（归一化）
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 排除自身（第 0 列）
    neighbor_indices = indices[:, 1:].astype(np.int64)
    neighbor_distances = distances[:, 1:].astype(np.float32)

    return neighbor_indices, neighbor_distances


if __name__ == "__main__":
    # 快速自测
    N, K, D = 32, 20, 352

    poi_emb = F.normalize(torch.randn(N, D), dim=-1)
    neighbor_embs = F.normalize(torch.randn(N, K, D), dim=-1)
    neighbor_dists = torch.rand(N, K) * 0.1  # 归一化距离

    encoder = SpatialAttentionEncoder(embedding_dim=D, num_heads=4)
    enhanced, weights = encoder(poi_emb, neighbor_embs, neighbor_dists)

    print(f"Enhanced emb: {enhanced.shape}")   # [32, 352]
    print(f"Attn weights: {weights.shape}")    # [32, 20]
    print(f"L2 norm check: {enhanced.norm(dim=-1).mean():.4f}")  # ~1.0
