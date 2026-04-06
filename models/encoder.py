# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 核心模型

核心能力：
1. 学习空间位置（坐标重构）
2. 学习相对距离（距离保持）
3. 学习邻接关系（邻居一致性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import SpatialEncoderConfig


class SpatialTopologyEncoder(nn.Module):
    """
    空间拓扑编码器 V2.3

    输入: POI特征 + 坐标
    输出: 空间embedding（保留拓扑关系）

    损失函数组合：
    - 坐标重构损失：保持绝对位置
    - 距离保持损失：保持相对距离
    - 邻居一致性损失：保持邻接关系
    """

    def __init__(self, config: SpatialEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        # Embedding层
        emb_dim = self.hidden_dim // 6
        self.category_emb = nn.Embedding(config.num_categories + 1, emb_dim)
        self.landuse_emb = nn.Embedding(config.num_landuses + 1, emb_dim)
        self.aoi_type_emb = nn.Embedding(config.num_aoi_types + 1, emb_dim)
        self.road_class_emb = nn.Embedding(config.num_road_classes + 1, emb_dim)
        self.num_proj = nn.Linear(3, emb_dim)
        self.coord_proj = nn.Linear(2, emb_dim)

        # 编码器
        input_dim = emb_dim * 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, config.embed_dim),
        )

        # 坐标解码器
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.embed_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),
        )

    def encode(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        编码：输入 → embedding

        Args:
            features: [N, 7] (category, landuse, aoi_type, road_class, density, entropy, road_dist)
            coords: [N, 2] (lng, lat)

        Returns:
            embedding: [N, D] 归一化后的向量
        """
        # Embedding
        cat_emb = self.category_emb(features[:, 0].long())
        lu_emb = self.landuse_emb(features[:, 1].long())
        aoi_emb = self.aoi_type_emb(features[:, 2].long())
        rc_emb = self.road_class_emb(features[:, 3].long())
        num_emb = self.num_proj(features[:, 4:7])

        # 坐标归一化
        coords_norm = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-8)
        coord_emb = self.coord_proj(coords_norm)

        # 拼接并编码
        x = torch.cat([cat_emb, lu_emb, aoi_emb, rc_emb, num_emb, coord_emb], dim=-1)
        z = self.encoder(x)

        return F.normalize(z, p=2, dim=-1)

    def decode_coord(self, z: torch.Tensor) -> torch.Tensor:
        """解码：embedding → 坐标"""
        return self.coord_decoder(z)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        z = self.encode(features, coords)
        coord_recon = self.decode_coord(z)
        return z, coord_recon


class DistancePreserveLoss(nn.Module):
    """
    距离保持损失

    让embedding空间中的距离关系与原始空间一致
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, pair_indices: torch.Tensor,
                spatial_dists: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, D] embeddings
            pair_indices: [P, 2] 采样的点对索引
            spatial_dists: [P] 归一化后的空间距离
        """
        z_i = z[pair_indices[:, 0]]
        z_j = z[pair_indices[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        return F.mse_loss(emb_dists, spatial_dists)


class NeighborConsistencyLoss(nn.Module):
    """
    邻居一致性损失

    让KNN邻居的embedding相似
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, knn_neighbors: List) -> torch.Tensor:
        """
        Args:
            z: [N, D] embeddings
            knn_neighbors: list of neighbor indices
        """
        total_loss = 0.0
        count = 0

        sample_size = min(1000, len(knn_neighbors))
        sample_indices = torch.randint(0, len(knn_neighbors), (sample_size,))

        for i in sample_indices:
            i = i.item()
            neighbors = knn_neighbors[i]
            if len(neighbors) > 0:
                z_center = z[i]
                z_neighbors = z[neighbors]
                cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_neighbors, dim=1)
                total_loss += (1 - cos_sim.mean())
                count += 1

        return total_loss / max(count, 1)


def sample_distance_pairs(coords: np.ndarray, num_pairs: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """采样点对并计算空间距离"""
    N = len(coords)

    idx_i = np.random.randint(0, N, num_pairs)
    idx_j = np.random.randint(0, N, num_pairs)

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    coords_i = coords[idx_i]
    coords_j = coords[idx_j]
    spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))

    # 归一化
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    return (
        torch.from_numpy(np.stack([idx_i, idx_j], axis=1)).long().to(device),
        torch.from_numpy(spatial_dists).float().to(device),
    )


def build_knn_neighbors(coords: np.ndarray, k: int = 10) -> List:
    """构建KNN邻居列表"""
    from sklearn.neighbors import kneighbors_graph

    adj = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    return [adj[i].nonzero()[1] for i in range(len(coords))]
