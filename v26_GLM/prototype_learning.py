# -*- coding: utf-8 -*-
"""
原型学习模块 - Phase 2

核心组件：
1. PrototypeLearning  - 可学习原型 + 原型损失（拉近同类/推远异类）
2. discover_prototypes - 分层 K-Means 初始化原型
3. PrototypeRetriever  - 推理时基于原型的快速检索

设计目标：
- 加速检索：O(P + M*N/P) << O(N)
- 提升 Region F1（类间分离）
- 显存友好：原型参数 ~140KB（可忽略）

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLearning(nn.Module):
    """
    原型学习模块

    每个功能区类别学习多个"原型" embedding（类别中心）。
    - 有标签样本：拉近同类原型，推远异类原型（Triplet-style）
    - 原型分散性：防止不同类别原型坍缩到同一位置

    参数预算：n_prototypes=100, D=352 → ~140KB（可忽略）
    """

    def __init__(
        self,
        n_prototypes: int = 100,
        embedding_dim: int = 352,
        n_classes: int = 6,
        temperature: float = 0.1,
        margin: float = 0.3,
    ):
        super().__init__()

        self.n_prototypes = n_prototypes
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.temperature = temperature
        self.margin = margin

        # 可学习的原型 embeddings
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))

        # 原型类别归属（固定）：均匀分配，每类 n_prototypes // n_classes 个
        proto_per_class = n_prototypes // n_classes
        prototype_classes = torch.zeros(n_prototypes, dtype=torch.long)
        for c in range(n_classes):
            start = c * proto_per_class
            end = start + proto_per_class if c < n_classes - 1 else n_prototypes
            prototype_classes[start:end] = c
        self.register_buffer("prototype_classes", prototype_classes)

    def get_normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            embeddings: [N, D] L2 归一化的 embedding
            labels:     [N]    功能区标签（0-5 有效，>=6 为未知）

        Returns:
            loss: 标量
            info: 统计字典
        """
        device = embeddings.device
        prototypes = self.get_normalized_prototypes()  # [P, D]

        # 只对有标签样本计算损失
        valid_mask = labels < self.n_classes
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device), {"proto_loss": 0.0, "diversity_loss": 0.0}

        valid_embs = embeddings[valid_mask]    # [M, D]
        valid_labels = labels[valid_mask]      # [M]

        # 相似度矩阵 [M, P]
        sim = torch.matmul(valid_embs, prototypes.T)

        # 同类 / 异类 mask [M, P]
        same_mask = (valid_labels.unsqueeze(1) == self.prototype_classes.unsqueeze(0))
        diff_mask = ~same_mask

        # 正样本：同类原型平均相似度
        pos_sim = (sim * same_mask.float()).sum(dim=1) / (same_mask.float().sum(dim=1) + 1e-8)

        # 负样本：异类原型最大相似度（最难负样本）
        neg_sim = (sim * diff_mask.float() + (~diff_mask).float() * -1e9).max(dim=1)[0]

        # Triplet loss with margin
        proto_loss = F.relu(neg_sim - pos_sim + self.margin).mean()

        # 原型分散性损失（不同类别原型之间保持距离）
        proto_sim = torch.matmul(prototypes, prototypes.T)  # [P, P]
        diff_proto_mask = (
            self.prototype_classes.unsqueeze(1) != self.prototype_classes.unsqueeze(0)
        ).float()
        diversity_loss = (proto_sim * diff_proto_mask).clamp(min=0).mean()

        total_loss = proto_loss + 0.1 * diversity_loss

        info = {
            "proto_loss": proto_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "pos_sim_mean": pos_sim.mean().item(),
            "neg_sim_mean": neg_sim.mean().item(),
        }
        return total_loss, info

    @torch.no_grad()
    def get_assignments(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        推理：返回每个 embedding 最近原型对应的类别。

        Args:
            embeddings: [N, D] L2 归一化

        Returns:
            pred_classes: [N] 预测功能区类别
        """
        prototypes = self.get_normalized_prototypes()
        sim = torch.matmul(embeddings, prototypes.T)  # [N, P]
        nearest = sim.argmax(dim=1)                   # [N]
        return self.prototype_classes[nearest]         # [N]

    def init_from_array(self, init_prototypes: np.ndarray) -> None:
        """用预计算的原型数组初始化参数。"""
        tensor = torch.from_numpy(init_prototypes).float()
        with torch.no_grad():
            self.prototypes.copy_(tensor)


# ============ 初始化工具 ============

def discover_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_prototypes: int = 100,
    n_classes: int = 6,
) -> np.ndarray:
    """
    分层 K-Means 初始化原型。

    策略：
    1. 按类别分组
    2. 每类内 K-Means 聚类
    3. 返回聚类中心（L2 归一化）

    Args:
        embeddings: [N, D]
        labels:     [N]  0-5 有效
        n_prototypes: 总原型数
        n_classes:    类别数

    Returns:
        init_prototypes: [n_prototypes, D] L2 归一化
    """
    from sklearn.cluster import KMeans

    D = embeddings.shape[1]
    proto_per_class = n_prototypes // n_classes
    init_prototypes = np.random.randn(n_prototypes, D).astype(np.float32)

    for c in range(n_classes):
        mask = labels == c
        start = c * proto_per_class
        end = start + proto_per_class if c < n_classes - 1 else n_prototypes
        n_needed = end - start

        if mask.sum() < 2:
            continue  # 保持随机初始化

        class_embs = embeddings[mask]
        n_clusters = min(n_needed, max(1, mask.sum() // 5))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
        kmeans.fit(class_embs)
        centers = kmeans.cluster_centers_  # [n_clusters, D]

        # 不够则重复填充
        if len(centers) < n_needed:
            repeats = (n_needed + len(centers) - 1) // len(centers)
            centers = np.tile(centers, (repeats, 1))[:n_needed]

        init_prototypes[start:end] = centers[:n_needed]

    # L2 归一化
    norms = np.linalg.norm(init_prototypes, axis=1, keepdims=True)
    return init_prototypes / (norms + 1e-8)


# ============ 快速检索 ============

class PrototypeRetriever:
    """
    基于原型的快速 POI 检索。

    推理流程：
    1. 找到 query 最近的 M 个原型
    2. 只在这些原型对应的 POI 中精确搜索
    复杂度：O(P + M*N/P) << O(N)
    """

    def __init__(self, top_m_prototypes: int = 10):
        self.top_m = top_m_prototypes
        self.prototypes: Optional[np.ndarray] = None       # [P, D]
        self.proto_to_poi: Optional[List[List[int]]] = None  # P 个列表

    def build_index(
        self,
        prototypes: np.ndarray,
        embeddings: np.ndarray,
        prototype_classes: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        构建原型 → POI 映射。

        Args:
            prototypes:       [P, D] 原型 embeddings
            embeddings:       [N, D] 所有 POI embeddings
            prototype_classes:[P]   原型类别
            labels:           [N]   POI 标签（-1 表示无标签）
        """
        self.prototypes = prototypes
        P = len(prototypes)
        N = len(embeddings)

        # 每个 POI 分配到最近原型
        sim = embeddings @ prototypes.T  # [N, P]
        assignments = sim.argmax(axis=1)  # [N]

        self.proto_to_poi = [[] for _ in range(P)]
        for poi_idx, proto_idx in enumerate(assignments):
            self.proto_to_poi[proto_idx].append(poi_idx)

    def search(
        self,
        query_emb: np.ndarray,
        all_embeddings: np.ndarray,
        k: int = 20,
    ) -> List[int]:
        """
        快速检索 Top-K 相似 POI。

        Args:
            query_emb:     [D] 查询 embedding
            all_embeddings:[N, D] 所有 POI embeddings
            k:             返回数量

        Returns:
            top_k_indices: List[int]
        """
        assert self.prototypes is not None, "先调用 build_index()"

        # Step 1: 找最近的 M 个原型
        proto_sim = query_emb @ self.prototypes.T  # [P]
        top_proto_idx = np.argsort(proto_sim)[::-1][:self.top_m]

        # Step 2: 收集候选 POI
        candidates = []
        for idx in top_proto_idx:
            candidates.extend(self.proto_to_poi[idx])
        candidates = list(set(candidates))

        if len(candidates) == 0:
            candidates = list(range(len(all_embeddings)))

        # Step 3: 精确搜索
        cand_embs = all_embeddings[candidates]
        cand_sim = query_emb @ cand_embs.T  # [C]
        top_k_local = np.argsort(cand_sim)[::-1][:k]

        return [candidates[i] for i in top_k_local]


if __name__ == "__main__":
    # 快速自测
    B, D, P = 64, 352, 100
    embs = F.normalize(torch.randn(B, D), dim=-1)
    labels = torch.randint(0, 7, (B,))

    proto = PrototypeLearning(n_prototypes=P, embedding_dim=D)
    loss, info = proto(embs, labels)
    print(f"Proto loss: {loss.item():.4f}, info: {info}")

    preds = proto.get_assignments(embs)
    print(f"Assignments shape: {preds.shape}, unique: {preds.unique()}")
