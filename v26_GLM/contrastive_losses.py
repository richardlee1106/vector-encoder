# -*- coding: utf-8 -*-
"""
对比学习损失函数 - 双塔架构专用

核心组件：
1. InfoNCELoss:  数值稳定的 InfoNCE 损失
2. SupConLoss:   监督对比学习（同类聚合，直接优化 Intra-class Recall）
3. PositiveNegativeSampler: 空间感知的正负样本采样器
4. DualTowerMultiTaskLoss:  集成 SupCon + Focal Region + 多任务损失

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失 - 数值稳定版

    使用 F.cross_entropy 内置的 log-sum-exp trick，避免数值溢出。

    损失公式：
        L = -log(exp(sim(q, pos)/τ) / Σ_j exp(sim(q, neg_j)/τ))

    等价于：
        L = -sim(q, pos)/τ + log(Σ_j exp(sim(q, neg_j)/τ))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 InfoNCE 损失。

        Args:
            query_emb: [B, D] 查询 embedding（L2 归一化）
            positive_emb: [B, D] 正样本 embedding（L2 归一化）
            negative_embs: [B, N, D] N 个负样本 embeddings（L2 归一化）

        Returns:
            loss: 标量损失
            info: 字典，包含 pos_sim 和 neg_sim 均值
        """
        batch_size = query_emb.size(0)
        num_neg = negative_embs.size(1)
        device = query_emb.device

        # 确保归一化
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        positive_emb = F.normalize(positive_emb, p=2, dim=-1)
        negative_embs = F.normalize(negative_embs.view(-1, negative_embs.size(-1)), p=2, dim=-1)
        negative_embs = negative_embs.view(batch_size, num_neg, -1)

        # 正样本相似度: [B]
        pos_sim = (query_emb * positive_emb).sum(dim=-1) / self.temperature

        # 负样本相似度: [B, N]
        neg_sim = torch.bmm(
            negative_embs,
            query_emb.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # 构建 logits: [B, N+1]，第一个位置是正样本
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # 标签: 全部为 0（正样本在第 0 个位置）
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # 使用 cross_entropy 计算（内置 log-sum-exp trick）
        loss = F.cross_entropy(logits, labels)

        # 统计信息
        with torch.no_grad():
            info = {
                "pos_sim_mean": pos_sim.mean().item(),
                "neg_sim_mean": neg_sim.mean().item(),
            }

        return loss, info


class SupConLoss(nn.Module):
    """
    监督对比学习损失（Supervised Contrastive Learning）

    核心思想：
    - 有标签样本：同类别全部作为正样本，异类别作为负样本
    - 直接优化类内聚合 → 提升 Intra-class Recall 和 Region F1

    与 InfoNCE 的区别：
    - InfoNCE: 每个 anchor 只有 1 个正样本
    - SupCon:  每个 anchor 有多个正样本（同类别所有样本）

    参考：Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """

    def __init__(self, temperature: float = 0.07, n_classes: int = 6):
        super().__init__()
        self.temperature = temperature
        self.n_classes = n_classes

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            embeddings: [B, D] L2 归一化的 embedding
            labels:     [B]    功能区标签（0~5 有效，>=6 为未知）

        Returns:
            loss: 标量
            info: 统计字典
        """
        device = embeddings.device

        # 只对有标签样本计算
        valid_mask = labels < self.n_classes
        if valid_mask.sum() < 4:
            return torch.tensor(0.0, device=device), {"supcon": 0.0, "n_valid": 0}

        valid_embs = F.normalize(embeddings[valid_mask], p=2, dim=-1)  # [M, D]
        valid_labels = labels[valid_mask]                               # [M]
        M = valid_embs.size(0)

        # 相似度矩阵 [M, M]，除以温度
        sim = torch.matmul(valid_embs, valid_embs.T) / self.temperature

        # 数值稳定：减去行最大值（不影响梯度）
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        # 正样本 mask：同类别，排除自身对角线
        pos_mask = (valid_labels.unsqueeze(1) == valid_labels.unsqueeze(0))  # [M, M]
        pos_mask.fill_diagonal_(False)

        # 排除自身的 exp(sim)
        self_mask = torch.eye(M, dtype=torch.bool, device=device)
        exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)  # [M, M]

        # log-sum-exp 分母（所有非自身样本）
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # [M, 1]

        # 每对正样本的 log-prob
        log_prob = sim - log_denom  # [M, M]

        # 对每个 anchor，对所有正样本取平均
        n_pos = pos_mask.float().sum(dim=1)  # [M]
        has_pos = n_pos > 0

        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device), {"supcon": 0.0, "n_valid": M}

        loss_per_anchor = -(log_prob * pos_mask.float())[has_pos].sum(dim=1) / n_pos[has_pos]
        loss = loss_per_anchor.mean()

        return loss, {
            "supcon": loss.item(),
            "n_valid": int(valid_mask.sum()),
            "n_with_pos": int(has_pos.sum()),
        }


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Focal Cross-Entropy Loss

    对难分类样本（少数类）加大惩罚，缓解类别不均衡。
    公式：FL = -(1 - p_t)^γ * log(p_t)

    Args:
        logits:          [N, C] 未归一化 logits
        labels:          [N]   类别标签
        weight:          [C]   类别权重（可选）
        gamma:           聚焦参数，越大越关注难样本
        label_smoothing: 标签平滑

    Returns:
        loss: 标量
    """
    # 先算标准 CE（per-sample）
    ce = F.cross_entropy(logits, labels, weight=weight,
                         label_smoothing=label_smoothing, reduction="none")
    # p_t = exp(-ce)
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma * ce).mean()
    return focal



    """
    空间感知的正负样本采样器

    正样本策略：
    1. K=5 空间最近邻
    2. 如果有标签，优先选择同类别

    负样本策略：
    1. 空间远离（距离 > threshold）
    2. 如果有标签，优先选择不同类别
    """

    def __init__(
        self,
        k_positive: int = 5,
        num_negatives: int = 64,
        negative_min_distance: float = 0.05,  # 归一化距离，约 5km
    ):
        self.k_positive = k_positive
        self.num_negatives = num_negatives
        self.negative_min_distance = negative_min_distance

    def precompute(
        self,
        coords: np.ndarray,
        region_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预计算正负样本索引（离线执行，保存到磁盘）。

        Args:
            coords: [N, 2] 坐标数组
            region_labels: [N] 功能区标签（0-5 有效，6 为未知）

        Returns:
            positive_indices: [N, K] 正样本索引
            negative_indices: [N, num_negatives] 负样本索引
        """
        from sklearn.neighbors import NearestNeighbors

        N = len(coords)
        device = 'cpu'

        # 归一化坐标（用于距离阈值）
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)
        coords_norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)

        # K 近邻（正样本候选）
        nbrs = NearestNeighbors(n_neighbors=self.k_positive + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        positive_indices = indices[:, 1:]  # 排除自身

        # 负样本采样
        negative_indices = np.zeros((N, self.num_negatives), dtype=np.int64)

        # 计算所有距离矩阵（分块避免 OOM）
        chunk_size = min(1000, N)

        for i in range(N):
            # 计算到所有点的距离
            dists = np.linalg.norm(coords_norm - coords_norm[i], axis=1)

            # 负样本候选：距离 > threshold
            far_mask = dists > self.negative_min_distance

            # 如果有标签，优先不同类别
            if region_labels is not None and region_labels[i] < 6:
                diff_class_mask = region_labels != region_labels[i]
                candidate_mask = far_mask & diff_class_mask
            else:
                candidate_mask = far_mask

            candidates = np.where(candidate_mask)[0]

            if len(candidates) >= self.num_negatives:
                neg_idx = np.random.choice(candidates, self.num_negatives, replace=False)
            else:
                # 不够则从远处随机补充
                remaining = self.num_negatives - len(candidates)
                far_candidates = np.where(far_mask)[0]
                if len(far_candidates) >= remaining:
                    extra = np.random.choice(far_candidates, remaining, replace=False)
                    neg_idx = np.concatenate([candidates, extra])
                else:
                    # 实在不够，从全局随机采样
                    all_idx = np.arange(N)
                    all_idx = np.setdiff1d(all_idx, [i])
                    neg_idx = np.random.choice(all_idx, self.num_negatives, replace=True)

            negative_indices[i] = neg_idx

        return positive_indices, negative_indices


class PositiveNegativeSampler:
    """
    空间感知的正负样本采样器（离线预计算，供 experiment_dual_tower 使用）

    正样本策略：K=5 空间最近邻（有标签时优先同类别）
    负样本策略：空间远离（距离 > threshold，有标签时优先不同类别）
    """

    def __init__(
        self,
        k_positive: int = 5,
        num_negatives: int = 64,
        negative_min_distance: float = 0.05,
    ):
        self.k_positive = k_positive
        self.num_negatives = num_negatives
        self.negative_min_distance = negative_min_distance

    def precompute(
        self,
        coords: np.ndarray,
        region_labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        N = len(coords)
        coord_min = coords.min(axis=0)
        coord_max = coords.max(axis=0)
        coords_norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)

        nbrs = NearestNeighbors(n_neighbors=self.k_positive + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        positive_indices = indices[:, 1:]

        negative_indices = np.zeros((N, self.num_negatives), dtype=np.int64)

        for i in range(N):
            dists = np.linalg.norm(coords_norm - coords_norm[i], axis=1)
            far_mask = dists > self.negative_min_distance

            if region_labels is not None and region_labels[i] < 6:
                diff_class_mask = region_labels != region_labels[i]
                candidate_mask = far_mask & diff_class_mask
            else:
                candidate_mask = far_mask

            candidates = np.where(candidate_mask)[0]

            if len(candidates) >= self.num_negatives:
                neg_idx = np.random.choice(candidates, self.num_negatives, replace=False)
            else:
                remaining = self.num_negatives - len(candidates)
                far_candidates = np.where(far_mask)[0]
                if len(far_candidates) >= remaining:
                    extra = np.random.choice(far_candidates, remaining, replace=False)
                    neg_idx = np.concatenate([candidates, extra])
                else:
                    all_idx = np.setdiff1d(np.arange(N), [i])
                    neg_idx = np.random.choice(all_idx, self.num_negatives, replace=True)

            negative_indices[i] = neg_idx

        return positive_indices, negative_indices


class DualTowerMultiTaskLoss(nn.Module):
    """
    双塔多任务损失函数 v2

    组合：
    1. 距离保持损失（Pearson + KNN）
    2. 坐标重构损失
    3. 方向分类损失
    4. 功能区分类损失（Focal Loss，缓解类别不均衡）
    5. 空间对比损失（InfoNCE，in-batch spatial negatives）
    6. 监督对比损失（SupCon，同类聚合 → 提升 Intra-class Recall）

    推荐权重：
    - distance_weight    = 0.5
    - contrastive_weight = 1.0
    - supcon_weight      = 1.5   ← 新增，直接优化类内聚合
    - direction_weight   = 1.5
    - region_weight      = 3.0   ← 提高，配合 Focal Loss
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
        temperature: float = 0.07,
        focal_gamma: float = 2.0,
        num_direction_classes: int = 8,
        num_region_classes: int = 6,
    ):
        super().__init__()

        self.distance_weight = distance_weight
        self.reconstruction_weight = reconstruction_weight
        self.direction_weight = direction_weight
        self.region_weight = region_weight
        self.contrastive_weight = contrastive_weight
        self.supcon_weight = supcon_weight
        self.focal_gamma = focal_gamma
        self.k = k_nearest
        self.gamma = 0.5

        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.supcon_loss = SupConLoss(temperature=temperature, n_classes=num_region_classes)

        # 功能区分类权重（少数类加权）
        region_class_weights = torch.tensor([0.3, 2.5, 1.0, 1.8, 1.5, 1.2])
        self.register_buffer("region_class_weights", region_class_weights)

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
        direction_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算多任务损失。

        Args:
            embeddings: [B, D] L2 归一化的 embedding
            pred_coords: [B, 2] 坐标重构
            pred_direction: [B, 8] 方向分类 logits
            pred_region: [B, 6] 功能区分类 logits
            true_coords: [B, 2] 真实坐标
            direction_labels: [B] 方向标签
            region_labels: [B] 功能区标签
            positive_indices: [B, K] 正样本索引（可选，用于对比学习）
            negative_indices: [B, N] 负样本索引（可选，用于对比学习）
            direction_weights: [B] 方向样本权重（可选）

        Returns:
            total_loss: 标量
            loss_dict: 各分量损失值
        """
        device = embeddings.device
        N = embeddings.size(0)
        loss_dict = {}

        # ============ 1. 距离保持损失（Pearson + KNN） ============
        sample_size = min(N, 2000)
        if sample_size < N:
            idx = torch.randperm(N, device=device)[:sample_size]
            emb_sample = embeddings[idx]
            coord_sample = true_coords[idx]
        else:
            emb_sample = embeddings
            coord_sample = true_coords

        emb_dist = torch.cdist(emb_sample, emb_sample, p=2)
        coord_dist = torch.cdist(coord_sample, coord_sample, p=2)

        emb_flat = emb_dist.view(-1)
        coord_flat = coord_dist.view(-1)

        emb_mean = emb_flat.mean()
        coord_mean = coord_flat.mean()
        emb_centered = emb_flat - emb_mean
        coord_centered = coord_flat - coord_mean

        emb_std = torch.sqrt((emb_centered ** 2).mean() + 1e-8)
        coord_std = torch.sqrt((coord_centered ** 2).mean() + 1e-8)

        pearson = (emb_centered * coord_centered).mean() / (emb_std * coord_std + 1e-8)
        pearson_loss = 1 - pearson

        # K 近邻约束
        k = min(self.k, N - 1)
        if k > 0:
            with torch.no_grad():
                coord_dist_full = torch.cdist(true_coords, true_coords, p=2)
                _, knn_idx = torch.topk(coord_dist_full, k=k + 1, largest=False, dim=1)
                knn_idx = knn_idx[:, 1:]

            neighbor_emb = embeddings[knn_idx]
            neighbor_coord = true_coords[knn_idx]

            pred_dist = torch.norm(embeddings.unsqueeze(1) - neighbor_emb, p=2, dim=-1)
            true_dist = torch.norm(true_coords.unsqueeze(1) - neighbor_coord, p=2, dim=-1)

            decay_weight = torch.exp(-self.gamma * true_dist)
            local_loss = ((pred_dist - true_dist) ** 2 * decay_weight).mean()
        else:
            local_loss = torch.tensor(0.0, device=device)

        l_distance = pearson_loss + 0.5 * local_loss
        loss_dict["distance"] = l_distance.item()

        # ============ 2. 坐标重构损失 ============
        l_reconstruct = F.mse_loss(pred_coords, true_coords)
        loss_dict["reconstruction"] = l_reconstruct.item()

        # ============ 3. 方向分类损失 ============
        l_direction = F.cross_entropy(
            pred_direction,
            direction_labels,
            label_smoothing=0.1,
        )
        loss_dict["direction"] = l_direction.item()

        # ============ 4. 功能区分类损失（Focal Loss，缓解类别不均衡） ============
        valid_mask = region_labels < 6
        if valid_mask.sum() > 0:
            valid_pred = pred_region[valid_mask]
            valid_labels = region_labels[valid_mask]
            weights = self.region_class_weights.to(device)
            l_region = focal_cross_entropy(
                valid_pred, valid_labels,
                weight=weights,
                gamma=self.focal_gamma,
                label_smoothing=0.05,
            )
        else:
            l_region = torch.tensor(0.0, device=device)
        loss_dict["region"] = l_region.item()

        # ============ 5. 空间对比损失（in-batch negatives，基于空间距离） ============
        if self.contrastive_weight > 0 and N >= 4:
            coord_dist_batch = torch.cdist(true_coords, true_coords, p=2)
            coord_dist_batch.fill_diagonal_(float('inf'))

            pos_idx_local = coord_dist_batch.argmin(dim=1)
            positive_emb = embeddings[pos_idx_local]

            num_neg = min(64, N - 1)
            neg_idx_local = coord_dist_batch.topk(k=num_neg, largest=True)[1]
            negative_embs = embeddings[neg_idx_local]

            l_contrastive, contrastive_info = self.contrastive_loss(
                query_emb=embeddings,
                positive_emb=positive_emb,
                negative_embs=negative_embs,
            )
            loss_dict["contrastive"] = l_contrastive.item()
            loss_dict["pos_sim"] = contrastive_info["pos_sim_mean"]
            loss_dict["neg_sim"] = contrastive_info["neg_sim_mean"]
        else:
            l_contrastive = torch.tensor(0.0, device=device)
            loss_dict["contrastive"] = 0.0
            loss_dict["pos_sim"] = 0.0
            loss_dict["neg_sim"] = 0.0

        # ============ 6. 监督对比损失（SupCon，同类聚合） ============
        if self.supcon_weight > 0:
            l_supcon, supcon_info = self.supcon_loss(embeddings, region_labels)
            loss_dict["supcon"] = l_supcon.item()
            loss_dict["supcon_n_valid"] = supcon_info.get("n_valid", 0)
        else:
            l_supcon = torch.tensor(0.0, device=device)
            loss_dict["supcon"] = 0.0

        # ============ 总损失 ============
        total_loss = (
            self.distance_weight      * l_distance +
            self.reconstruction_weight* l_reconstruct +
            self.direction_weight     * l_direction +
            self.region_weight        * l_region +
            self.contrastive_weight   * l_contrastive +
            self.supcon_weight        * l_supcon
        )
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试 InfoNCE 损失
    batch_size = 64
    embedding_dim = 352
    num_negatives = 32

    query = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negatives = torch.randn(batch_size, num_negatives, embedding_dim)

    loss_fn = InfoNCELoss(temperature=0.07)
    loss, info = loss_fn(query, positive, negatives)

    print(f"InfoNCE Loss: {loss.item():.4f}")
    print(f"Pos sim: {info['pos_sim_mean']:.4f}, Neg sim: {info['neg_sim_mean']:.4f}")

    # 测试采样器
    print("\nTesting PositiveNegativeSampler...")
    coords = np.random.randn(1000, 2).astype(np.float32)
    labels = np.random.randint(0, 7, 1000)

    sampler = PositiveNegativeSampler(k_positive=5, num_negatives=64)
    pos_idx, neg_idx = sampler.precompute(coords, labels)

    print(f"Positive indices: {pos_idx.shape}")
    print(f"Negative indices: {neg_idx.shape}")
