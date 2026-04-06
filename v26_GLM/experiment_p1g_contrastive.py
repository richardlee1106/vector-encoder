# -*- coding: utf-8 -*-
"""
P1C-Fix: 集成多方案方向训练 + 功能区对比学习

目标：
1. 恢复 P0 的方向训练效果 (DirAcc > 60%)
2. 保持 P1 的功能区对比学习 (Region F1 > 30%)

损失函数：
L_total = 3.0 * L_distance
        + 1.0 * L_reconstruct
        + 2.0 * L_direction (neighbor 0.6 + global 0.4)
        + 0.3 * L_region (contrastive)

Author: GLM (Qianfan Code)
Date: 2026-03-17
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import random
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
# tqdm removed to reduce console output

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import CellEncoderMLP, build_mlp_encoder, count_parameters
from spatial_encoder.v26_GLM.direction_supervision import (
    MultiSchemeDirectionSupervision,
    DirectionScheme,
    compute_global_center_direction,
    compute_neighbor_relative_direction,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ P0: 多方案方向损失 ============

class MultiSchemeDirectionLoss(nn.Module):
    """多方案联合方向损失（P0）"""

    def __init__(
        self,
        neighbor_weight: float = 0.6,
        global_center_weight: float = 0.4,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.neighbor_weight = neighbor_weight
        self.global_center_weight = global_center_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def focal_loss(self, pred_logits, targets, weight=None):
        ce_loss = F.cross_entropy(
            pred_logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        probs = F.softmax(pred_logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss = focal_weight * ce_loss
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    def forward(
        self,
        pred_logits: torch.Tensor,
        neighbor_labels: torch.Tensor,
        neighbor_valid: torch.Tensor,
        global_labels: torch.Tensor,
        global_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        device = pred_logits.device
        loss_dict = {}

        # 邻居方向损失
        neighbor_loss = torch.tensor(0.0, device=device)
        neighbor_mask = neighbor_valid.bool()
        if neighbor_mask.sum() > 0:
            pred_masked = pred_logits[neighbor_mask]
            labels_masked = neighbor_labels[neighbor_mask]
            neighbor_loss = self.focal_loss(pred_masked, labels_masked)
        loss_dict["neighbor_dir"] = neighbor_loss.item()

        # 全局方向损失
        global_loss = torch.tensor(0.0, device=device)
        global_mask = global_valid.bool()
        if global_mask.sum() > 0:
            pred_masked = pred_logits[global_mask]
            labels_masked = global_labels[global_mask]
            global_loss = self.focal_loss(pred_masked, labels_masked)
        loss_dict["global_dir"] = global_loss.item()

        # 加权组合
        total_loss = self.neighbor_weight * neighbor_loss + self.global_center_weight * global_loss
        loss_dict["total_dir"] = total_loss.item()

        return total_loss, loss_dict


# ============ P1: 功能区分类损失（P1F: 交叉熵替代对比学习） ============

class RegionClassificationLossP1F(nn.Module):
    """
    P1F: 功能区分类损失（交叉熵）

    替代对比学习原因：
    1. 对比学习需要 batch 内正样本对，17% 标签密度效率太低
    2. 交叉熵每个有标签样本直接提供梯度信号
    3. 效率提升约 8-10 倍
    """

    def __init__(
        self,
        num_classes: int = 6,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 6,  # 未知标签
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # 类别权重：居住类权重低（样本多），商业类权重高（样本少）
        if class_weights is None:
            class_weights = torch.tensor([0.3, 2.0, 1.0, 1.5, 1.5, 1.2])
        self.register_buffer("class_weights", class_weights)

        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,  # [batch, 6]
        region_labels: torch.Tensor,  # [batch]
    ) -> Tuple[torch.Tensor, Dict]:
        device = pred_logits.device

        # 确保损失函数在正确的设备上
        if self.loss_fn.weight.device != device:
            self.loss_fn.weight = self.loss_fn.weight.to(device)

        # 计算损失（ignore_index=6 自动过滤未知标签）
        loss = self.loss_fn(pred_logits, region_labels)

        # 计算准确率（仅有效标签）
        valid_mask = region_labels != self.ignore_index
        if valid_mask.sum() > 0:
            pred_classes = pred_logits[valid_mask].argmax(dim=-1)
            true_classes = region_labels[valid_mask]
            accuracy = (pred_classes == true_classes).float().mean()
        else:
            accuracy = torch.tensor(0.0, device=device)

        return loss, {
            "region": loss.item(),
            "region_acc": accuracy.item() * 100,
        }


# ============ 集成多任务损失 ============

class IntegratedMultiTaskLoss(nn.Module):
    """
    P1F 集成多任务损失 - 权重翻转版

    关键修改：
    - distance_weight: 3.0 → 0.5（不再霸占梯度）
    - region_weight: 0.3 → 2.0（让分类信号占主导）

    L_total = 0.5 * L_distance
            + 0.5 * L_reconstruct
            + 1.5 * L_direction
            + 2.0 * L_region（交叉熵）
    """

    def __init__(
        self,
        k_nearest: int = 85,
        distance_weight: float = 0.5,    # P1F: 从 3.0 降到 0.5
        reconstruction_weight: float = 0.5,  # P1F: 从 1.0 降到 0.5
        direction_weight: float = 1.5,   # P1F: 保持方向
        region_weight: float = 2.0,      # P1F: 从 0.3 提到 2.0
        neighbor_dir_weight: float = 0.6,
        global_dir_weight: float = 0.4,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        self.distance_weight = distance_weight
        self.reconstruction_weight = reconstruction_weight
        self.direction_weight = direction_weight
        self.region_weight = region_weight

        self.k = k_nearest
        self.gamma = 0.5

        # P0: 方向损失
        self.direction_loss = MultiSchemeDirectionLoss(
            neighbor_weight=neighbor_dir_weight,
            global_center_weight=global_dir_weight,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
        )

        # P1F: 功能区分类损失（交叉熵替代对比学习）
        self.region_loss = RegionClassificationLossP1F(
            num_classes=6,
            class_weights=torch.tensor([0.3, 2.0, 1.0, 1.5, 1.5, 1.2]),
            ignore_index=6,
            label_smoothing=0.1,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_direction: torch.Tensor,
        pred_region: torch.Tensor,  # P1F: 新增，region 分类 logits
        true_coords: torch.Tensor,
        neighbor_dir_labels: torch.Tensor,
        neighbor_dir_valid: torch.Tensor,
        global_dir_labels: torch.Tensor,
        global_dir_valid: torch.Tensor,
        region_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        device = embeddings.device
        N = embeddings.size(0)
        loss_dict = {}

        # ============ 1. 距离保持损失 ============
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

        # K近邻约束
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

        # ============ 3. P0: 多方案方向损失 ============
        l_direction, dir_dict = self.direction_loss(
            pred_logits=pred_direction,
            neighbor_labels=neighbor_dir_labels,
            neighbor_valid=neighbor_dir_valid,
            global_labels=global_dir_labels,
            global_valid=global_dir_valid,
        )
        loss_dict.update(dir_dict)

        # ============ 4. P1F: 功能区分类损失（交叉熵） ============
        l_region, region_dict = self.region_loss(pred_region, region_labels)
        loss_dict.update(region_dict)

        # ============ 总损失 ============
        total_loss = (
            self.distance_weight * l_distance +
            self.reconstruction_weight * l_reconstruct +
            self.direction_weight * l_direction +
            self.region_weight * l_region
        )
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


# ============ 数据集 ============

class CellDatasetP1C(Dataset):
    """P1C集成数据集"""

    def __init__(self, data: Dict):
        self.point_features = torch.tensor(data["point_features"], dtype=torch.float32)
        self.line_features = torch.tensor(data["line_features"], dtype=torch.float32)
        self.polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)

        self.neighbor_dir_labels = torch.tensor(data["neighbor_dir_labels"], dtype=torch.long)
        self.neighbor_dir_valid = torch.tensor(data["neighbor_dir_valid"], dtype=torch.bool)
        self.global_dir_labels = torch.tensor(data["global_dir_labels"], dtype=torch.long)
        self.global_dir_valid = torch.tensor(data["global_dir_valid"], dtype=torch.bool)

        self.region_labels = torch.tensor(data["region_labels"], dtype=torch.long)
        # 原始标签用于评估（不受伪标签影响）
        self.original_region_labels = torch.tensor(
            data.get("original_region_labels", data["region_labels"]), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.point_features)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "point_feat": self.point_features[idx],
            "line_feat": self.line_features[idx],
            "polygon_feat": self.polygon_features[idx],
            "direction_feat": self.direction_features[idx],
            "coords": self.coords[idx],
            "neighbor_dir_label": self.neighbor_dir_labels[idx],
            "neighbor_dir_valid": self.neighbor_dir_valid[idx],
            "global_dir_label": self.global_dir_labels[idx],
            "global_dir_valid": self.global_dir_valid[idx],
            "region_label": self.region_labels[idx],
            "original_region_label": self.original_region_labels[idx],
            "idx": idx,
        }


def collate_fn_p1c(batch: List[Dict]) -> Dict:
    return {
        "point_feat": torch.stack([item["point_feat"] for item in batch]),
        "line_feat": torch.stack([item["line_feat"] for item in batch]),
        "polygon_feat": torch.stack([item["polygon_feat"] for item in batch]),
        "direction_feat": torch.stack([item["direction_feat"] for item in batch]),
        "coords": torch.stack([item["coords"] for item in batch]),
        "neighbor_dir_label": torch.stack([item["neighbor_dir_label"] for item in batch]),
        "neighbor_dir_valid": torch.stack([item["neighbor_dir_valid"] for item in batch]),
        "global_dir_label": torch.stack([item["global_dir_label"] for item in batch]),
        "global_dir_valid": torch.stack([item["global_dir_valid"] for item in batch]),
        "region_label": torch.stack([item["region_label"] for item in batch]),
        "original_region_label": torch.stack([item["original_region_label"] for item in batch]),
    }


# ============ 数据加载 ============

def load_data_integrated(config: V26ProConfig, sample_ratio: float = 0.1, use_neighbor_feats: bool = True, use_pseudo_labels: bool = False) -> Dict:
    """
    加载集成数据（方向 + 功能区 + 邻域特征 + 伪标签）

    P1E: 支持加载邻域聚合特征，扩展输入维度 [72] → [112]
    P1E+: 支持伪标签策略，解决标签稀疏问题（默认禁用，噪声太大）
    """
    try:
        from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
        from pathlib import Path

        print(f"Loading data (sample_ratio={sample_ratio})...")

        # 加载基础数据
        base_data = load_dataset_for_training(config=config, sample_ratio=sample_ratio)
        coords = base_data["coords"]
        neighbor_indices = base_data["neighbor_indices"]
        N = len(coords)

        print(f"  Loaded {N} cells")

        # P1E: 加载邻域特征
        neighbor_feats = None
        if use_neighbor_feats:
            neighbor_feat_path = Path(__file__).parent / "p1e_neighbor_feat.npy"
            if neighbor_feat_path.exists():
                neighbor_feats = np.load(neighbor_feat_path)
                print(f"  P1E: Loaded neighbor features {neighbor_feats.shape}")
                # 如果采样，需要截取
                if len(neighbor_feats) != N:
                    print(f"  Warning: neighbor_feats size mismatch ({len(neighbor_feats)} vs {N}), regenerating...")
                    neighbor_feats = None

        # 如果邻域特征不存在或大小不匹配，实时计算
        if use_neighbor_feats and neighbor_feats is None:
            print("  P1E: Computing neighbor features on-the-fly...")
            from spatial_encoder.v26_GLM.p1e_neighbor_features import compute_neighbor_features
            neighbor_feats = compute_neighbor_features(
                coords=coords,
                point_features=base_data["point_features"],
                line_features=base_data["line_features"],
                region_labels=base_data["region_labels"],
                k_neighbors=6,
            )

        # 保留原始标签
        region_labels = base_data["region_labels"]
        original_labels = region_labels.copy()  # 用于评估

        # P1E+: 伪标签策略（默认禁用）
        if use_pseudo_labels:
            from spatial_encoder.v26_GLM.p1e_pseudo_labels import generate_pseudo_labels, analyze_pseudo_labels
            pseudo_labels, pseudo_mask = generate_pseudo_labels(
                base_data["point_features"],
                region_labels,
            )
            stats = analyze_pseudo_labels(region_labels, pseudo_labels, pseudo_mask)
            print(f"  P1E+: Pseudo-labels generated: {stats['pseudo_generated']:,}")
            print(f"  P1E+: Valid label coverage: {stats['original_ratio']:.1%} → {stats['pseudo_ratio']:.1%}")
            region_labels = pseudo_labels  # 用于训练

        # 计算邻居相对方向
        print("  Computing neighbor-relative direction...")
        neighbor_dir = compute_neighbor_relative_direction(coords, neighbor_indices)
        neighbor_labels = np.zeros(N, dtype=np.int64)
        neighbor_valid = np.zeros(N, dtype=bool)
        for i in range(N):
            valid_neighbors = neighbor_dir.valid_mask[i]
            if valid_neighbors.any():
                first_valid = np.where(valid_neighbors)[0][0]
                neighbor_labels[i] = neighbor_dir.labels[i, first_valid]
                neighbor_valid[i] = True

        # 计算全局中心方向
        print("  Computing global-center direction...")
        global_dir = compute_global_center_direction(coords)
        global_labels = global_dir.labels
        global_valid = global_dir.valid_mask

        # P1E: 拼接邻域特征到 point_features
        point_features = base_data["point_features"]
        if neighbor_feats is not None:
            point_features = np.concatenate([point_features, neighbor_feats], axis=1)
            print(f"  P1E: Extended point_features to {point_features.shape}")

        return {
            "point_features": point_features,
            "line_features": base_data["line_features"],
            "polygon_features": base_data["polygon_features"],
            "direction_features": base_data["direction_features"],
            "coords": coords,
            "neighbor_dir_labels": neighbor_labels,
            "neighbor_dir_valid": neighbor_valid,
            "global_dir_labels": global_labels,
            "global_dir_valid": global_valid,
            "region_labels": region_labels,  # 用于训练
            "original_region_labels": original_labels,  # 用于评估
        }

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# ============ 完整评估函数 ============

@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """完整评估：方向 + 功能区 + 空间"""
    model.eval()

    # 方向统计
    neighbor_correct = 0
    neighbor_total = 0
    global_correct = 0
    global_total = 0

    all_embeddings = []
    all_coords = []
    all_region_labels = []
    all_region_preds = []  # P1F-Fix: 收集分类头的直接预测

    for batch in loader:
        point_feat = batch["point_feat"].to(device)
        line_feat = batch["line_feat"].to(device)
        polygon_feat = batch["polygon_feat"].to(device)
        direction_feat = batch["direction_feat"].to(device)
        coords = batch["coords"].to(device)

        emb, dir_pred, reg_pred, coord_pred = model(
            point_feat, line_feat, polygon_feat, direction_feat
        )

        # 方向准确率
        pred_dirs = dir_pred.argmax(dim=-1).cpu()

        neighbor_label = batch["neighbor_dir_label"]
        neighbor_valid = batch["neighbor_dir_valid"]
        neighbor_mask = neighbor_valid.bool()
        if neighbor_mask.sum() > 0:
            neighbor_correct += (pred_dirs[neighbor_mask] == neighbor_label[neighbor_mask]).sum().item()
            neighbor_total += neighbor_mask.sum().item()

        global_label = batch["global_dir_label"]
        global_valid = batch["global_dir_valid"]
        global_mask = global_valid.bool()
        if global_mask.sum() > 0:
            global_correct += (pred_dirs[global_mask] == global_label[global_mask]).sum().item()
            global_total += global_mask.sum().item()

        all_embeddings.append(emb.cpu())
        all_coords.append(coords.cpu())
        all_region_labels.append(batch["original_region_label"])  # 使用原始标签评估
        all_region_preds.append(reg_pred.cpu())  # P1F-Fix: 收集分类头预测

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_coords = torch.cat(all_coords, dim=0).numpy()
    all_region_labels = torch.cat(all_region_labels, dim=0).numpy()
    all_region_preds = torch.cat(all_region_preds, dim=0).numpy()  # [N, 6]

    results = {
        "neighbor_dir_acc": neighbor_correct / max(neighbor_total, 1) * 100,
        "global_dir_acc": global_correct / max(global_total, 1) * 100,
    }

    # Pearson & Spearman
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

    sample_size = min(1000, len(all_embeddings))  # 减少采样量
    idx = np.random.choice(len(all_embeddings), sample_size, replace=False)

    emb_dist = pairwise_distances(all_embeddings[idx], metric='euclidean').flatten()
    coord_dist = pairwise_distances(all_coords[idx], metric='euclidean').flatten()

    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    results["pearson"] = pearson
    results["spearman"] = spearman

    # Overlap@K
    k = 20
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(all_coords)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(all_embeddings)

    _, coord_idx = nbrs_coord.kneighbors(all_coords)
    _, emb_idx = nbrs_emb.kneighbors(all_embeddings)

    coord_idx = coord_idx[:, 1:]
    emb_idx = emb_idx[:, 1:]

    overlaps = []
    recall_at_k = []
    for i in range(len(all_coords)):
        overlap = len(set(coord_idx[i]) & set(emb_idx[i])) / k
        overlaps.append(overlap)
        # Recall@K: 真实邻居中有多少被找回
        recall = len(set(coord_idx[i]) & set(emb_idx[i])) / len(set(coord_idx[i]))
        recall_at_k.append(recall)
    results["overlap"] = np.mean(overlaps)
    results["recall_at_k"] = np.mean(recall_at_k)

    # Region F1 - 两种评估方式
    from sklearn.metrics import f1_score
    valid_mask = all_region_labels < 6  # 排除未知
    if valid_mask.sum() > 0:
        true_labels = all_region_labels[valid_mask]

        # 方式1: embedding 聚类 F1（KMeans，与 L1/L2 指标对齐）
        from sklearn.cluster import KMeans
        n_clusters = 6
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        pred_clusters = kmeans.fit_predict(all_embeddings[valid_mask])

        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                cost_matrix[i, j] = -np.sum((pred_clusters == i) & (true_labels == j))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {row: col for row, col in zip(row_ind, col_ind)}

        mapped_preds = np.array([mapping.get(c, c) for c in pred_clusters])
        region_f1 = f1_score(true_labels, mapped_preds, average='macro') * 100
        results["region_f1"] = region_f1

        # 方式2: 分类头直接 F1（region_head 的真实分类能力）
        # P1F-Fix 核心：region_head 从 hidden 层分支，这里直接测其预测
        clf_preds = all_region_preds[valid_mask].argmax(axis=-1)
        region_clf_f1 = f1_score(true_labels, clf_preds, average='macro') * 100
        region_clf_acc = (clf_preds == true_labels).mean() * 100
        results["region_clf_f1"] = region_clf_f1
        results["region_clf_acc"] = region_clf_acc
    else:
        results["region_f1"] = 0.0
        results["region_clf_f1"] = 0.0
        results["region_clf_acc"] = 0.0

    # Region Sep
    valid_labels = all_region_labels[valid_mask]
    valid_embeddings = all_embeddings[valid_mask]

    unique_labels = np.unique(valid_labels)
    if len(unique_labels) > 1:
        intra_dist = []
        inter_dist = []

        for label in unique_labels:
            label_mask = valid_labels == label
            label_embs = valid_embeddings[label_mask]
            if len(label_embs) > 1:
                label_dists = pairwise_distances(label_embs)
                intra_dist.append(label_dists[np.triu_indices(len(label_embs), k=1)].mean())

        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                mask1 = valid_labels == label1
                mask2 = valid_labels == label2
                dists = pairwise_distances(valid_embeddings[mask1], valid_embeddings[mask2])
                inter_dist.append(dists.mean())

        if intra_dist and inter_dist:
            results["region_sep"] = np.mean(inter_dist) / (np.mean(intra_dist) + 1e-8)
        else:
            results["region_sep"] = 0.0
    else:
        results["region_sep"] = 0.0

    return results


# ============ 训练器 ============

class TrainerP1C:
    """P1C集成训练器"""

    def __init__(self, model, criterion, config, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.learning_rate * 0.01,
        )

        self.best_loss = float("inf")
        self.history = []

    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in loader:
            point_feat = batch["point_feat"].to(self.device, non_blocking=True)
            line_feat = batch["line_feat"].to(self.device, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(self.device, non_blocking=True)
            direction_feat = batch["direction_feat"].to(self.device, non_blocking=True)
            coords = batch["coords"].to(self.device, non_blocking=True)
            neighbor_dir_label = batch["neighbor_dir_label"].to(self.device, non_blocking=True)
            neighbor_dir_valid = batch["neighbor_dir_valid"].to(self.device, non_blocking=True)
            global_dir_label = batch["global_dir_label"].to(self.device, non_blocking=True)
            global_dir_valid = batch["global_dir_valid"].to(self.device, non_blocking=True)
            region_label = batch["region_label"].to(self.device, non_blocking=True)

            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,  # P1F: 传递 region logits
                true_coords=coords,
                neighbor_dir_labels=neighbor_dir_label,
                neighbor_dir_valid=neighbor_dir_valid,
                global_dir_labels=global_dir_label,
                global_dir_valid=global_dir_valid,
                region_labels=region_label,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict["total"]
            n_batches += 1

        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch in loader:
            point_feat = batch["point_feat"].to(self.device)
            line_feat = batch["line_feat"].to(self.device)
            polygon_feat = batch["polygon_feat"].to(self.device)
            direction_feat = batch["direction_feat"].to(self.device)
            coords = batch["coords"].to(self.device)
            neighbor_dir_label = batch["neighbor_dir_label"].to(self.device)
            neighbor_dir_valid = batch["neighbor_dir_valid"].to(self.device)
            global_dir_label = batch["global_dir_label"].to(self.device)
            global_dir_valid = batch["global_dir_valid"].to(self.device)
            region_label = batch["region_label"].to(self.device)

            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,  # P1F: 传递 region logits
                true_coords=coords,
                neighbor_dir_labels=neighbor_dir_label,
                neighbor_dir_valid=neighbor_dir_valid,
                global_dir_labels=global_dir_label,
                global_dir_valid=global_dir_valid,
                region_labels=region_label,
            )

            total_loss += loss_dict["total"]
            n_batches += 1

        return {"loss": total_loss / n_batches}

    def train(self, train_loader, val_loader, epochs: int = 30):
        print(f"\n{'='*60}")
        print(f"P1C-Fix Training: Direction + Region")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            # 每15个epoch完整评估并打印
            if (epoch + 1) % 15 == 0 or epoch == 0:
                eval_results = evaluate_full(self.model, val_loader, self.device)

                print(f"Epoch {epoch+1:2d}/{epochs}: "
                      f"DirAcc={eval_results['global_dir_acc']:.1f}%, "
                      f"RegionF1={eval_results['region_f1']:.1f}%, "
                      f"ClfF1={eval_results.get('region_clf_f1', 0):.1f}%, "
                      f"ClfAcc={eval_results.get('region_clf_acc', 0):.1f}%, "
                      f"RegionSep={eval_results['region_sep']:.2f}, "
                      f"Pearson={eval_results['pearson']:.4f}")

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            })

            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]

        print(f"\nTraining completed. Best val loss: {self.best_loss:.4f}")

        # 最终评估
        print("\n" + "="*60)
        print("Final Evaluation:")
        results = evaluate_full(self.model, val_loader, self.device)

        print(f"  DirAcc: {results['global_dir_acc']:.2f}%")
        print(f"  Region F1 (KMeans): {results['region_f1']:.2f}%")
        print(f"  Region F1 (分类头): {results.get('region_clf_f1', 0):.2f}%")
        print(f"  Region Clf Acc: {results.get('region_clf_acc', 0):.2f}%")
        print(f"  Region Sep: {results['region_sep']:.4f}")
        print(f"  Pearson: {results['pearson']:.4f}")
        print(f"  Spearman: {results['spearman']:.4f}")
        print(f"  Overlap@K: {results['overlap']*100:.2f}%")
        print(f"  Recall@20: {results['recall_at_k']*100:.2f}%")
        print("="*60)

        return results


# ============ 主函数 ============

def run_p1c_experiment(
    sample_ratio: float = 0.1,
    epochs: int = 30,
    batch_size: int = 16384,
    use_neighbor_feats: bool = False,  # P1E: 禁用邻域特征（已证明无效）
    use_pseudo_labels: bool = True,   # P1F: 启用伪标签（44% 覆盖率）
):
    print(f"\n{'#'*60}")
    print(f"# P1F: Loss Weight Surgery + Classification Head")
    print(f"# Key changes: distance_weight=0.5, region_weight=2.0")
    print(f"# Cross-entropy replaces contrastive learning")
    if use_pseudo_labels:
        print(f"# P1F: Pseudo-labels ENABLED (44% coverage)")
    print(f"# Sample ratio: {sample_ratio}")
    print(f"# Epochs: {epochs}")
    print(f"{'#'*60}\n")

    config = DEFAULT_PRO_CONFIG
    config.training.batch_size = batch_size
    config.training.num_epochs = epochs

    set_seed(config.training.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载数据
    data = load_data_integrated(
        config, sample_ratio,
        use_neighbor_feats=use_neighbor_feats,
        use_pseudo_labels=use_pseudo_labels,
    )
    n_cells = len(data["coords"])

    # P1E: 动态更新 point_feature_dim
    actual_point_dim = data["point_features"].shape[1]
    if actual_point_dim != config.model.point_feature_dim:
        print(f"P1E: Updating point_feature_dim from {config.model.point_feature_dim} to {actual_point_dim}")
        config.model.point_feature_dim = actual_point_dim
        # 更新 neighbor_feature_dim
        config.model.neighbor_feature_dim = actual_point_dim - 32

    dataset = CellDatasetP1C(data)
    train_size = int(0.9 * n_cells)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, n_cells - train_size]
    )

    actual_batch_size = min(batch_size, train_size // 2)
    actual_batch_size = max(actual_batch_size, 64)

    train_loader = DataLoader(
        train_set,
        batch_size=actual_batch_size,
        shuffle=True,
        collate_fn=collate_fn_p1c,
        num_workers=0,  # 减少内存使用
        pin_memory=False,  # 减少内存使用
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collate_fn_p1c,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Dataset: {n_cells} cells, train={train_size}, val={n_cells - train_size}")
    print(f"Batch size: {actual_batch_size}")

    # 模型
    model = build_mlp_encoder(config)
    params = count_parameters(model)
    print(f"Model: {params['total_m']:.1f}M parameters")

    # 损失函数
    # P1F: 权重翻转 - 让 region 信号占主导
    # P1F-Final 最终平衡权重（经架构解耦验证）
    distance_weight = 0.5    # 稍微拉回，确保 Pearson > 0.97
    reconstruction_weight = 0.3
    direction_weight = 1.5   # 确保方向拓扑在 hidden 层能学出来
    region_weight = 1.5      # 分类头已达 40%，不再需要 3.0 的暴力权重，给其他任务腾空间

    criterion = IntegratedMultiTaskLoss(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=distance_weight,
        reconstruction_weight=reconstruction_weight,
        direction_weight=direction_weight,
        region_weight=region_weight,
        neighbor_dir_weight=0.6,
        global_dir_weight=0.4,
        use_focal=True,
        focal_gamma=2.0,
    )
    print(f"P1F Loss weights: distance={distance_weight}, reconstruction={reconstruction_weight}, direction={direction_weight}, region={region_weight}")

    # 训练
    trainer = TrainerP1C(model, criterion, config, device)
    results = trainer.train(train_loader, val_loader, epochs)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P1F: Loss Weight Surgery + Classification")
    parser.add_argument("--sample", type=float, default=0.1, help="Sample ratio")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16384, help="Batch size")
    parser.add_argument("--neighbor", action="store_true", help="Enable P1E neighbor features (disabled by default)")
    parser.add_argument("--no-pseudo", action="store_true", help="Disable P1F pseudo labels")
    args = parser.parse_args()

    run_p1c_experiment(
        sample_ratio=args.sample,
        epochs=args.epochs,
        batch_size=args.batch,
        use_neighbor_feats=args.neighbor,
        use_pseudo_labels=not args.no_pseudo,
    )
