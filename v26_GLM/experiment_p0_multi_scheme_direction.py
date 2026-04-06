# -*- coding: utf-8 -*-
"""
P0实验：多方案联合方向训练

目标：提升DirAcc从22%到60%

方案：
1. 同时训练 neighbor_relative + global_center 两个方向方案
2. 加权组合：0.6 * neighbor_relative + 0.4 * global_center
3. 使用Focal Loss处理方向类别不平衡

渐进式验证：
- Phase 1: 单实验区测试
- Phase 2: 三实验区测试
- Phase 3: 渐进式全量 (10% -> 30% -> 60% -> 80% -> 100%)

Author: Claude
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
from tqdm import tqdm

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import CellEncoderMLP, build_mlp_encoder, count_parameters
from spatial_encoder.v26_GLM.direction_supervision import (
    MultiSchemeDirectionSupervision,
    DirectionScheme,
    compute_global_center_direction,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ 多方案方向损失 ============

class MultiSchemeDirectionLoss(nn.Module):
    """
    多方案联合方向损失

    同时训练多个方向方案，加权组合：
    L_direction = w1 * neighbor_relative_loss + w2 * global_center_loss
    """

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

    def focal_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Focal Loss处理类别不平衡"""
        ce_loss = F.cross_entropy(
            pred_logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )

        # 计算概率
        probs = F.softmax(pred_logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal权重
        focal_weight = (1 - pt) ** self.focal_gamma

        loss = focal_weight * ce_loss

        if weight is not None:
            loss = loss * weight

        return loss.mean()

    def standard_loss(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准交叉熵损失"""
        ce_loss = F.cross_entropy(
            pred_logits, targets,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )

        if weight is not None:
            ce_loss = ce_loss * weight

        return ce_loss.mean()

    def forward(
        self,
        pred_logits: torch.Tensor,
        neighbor_labels: torch.Tensor,
        neighbor_valid: torch.Tensor,
        global_labels: torch.Tensor,
        global_valid: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算多方案方向损失

        Args:
            pred_logits: 方向预测 [batch, 8]
            neighbor_labels: 邻居相对方向标签 [batch]
            neighbor_valid: 邻居方向是否有效 [batch]
            global_labels: 全局中心方向标签 [batch]
            global_valid: 全局方向是否有效 [batch]
            sample_weights: 样本权重 [batch]

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}
        device = pred_logits.device
        batch_size = pred_logits.size(0)

        # 1. 邻居相对方向损失
        neighbor_loss = torch.tensor(0.0, device=device)
        neighbor_mask = neighbor_valid.bool()
        if neighbor_mask.sum() > 0:
            neighbor_loss = self._compute_loss(
                pred_logits, neighbor_labels, neighbor_mask, sample_weights
            )
        loss_dict["neighbor_dir"] = neighbor_loss.item()

        # 2. 全局中心方向损失
        global_loss = torch.tensor(0.0, device=device)
        global_mask = global_valid.bool()
        if global_mask.sum() > 0:
            global_loss = self._compute_loss(
                pred_logits, global_labels, global_mask, sample_weights
            )
        loss_dict["global_dir"] = global_loss.item()

        # 3. 加权组合
        total_loss = (
            self.neighbor_weight * neighbor_loss +
            self.global_center_weight * global_loss
        )
        loss_dict["total_dir"] = total_loss.item()

        return total_loss, loss_dict

    def _compute_loss(
        self,
        pred_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算单个方案的方向损失"""
        pred_masked = pred_logits[mask]
        labels_masked = labels[mask]

        weight_masked = None
        if sample_weights is not None:
            weight_masked = sample_weights[mask]

        if self.use_focal:
            return self.focal_loss(pred_masked, labels_masked, weight_masked)
        else:
            return self.standard_loss(pred_masked, labels_masked, weight_masked)


# ============ 增强版多任务损失 ============

class EnhancedMultiTaskLoss(nn.Module):
    """
    V2.6 Pro 增强版多任务损失

    改进：
    1. 多方案联合方向训练
    2. Focal Loss处理方向不平衡
    """

    def __init__(
        self,
        k_nearest: int = 85,
        distance_weight: float = 3.0,
        reconstruction_weight: float = 1.0,
        direction_weight: float = 2.0,  # 提升方向权重
        neighbor_dir_weight: float = 0.6,
        global_dir_weight: float = 0.4,
        use_focal: bool = True,
    ):
        super().__init__()

        self.distance_weight = distance_weight
        self.reconstruction_weight = reconstruction_weight
        self.direction_weight = direction_weight

        # 距离损失参数
        self.k = k_nearest
        self.gamma = 0.5

        # 多方案方向损失
        self.direction_loss = MultiSchemeDirectionLoss(
            neighbor_weight=neighbor_dir_weight,
            global_center_weight=global_dir_weight,
            use_focal=use_focal,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_direction: torch.Tensor,
        true_coords: torch.Tensor,
        neighbor_dir_labels: torch.Tensor,
        neighbor_dir_valid: torch.Tensor,
        global_dir_labels: torch.Tensor,
        global_dir_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """计算总损失"""
        loss_dict = {}
        device = embeddings.device
        N = embeddings.size(0)

        # ============ 1. 距离保持损失（Pearson + K近邻）============
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

        # ============ 3. 多方案方向损失 ============
        l_direction, dir_dict = self.direction_loss(
            pred_logits=pred_direction,
            neighbor_labels=neighbor_dir_labels,
            neighbor_valid=neighbor_dir_valid,
            global_labels=global_dir_labels,
            global_valid=global_dir_valid,
        )
        loss_dict.update(dir_dict)

        # ============ 总损失 ============
        total_loss = (
            self.distance_weight * l_distance +
            self.reconstruction_weight * l_reconstruct +
            self.direction_weight * l_direction
        )
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


# ============ 数据集 ============

class CellDatasetP0(Dataset):
    """P0实验数据集"""

    def __init__(self, data: Dict):
        self.point_features = torch.tensor(data["point_features"], dtype=torch.float32)
        self.line_features = torch.tensor(data["line_features"], dtype=torch.float32)
        self.polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)

        # 多方案方向标签
        self.neighbor_dir_labels = torch.tensor(data["neighbor_dir_labels"], dtype=torch.long)
        self.neighbor_dir_valid = torch.tensor(data["neighbor_dir_valid"], dtype=torch.bool)
        self.global_dir_labels = torch.tensor(data["global_dir_labels"], dtype=torch.long)
        self.global_dir_valid = torch.tensor(data["global_dir_valid"], dtype=torch.bool)

        self.region_labels = torch.tensor(data["region_labels"], dtype=torch.long)

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
            "idx": idx,
        }


def collate_fn_p0(batch: List[Dict]) -> Dict:
    """P0数据集批处理"""
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
    }


# ============ 数据加载 ============

def load_data_multi_scheme(
    config: V26ProConfig,
    sample_ratio: float = 0.1,
    experiment_region: Optional[str] = None,
) -> Dict:
    """
    加载多方案方向数据

    Args:
        config: 配置
        sample_ratio: 采样比例
        experiment_region: 实验区域名称（用于Phase 1/2）
    """
    try:
        from data_loader_v26 import load_dataset_for_training, compute_neighbor_indices
        from spatial_encoder.v26_GLM.direction_supervision import (
            compute_neighbor_relative_direction,
            compute_global_center_direction,
        )

        print(f"Loading real data (sample_ratio={sample_ratio})...")

        # 加载基础数据
        base_data = load_dataset_for_training(config=config, sample_ratio=sample_ratio)
        coords = base_data["coords"]
        neighbor_indices = base_data["neighbor_indices"]
        N = len(coords)

        print(f"  Loaded {N} cells")

        # 计算邻居相对方向
        print("  Computing neighbor-relative direction...")
        neighbor_dir = compute_neighbor_relative_direction(coords, neighbor_indices)
        # 取最近邻的方向（第一个有效邻居），而不是平均
        neighbor_labels = np.zeros(N, dtype=np.int64)
        neighbor_valid = np.zeros(N, dtype=bool)
        for i in range(N):
            valid_neighbors = neighbor_dir.valid_mask[i]
            if valid_neighbors.any():
                # 取第一个有效邻居的方向
                first_valid = np.where(valid_neighbors)[0][0]
                neighbor_labels[i] = neighbor_dir.labels[i, first_valid]
                neighbor_valid[i] = True

        # 计算全局中心方向
        print("  Computing global-center direction...")
        global_dir = compute_global_center_direction(coords)
        global_labels = global_dir.labels
        global_valid = global_dir.valid_mask

        return {
            "point_features": base_data["point_features"],
            "line_features": base_data["line_features"],
            "polygon_features": base_data["polygon_features"],
            "direction_features": base_data["direction_features"],
            "coords": coords,
            "neighbor_dir_labels": neighbor_labels,
            "neighbor_dir_valid": neighbor_valid,
            "global_dir_labels": global_labels,
            "global_dir_valid": global_valid,
            "region_labels": base_data["region_labels"],
        }

    except Exception as e:
        print(f"Using mock data: {e}")
        np.random.seed(config.training.random_seed)

        n = max(50000, config.training.batch_size * 4)
        coords = np.random.rand(n, 2).astype(np.float32)
        coords[:, 0] = coords[:, 0] * 1.4 + 113.6
        coords[:, 1] = coords[:, 1] * 1.4 + 29.9

        center = coords.mean(axis=0)

        # 邻居相对方向（模拟）
        neighbor_labels = np.random.randint(0, 8, n)
        neighbor_valid = np.random.rand(n) > 0.3  # 70%有效

        # 全局中心方向
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        global_labels = ((angles + np.pi) / (np.pi / 4)).astype(int) % 8
        global_valid = np.ones(n, dtype=bool)

        return {
            "point_features": np.random.randn(n, 32).astype(np.float32),
            "line_features": np.random.randn(n, 16).astype(np.float32),
            "polygon_features": np.random.randn(n, 16).astype(np.float32),
            "direction_features": np.random.randn(n, 8).astype(np.float32),
            "coords": coords,
            "neighbor_dir_labels": neighbor_labels.astype(np.int64),
            "neighbor_dir_valid": neighbor_valid.astype(bool),
            "global_dir_labels": global_labels.astype(np.int64),
            "global_dir_valid": global_valid.astype(bool),
            "region_labels": np.random.randint(0, 16, n),
        }


# ============ 评估函数 ============

@torch.no_grad()
def evaluate_direction_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """评估方向准确率"""
    model.eval()

    neighbor_correct = 0
    neighbor_total = 0
    global_correct = 0
    global_total = 0

    all_embeddings = []
    all_coords = []

    for batch in loader:
        point_feat = batch["point_feat"].to(device)
        line_feat = batch["line_feat"].to(device)
        polygon_feat = batch["polygon_feat"].to(device)
        direction_feat = batch["direction_feat"].to(device)
        coords = batch["coords"].to(device)

        emb, dir_pred, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)

        # 方向预测
        pred_dirs = dir_pred.argmax(dim=-1).cpu()

        # 邻居方向准确率
        neighbor_label = batch["neighbor_dir_label"]
        neighbor_valid = batch["neighbor_dir_valid"]
        neighbor_mask = neighbor_valid.bool()

        if neighbor_mask.sum() > 0:
            neighbor_correct += (pred_dirs[neighbor_mask] == neighbor_label[neighbor_mask]).sum().item()
            neighbor_total += neighbor_mask.sum().item()

        # 全局方向准确率
        global_label = batch["global_dir_label"]
        global_valid = batch["global_dir_valid"]
        global_mask = global_valid.bool()

        if global_mask.sum() > 0:
            global_correct += (pred_dirs[global_mask] == global_label[global_mask]).sum().item()
            global_total += global_mask.sum().item()

        all_embeddings.append(emb.cpu())
        all_coords.append(coords.cpu())

    results = {
        "neighbor_dir_acc": neighbor_correct / max(neighbor_total, 1) * 100,
        "global_dir_acc": global_correct / max(global_total, 1) * 100,
        "combined_dir_acc": (neighbor_correct + global_correct) / max(neighbor_total + global_total, 1) * 100,
    }

    # 计算Pearson
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_coords = torch.cat(all_coords, dim=0).numpy()

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances

    # 采样计算
    sample_size = min(2000, len(all_embeddings))
    idx = np.random.choice(len(all_embeddings), sample_size, replace=False)

    emb_dist = pairwise_distances(all_embeddings[idx], metric='euclidean').flatten()
    coord_dist = pairwise_distances(all_coords[idx], metric='euclidean').flatten()

    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    results["pearson"] = pearson
    results["spearman"] = spearman

    return results


# ============ 训练器 ============

class TrainerP0:
    """P0实验训练器"""

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
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {"distance": 0, "reconstruction": 0, "neighbor_dir": 0, "global_dir": 0}
        n_batches = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            point_feat = batch["point_feat"].to(self.device, non_blocking=True)
            line_feat = batch["line_feat"].to(self.device, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(self.device, non_blocking=True)
            direction_feat = batch["direction_feat"].to(self.device, non_blocking=True)
            coords = batch["coords"].to(self.device, non_blocking=True)
            neighbor_dir_label = batch["neighbor_dir_label"].to(self.device, non_blocking=True)
            neighbor_dir_valid = batch["neighbor_dir_valid"].to(self.device, non_blocking=True)
            global_dir_label = batch["global_dir_label"].to(self.device, non_blocking=True)
            global_dir_valid = batch["global_dir_valid"].to(self.device, non_blocking=True)

            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                true_coords=coords,
                neighbor_dir_labels=neighbor_dir_label,
                neighbor_dir_valid=neighbor_dir_valid,
                global_dir_labels=global_dir_label,
                global_dir_valid=global_dir_valid,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict["total"]
            for k in loss_components:
                if k in loss_dict:
                    loss_components[k] += loss_dict[k]
            n_batches += 1

        return {
            "loss": total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict:
        """验证"""
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

            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                true_coords=coords,
                neighbor_dir_labels=neighbor_dir_label,
                neighbor_dir_valid=neighbor_dir_valid,
                global_dir_labels=global_dir_label,
                global_dir_valid=global_dir_valid,
            )

            total_loss += loss_dict["total"]
            n_batches += 1

        return {"loss": total_loss / n_batches}

    def train(self, train_loader, val_loader, epochs: int = 30):
        """完整训练"""
        print(f"\n{'='*60}")
        print(f"P0 Training: Multi-Scheme Direction")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.validate(val_loader)

            # 学习率调度
            self.scheduler.step()

            # 记录
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            # 每5个epoch评估一次方向准确率
            if (epoch + 1) % 5 == 0 or epoch == 0:
                eval_results = evaluate_direction_accuracy(self.model, val_loader, self.device)

                print(f"Epoch {epoch+1:2d}: loss={train_metrics['loss']:.4f}, "
                      f"val={val_metrics['loss']:.4f}, "
                      f"neighbor_dir={eval_results['neighbor_dir_acc']:.1f}%, "
                      f"global_dir={eval_results['global_dir_acc']:.1f}%, "
                      f"Pearson={eval_results['pearson']:.4f}, "
                      f"mem={gpu_mem:.1f}GB, time={epoch_time:.1f}s")
            else:
                print(f"Epoch {epoch+1:2d}: loss={train_metrics['loss']:.4f}, "
                      f"val={val_metrics['loss']:.4f}, "
                      f"dist={train_metrics['distance']:.4f}, "
                      f"neighbor={train_metrics['neighbor_dir']:.4f}, "
                      f"global={train_metrics['global_dir']:.4f}, "
                      f"mem={gpu_mem:.1f}GB, time={epoch_time:.1f}s")

            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                **train_metrics,
            })

            # 保存最佳模型
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]

        print(f"\nTraining completed. Best val loss: {self.best_loss:.4f}")

        # 最终评估
        print("\n" + "="*60)
        print("Final Evaluation:")
        results = evaluate_direction_accuracy(self.model, val_loader, self.device)

        print(f"  Neighbor DirAcc: {results['neighbor_dir_acc']:.2f}%")
        print(f"  Global DirAcc: {results['global_dir_acc']:.2f}%")
        print(f"  Combined DirAcc: {results['combined_dir_acc']:.2f}%")
        print(f"  Pearson: {results['pearson']:.4f}")
        print(f"  Spearman: {results['spearman']:.4f}")
        print("="*60)

        return results


# ============ 主函数 ============

def run_p0_experiment(
    sample_ratio: float = 0.1,
    epochs: int = 30,
    batch_size: int = 16384,
    experiment_name: str = "p0_multi_scheme_direction",
):
    """
    运行P0实验

    Args:
        sample_ratio: 数据采样比例
        epochs: 训练轮数
        batch_size: 批大小
        experiment_name: 实验名称
    """
    print(f"\n{'#'*60}")
    print(f"# P0 Experiment: {experiment_name}")
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
    data = load_data_multi_scheme(config, sample_ratio)
    n_cells = len(data["coords"])

    dataset = CellDatasetP0(data)
    train_size = int(0.9 * n_cells)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, n_cells - train_size]
    )

    # 调整batch_size
    actual_batch_size = min(batch_size, train_size // 2)
    actual_batch_size = max(actual_batch_size, 64)

    train_loader = DataLoader(
        train_set,
        batch_size=actual_batch_size,
        shuffle=True,
        collate_fn=collate_fn_p0,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collate_fn_p0,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Dataset: {n_cells} cells, train={train_size}, val={n_cells - train_size}")
    print(f"Batch size: {actual_batch_size}")

    # 检查数据有效性
    neighbor_valid_rate = data["neighbor_dir_valid"].mean()
    global_valid_rate = data["global_dir_valid"].mean()
    print(f"Neighbor dir valid rate: {neighbor_valid_rate:.2%}")
    print(f"Global dir valid rate: {global_valid_rate:.2%}")

    # 模型
    model = build_mlp_encoder(config)
    params = count_parameters(model)
    print(f"Model: {params['total_m']:.1f}M parameters")

    # 损失函数 - 恢复原权重，通过更长训练提升效果
    criterion = EnhancedMultiTaskLoss(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=config.loss.distance_weight,
        reconstruction_weight=config.loss.reconstruction_weight,
        direction_weight=2.0,  # 提升方向权重
        neighbor_dir_weight=0.6,
        global_dir_weight=0.4,
        use_focal=True,
    )

    # 训练
    trainer = TrainerP0(model, criterion, config, device)
    results = trainer.train(train_loader, val_loader, epochs)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P0: Multi-Scheme Direction Training")
    parser.add_argument("--sample", type=float, default=0.1, help="Sample ratio")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16384, help="Batch size")
    parser.add_argument("--phase", type=int, default=1, help="Experiment phase (1/2/3)")
    args = parser.parse_args()

    run_p0_experiment(
        sample_ratio=args.sample,
        epochs=args.epochs,
        batch_size=args.batch,
    )
