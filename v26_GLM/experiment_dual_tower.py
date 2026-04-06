# -*- coding: utf-8 -*-
"""
Phase 1: 双塔架构 + 负采样对比学习实验

目标：
- Intra-class Recall: 24.2% → 50-65%
- 保持 L1/L2/L3 指标 (Pearson > 0.95, DirAcc > 60%)

核心改进：
1. 双塔架构：共享 POI 编码器 + 上下文聚合器
2. 负采样对比学习：空间感知的正负样本策略
3. 多任务损失：对比学习 + 距离保持 + 方向 + 分类

运行：
  python experiment_dual_tower.py --sample 0.1 --epochs 30  # 快速验证
  python experiment_dual_tower.py --sample 1.0 --epochs 80  # 全量训练

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.dual_tower_encoder import DualTowerEncoder, build_dual_tower_encoder
from spatial_encoder.v26_GLM.encoder_v26_mlp import count_parameters
from spatial_encoder.v26_GLM.contrastive_losses import (
    DualTowerMultiTaskLoss,
    PositiveNegativeSampler,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ 数据集 ============

class DualTowerDataset(Dataset):
    """双塔训练数据集"""

    def __init__(self, data: Dict):
        self.point_features = torch.tensor(data["point_features"], dtype=torch.float32)
        self.line_features = torch.tensor(data["line_features"], dtype=torch.float32)
        self.polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)
        self.direction_labels = torch.tensor(data["global_dir_labels"], dtype=torch.long)
        self.direction_valid = torch.tensor(data["global_dir_valid"], dtype=torch.bool)
        self.region_labels = torch.tensor(data["region_labels"], dtype=torch.long)
        self.neighbor_indices = torch.tensor(data["neighbor_indices"], dtype=torch.long)

        # 正负样本索引（预计算）
        self.positive_indices = torch.tensor(data["positive_indices"], dtype=torch.long)
        self.negative_indices = torch.tensor(data["negative_indices"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.point_features)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "point_feat": self.point_features[idx],
            "line_feat": self.line_features[idx],
            "polygon_feat": self.polygon_features[idx],
            "direction_feat": self.direction_features[idx],
            "coords": self.coords[idx],
            "direction_label": self.direction_labels[idx],
            "direction_valid": self.direction_valid[idx],
            "region_label": self.region_labels[idx],
            "neighbor_idx": self.neighbor_indices[idx],
            "positive_idx": self.positive_indices[idx],
            "negative_idx": self.negative_indices[idx],
            "idx": idx,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "point_feat": torch.stack([item["point_feat"] for item in batch]),
        "line_feat": torch.stack([item["line_feat"] for item in batch]),
        "polygon_feat": torch.stack([item["polygon_feat"] for item in batch]),
        "direction_feat": torch.stack([item["direction_feat"] for item in batch]),
        "coords": torch.stack([item["coords"] for item in batch]),
        "direction_label": torch.stack([item["direction_label"] for item in batch]),
        "direction_valid": torch.stack([item["direction_valid"] for item in batch]),
        "region_label": torch.stack([item["region_label"] for item in batch]),
        "neighbor_idx": torch.stack([item["neighbor_idx"] for item in batch]),
        "positive_idx": torch.stack([item["positive_idx"] for item in batch]),
        "negative_idx": torch.stack([item["negative_idx"] for item in batch]),
    }


# ============ 数据加载 ============

def load_data_dual_tower(
    config: V26ProConfig,
    sample_ratio: float = 0.1,
    precompute_samples: bool = True,
) -> Dict:
    """
    加载双塔训练数据（包含正负样本索引）

    复用 experiment_p1c_integrated.py 的数据加载逻辑
    """
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
    from spatial_encoder.v26_GLM.direction_supervision import compute_global_center_direction

    print(f"Loading data (sample_ratio={sample_ratio})...")

    # 加载基础数据
    base_data = load_dataset_for_training(config=config, sample_ratio=sample_ratio)
    coords = base_data["coords"]
    region_labels = base_data["region_labels"]
    neighbor_indices = base_data["neighbor_indices"]
    N = len(coords)

    print(f"  Loaded {N} cells")

    # 计算全局方向标签
    print("  Computing global-center direction...")
    global_dir = compute_global_center_direction(coords)
    global_labels = global_dir.labels
    global_valid = global_dir.valid_mask

    # 预计算正负样本索引
    if precompute_samples:
        print("  Precomputing positive/negative samples...")
        sampler = PositiveNegativeSampler(
            k_positive=config.dual_tower.positive_k,
            num_negatives=config.dual_tower.num_negatives,
            negative_min_distance=config.dual_tower.negative_min_distance,
        )
        pos_idx, neg_idx = sampler.precompute(coords, region_labels)
        print(f"  Positive: {pos_idx.shape}, Negative: {neg_idx.shape}")
    else:
        pos_idx = np.zeros((N, config.dual_tower.positive_k), dtype=np.int64)
        neg_idx = np.zeros((N, config.dual_tower.num_negatives), dtype=np.int64)

    return {
        "point_features": base_data["point_features"],
        "line_features": base_data["line_features"],
        "polygon_features": base_data["polygon_features"],
        "direction_features": base_data["direction_features"],
        "coords": coords,
        "global_dir_labels": global_labels,
        "global_dir_valid": global_valid,
        "region_labels": region_labels,
        "neighbor_indices": neighbor_indices,
        "positive_indices": pos_idx,
        "negative_indices": neg_idx,
        "metadata": base_data.get("metadata"),
    }


# ============ 评估函数 ============

@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """完整评估：L1-L3 指标"""
    model.eval()

    all_embeddings = []
    all_coords = []
    all_region_labels = []
    all_region_preds = []
    all_dir_preds = []
    all_dir_labels = []

    for batch in loader:
        point_feat = batch["point_feat"].to(device)
        line_feat = batch["line_feat"].to(device)
        polygon_feat = batch["polygon_feat"].to(device)
        direction_feat = batch["direction_feat"].to(device)

        emb, dir_pred, reg_pred, coord_pred = model(
            point_feat, line_feat, polygon_feat, direction_feat
        )

        all_embeddings.append(emb.cpu())
        all_coords.append(batch["coords"])
        all_region_labels.append(batch["region_label"])
        all_region_preds.append(reg_pred.cpu())
        all_dir_preds.append(dir_pred.cpu())
        all_dir_labels.append(batch["direction_label"])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_coords = torch.cat(all_coords, dim=0).numpy()
    all_region_labels = torch.cat(all_region_labels, dim=0).numpy()
    all_region_preds = torch.cat(all_region_preds, dim=0).numpy()
    all_dir_preds = torch.cat(all_dir_preds, dim=0).numpy()
    all_dir_labels = torch.cat(all_dir_labels, dim=0).numpy()

    results = {}

    # ============ L1: Pearson & Spearman ============
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances

    sample_size = min(1000, len(all_embeddings))
    idx = np.random.choice(len(all_embeddings), sample_size, replace=False)

    emb_dist = pairwise_distances(all_embeddings[idx], metric='euclidean').flatten()
    coord_dist = pairwise_distances(all_coords[idx], metric='euclidean').flatten()

    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    results["pearson"] = pearson
    results["spearman"] = spearman

    # ============ L2: Overlap@K & Recall@20 ============
    from sklearn.neighbors import NearestNeighbors

    k = 20
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(all_coords)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(all_embeddings)

    _, coord_idx = nbrs_coord.kneighbors(all_coords)
    _, emb_idx = nbrs_emb.kneighbors(all_embeddings)

    coord_idx = coord_idx[:, 1:]
    emb_idx = emb_idx[:, 1:]

    overlaps = []
    for i in range(len(all_coords)):
        overlap = len(set(coord_idx[i]) & set(emb_idx[i])) / k
        overlaps.append(overlap)

    results["overlap"] = np.mean(overlaps)

    # ============ L3: Direction Accuracy ============
    dir_pred_classes = all_dir_preds.argmax(axis=-1)
    dir_acc = (dir_pred_classes == all_dir_labels).mean() * 100
    results["dir_acc"] = dir_acc

    # ============ L3: Region F1 ============
    from sklearn.metrics import f1_score

    valid_mask = all_region_labels < 6
    if valid_mask.sum() > 0:
        true_labels = all_region_labels[valid_mask]
        pred_labels = all_region_preds[valid_mask].argmax(axis=-1)

        region_clf_f1 = f1_score(true_labels, pred_labels, average='macro') * 100
        region_clf_acc = (pred_labels == true_labels).mean() * 100
        results["region_clf_f1"] = region_clf_f1
        results["region_clf_acc"] = region_clf_acc
    else:
        results["region_clf_f1"] = 0.0
        results["region_clf_acc"] = 0.0

    # ============ L3: Intra-class Recall ============
    # 使用 embedding 找同类 POI 的召回率
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(all_embeddings)
    _, emb_knn_idx = nbrs.kneighbors(all_embeddings)
    emb_knn_idx = emb_knn_idx[:, 1:]  # 排除自身

    recalls = []
    for i in range(len(all_embeddings)):
        true_label = all_region_labels[i]
        if true_label >= 6:
            continue
        neighbor_labels = all_region_labels[emb_knn_idx[i]]
        same_class = (neighbor_labels == true_label).sum()
        recalls.append(same_class / k)

    if recalls:
        results["intra_class_recall"] = np.mean(recalls) * 100
    else:
        results["intra_class_recall"] = 0.0

    return results


# ============ 训练器 ============

class DualTowerTrainer:
    """双塔训练器"""

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

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0
        n_batches = 0
        loss_sums = {}

        for batch in loader:
            point_feat = batch["point_feat"].to(self.device, non_blocking=True)
            line_feat = batch["line_feat"].to(self.device, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(self.device, non_blocking=True)
            direction_feat = batch["direction_feat"].to(self.device, non_blocking=True)
            coords = batch["coords"].to(self.device, non_blocking=True)
            direction_label = batch["direction_label"].to(self.device, non_blocking=True)
            region_label = batch["region_label"].to(self.device, non_blocking=True)
            positive_idx = batch["positive_idx"].to(self.device, non_blocking=True)
            negative_idx = batch["negative_idx"].to(self.device, non_blocking=True)

            # 前向传播（无上下文，标准路径）
            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            # 计算损失
            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,
                true_coords=coords,
                direction_labels=direction_label,
                region_labels=region_label,
                positive_indices=positive_idx,
                negative_indices=negative_idx,
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict["total"]
            n_batches += 1

            # 累积各项损失
            for k, v in loss_dict.items():
                if k not in loss_sums:
                    loss_sums[k] = 0
                loss_sums[k] += v

        # 平均损失
        avg_loss_dict = {k: v / n_batches for k, v in loss_sums.items()}
        return {"loss": total_loss / n_batches, "loss_dict": avg_loss_dict}

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
            direction_label = batch["direction_label"].to(self.device)
            region_label = batch["region_label"].to(self.device)
            positive_idx = batch["positive_idx"].to(self.device)
            negative_idx = batch["negative_idx"].to(self.device)

            emb, dir_pred, reg_pred, coord_pred = self.model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, _ = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,
                true_coords=coords,
                direction_labels=direction_label,
                region_labels=region_label,
                positive_indices=positive_idx,
                negative_indices=negative_idx,
            )

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / n_batches}

    def train(self, train_loader, val_loader, epochs: int = 30):
        print(f"\n{'='*60}")
        print(f"DualTower Training: Contrastive + Multi-task")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            # 每15个epoch完整评估
            if (epoch + 1) % 15 == 0 or epoch == 0:
                eval_results = evaluate_full(self.model, val_loader, self.device)

                print(f"Epoch {epoch+1:2d}/{epochs}: "
                      f"Pearson={eval_results['pearson']:.4f}, "
                      f"DirAcc={eval_results['dir_acc']:.1f}%, "
                      f"ClfF1={eval_results['region_clf_f1']:.1f}%, "
                      f"IntraRecall={eval_results['intra_class_recall']:.1f}%")

                # 打印详细损失
                loss_dict = train_metrics.get("loss_dict", {})
                print(f"  Losses: dist={loss_dict.get('distance',0):.4f}, "
                      f"dir={loss_dict.get('direction',0):.4f}, "
                      f"reg={loss_dict.get('region',0):.4f}, "
                      f"contra={loss_dict.get('contrastive',0):.4f}, "
                      f"supcon={loss_dict.get('supcon',0):.4f}")

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

        print(f"  Pearson: {results['pearson']:.4f}")
        print(f"  Spearman: {results['spearman']:.4f}")
        print(f"  Overlap@K: {results['overlap']*100:.2f}%")
        print(f"  DirAcc: {results['dir_acc']:.2f}%")
        print(f"  Region Clf F1: {results['region_clf_f1']:.2f}%")
        print(f"  Region Clf Acc: {results['region_clf_acc']:.2f}%")
        print(f"  Intra-class Recall: {results['intra_class_recall']:.2f}%")
        print("="*60)

        return results


# ============ 主函数 ============

def run_dual_tower_experiment(
    sample_ratio: float = 0.1,
    epochs: int = 30,
    batch_size: int = 16384,
    contrastive_weight: float = 1.0,
    supcon_weight: float = 1.5,
    region_weight: float = 3.0,
):
    print(f"\n{'#'*60}")
    print(f"# Phase 1: Dual-Tower + SupCon + Focal Region")
    print(f"# Sample ratio: {sample_ratio}")
    print(f"# Epochs: {epochs}")
    print(f"# contrastive={contrastive_weight}, supcon={supcon_weight}, region={region_weight}")
    print(f"{'#'*60}\n")

    config = DEFAULT_PRO_CONFIG
    config.training.batch_size = batch_size
    config.training.num_epochs = epochs
    config.dual_tower.contrastive_weight = contrastive_weight

    set_seed(config.training.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载数据
    data = load_data_dual_tower(config, sample_ratio)
    n_cells = len(data["coords"])

    dataset = DualTowerDataset(data)
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
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=actual_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Dataset: {n_cells} cells, train={train_size}, val={n_cells - train_size}")
    print(f"Batch size: {actual_batch_size}")

    # 模型
    model = build_dual_tower_encoder(config)
    params = count_parameters(model)
    print(f"Model: {params['total_m']:.1f}M parameters")

    # 损失函数
    criterion = DualTowerMultiTaskLoss(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=config.loss.distance_weight,
        reconstruction_weight=config.loss.reconstruction_weight,
        direction_weight=config.loss.direction_weight,
        region_weight=region_weight,
        contrastive_weight=contrastive_weight,
        supcon_weight=supcon_weight,
        temperature=config.dual_tower.contrastive_temperature,
    )

    print(f"Loss weights: dist={config.loss.distance_weight}, "
          f"dir={config.loss.direction_weight}, reg={region_weight}, "
          f"contra={contrastive_weight}, supcon={supcon_weight}")

    # 训练
    trainer = DualTowerTrainer(model, criterion, config, device)
    results = trainer.train(train_loader, val_loader, epochs)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Dual-Tower + SupCon + Focal Region")
    parser.add_argument("--sample",      type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=16384)
    parser.add_argument("--contrastive", type=float, default=1.0,  help="Spatial InfoNCE weight")
    parser.add_argument("--supcon",      type=float, default=1.5,  help="SupCon weight")
    parser.add_argument("--region",      type=float, default=3.0,  help="Region Focal loss weight")
    args = parser.parse_args()

    run_dual_tower_experiment(
        sample_ratio=args.sample,
        epochs=args.epochs,
        batch_size=args.batch,
        contrastive_weight=args.contrastive,
        supcon_weight=args.supcon,
        region_weight=args.region,
    )
