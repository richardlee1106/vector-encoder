# -*- coding: utf-8 -*-
"""
三镇渐进式训练脚本

特性：
1. 渐进式训练：10% → 30% → 50% → 80% → 100%
2. 梯度累积：小批次高效训练
3. 双塔架构 + 对比学习

运行：
  python train_town_encoder.py                    # 完整渐进式训练
  python train_town_encoder.py --start-phase 2    # 从 Phase 2 开始
  python train_town_encoder.py --phase 0 --sample 0.1  # 单阶段测试

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import CellEncoderMLP, count_parameters
from spatial_encoder.v26_GLM.data_loader_town import load_town_dataset
from spatial_encoder.v26_GLM.losses_v26_pro import MultiTaskLossPro
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

class TownDataset(Dataset):
    """三镇训练数据集"""

    def __init__(self, data: Dict):
        self.point_features = torch.tensor(data["point_features"], dtype=torch.float32)
        self.line_features = torch.tensor(data["line_features"], dtype=torch.float32)
        self.polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)
        self.direction_labels = torch.tensor(data["direction_labels"], dtype=torch.long)
        self.direction_valid = torch.tensor(data["direction_valid"], dtype=torch.bool)
        self.region_labels = torch.tensor(data["region_labels"], dtype=torch.long)
        self.neighbor_indices = torch.tensor(data["neighbor_indices"], dtype=torch.long)

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
    }


# ============ 评估函数 ============

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """完整评估"""
    model.eval()

    all_embeddings = []
    all_coords = []
    all_region_labels = []
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
        all_dir_preds.append(dir_pred.cpu())
        all_dir_labels.append(batch["direction_label"])

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_coords = torch.cat(all_coords, dim=0).numpy()
    all_region_labels = torch.cat(all_region_labels, dim=0).numpy()
    all_dir_preds = torch.cat(all_dir_preds, dim=0).numpy()
    all_dir_labels = torch.cat(all_dir_labels, dim=0).numpy()

    results = {}

    # L1: Pearson & Spearman
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances

    sample_size = min(500, len(all_embeddings))
    idx = np.random.choice(len(all_embeddings), sample_size, replace=False)

    emb_dist = pairwise_distances(all_embeddings[idx], metric='euclidean').flatten()
    coord_dist = pairwise_distances(all_coords[idx], metric='euclidean').flatten()

    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    results["pearson"] = pearson
    results["spearman"] = spearman

    # L2: Overlap@K
    from sklearn.neighbors import NearestNeighbors

    n_samples = len(all_coords)
    k = min(20, n_samples - 1)  # 适应小样本
    if k < 2:
        results["overlap"] = 0.0
    else:
        nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(all_coords)
        nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(all_embeddings)

        _, coord_idx = nbrs_coord.kneighbors(all_coords)
        _, emb_idx = nbrs_emb.kneighbors(all_embeddings)

        coord_idx = coord_idx[:, 1:]
        emb_idx = emb_idx[:, 1:]

        overlaps = []
        for i in range(n_samples):
            overlap = len(set(coord_idx[i]) & set(emb_idx[i])) / k
            overlaps.append(overlap)

        results["overlap"] = np.mean(overlaps)

    # L3: Direction Accuracy
    dir_pred_classes = all_dir_preds.argmax(axis=-1)
    dir_acc = (dir_pred_classes == all_dir_labels).mean() * 100
    results["dir_acc"] = dir_acc

    # L3: Region F1
    from sklearn.metrics import f1_score

    valid_mask = all_region_labels < 6
    if valid_mask.sum() > 0:
        true_labels = all_region_labels[valid_mask]
        pred_labels = all_dir_preds[valid_mask].argmax(axis=-1)
        region_f1 = f1_score(true_labels, pred_labels, average='macro') * 100
        results["region_f1"] = region_f1
    else:
        results["region_f1"] = 0.0

    # L3: Intra-class Recall
    n_samples = len(all_embeddings)
    k_recall = min(20, n_samples - 1)
    if k_recall < 2:
        results["intra_recall"] = 0.0
    else:
        nbrs = NearestNeighbors(n_neighbors=k_recall+1, metric='cosine').fit(all_embeddings)
        _, emb_knn_idx = nbrs.kneighbors(all_embeddings)
        emb_knn_idx = emb_knn_idx[:, 1:]

        recalls = []
        for i in range(n_samples):
            true_label = all_region_labels[i]
            if true_label >= 6:
                continue
            neighbor_labels = all_region_labels[emb_knn_idx[i]]
            same_class = (neighbor_labels == true_label).sum()
            recalls.append(same_class / k_recall)

        if recalls:
            results["intra_recall"] = np.mean(recalls) * 100
        else:
            results["intra_recall"] = 0.0

    return results


# ============ 训练器 ============

class ProgressiveTrainer:
    """渐进式训练器（支持梯度累积）"""

    def __init__(self, model, criterion, config, device, save_dir: Path):
        self.model = model.to(device)
        self.criterion = criterion
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.accumulation_steps = config.training.accumulation_steps

        self.history = []
        self.best_pearson = 0.0
        self.best_epoch = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0
        n_accumulated = 0
        loss_sums = {}

        for batch_idx, batch in enumerate(loader):
            point_feat = batch["point_feat"].to(self.device, non_blocking=True)
            line_feat = batch["line_feat"].to(self.device, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(self.device, non_blocking=True)
            direction_feat = batch["direction_feat"].to(self.device, non_blocking=True)
            coords = batch["coords"].to(self.device, non_blocking=True)
            direction_label = batch["direction_label"].to(self.device, non_blocking=True)
            region_label = batch["region_label"].to(self.device, non_blocking=True)

            # 前向传播
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
            )

            # 梯度累积：损失除以累积步数
            loss = loss / self.accumulation_steps
            loss.backward()

            n_accumulated += 1

            # 累积足够步数后更新参数
            if n_accumulated >= self.accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                n_accumulated = 0

            total_loss += loss_dict["total"]
            for k, v in loss_dict.items():
                if k not in loss_sums:
                    loss_sums[k] = 0
                loss_sums[k] += v

        # 处理剩余的累积梯度
        if n_accumulated > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        n_batches = len(loader)
        avg_loss_dict = {k: v / n_batches for k, v in loss_sums.items()}
        return {"loss": total_loss / n_batches, "loss_dict": avg_loss_dict}

    def validate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                point_feat = batch["point_feat"].to(self.device)
                line_feat = batch["line_feat"].to(self.device)
                polygon_feat = batch["polygon_feat"].to(self.device)
                direction_feat = batch["direction_feat"].to(self.device)
                coords = batch["coords"].to(self.device)
                direction_label = batch["direction_label"].to(self.device)
                region_label = batch["region_label"].to(self.device)

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
                )

                total_loss += loss.item()
                n_batches += 1

        return {"loss": total_loss / n_batches}

    def train_phase(self, train_loader, val_loader, epochs: int, phase_name: str) -> Dict:
        """训练一个阶段"""

        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.config.training.learning_rate * 0.01,
        )

        print(f"\n{'='*60}")
        print(f"{phase_name}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Accumulation steps: {self.accumulation_steps}")
        print(f"  Effective batch: {train_loader.batch_size * self.accumulation_steps}")
        print(f"{'='*60}\n")

        best_phase_pearson = 0.0

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader)

            scheduler.step()

            epoch_time = time.time() - epoch_start

            # 每 10 个 epoch 完整评估
            if (epoch + 1) % 10 == 0 or epoch == 0:
                eval_results = evaluate(self.model, val_loader, self.device)

                print(f"Epoch {epoch+1:2d}/{epochs}: "
                      f"Pearson={eval_results['pearson']:.4f}, "
                      f"DirAcc={eval_results['dir_acc']:.1f}%, "
                      f"Overlap={eval_results['overlap']*100:.1f}%, "
                      f"IntraRecall={eval_results['intra_recall']:.1f}%")

                if eval_results['pearson'] > best_phase_pearson:
                    best_phase_pearson = eval_results['pearson']

                if eval_results['pearson'] > self.best_pearson:
                    self.best_pearson = eval_results['pearson']
                    self.best_epoch = epoch + 1
                    self.save_model("best_model.pt")

        return {"best_pearson": best_phase_pearson}

    def save_model(self, filename: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_pearson": self.best_pearson,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }, self.save_dir / filename)


# ============ 主函数 ============

def run_progressive_training(
    start_phase: int = 0,
    single_phase: int = None,
    sample_ratio: float = None,
):
    """
    渐进式训练

    Args:
        start_phase: 从哪个阶段开始（0-4）
        single_phase: 只运行指定阶段（用于测试）
        sample_ratio: 指定采样比例（覆盖默认值）
    """
    config = DEFAULT_PRO_CONFIG
    set_seed(config.training.seed if hasattr(config.training, 'seed') else 42)

    # 渐进式训练配置
    phases = [
        {"name": "Phase 1 (10%)", "ratio": 0.1, "epochs": 30},
        {"name": "Phase 2 (30%)", "ratio": 0.3, "epochs": 40},
        {"name": "Phase 3 (50%)", "ratio": 0.5, "epochs": 50},
        {"name": "Phase 4 (80%)", "ratio": 0.8, "epochs": 60},
        {"name": "Phase 5 (100%)", "ratio": 1.0, "epochs": 80},
    ]

    # 单阶段测试模式
    if single_phase is not None:
        phases = [phases[single_phase]]
        start_phase = 0

    # 指定采样比例
    if sample_ratio is not None:
        phases = [{"name": f"Custom ({sample_ratio*100:.0f}%)", "ratio": sample_ratio, "epochs": 50}]
        start_phase = 0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建模型
    model = CellEncoderMLP(
        point_feat_dim=config.model.point_feature_dim,
        line_feat_dim=config.model.line_feature_dim,
        polygon_feat_dim=config.model.polygon_feature_dim,
        direction_feat_dim=config.model.direction_feature_dim,
        hidden_dim=config.model.hidden_dim,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_encoder_layers,
        num_direction_classes=config.model.num_direction_classes,
        num_region_classes=config.model.num_region_classes,
    )
    params = count_parameters(model)
    print(f"Model: {params['total_m']:.2f}M parameters")

    # 损失函数
    criterion = MultiTaskLossPro(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=config.loss.distance_weight,
        reconstruction_weight=config.loss.reconstruction_weight,
        direction_weight=config.loss.direction_weight,
        region_weight=config.loss.region_weight,
    )

    # 保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.training.save_dir) / timestamp

    # 训练器
    trainer = ProgressiveTrainer(model, criterion, config, device, save_dir)

    # 渐进式训练
    all_results = []

    for phase_idx, phase in enumerate(phases):
        if phase_idx < start_phase:
            print(f"Skipping {phase['name']}...")
            continue

        print(f"\n{'#'*60}")
        print(f"# {phase['name']}")
        print(f"# Sample ratio: {phase['ratio']}")
        print(f"{'#'*60}")

        # 加载数据
        data = load_town_dataset(config=config, sample_ratio=phase['ratio'])
        n_cells = len(data['coords'])

        dataset = TownDataset(data)

        # 划分训练/验证
        train_size = int(0.9 * n_cells)
        val_size = n_cells - train_size
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # 计算实际 batch size
        actual_batch = min(config.training.batch_size, train_size // 2)
        actual_batch = max(actual_batch, 32)

        train_loader = DataLoader(
            train_set,
            batch_size=actual_batch,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=actual_batch,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        print(f"  Train: {train_size}, Val: {val_size}")
        print(f"  Batch: {actual_batch}, Effective: {actual_batch * config.training.accumulation_steps}")

        # 训练
        result = trainer.train_phase(
            train_loader, val_loader,
            epochs=phase['epochs'],
            phase_name=phase['name']
        )
        result['phase'] = phase['name']
        result['ratio'] = phase['ratio']
        all_results.append(result)

        # 保存阶段模型
        trainer.save_model(f"phase{phase_idx+1}_model.pt")

    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    # 加载最佳模型
    checkpoint = torch.load(save_dir / "best_model.pt", weights_only=False)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    # 用全量数据评估
    full_data = load_town_dataset(config=config, sample_ratio=1.0)
    full_dataset = TownDataset(full_data)
    full_loader = DataLoader(
        full_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )

    final_results = evaluate(trainer.model, full_loader, device)

    print(f"  Pearson: {final_results['pearson']:.4f}")
    print(f"  Spearman: {final_results['spearman']:.4f}")
    print(f"  Overlap@K: {final_results['overlap']*100:.2f}%")
    print(f"  DirAcc: {final_results['dir_acc']:.2f}%")
    print(f"  Region F1: {final_results['region_f1']:.2f}%")
    print(f"  Intra-class Recall: {final_results['intra_recall']:.2f}%")

    # 保存结果
    results_file = save_dir / "training_results.json"

    def convert_to_native(obj):
        """递归转换为 Python 原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy/torch types
            return obj.item()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        else:
            return obj

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "phases": convert_to_native(all_results),
            "final": convert_to_native(final_results),
            "config": {
                "hidden_dim": config.model.hidden_dim,
                "embedding_dim": config.model.embedding_dim,
                "batch_size": config.training.batch_size,
                "accumulation_steps": config.training.accumulation_steps,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三镇渐进式训练")
    parser.add_argument("--start-phase", type=int, default=0, help="从哪个阶段开始（0-4）")
    parser.add_argument("--phase", type=int, default=None, help="只运行指定阶段（用于测试）")
    parser.add_argument("--sample", type=float, default=None, help="指定采样比例")
    args = parser.parse_args()

    run_progressive_training(
        start_phase=args.start_phase,
        single_phase=args.phase,
        sample_ratio=args.sample,
    )
