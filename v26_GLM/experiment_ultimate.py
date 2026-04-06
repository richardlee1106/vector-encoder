# -*- coding: utf-8 -*-
"""
Phase 3: 终极空间编码器训练脚本

架构：双塔 + 原型 + 时空注意力
目标：
  L3 超预期：DirAcc > 70%, Region F1 > 50%
  L4 初探：Range IoU > 30%（为后续专项优化打基础）

运行：
  python experiment_ultimate.py --sample 0.1 --epochs 30   # 快速验证
  python experiment_ultimate.py --sample 1.0 --epochs 80   # 全量训练

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.ultimate_encoder import UltimateSpatialEncoder, UltimateLoss, build_ultimate_encoder
from spatial_encoder.v26_GLM.encoder_v26_mlp import count_parameters
from spatial_encoder.v26_GLM.contrastive_losses import PositiveNegativeSampler
from spatial_encoder.v26_GLM.spatial_attention_encoder import precompute_neighbors
from spatial_encoder.v26_GLM.prototype_learning import discover_prototypes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ 数据集 ============

class UltimateDataset(Dataset):
    """终极训练数据集（含邻居距离）"""

    def __init__(self, data: Dict):
        self.point_features    = torch.tensor(data["point_features"],    dtype=torch.float32)
        self.line_features     = torch.tensor(data["line_features"],     dtype=torch.float32)
        self.polygon_features  = torch.tensor(data["polygon_features"],  dtype=torch.float32)
        self.direction_features= torch.tensor(data["direction_features"],dtype=torch.float32)
        self.coords            = torch.tensor(data["coords"],            dtype=torch.float32)
        self.direction_labels  = torch.tensor(data["global_dir_labels"], dtype=torch.long)
        self.region_labels     = torch.tensor(data["region_labels"],     dtype=torch.long)
        self.neighbor_indices  = torch.tensor(data["neighbor_indices"],  dtype=torch.long)
        self.neighbor_distances= torch.tensor(data["neighbor_distances"],dtype=torch.float32)
        self.positive_indices  = torch.tensor(data["positive_indices"],  dtype=torch.long)
        self.negative_indices  = torch.tensor(data["negative_indices"],  dtype=torch.long)

        # 拼接完整特征（72 维），用于邻居编码
        self.all_features = torch.cat([
            self.point_features,
            self.line_features,
            self.polygon_features,
            self.direction_features,
        ], dim=-1)  # [N, 72]

    def __len__(self) -> int:
        return len(self.point_features)

    def __getitem__(self, idx: int) -> Dict:
        # 邻居特征：从全局特征表中查找
        neighbor_idx = self.neighbor_indices[idx]  # [K]
        neighbor_feats = self.all_features[neighbor_idx]  # [K, 72]

        return {
            "point_feat":       self.point_features[idx],
            "line_feat":        self.line_features[idx],
            "polygon_feat":     self.polygon_features[idx],
            "direction_feat":   self.direction_features[idx],
            "coords":           self.coords[idx],
            "direction_label":  self.direction_labels[idx],
            "region_label":     self.region_labels[idx],
            "neighbor_feats":   neighbor_feats,
            "neighbor_dists":   self.neighbor_distances[idx],
            "positive_idx":     self.positive_indices[idx],
            "negative_idx":     self.negative_indices[idx],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


# ============ 数据加载 ============

def load_data_ultimate(config: V26ProConfig, sample_ratio: float = 0.1) -> Dict:
    """加载终极训练数据（含邻居距离 + 正负样本）"""
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
    from spatial_encoder.v26_GLM.direction_supervision import compute_global_center_direction

    print(f"Loading data (sample_ratio={sample_ratio})...")
    base_data = load_dataset_for_training(config=config, sample_ratio=sample_ratio)
    coords = base_data["coords"]
    region_labels = base_data["region_labels"]
    N = len(coords)
    print(f"  Loaded {N} cells")

    # 全局方向标签
    print("  Computing direction labels...")
    global_dir = compute_global_center_direction(coords)

    # 预计算 K 近邻（含距离）
    k = config.spatial_attention.context_k
    print(f"  Precomputing {k}-NN with distances...")
    neighbor_indices, neighbor_distances = precompute_neighbors(coords, k=k)

    # 归一化距离到 [0, 1]
    max_dist = neighbor_distances.max() + 1e-8
    neighbor_distances = neighbor_distances / max_dist

    # 预计算正负样本
    print("  Precomputing positive/negative samples...")
    sampler = PositiveNegativeSampler(
        k_positive=config.dual_tower.positive_k,
        num_negatives=config.dual_tower.num_negatives,
        negative_min_distance=config.dual_tower.negative_min_distance,
    )
    pos_idx, neg_idx = sampler.precompute(coords, region_labels)

    return {
        "point_features":    base_data["point_features"],
        "line_features":     base_data["line_features"],
        "polygon_features":  base_data["polygon_features"],
        "direction_features":base_data["direction_features"],
        "coords":            coords,
        "global_dir_labels": global_dir.labels,
        "region_labels":     region_labels,
        "neighbor_indices":  neighbor_indices,
        "neighbor_distances":neighbor_distances,
        "positive_indices":  pos_idx,
        "negative_indices":  neg_idx,
    }


# ============ 评估 ============

@torch.no_grad()
def evaluate(model: UltimateSpatialEncoder, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()

    all_embs, all_coords, all_region_labels = [], [], []
    all_dir_preds, all_dir_labels, all_reg_preds = [], [], []

    for batch in loader:
        point_feat  = batch["point_feat"].to(device)
        line_feat   = batch["line_feat"].to(device)
        polygon_feat= batch["polygon_feat"].to(device)
        dir_feat    = batch["direction_feat"].to(device)

        emb, dir_pred, reg_pred, _ = model.forward_simple(
            point_feat, line_feat, polygon_feat, dir_feat
        )

        all_embs.append(emb.cpu())
        all_coords.append(batch["coords"])
        all_region_labels.append(batch["region_label"])
        all_dir_preds.append(dir_pred.cpu())
        all_dir_labels.append(batch["direction_label"])
        all_reg_preds.append(reg_pred.cpu())

    all_embs   = torch.cat(all_embs).numpy()
    all_coords = torch.cat(all_coords).numpy()
    all_rlabels= torch.cat(all_region_labels).numpy()
    all_dpreds = torch.cat(all_dir_preds).numpy()
    all_dlabels= torch.cat(all_dir_labels).numpy()
    all_rpreds = torch.cat(all_reg_preds).numpy()

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances, f1_score
    from sklearn.neighbors import NearestNeighbors

    results = {}

    # L1
    n = min(1000, len(all_embs))
    idx = np.random.choice(len(all_embs), n, replace=False)
    ed = pairwise_distances(all_embs[idx]).flatten()
    cd = pairwise_distances(all_coords[idx]).flatten()
    results["pearson"],  _ = pearsonr(ed, cd)
    results["spearman"], _ = spearmanr(ed, cd)

    # L2
    k = 20
    _, ci = NearestNeighbors(n_neighbors=k+1).fit(all_coords).kneighbors(all_coords)
    _, ei = NearestNeighbors(n_neighbors=k+1).fit(all_embs).kneighbors(all_embs)
    ci, ei = ci[:, 1:], ei[:, 1:]
    results["overlap"] = np.mean([len(set(ci[i]) & set(ei[i])) / k for i in range(len(all_embs))])

    # L3 方向
    results["dir_acc"] = (all_dpreds.argmax(-1) == all_dlabels).mean() * 100

    # L3 Region F1
    valid = all_rlabels < 6
    if valid.sum() > 0:
        results["region_f1"]  = f1_score(all_rlabels[valid], all_rpreds[valid].argmax(-1), average="macro") * 100
        results["region_acc"] = (all_rpreds[valid].argmax(-1) == all_rlabels[valid]).mean() * 100
    else:
        results["region_f1"] = results["region_acc"] = 0.0

    # L3 Intra-class Recall
    _, knn_idx = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(all_embs).kneighbors(all_embs)
    knn_idx = knn_idx[:, 1:]
    recalls = [
        (all_rlabels[knn_idx[i]] == all_rlabels[i]).sum() / k
        for i in range(len(all_embs)) if all_rlabels[i] < 6
    ]
    results["intra_recall"] = np.mean(recalls) * 100 if recalls else 0.0

    return results


# ============ 训练器 ============

class UltimateTrainer:

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
        self.best_pearson = 0.0
        self.best_state = None

    def train_epoch(self, loader: DataLoader, epoch: int, use_context: bool = True) -> Dict:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        loss_sums: Dict[str, float] = {}

        for batch in loader:
            dev = self.device
            point_feat   = batch["point_feat"].to(dev, non_blocking=True)
            line_feat    = batch["line_feat"].to(dev, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(dev, non_blocking=True)
            dir_feat     = batch["direction_feat"].to(dev, non_blocking=True)
            coords       = batch["coords"].to(dev, non_blocking=True)
            dir_label    = batch["direction_label"].to(dev, non_blocking=True)
            reg_label    = batch["region_label"].to(dev, non_blocking=True)
            pos_idx      = batch["positive_idx"].to(dev, non_blocking=True)
            neg_idx      = batch["negative_idx"].to(dev, non_blocking=True)

            neighbor_feats = batch["neighbor_feats"].to(dev, non_blocking=True) if use_context else None
            neighbor_dists = batch["neighbor_dists"].to(dev, non_blocking=True) if use_context else None

            emb, dir_pred, reg_pred, coord_pred, proto_loss = self.model(
                point_feat, line_feat, polygon_feat, dir_feat,
                neighbor_feats, neighbor_dists, reg_label,
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,
                true_coords=coords,
                direction_labels=dir_label,
                region_labels=reg_label,
                positive_indices=pos_idx,
                negative_indices=neg_idx,
                proto_loss=proto_loss,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict["total"]
            n_batches += 1
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        self.scheduler.step()
        return {k: v / n_batches for k, v in loss_sums.items()}

    def train(self, train_loader, val_loader, epochs: int) -> Dict:
        print(f"\n{'='*60}")
        print(f"Ultimate Encoder Training: Attention + Prototype + Contrastive")
        print(f"  Epochs: {epochs}, Batch: {train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()
            # 前 5 个 epoch 不启用上下文（warmup）
            use_context = epoch >= self.config.dual_tower.warmup_contrastive_epochs
            loss_dict = self.train_epoch(train_loader, epoch, use_context=use_context)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                results = evaluate(self.model, val_loader, self.device)
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Pearson={results['pearson']:.4f} | "
                    f"DirAcc={results['dir_acc']:.1f}% | "
                    f"RegF1={results['region_f1']:.1f}% | "
                    f"IntraRecall={results['intra_recall']:.1f}% | "
                    f"GPU={gpu_mem:.1f}GB | "
                    f"{time.time()-t0:.0f}s"
                )
                print(
                    f"  Losses: dist={loss_dict.get('distance',0):.3f} "
                    f"dir={loss_dict.get('direction',0):.3f} "
                    f"reg={loss_dict.get('region',0):.3f} "
                    f"contra={loss_dict.get('contrastive',0):.3f} "
                    f"proto={loss_dict.get('prototype',0):.3f}"
                )

                # 保存最佳模型（按 Pearson）
                if results['pearson'] > self.best_pearson:
                    self.best_pearson = results['pearson']
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    print(f"  [best] Best model saved (Pearson={self.best_pearson:.4f})")

        # 最终评估
        print("\n" + "="*60)
        print("Final Evaluation:")
        results = evaluate(self.model, val_loader, self.device)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # 保存最佳模型
        if self.best_state is not None:
            save_dir = Path(__file__).parent / "saved_models" / "cell_encoder"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.best_state, save_dir / "best_model.pt")
            print(f"  Best model saved to {save_dir / 'best_model.pt'}")
            print(f"  Best Pearson: {self.best_pearson:.4f}")
        print("="*60)
        return results


# ============ 主函数 ============

def run_ultimate_experiment(
    sample_ratio: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
    prototype_weight: float = 0.5,
    contrastive_weight: float = 1.0,
) -> Dict:
    print(f"\n{'#'*60}")
    print(f"# Phase 3: Ultimate Encoder (Attention + Prototype + Contrastive)")
    print(f"# sample={sample_ratio}, epochs={epochs}, batch={batch_size}")
    print(f"{'#'*60}\n")

    config = DEFAULT_PRO_CONFIG
    config.training.batch_size = batch_size
    config.training.num_epochs = epochs
    config.dual_tower.contrastive_weight = contrastive_weight
    config.prototype.prototype_weight = prototype_weight

    set_seed(config.training.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    data = load_data_ultimate(config, sample_ratio)
    N = len(data["coords"])

    dataset = UltimateDataset(data)
    train_size = int(0.9 * N)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, N - train_size]
    )

    actual_bs = max(64, min(batch_size, train_size // 4))
    train_loader = DataLoader(train_set, batch_size=actual_bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=actual_bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"Dataset: {N} cells, train={train_size}, val={N-train_size}, batch={actual_bs}")

    # 构建模型
    model = build_ultimate_encoder(config)
    params = count_parameters(model)
    print(f"Model: {params['total_m']:.2f}M parameters")

    # 损失函数
    criterion = UltimateLoss(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=config.loss.distance_weight,
        reconstruction_weight=config.loss.reconstruction_weight,
        direction_weight=config.loss.direction_weight,
        region_weight=config.loss.region_weight,
        contrastive_weight=contrastive_weight,
        prototype_weight=prototype_weight,
    )

    # 训练
    trainer = UltimateTrainer(model, criterion, config, device)
    return trainer.train(train_loader, val_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Ultimate Spatial Encoder")
    parser.add_argument("--sample",      type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch",       type=int,   default=256)
    parser.add_argument("--proto",       type=float, default=0.5,  help="Prototype loss weight")
    parser.add_argument("--contrastive", type=float, default=1.0,  help="Contrastive loss weight")
    args = parser.parse_args()

    run_ultimate_experiment(
        sample_ratio=args.sample,
        epochs=args.epochs,
        batch_size=args.batch,
        prototype_weight=args.proto,
        contrastive_weight=args.contrastive,
    )
