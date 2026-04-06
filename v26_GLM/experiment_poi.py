# -*- coding: utf-8 -*-
"""
POI 级空间编码器训练脚本

数据：565K POI（替代 1828 Cell），标签来自 AOI/Landuse/POI 三层分层映射
架构：UltimateSpatialEncoder（双塔 + 原型 + 时空注意力）
目标：
  Region F1 > 50%（SupCon 正样本对从 ~6 → ~45，信号充足）
  Intra-class Recall > 50%
  保持 L1/L2 指标

运行：
  python -m spatial_encoder.v26_GLM.experiment_poi --sample 0.05 --epochs 10   # 快速验证
  python -m spatial_encoder.v26_GLM.experiment_poi --sample 0.1  --epochs 30   # 小样本
  python -m spatial_encoder.v26_GLM.experiment_poi --sample 1.0  --epochs 80   # 全量

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

from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.ultimate_encoder import (
    UltimateSpatialEncoder, UltimateLoss, build_ultimate_encoder,
)
from spatial_encoder.v26_GLM.encoder_v26_mlp import count_parameters
from spatial_encoder.v26_GLM.data_loader_poi import POIDataLoader
from spatial_encoder.v26_GLM.direction_supervision import compute_global_center_direction
from spatial_encoder.v26_GLM.spatial_attention_encoder import precompute_neighbors


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============ 数据集 ============

class POIDataset(Dataset):
    """POI 级训练数据集"""

    def __init__(self, data: Dict):
        self.point_features     = torch.tensor(data["point_features"],     dtype=torch.float32)
        self.line_features      = torch.tensor(data["line_features"],      dtype=torch.float32)
        self.polygon_features   = torch.tensor(data["polygon_features"],   dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords             = torch.tensor(data["coords"],             dtype=torch.float32)
        self.direction_labels   = torch.tensor(data["direction_labels"],   dtype=torch.long)
        self.region_labels      = torch.tensor(data["region_labels"],      dtype=torch.long)
        self.neighbor_indices   = torch.tensor(data["neighbor_indices"],   dtype=torch.long)
        self.neighbor_distances = torch.tensor(data["neighbor_distances"], dtype=torch.float32)

        # 全局特征表，用于邻居特征查找
        self.all_features = torch.cat([
            self.point_features,
            self.line_features,
            self.polygon_features,
            self.direction_features,
        ], dim=-1)  # [N, 72]

        # Cell 宏观 embedding（层次化多尺度，可选）
        if data.get("cell_embeddings") is not None:
            self.cell_embeddings    = torch.tensor(data["cell_embeddings"],    dtype=torch.float32)
            self.cell_dist_to_center= torch.tensor(data["cell_dist_to_center"],dtype=torch.float32)
        else:
            self.cell_embeddings     = None
            self.cell_dist_to_center = None

    def __len__(self) -> int:
        return len(self.point_features)

    def __getitem__(self, idx: int) -> Dict:
        neighbor_idx   = self.neighbor_indices[idx]       # [K]
        neighbor_feats = self.all_features[neighbor_idx]  # [K, 72]
        item = {
            "point_feat":      self.point_features[idx],
            "line_feat":       self.line_features[idx],
            "polygon_feat":    self.polygon_features[idx],
            "direction_feat":  self.direction_features[idx],
            "coords":          self.coords[idx],
            "direction_label": self.direction_labels[idx],
            "region_label":    self.region_labels[idx],
            "neighbor_feats":  neighbor_feats,
            "neighbor_dists":  self.neighbor_distances[idx],
        }
        if self.cell_embeddings is not None:
            item["cell_emb"]  = self.cell_embeddings[idx]
            item["cell_dist"] = self.cell_dist_to_center[idx]
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


# ============ 数据加载 ============

def load_poi_data(
    sample_ratio: float = 0.1,
    limit: Optional[int] = None,
    city_filter: Optional[List[str]] = None,
    k_attn: int = 20,
    cell_model_path: Optional[str] = None,
) -> Dict:
    """
    加载 POI 级训练数据。

    Args:
        sample_ratio:    采样比例
        limit:           最大样本数
        city_filter:     按区过滤（如 ['武昌区','汉阳区']）
        k_attn:          时空注意力的邻居数 K
        cell_model_path: Cell 级模型路径（启用层次化多尺度，可选）
    """
    loader = POIDataLoader(k_neighbors=50)
    result = loader.load(
        sample_ratio=sample_ratio,
        limit=limit,
        city_filter=city_filter,
        cell_model_path=cell_model_path,
    )
    pt, ln, pg, dr, coords, region_labels_raw, metadata, cell_embeddings, cell_dist = result
    N = len(coords)
    print(f"  {N:,} POIs loaded")

    # -1（无地块覆盖）→ 6（未知），兼容 SupCon / Focal Loss
    region_labels = region_labels_raw.copy()
    region_labels[region_labels == -1] = 6

    # 方向标签（相对城市中心）
    print("  Computing direction labels...")
    global_dir = compute_global_center_direction(coords)
    direction_labels = global_dir.labels  # [N]

    # 预计算时空注意力 K 近邻（含距离）
    print(f"  Precomputing {k_attn}-NN for spatial attention...")
    neighbor_indices, neighbor_distances = precompute_neighbors(coords, k=k_attn)
    max_dist = neighbor_distances.max() + 1e-8
    neighbor_distances = (neighbor_distances / max_dist).astype(np.float32)

    # 打印标签分布
    from collections import Counter
    from spatial_encoder.v26_GLM.data_loader_v26 import MERGED_REGION_NAMES
    dist = Counter(region_labels.tolist())
    print("  Label distribution:")
    for label in sorted(dist):
        name = MERGED_REGION_NAMES.get(label, "未知")
        print(f"    {label} {name}: {dist[label]:,} ({dist[label]/N*100:.1f}%)")

    return {
        "point_features":     pt,
        "line_features":      ln,
        "polygon_features":   pg,
        "direction_features": dr,
        "coords":             coords,
        "direction_labels":   direction_labels,
        "region_labels":      region_labels,
        "neighbor_indices":   neighbor_indices,
        "neighbor_distances": neighbor_distances,
        "metadata":           metadata,
        "cell_embeddings":    cell_embeddings,
        "cell_dist_to_center":cell_dist,
    }


# ============ 评估 ============

@torch.no_grad()
def evaluate(model: UltimateSpatialEncoder, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()

    all_embs, all_coords, all_rlabels = [], [], []
    all_dpreds, all_dlabels, all_rpreds = [], [], []

    for batch in loader:
        emb, dir_pred, reg_pred, _ = model.forward_simple(
            batch["point_feat"].to(device),
            batch["line_feat"].to(device),
            batch["polygon_feat"].to(device),
            batch["direction_feat"].to(device),
        )
        all_embs.append(emb.cpu())
        all_coords.append(batch["coords"])
        all_rlabels.append(batch["region_label"])
        all_dpreds.append(dir_pred.cpu())
        all_dlabels.append(batch["direction_label"])
        all_rpreds.append(reg_pred.cpu())

    all_embs   = torch.cat(all_embs).numpy()
    all_coords = torch.cat(all_coords).numpy()
    all_rlabels= torch.cat(all_rlabels).numpy()
    all_dpreds = torch.cat(all_dpreds).numpy()
    all_dlabels= torch.cat(all_dlabels).numpy()
    all_rpreds = torch.cat(all_rpreds).numpy()

    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances, f1_score
    from sklearn.neighbors import NearestNeighbors

    results = {}

    # L1: Pearson / Spearman（采样 2000 避免 OOM）
    n = min(2000, len(all_embs))
    idx = np.random.choice(len(all_embs), n, replace=False)
    ed = pairwise_distances(all_embs[idx]).flatten()
    cd = pairwise_distances(all_coords[idx]).flatten()
    results["pearson"],  _ = pearsonr(ed, cd)
    results["spearman"], _ = spearmanr(ed, cd)

    # L2: Overlap@20
    k = 20
    _, ci = NearestNeighbors(n_neighbors=k+1).fit(all_coords).kneighbors(all_coords)
    _, ei = NearestNeighbors(n_neighbors=k+1).fit(all_embs).kneighbors(all_embs)
    ci, ei = ci[:, 1:], ei[:, 1:]
    results["overlap"] = np.mean([len(set(ci[i]) & set(ei[i])) / k for i in range(len(all_embs))])

    # L3: 方向准确率
    results["dir_acc"] = (all_dpreds.argmax(-1) == all_dlabels).mean() * 100

    # L3: Region F1（仅有效标签 0-5）
    valid = all_rlabels < 6
    if valid.sum() > 10:
        results["region_f1"]  = f1_score(
            all_rlabels[valid], all_rpreds[valid].argmax(-1),
            average="macro", zero_division=0,
        ) * 100
        results["region_acc"] = (all_rpreds[valid].argmax(-1) == all_rlabels[valid]).mean() * 100
    else:
        results["region_f1"] = results["region_acc"] = 0.0

    # L3: Intra-class Recall@20
    _, knn_idx = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(all_embs).kneighbors(all_embs)
    knn_idx = knn_idx[:, 1:]
    recalls = [
        (all_rlabels[knn_idx[i]] == all_rlabels[i]).sum() / k
        for i in range(len(all_embs)) if all_rlabels[i] < 6
    ]
    results["intra_recall"] = np.mean(recalls) * 100 if recalls else 0.0

    return results


# ============ 训练器 ============

class POITrainer:

    def __init__(self, model, criterion, config, device):
        self.model     = model.to(device)
        self.criterion = criterion
        self.config    = config
        self.device    = device

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
        self.best_f1   = 0.0
        self.best_state= None

    def train_epoch(self, loader: DataLoader, use_context: bool = True) -> Dict:
        self.model.train()
        loss_sums: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            dev = self.device
            point_feat   = batch["point_feat"].to(dev, non_blocking=True)
            line_feat    = batch["line_feat"].to(dev, non_blocking=True)
            polygon_feat = batch["polygon_feat"].to(dev, non_blocking=True)
            dir_feat     = batch["direction_feat"].to(dev, non_blocking=True)
            coords       = batch["coords"].to(dev, non_blocking=True)
            dir_label    = batch["direction_label"].to(dev, non_blocking=True)
            reg_label    = batch["region_label"].to(dev, non_blocking=True)

            neighbor_feats = batch["neighbor_feats"].to(dev, non_blocking=True) if use_context else None
            neighbor_dists = batch["neighbor_dists"].to(dev, non_blocking=True) if use_context else None
            cell_emb  = batch["cell_emb"].to(dev, non_blocking=True)  if "cell_emb"  in batch else None
            cell_dist = batch["cell_dist"].to(dev, non_blocking=True) if "cell_dist" in batch else None

            emb, dir_pred, reg_pred, coord_pred, proto_loss = self.model(
                point_feat, line_feat, polygon_feat, dir_feat,
                neighbor_feats, neighbor_dists, reg_label,
                cell_context=cell_emb, cell_distances=cell_dist,
            )

            loss, loss_dict = self.criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                pred_region=reg_pred,
                true_coords=coords,
                direction_labels=dir_label,
                region_labels=reg_label,
                proto_loss=proto_loss,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            n_batches += 1
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v

        self.scheduler.step()
        return {k: v / n_batches for k, v in loss_sums.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict:
        print(f"\n{'='*60}")
        print(f"POI-level Training: {epochs} epochs, batch={train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()
            # warmup: 前 5 epoch 不启用时空注意力上下文
            use_context = epoch >= 5
            loss_dict = self.train_epoch(train_loader, use_context=use_context)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                results = evaluate(self.model, val_loader, self.device)
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

                print(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Pearson={results['pearson']:.4f} | "
                    f"Overlap={results['overlap']*100:.1f}% | "
                    f"DirAcc={results['dir_acc']:.1f}% | "
                    f"RegF1={results['region_f1']:.1f}% | "
                    f"IntraRecall={results['intra_recall']:.1f}% | "
                    f"GPU={gpu_mem:.1f}GB | {time.time()-t0:.0f}s"
                )
                print(
                    f"  Losses: dist={loss_dict.get('distance',0):.3f} "
                    f"dir={loss_dict.get('direction',0):.3f} "
                    f"reg={loss_dict.get('region',0):.3f} "
                    f"contra={loss_dict.get('contrastive',0):.3f} "
                    f"supcon={loss_dict.get('supcon',0):.3f} "
                    f"proto={loss_dict.get('prototype',0):.3f}"
                )

                # 保存最佳模型（按 Region F1）
                if results["region_f1"] > self.best_f1:
                    self.best_f1 = results["region_f1"]
                    save_dir = Path(__file__).parent / "saved_models" / "poi_encoder"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_dir / "best_model.pt")
                    print(f"  [best] Best model saved (RegF1={self.best_f1:.1f}%)")

        # 最终评估
        print("\n" + "="*60)
        print("Final Evaluation:")
        results = evaluate(self.model, val_loader, self.device)
        print(f"  Pearson:        {results['pearson']:.4f}")
        print(f"  Spearman:       {results['spearman']:.4f}")
        print(f"  Overlap@20:     {results['overlap']*100:.2f}%")
        print(f"  DirAcc:         {results['dir_acc']:.2f}%")
        print(f"  Region F1:      {results['region_f1']:.2f}%")
        print(f"  Region Acc:     {results['region_acc']:.2f}%")
        print(f"  IntraRecall@20: {results['intra_recall']:.2f}%")
        print("="*60)
        return results


# ============ 主函数 ============

def run_poi_experiment(
    sample_ratio: float = 0.1,
    epochs: int = 30,
    batch_size: int = 512,
    supcon_weight: float = 1.5,
    region_weight: float = 3.0,
    prototype_weight: float = 0.5,
    contrastive_weight: float = 1.0,
    city_filter: Optional[List[str]] = None,
    cell_model_path: Optional[str] = None,
) -> Dict:
    print(f"\n{'#'*60}")
    print(f"# POI-level Spatial Encoder")
    print(f"# sample={sample_ratio}, epochs={epochs}, batch={batch_size}")
    print(f"# supcon={supcon_weight}, region={region_weight}, proto={prototype_weight}")
    if cell_model_path:
        print(f"# cell_model={cell_model_path} [hierarchical multi-scale]")
    print(f"{'#'*60}\n")

    config = DEFAULT_PRO_CONFIG
    config.training.num_epochs = epochs

    set_seed(config.training.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 加载数据
    print("\nLoading POI data...")
    data = load_poi_data(
        sample_ratio=sample_ratio,
        k_attn=config.spatial_attention.context_k,
        city_filter=city_filter,
        cell_model_path=cell_model_path,
    )
    N = len(data["coords"])

    dataset = POIDataset(data)
    train_size = int(0.9 * N)
    val_size   = N - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    actual_bs = max(64, min(batch_size, train_size // 4))
    train_loader = DataLoader(
        train_set, batch_size=actual_bs, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=actual_bs * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"\nDataset: {N:,} POIs | train={train_size:,} | val={val_size:,} | batch={actual_bs}")

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
        region_weight=region_weight,
        contrastive_weight=contrastive_weight,
        supcon_weight=supcon_weight,
        prototype_weight=prototype_weight,
        temperature=config.dual_tower.contrastive_temperature,
    )
    print(f"Loss: dist={config.loss.distance_weight} dir={config.loss.direction_weight} "
          f"reg={region_weight} contra={contrastive_weight} "
          f"supcon={supcon_weight} proto={prototype_weight}")

    trainer = POITrainer(model, criterion, config, device)
    return trainer.train(train_loader, val_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POI-level Spatial Encoder")
    parser.add_argument("--sample",     type=float, default=0.1,  help="采样比例")
    parser.add_argument("--epochs",     type=int,   default=30,   help="训练轮数")
    parser.add_argument("--batch",      type=int,   default=512,  help="批大小")
    parser.add_argument("--supcon",     type=float, default=1.5,  help="SupCon 权重")
    parser.add_argument("--region",     type=float, default=3.0,  help="Region Focal 权重")
    parser.add_argument("--proto",      type=float, default=0.5,  help="Prototype 权重")
    parser.add_argument("--contrastive",type=float, default=1.0,  help="InfoNCE 权重")
    parser.add_argument("--cell_model", type=str,   default=None, help="Cell 级模型路径（启用层次化多尺度）")
    args = parser.parse_args()

    run_poi_experiment(
        sample_ratio=args.sample,
        epochs=args.epochs,
        batch_size=args.batch,
        supcon_weight=args.supcon,
        region_weight=args.region,
        prototype_weight=args.proto,
        contrastive_weight=args.contrastive,
        cell_model_path=args.cell_model,
    )
