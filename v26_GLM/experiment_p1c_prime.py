# -*- coding: utf-8 -*-
"""
P1C': 使用伪标签重训练

目标：使用 P1D 传播后的伪标签（覆盖率44.33%）重训练模型
期望：Region F1 突破 30%

Author: GLM (Qianfan Code)
Date: 2026-03-17
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, silhouette_score
from scipy.stats import pearsonr, spearmanr
from collections import Counter
import time

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder, count_parameters
from spatial_encoder.v26_GLM.experiment_p1c_integrated import (
    load_data_integrated,
    CellDatasetP1C,
    collate_fn_p1c,
    MultiSchemeDirectionLoss,
    evaluate_full,
    set_seed,
)
from spatial_encoder.v26_GLM.losses_v26_pro import KNNDistanceLoss, CoordinateReconstructionLoss


def load_pseudo_labels():
    """加载 P1D 生成的伪标签"""
    pseudo_path = Path(__file__).parent / "p1d_output" / "pseudo_labels.npy"
    confidence_path = Path(__file__).parent / "p1d_output" / "confidence.npy"

    pseudo_labels = np.load(pseudo_path)
    confidence = np.load(confidence_path)

    print(f"Loaded pseudo labels: {pseudo_labels.shape}")
    print(f"  Coverage: {(pseudo_labels < 6).sum() / len(pseudo_labels) * 100:.2f}%")

    return pseudo_labels, confidence


class WeightedRegionContrastiveLoss(nn.Module):
    """加权功能区对比损失，使用置信度加权"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, confidence):
        # 只用有标签的样本（label < 6）
        valid_mask = labels < 6
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=embeddings.device), {}

        valid_emb = embeddings[valid_indices]
        valid_labels = labels[valid_indices]
        valid_conf = confidence[valid_indices]

        n = len(valid_indices)

        # 计算相似度矩阵
        emb_norm = torch.nn.functional.normalize(valid_emb, p=2, dim=1)
        sim_matrix = torch.mm(emb_norm, emb_norm.t()) / self.temperature

        # 创建标签匹配矩阵
        label_matrix = valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)

        # 排除对角线
        mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        label_matrix = label_matrix & mask

        # 加权：高置信度样本权重更高
        weight_matrix = valid_conf.unsqueeze(0) * valid_conf.unsqueeze(1)
        weight_matrix = weight_matrix * mask.float()

        # 正样本对损失
        pos_mask = label_matrix.float()
        if pos_mask.sum() > 0:
            pos_sim = (sim_matrix * pos_mask * weight_matrix).sum() / (pos_mask * weight_matrix).sum()
        else:
            pos_sim = torch.tensor(0.0, device=embeddings.device)

        # 负样本对损失
        neg_mask = (~label_matrix).float() * mask.float()
        if neg_mask.sum() > 0:
            neg_sim = (sim_matrix * neg_mask * weight_matrix).sum() / (neg_mask * weight_matrix).sum()
        else:
            neg_sim = torch.tensor(0.0, device=embeddings.device)

        # 对比损失
        loss = -pos_sim + torch.log(torch.exp(sim_matrix * mask).sum(dim=1).mean())

        return loss, {"pos_sim": pos_sim.item(), "neg_sim": neg_sim.item()}


class IntegratedMultiTaskLossP1CPrime(nn.Module):
    """P1C' 集成多任务损失"""

    def __init__(self, k_nearest: int = 85,
                 distance_weight: float = 3.0,
                 reconstruction_weight: float = 1.0,
                 direction_weight: float = 2.0,
                 region_weight: float = 0.2):
        super().__init__()
        self.distance_weight = distance_weight
        self.reconstruction_weight = reconstruction_weight
        self.direction_weight = direction_weight
        self.region_weight = region_weight

        self.distance_loss = KNNDistanceLoss(k=k_nearest)
        self.reconstruct_loss = CoordinateReconstructionLoss()
        self.direction_loss = MultiSchemeDirectionLoss()
        self.region_loss = WeightedRegionContrastiveLoss(temperature=0.1)

    def forward(self, embeddings, pred_coords, pred_direction, true_coords,
                neighbor_dir_labels, neighbor_dir_valid, global_dir_labels, global_dir_valid,
                region_labels, confidence):
        total_loss = 0.0
        metrics = {}

        # 1. 距离保持损失
        l_dist = self.distance_loss(embeddings, true_coords)
        total_loss = total_loss + self.distance_weight * l_dist
        metrics["distance_loss"] = l_dist.item()

        # 2. 坐标重构损失
        l_recon = self.reconstruct_loss(pred_coords, true_coords)
        total_loss = total_loss + self.reconstruction_weight * l_recon
        metrics["reconstruct_loss"] = l_recon.item()

        # 3. 多方案方向损失
        l_dir, dir_metrics = self.direction_loss(
            pred_direction, neighbor_dir_labels, neighbor_dir_valid,
            global_dir_labels, global_dir_valid
        )
        total_loss = total_loss + self.direction_weight * l_dir
        metrics["direction_loss"] = l_dir.item()

        # 4. 加权功能区对比损失
        l_region, m_region = self.region_loss(embeddings, region_labels, confidence)
        total_loss = total_loss + self.region_weight * l_region
        metrics["region_loss"] = l_region.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics


class CellDatasetP1CPrime(Dataset):
    """P1C' 数据集，使用伪标签和置信度"""

    def __init__(self, data: dict, pseudo_labels: np.ndarray, confidence: np.ndarray):
        self.point_features = torch.tensor(data["point_features"], dtype=torch.float32)
        self.line_features = torch.tensor(data["line_features"], dtype=torch.float32)
        self.polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
        self.direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)
        self.pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
        self.confidence = torch.tensor(confidence, dtype=torch.float32)

        self.neighbor_dir_labels = torch.tensor(data["neighbor_dir_labels"], dtype=torch.long)
        self.neighbor_dir_valid = torch.tensor(data["neighbor_dir_valid"], dtype=torch.bool)
        self.global_dir_labels = torch.tensor(data["global_dir_labels"], dtype=torch.long)
        self.global_dir_valid = torch.tensor(data["global_dir_valid"], dtype=torch.bool)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return {
            "point_feat": self.point_features[idx],
            "line_feat": self.line_features[idx],
            "polygon_feat": self.polygon_features[idx],
            "direction_feat": self.direction_features[idx],
            "coords": self.coords[idx],
            "region_label": self.pseudo_labels[idx],
            "confidence": self.confidence[idx],
            "neighbor_dir_label": self.neighbor_dir_labels[idx],
            "neighbor_dir_valid": self.neighbor_dir_valid[idx],
            "global_dir_label": self.global_dir_labels[idx],
            "global_dir_valid": self.global_dir_valid[idx],
        }


def collate_fn_p1c_prime(batch):
    return {
        "point_feat": torch.stack([b["point_feat"] for b in batch]),
        "line_feat": torch.stack([b["line_feat"] for b in batch]),
        "polygon_feat": torch.stack([b["polygon_feat"] for b in batch]),
        "direction_feat": torch.stack([b["direction_feat"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "region_label": torch.stack([b["region_label"] for b in batch]),
        "confidence": torch.stack([b["confidence"] for b in batch]),
        "neighbor_dir_label": torch.stack([b["neighbor_dir_label"] for b in batch]),
        "neighbor_dir_valid": torch.stack([b["neighbor_dir_valid"] for b in batch]),
        "global_dir_label": torch.stack([b["global_dir_label"] for b in batch]),
        "global_dir_valid": torch.stack([b["global_dir_valid"] for b in batch]),
    }


def train_p1c_prime(epochs: int = 80, batch_size: int = 16384, region_weight: float = 0.2):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 加载数据
    config = DEFAULT_PRO_CONFIG
    print("Loading data...")
    data = load_data_integrated(config, sample_ratio=1.0)

    # 加载伪标签
    pseudo_labels, confidence = load_pseudo_labels()

    # 创建数据集
    dataset = CellDatasetP1CPrime(data, pseudo_labels, confidence)
    n_cells = len(dataset)

    train_size = int(0.9 * n_cells)
    val_size = n_cells - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn_p1c_prime, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_p1c_prime, num_workers=0, pin_memory=False)

    print(f"Dataset: {n_cells} cells, train={train_size}, val={val_size}")
    print(f"Batch size: {batch_size}")

    # 模型
    model = build_mlp_encoder(config).to(device)
    n_params = count_parameters(model)
    print(f"Model: {n_params['total_m']:.1f}M parameters")

    # 损失函数
    criterion = IntegratedMultiTaskLossP1CPrime(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=3.0,
        reconstruction_weight=1.0,
        direction_weight=2.0,
        region_weight=region_weight,
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.training.learning_rate * 0.01)

    print(f"\n{'='*60}")
    print(f"P1C' Training with Pseudo Labels")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Region weight: {region_weight}")
    print(f"  Label coverage: {(pseudo_labels < 6).sum() / len(pseudo_labels) * 100:.2f}%")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            point_feat = batch["point_feat"].to(device)
            line_feat = batch["line_feat"].to(device)
            polygon_feat = batch["polygon_feat"].to(device)
            direction_feat = batch["direction_feat"].to(device)
            coords = batch["coords"].to(device)
            region_label = batch["region_label"].to(device)
            confidence_batch = batch["confidence"].to(device)
            neighbor_dir_label = batch["neighbor_dir_label"].to(device)
            neighbor_dir_valid = batch["neighbor_dir_valid"].to(device)
            global_dir_label = batch["global_dir_label"].to(device)
            global_dir_valid = batch["global_dir_valid"].to(device)

            emb, dir_pred, reg_pred, coord_pred = model(
                point_feat, line_feat, polygon_feat, direction_feat
            )

            loss, _ = criterion(
                embeddings=emb,
                pred_coords=coord_pred,
                pred_direction=dir_pred,
                true_coords=coords,
                neighbor_dir_labels=neighbor_dir_label,
                neighbor_dir_valid=neighbor_dir_valid,
                global_dir_labels=global_dir_label,
                global_dir_valid=global_dir_valid,
                region_labels=region_label,
                confidence=confidence_batch,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # 每15个epoch评估
        if (epoch + 1) % 15 == 0 or epoch == 0:
            # 使用原始 evaluate_full 但需要转换数据格式
            model.eval()
            all_embeddings = []
            all_coords = []
            all_region_labels = []
            all_global_dir = []
            all_global_dir_valid = []

            correct_global = 0
            total_global = 0

            with torch.no_grad():
                for batch in val_loader:
                    point_feat = batch["point_feat"].to(device)
                    line_feat = batch["line_feat"].to(device)
                    polygon_feat = batch["polygon_feat"].to(device)
                    direction_feat = batch["direction_feat"].to(device)
                    coords = batch["coords"].to(device)
                    region_label = batch["region_label"]
                    global_dir_label = batch["global_dir_label"]
                    global_dir_valid = batch["global_dir_valid"]

                    emb, dir_pred, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)

                    all_embeddings.append(emb.cpu())
                    all_coords.append(coords.cpu())
                    all_region_labels.append(region_label)
                    all_global_dir.append(global_dir_label)
                    all_global_dir_valid.append(global_dir_valid)

                    dir_pred_labels = dir_pred.argmax(dim=1)
                    valid_mask = global_dir_valid.to(device)
                    global_dir_label_dev = global_dir_label.to(device)
                    correct = (dir_pred_labels[valid_mask] == global_dir_label_dev[valid_mask]).sum()
                    correct_global += correct.item()
                    total_global += valid_mask.sum().item()

            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            coords_np = torch.cat(all_coords, dim=0).numpy()
            region_labels_np = torch.cat(all_region_labels, dim=0).numpy()

            # Pearson
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            sim_matrix = np.dot(emb_norm, emb_norm.T)
            emb_dist = np.sqrt(2 - 2 * sim_matrix + 1e-8)
            coord_dist = np.sqrt(((coords_np[:, None, :] - coords_np[None, :, :]) ** 2).sum(axis=2))
            mask_triu = np.triu(np.ones(len(embeddings), dtype=bool), k=1)
            pearson, _ = pearsonr(emb_dist[mask_triu], coord_dist[mask_triu])

            # Region F1 & Sep
            valid_mask_np = region_labels_np < 6
            if valid_mask_np.sum() > 100:
                sil = silhouette_score(embeddings[valid_mask_np], region_labels_np[valid_mask_np])
            else:
                sil = 0

            class_centers = {}
            for c in range(6):
                mask_c = region_labels_np == c
                if mask_c.sum() > 0:
                    class_centers[c] = embeddings[mask_c].mean(axis=0)

            intra_dist = []
            for c in range(6):
                mask_c = region_labels_np == c
                if mask_c.sum() > 1:
                    dists = np.linalg.norm(embeddings[mask_c] - class_centers[c], axis=1)
                    intra_dist.append(dists.mean())

            inter_dist = []
            classes = list(class_centers.keys())
            for i, c1 in enumerate(classes):
                for c2 in classes[i+1:]:
                    inter_dist.append(np.linalg.norm(class_centers[c1] - class_centers[c2]))

            if len(intra_dist) > 0 and len(inter_dist) > 0:
                region_sep = np.mean(inter_dist) / (np.mean(intra_dist) + 1e-8)
            else:
                region_sep = 0

            global_dir_acc = correct_global / total_global * 100 if total_global > 0 else 0
            coverage = valid_mask_np.sum() / len(region_labels_np) * 100

            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"DirAcc={global_dir_acc:.1f}%, "
                  f"RegionSep={region_sep:.2f}, "
                  f"Pearson={pearson:.4f}, "
                  f"Coverage={coverage:.1f}%")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} min")

    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation:")
    print(f"  DirAcc: {global_dir_acc:.2f}%")
    print(f"  Region Sep: {region_sep:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    print(f"  Label Coverage: {coverage:.2f}%")
    print("="*60)

    # 保存模型
    save_path = Path(__file__).parent / "p1d_output" / "p1c_prime_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return {
        "global_dir_acc": global_dir_acc,
        "region_sep": region_sep,
        "pearson": pearson,
        "coverage": coverage,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P1C': Training with Pseudo Labels")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--region-weight", type=float, default=0.2)
    args = parser.parse_args()

    train_p1c_prime(
        epochs=args.epochs,
        batch_size=args.batch,
        region_weight=args.region_weight,
    )
