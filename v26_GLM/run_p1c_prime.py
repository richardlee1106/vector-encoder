# -*- coding: utf-8 -*-
"""
P1C': 使用伪标签重训练（简化版）

直接修改P1C集成训练，用伪标签替换原始标签

Author: Claude
Date: 2026-03-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

# 加载伪标签
print("Loading pseudo labels...")
pseudo_labels = np.load(Path(__file__).parent / "p1d_output" / "pseudo_labels.npy")
print(f"  Coverage: {(pseudo_labels < 6).sum() / len(pseudo_labels) * 100:.2f}%")

# 导入P1C集成训练模块
from spatial_encoder.v26_GLM.experiment_p1c_integrated import (
    load_data_integrated,
    CellDatasetP1C,
    collate_fn_p1c,
    set_seed,
    evaluate_full,
)
from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder
from spatial_encoder.v26_GLM.losses_v26_pro import KNNDistanceLoss, CoordinateReconstructionLoss
from spatial_encoder.v26_GLM.experiment_p1c_integrated import MultiSchemeDirectionLoss, RegionContrastiveLossP1C

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import time


class IntegratedLossP1CPrime(nn.Module):
    """P1C'集成损失"""

    def __init__(self, k_neighbors=85, region_weight=0.2):
        super().__init__()
        self.distance_loss = KNNDistanceLoss(k=k_neighbors)
        self.recon_loss = CoordinateReconstructionLoss()
        self.dir_loss = MultiSchemeDirectionLoss()
        self.region_loss = RegionContrastiveLossP1C(temperature=0.07)

        self.distance_weight = 3.0
        self.recon_weight = 1.0
        self.dir_weight = 2.0
        self.region_weight = region_weight

    def forward(self, embeddings, pred_coords, pred_direction, true_coords,
                neighbor_dir_labels, neighbor_dir_valid, global_dir_labels, global_dir_valid,
                region_labels):
        # 距离损失
        l_dist = self.distance_loss(embeddings, true_coords)

        # 重构损失
        l_recon = self.recon_loss(pred_coords, true_coords)

        # 方向损失
        l_dir, _ = self.dir_loss(pred_direction, neighbor_dir_labels, neighbor_dir_valid,
                                  global_dir_labels, global_dir_valid)

        # 功能区对比损失
        l_region, _ = self.region_loss(embeddings, region_labels)

        total = (self.distance_weight * l_dist +
                 self.recon_weight * l_recon +
                 self.dir_weight * l_dir +
                 self.region_weight * l_region)

        return total, {
            "dist": l_dist.item(),
            "recon": l_recon.item(),
            "dir": l_dir.item(),
            "region": l_region.item(),
        }


def train_p1c_prime(epochs=80, batch_size=16384, region_weight=0.2):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = DEFAULT_PRO_CONFIG

    # 加载数据
    print("Loading data...")
    data = load_data_integrated(config, sample_ratio=1.0)

    # 替换为伪标签
    data["region_labels"] = pseudo_labels
    print(f"Replaced region_labels with pseudo labels (coverage={pseudo_labels.sum()})")

    n_cells = len(data["coords"])

    dataset = CellDatasetP1C(data)
    train_size = int(0.9 * n_cells)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, n_cells - train_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn_p1c, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_p1c, num_workers=0, pin_memory=False)

    print(f"Dataset: {n_cells} cells, train={train_size}, val={n_cells - train_size}")

    # 模型
    model = build_mlp_encoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters")

    # 损失函数
    criterion = IntegratedLossP1CPrime(
        k_neighbors=config.loss.k_nearest_neighbors,
        region_weight=region_weight,
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate,
                       weight_decay=config.training.weight_decay)
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
            eval_results = evaluate_full(model, val_loader, device)
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"DirAcc={eval_results['global_dir_acc']:.1f}%, "
                  f"RegionF1={eval_results['region_f1']:.1f}%, "
                  f"RegionSep={eval_results['region_sep']:.2f}, "
                  f"Pearson={eval_results['pearson']:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time/60:.1f} min")

    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation:")
    final_results = evaluate_full(model, val_loader, device)
    print(f"  DirAcc: {final_results['global_dir_acc']:.2f}%")
    print(f"  Region F1: {final_results['region_f1']:.2f}%")
    print(f"  Region Sep: {final_results['region_sep']:.4f}")
    print(f"  Pearson: {final_results['pearson']:.4f}")
    print(f"  Spearman: {final_results['spearman']:.4f}")
    print(f"  Overlap@K: {final_results['overlap_at_k']:.2f}%")
    print(f"  Recall@20: {final_results['recall_at_20']:.2f}%")
    print("="*60)

    # 保存模型
    save_path = Path(__file__).parent / "p1d_output" / "p1c_prime_model_v2.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'results': final_results,
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--region-weight", type=float, default=0.2)
    args = parser.parse_args()

    train_p1c_prime(
        epochs=args.epochs,
        batch_size=args.batch,
        region_weight=args.region_weight,
    )
