# -*- coding: utf-8 -*-
"""
P1D: 基于当前 embedding 的标签传播

目标：将有标签覆盖率从 17.19% 提升到 40-60%

步骤：
1. 训练/加载模型
2. 生成全量 embedding
3. KNN 标签传播
4. 输出诊断报告
5. 保存伪标签

Author: GLM (Qianfan Code)
Date: 2026-03-17
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import matplotlib.pyplot as plt

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder, count_parameters
from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training


def train_and_save_model(config: V26ProConfig, epochs: int = 80, save_path: str = None):
    """训练模型并保存"""
    from spatial_encoder.v26_GLM.experiment_p1c_integrated import (
        load_data_integrated,
        CellDatasetP1C,
        collate_fn_p1c,
        IntegratedMultiTaskLoss,
        evaluate_full,
        set_seed,
    )

    set_seed(config.training.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    data = load_data_integrated(config, sample_ratio=1.0)
    n_cells = len(data["coords"])

    dataset = CellDatasetP1C(data)
    train_size = int(0.9 * n_cells)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, n_cells - train_size])

    train_loader = DataLoader(train_set, batch_size=16384, shuffle=True,
                               collate_fn=collate_fn_p1c, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=16384, shuffle=False,
                             collate_fn=collate_fn_p1c, num_workers=0, pin_memory=False)

    # 模型
    model = build_mlp_encoder(config).to(device)
    criterion = IntegratedMultiTaskLoss(
        k_nearest=config.loss.k_nearest_neighbors,
        distance_weight=3.0,
        reconstruction_weight=1.0,
        direction_weight=2.0,
        region_weight=0.3,
    )

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.training.learning_rate * 0.01)

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            point_feat = batch["point_feat"].to(device)
            line_feat = batch["line_feat"].to(device)
            polygon_feat = batch["polygon_feat"].to(device)
            direction_feat = batch["direction_feat"].to(device)
            coords = batch["coords"].to(device)
            neighbor_dir_label = batch["neighbor_dir_label"].to(device)
            neighbor_dir_valid = batch["neighbor_dir_valid"].to(device)
            global_dir_label = batch["global_dir_label"].to(device)
            global_dir_valid = batch["global_dir_valid"].to(device)
            region_label = batch["region_label"].to(device)

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

        scheduler.step()

        # 每15个epoch打印
        if (epoch + 1) % 15 == 0 or epoch == 0:
            eval_results = evaluate_full(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs}: DirAcc={eval_results['global_dir_acc']:.1f}%, "
                  f"RegionF1={eval_results['region_f1']:.1f}%, RegionSep={eval_results['region_sep']:.2f}")

    # 保存模型
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__,
        }, save_path)
        print(f"Model saved to {save_path}")

    return model, data, device


def load_model_for_inference(config: V26ProConfig, model_path: str):
    """加载已训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp_encoder(config).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, device


def generate_embeddings(model, data: dict, device, batch_size: int = 16384):
    """生成全量 embedding"""
    model.eval()

    point_features = torch.tensor(data["point_features"], dtype=torch.float32)
    line_features = torch.tensor(data["line_features"], dtype=torch.float32)
    polygon_features = torch.tensor(data["polygon_features"], dtype=torch.float32)
    direction_features = torch.tensor(data["direction_features"], dtype=torch.float32)

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(point_features), batch_size):
            batch_point = point_features[i:i+batch_size].to(device)
            batch_line = line_features[i:i+batch_size].to(device)
            batch_polygon = polygon_features[i:i+batch_size].to(device)
            batch_dir = direction_features[i:i+batch_size].to(device)

            emb, _, _, _ = model(batch_point, batch_line, batch_polygon, batch_dir)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings


def label_propagation(embeddings: np.ndarray, labels: np.ndarray,
                      k: int = 20, confidence_threshold: float = 0.6):
    """
    全局 KNN 标签传播

    Args:
        embeddings: (N, D) embedding 矩阵
        labels: (N,) 原始标签 (6 = 未知)
        k: 近邻数（全局KNN，然后统计有标签邻居）
        confidence_threshold: 置信度阈值

    Returns:
        new_labels: 传播后的标签
        confidence: 每个样本的置信度
    """
    n_samples = len(labels)
    new_labels = labels.copy()
    confidence = np.zeros(n_samples)

    # 有标签和无标签的索引
    labeled_mask = labels < 6  # 6 是未知
    unlabeled_mask = labels == 6

    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    print(f"\n=== Label Propagation (全局KNN) ===")
    print(f"  Labeled samples: {labeled_mask.sum()} ({labeled_mask.sum()/n_samples*100:.2f}%)")
    print(f"  Unlabeled samples: {unlabeled_mask.sum()} ({unlabeled_mask.sum()/n_samples*100:.2f}%)")
    print(f"  K = {k}, confidence threshold = {confidence_threshold}")

    # 关键修复：对整个 embedding 空间建立 KNN（包含有标签 + 无标签）
    # k+1 是为了排除自身
    print(f"  Building global KNN on {n_samples} samples...")
    nbrs_all = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embeddings)

    # 对每个无标签样本找 K+1 个全局近邻（包含自身）
    unlabeled_embeddings = embeddings[unlabeled_indices]
    all_distances, all_indices = nbrs_all.kneighbors(unlabeled_embeddings)

    # 统计并传播
    propagated_count = 0
    confidence_bins = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.0: 0}

    for i, (idx_list, dist_list) in enumerate(zip(all_indices, all_distances)):
        unlabeled_idx = unlabeled_indices[i]

        # 排除自身（第一个邻居是自己）
        neighbor_indices = all_indices[i][1:k+1]

        # 获取邻居的标签
        neighbor_labels_all = labels[neighbor_indices]

        # 只取有标签的邻居（排除其他无标签点）
        labeled_neighbor_labels = neighbor_labels_all[neighbor_labels_all < 6]

        if len(labeled_neighbor_labels) == 0:
            # 周围全是未知，跳过
            confidence[unlabeled_idx] = 0
            continue

        # 统计类别分布
        label_counts = Counter(labeled_neighbor_labels.tolist())
        most_common_label, most_common_count = label_counts.most_common(1)[0]

        # 计算置信度 = 最多类别占「有标签邻居」的比例
        conf = most_common_count / len(labeled_neighbor_labels)
        confidence[unlabeled_idx] = conf

        # 统计置信度分布
        for threshold in sorted(confidence_bins.keys(), reverse=True):
            if conf >= threshold:
                confidence_bins[threshold] += 1
                break

        # 如果置信度足够高，传播标签
        if conf >= confidence_threshold:
            new_labels[unlabeled_idx] = most_common_label
            propagated_count += 1

    print(f"\n  Propagated labels: {propagated_count}")
    print(f"  New coverage: {(labeled_mask.sum() + propagated_count) / n_samples * 100:.2f}%")

    print(f"\n  Confidence distribution:")
    for threshold in sorted(confidence_bins.keys()):
        print(f"    >= {threshold}: {confidence_bins[threshold]} samples")

    return new_labels, confidence


def generate_report(original_labels: np.ndarray, new_labels: np.ndarray,
                    confidence: np.ndarray, save_dir: str):
    """生成诊断报告"""

    # 类别名称
    class_names = {
        0: "居住类",
        1: "商业类",
        2: "工业类",
        3: "教育类",
        4: "公共类",
        5: "自然类",
        6: "未知",
    }

    n_samples = len(original_labels)

    print("\n" + "="*60)
    print("P1D Label Propagation Report")
    print("="*60)

    # 1. 覆盖率对比
    original_coverage = (original_labels < 6).sum() / n_samples * 100
    new_coverage = (new_labels < 6).sum() / n_samples * 100

    print(f"\n1. Coverage Comparison:")
    print(f"   Original: {original_coverage:.2f}% ({(original_labels < 6).sum()} samples)")
    print(f"   After propagation: {new_coverage:.2f}% ({(new_labels < 6).sum()} samples)")
    print(f"   Increase: +{new_coverage - original_coverage:.2f}% ({(new_labels < 6).sum() - (original_labels < 6).sum()} samples)")

    # 2. 各类别分布
    print(f"\n2. Class Distribution:")
    print(f"   {'Class':<10} {'Original':<12} {'After Prop':<12} {'Change':<10}")
    print(f"   {'-'*44}")

    for class_id in range(7):
        orig_count = (original_labels == class_id).sum()
        new_count = (new_labels == class_id).sum()
        change = new_count - orig_count
        print(f"   {class_names[class_id]:<10} {orig_count:<12} {new_count:<12} {'+' if change >= 0 else ''}{change:<10}")

    # 3. 置信度分布
    print(f"\n3. Confidence Distribution (propagated samples only):")
    propagated_mask = (original_labels == 6) & (new_labels < 6)
    if propagated_mask.sum() > 0:
        propagated_conf = confidence[propagated_mask]
        print(f"   Mean: {propagated_conf.mean():.3f}")
        print(f"   Median: {np.median(propagated_conf):.3f}")
        print(f"   High confidence (>0.9): {(propagated_conf > 0.9).sum()} ({(propagated_conf > 0.9).sum()/len(propagated_conf)*100:.1f}%)")
        print(f"   Medium confidence (0.7-0.9): {((propagated_conf >= 0.7) & (propagated_conf <= 0.9)).sum()}")

    print("="*60)

    # 保存报告
    report = {
        "original_coverage": original_coverage,
        "new_coverage": new_coverage,
        "propagated_count": int((new_labels < 6).sum() - (original_labels < 6).sum()),
        "class_distribution": {
            class_names[i]: {
                "original": int((original_labels == i).sum()),
                "after": int((new_labels == i).sum()),
            }
            for i in range(7)
        },
        "confidence_stats": {
            "mean": float(propagated_conf.mean()) if propagated_mask.sum() > 0 else 0,
            "median": float(np.median(propagated_conf)) if propagated_mask.sum() > 0 else 0,
        }
    }

    with open(Path(save_dir) / "p1d_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 保存伪标签
    np.save(Path(save_dir) / "pseudo_labels.npy", new_labels)
    np.save(Path(save_dir) / "confidence.npy", confidence)

    print(f"\nSaved: p1d_report.json, pseudo_labels.npy, confidence.npy")

    return report


def run_p1d(model_path: str = None, k: int = 20, confidence_threshold: float = 0.6):
    """运行 P1D 标签传播"""

    save_dir = Path(__file__).parent / "p1d_output"
    save_dir.mkdir(exist_ok=True)

    config = DEFAULT_PRO_CONFIG

    # 1. 加载或训练模型
    if model_path and Path(model_path).exists():
        model, device = load_model_for_inference(config, model_path)
        # 加载数据
        data = load_dataset_for_training(config=config, sample_ratio=1.0)
    else:
        print("No saved model found. Training from scratch...")
        model_path = str(save_dir / "p1c_model.pt")
        model, data, device = train_and_save_model(config, epochs=80, save_path=model_path)

    # 2. 生成 embedding
    print("\n=== Generating Embeddings ===")
    embeddings = generate_embeddings(model, data, device)

    # 3. 标签传播
    original_labels = data["region_labels"]
    new_labels, confidence = label_propagation(
        embeddings, original_labels,
        k=k,
        confidence_threshold=confidence_threshold
    )

    # 4. 生成报告
    report = generate_report(original_labels, new_labels, confidence, save_dir)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P1D: Label Propagation")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--k", type=int, default=20, help="K neighbors (global KNN)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold")
    args = parser.parse_args()

    run_p1d(model_path=args.model, k=args.k, confidence_threshold=args.threshold)
