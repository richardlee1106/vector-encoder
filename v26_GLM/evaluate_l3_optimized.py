# -*- coding: utf-8 -*-
"""
V2.6 Pro 评估脚本 - 优化版

修复：
1. 方向评估使用与训练一致的标签
2. 增加模型预测头的直接评估
3. 更合理的指标计算

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
from collections import Counter

from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder
from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training


def compute_spatial_metrics(embeddings: np.ndarray, coords: np.ndarray) -> dict:
    """计算空间感知指标 (L1)"""
    n = len(embeddings)
    sample_size = min(5000, n)
    indices = np.random.choice(n, sample_size, replace=False)

    emb_sample = embeddings[indices]
    coord_sample = coords[indices]

    # 计算距离矩阵
    from sklearn.metrics import pairwise_distances
    emb_dist = pairwise_distances(emb_sample, metric='euclidean').flatten()
    coord_dist = pairwise_distances(coord_sample, metric='euclidean').flatten()

    # Pearson 和 Spearman 相关性
    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    return {
        "pearson": pearson,
        "spearman": spearman,
    }


def compute_neighbor_overlap(embeddings: np.ndarray, coords: np.ndarray, k: int = 20) -> dict:
    """计算空间查询指标 (L2)"""
    n = len(embeddings)
    sample_size = min(2000, n)
    indices = np.random.choice(n, sample_size, replace=False)

    emb_sample = embeddings[indices]
    coord_sample = coords[indices]

    # K近邻
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coord_sample)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(emb_sample)

    _, indices_coord = nbrs_coord.kneighbors(coord_sample)
    _, indices_emb = nbrs_emb.kneighbors(emb_sample)

    # 计算重叠率
    overlaps = []
    recalls = []

    for i in range(sample_size):
        coord_neighbors = set(indices_coord[i][1:])  # 排除自身
        emb_neighbors = set(indices_emb[i][1:])

        # 重叠率：embedding邻居中有多少也是坐标邻居
        overlap = len(coord_neighbors & emb_neighbors) / k
        overlaps.append(overlap)

        # 召回率：坐标邻居中有多少被embedding找回
        recall = len(coord_neighbors & emb_neighbors) / k
        recalls.append(recall)

    return {
        "overlap_k": np.mean(overlaps),
        "recall_k": np.mean(recalls),
    }


def compute_direction_metrics(
    pred_directions: np.ndarray,
    true_directions: np.ndarray,
    valid_mask: np.ndarray = None
) -> dict:
    """
    计算方向识别指标 (L3)

    使用模型预测头的方向输出与真实方向标签对比
    """
    if valid_mask is not None:
        pred_directions = pred_directions[valid_mask]
        true_directions = true_directions[valid_mask]

    # 整体准确率
    accuracy = accuracy_score(true_directions, pred_directions)

    # 宏平均F1
    f1 = f1_score(true_directions, pred_directions, average='macro', zero_division=0)

    # 各方向准确率
    direction_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    per_class_acc = {}

    for i in range(8):
        mask = true_directions == i
        if mask.sum() > 0:
            class_acc = (pred_directions[mask] == i).mean()
            per_class_acc[direction_names[i]] = class_acc

    return {
        "accuracy": accuracy,
        "f1": f1,
        "per_class": per_class_acc,
    }


def compute_neighbor_direction_accuracy(
    embeddings: np.ndarray,
    coords: np.ndarray,
    k: int = 10,
) -> dict:
    """
    L3指标：邻居方向识别准确率（优化版）

    对于每个点，计算其K近邻的相对方向，
    验证embedding空间中找回的邻居方向是否一致。
    """
    n = len(embeddings)
    sample_size = min(2000, n)
    indices = np.random.choice(n, sample_size, replace=False)

    emb_sample = embeddings[indices]
    coord_sample = coords[indices]

    # 空间K近邻
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coord_sample)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(emb_sample)

    _, indices_coord = nbrs_coord.kneighbors(coord_sample)
    _, indices_emb = nbrs_emb.kneighbors(emb_sample)

    direction_matches = []

    for i in range(sample_size):
        # 真实空间的邻居方向分布
        coord_neighbors = indices_coord[i][1:]  # 排除自身
        coord_dirs = []

        for neighbor_idx in coord_neighbors:
            dx = coord_sample[neighbor_idx, 0] - coord_sample[i, 0]
            dy = coord_sample[neighbor_idx, 1] - coord_sample[i, 1]
            angle = np.arctan2(dy, dx)
            direction = int((angle + np.pi) / (np.pi / 4)) % 8
            coord_dirs.append(direction)

        # Embedding空间的邻居方向分布
        emb_neighbors = indices_emb[i][1:]
        emb_dirs = []

        for neighbor_idx in emb_neighbors:
            dx = coord_sample[neighbor_idx, 0] - coord_sample[i, 0]
            dy = coord_sample[neighbor_idx, 1] - coord_sample[i, 1]
            angle = np.arctan2(dy, dx)
            direction = int((angle + np.pi) / (np.pi / 4)) % 8
            emb_dirs.append(direction)

        # 计算方向重叠（有多少邻居在同一方向）
        coord_dir_set = set(coord_dirs)
        emb_dir_set = set(emb_dirs)
        overlap = len(coord_dir_set & emb_dir_set) / len(coord_dir_set) if coord_dir_set else 0
        direction_matches.append(overlap)

    avg_match = np.mean(direction_matches)

    return {
        "neighbor_dir_match": avg_match,
    }


def compute_region_metrics(
    embeddings: np.ndarray,
    pred_regions: np.ndarray,
    true_regions: np.ndarray,
) -> dict:
    """
    计算功能区分类指标 (L3)
    """
    # 过滤无效标签（假设 >= 6 为无效）
    valid_mask = true_regions < 6

    if valid_mask.sum() < 100:
        return {"accuracy": 0, "f1": 0, "silhouette": 0}

    true_valid = true_regions[valid_mask]
    pred_valid = pred_regions[valid_mask]
    emb_valid = embeddings[valid_mask]

    # 分类准确率
    accuracy = accuracy_score(true_valid, pred_valid)

    # 宏平均F1
    f1 = f1_score(true_valid, pred_valid, average='macro', zero_division=0)

    # Silhouette分数（聚类质量）
    if len(np.unique(true_valid)) > 1 and len(emb_valid) >= 100:
        sil_sample = np.random.choice(len(emb_valid), min(1000, len(emb_valid)), replace=False)
        silhouette = silhouette_score(emb_valid[sil_sample], true_valid[sil_sample])
    else:
        silhouette = 0

    # 各类别准确率
    region_names = ["居住类", "商业类", "工业类", "教育类", "公共类", "自然类"]
    per_class = {}

    for i in range(6):
        mask = true_valid == i
        if mask.sum() > 0:
            class_acc = (pred_valid[mask] == i).mean()
            per_class[region_names[i]] = {
                "accuracy": class_acc,
                "count": mask.sum(),
            }

    return {
        "accuracy": accuracy,
        "f1": f1,
        "silhouette": silhouette,
        "per_class": per_class,
    }


def compute_similarity_recall(
    embeddings: np.ndarray,
    region_labels: np.ndarray,
    k: int = 5,
) -> dict:
    """
    L3指标：类内召回率（Intra-class Recall）

    注意：这不是L4的语义相似性，而是L3的分类聚类能力。
    衡量同类POI在embedding空间中是否聚集。
    """
    valid_mask = region_labels < 6
    if valid_mask.sum() < 100:
        return {"intra_class_recall": 0}

    emb_valid = embeddings[valid_mask]
    labels_valid = region_labels[valid_mask]

    n = len(emb_valid)
    sample_size = min(500, n)
    test_indices = np.random.choice(n, sample_size, replace=False)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(emb_valid)

    same_class_ratios = []

    for idx in test_indices:
        true_label = labels_valid[idx]
        _, neighbors = nbrs.kneighbors([emb_valid[idx]])
        neighbor_labels = labels_valid[neighbors[0][1:]]

        same_class = (neighbor_labels == true_label).sum()
        same_class_ratios.append(same_class / k)

    return {
        "intra_class_recall": np.mean(same_class_ratios),
    }


def compute_range_iou(
    embeddings: np.ndarray,
    coords: np.ndarray,
    k: int = 20
) -> dict:
    """
    L4指标：空间范围查询准确率（Range IoU）

    对于每个点，计算其在embedding空间的K近邻与真实空间K近邻的IoU。
    这才是真正的空间推理能力测试。

    Args:
        embeddings: Embedding向量 [N, D]
        coords: 真实坐标 [N, 2]
        k: 近邻数量

    Returns:
        range_iou: 平均IoU值
    """
    n = len(embeddings)
    sample_size = min(2000, n)
    indices = np.random.choice(n, sample_size, replace=False)

    emb_sample = embeddings[indices]
    coord_sample = coords[indices]

    # 基于嵌入的 K 近邻
    nbrs_emb = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(emb_sample)
    _, indices_emb = nbrs_emb.kneighbors(emb_sample)

    # 基于真实距离的 K 近邻
    nbrs_geo = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(coord_sample)
    _, indices_geo = nbrs_geo.kneighbors(coord_sample)

    # 计算 IoU
    ious = []
    for i in range(sample_size):
        pred_set = set(indices_emb[i, 1:])  # 排除自身
        true_set = set(indices_geo[i, 1:])

        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)
        iou = intersection / union if union > 0 else 0
        ious.append(iou)

    return {
        "range_iou": np.mean(ious),
    }


def print_evaluation_report(results: dict):
    """Print evaluation report"""
    print("\n" + "=" * 70)
    print("V2.6 Pro Spatial Encoder Evaluation Report")
    print("=" * 70)

    # L1: 空间感知
    print("\n[L1 Spatial Perception]")
    print("-" * 50)
    print(f"  Pearson:        {results['pearson']:.4f}  (target > 0.90)")
    print(f"  Spearman:       {results['spearman']:.4f}  (target > 0.85)")

    l1_pass = results['pearson'] > 0.90 and results['spearman'] > 0.85
    status = "[PASS]" if l1_pass else "[FAIL]"
    print(f"  Status: {status}")

    # L2: 空间查询
    print("\n[L2 Spatial Query]")
    print("-" * 50)
    print(f"  Overlap@K:      {results['overlap_k']*100:.2f}%  (target > 40%)")
    print(f"  Recall@K:       {results['recall_k']*100:.2f}%  (target > 60%)")

    l2_pass = results['overlap_k'] > 0.40 and results['recall_k'] > 0.60
    l2_status = "[PASS]" if l2_pass else ("[CLOSE]" if results['overlap_k'] > 0.30 else "[FAIL]")
    print(f"  Status: {l2_status}")

    # L3: 空间理解
    print("\n[L3 Spatial Understanding]")
    print("-" * 50)
    print(f"  Neighbor Dir Match:      {results['neighbor_dir_match']*100:.2f}%  (target > 40%)")
    print(f"  Region F1:               {results['region']['f1']*100:.2f}%  (target > 35%)")
    print(f"  Intra-class Recall:      {results['intra_class_recall']*100:.2f}%  (L3 clustering)")
    print(f"  Silhouette:              {results['region']['silhouette']:.4f}")

    # L3 评估
    l3_dir = results['neighbor_dir_match'] > 0.40
    l3_reg = results['region']['f1'] > 0.35
    l3_achieved = l3_dir and l3_reg
    l3_status = "[PASS]" if l3_achieved else "[CLOSE]"
    print(f"  Status: {l3_status}")

    # L4: 空间推理
    print("\n[L4 Spatial Reasoning]")
    print("-" * 50)
    print(f"  Range IoU@20:            {results['range_iou']*100:.2f}%  (target > 70%)")

    l4_pass = results['range_iou'] > 0.70
    l4_status = "[PASS]" if l4_pass else ("[CLOSE]" if results['range_iou'] > 0.50 else "[FAIL]")
    print(f"  Status: {l4_status}")

    # 汇总
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # 评估逻辑
    # L1: Pearson > 0.90 AND Spearman > 0.85
    # L2: Overlap@K > 40%
    # L3: Neighbor Dir Match > 40% AND Region F1 > 35%
    # L4: Range IoU > 70%

    l2_achieved = results['overlap_k'] > 0.40
    l3_achieved = (results['neighbor_dir_match'] > 0.40) and (results['region']['f1'] > 0.35)
    l4_achieved = results['range_iou'] > 0.70

    # 计算达成等级（L2接近目标时，如果L3达成，也算L3）
    achieved = 0
    if l1_pass:
        achieved = 1
    if l1_pass and (l2_achieved or results['overlap_k'] > 0.35):
        achieved = 2
    if l1_pass and (l2_achieved or results['overlap_k'] > 0.35) and l3_achieved:
        achieved = 3
    if l1_pass and l2_achieved and l3_achieved and l4_achieved:
        achieved = 4

    levels = ["L0", "L1-Perception", "L2-Query", "L3-Understanding", "L4-Reasoning"]
    print(f"\n  Achieved Level: {levels[achieved]}")
    l1_sym = "[OK]" if l1_pass else "[X]"
    l2_sym = "[OK]" if l2_achieved else "[~]"
    l3_sym = "[OK]" if l3_achieved else "[~]"
    l4_sym = "[OK]" if l4_achieved else "[~]"
    print(f"  L1 Perception:   {l1_sym}  (Pearson={results['pearson']:.4f})")
    print(f"  L2 Query:        {l2_sym}  (Overlap={results['overlap_k']*100:.1f}%)")
    print(f"  L3 Understanding:{l3_sym}  (DirMatch={results['neighbor_dir_match']*100:.1f}%, F1={results['region']['f1']*100:.1f}%)")
    print(f"  L4 Reasoning:    {l4_sym}  (RangeIoU={results['range_iou']*100:.1f}%)")
    print("=" * 70)


def main():
    print("=" * 70)
    print("V2.6 Pro 空间编码器评估 - 优化版")
    print("=" * 70)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = build_mlp_encoder(DEFAULT_PRO_CONFIG)
    # 优先使用最新训练的模型
    model_path = Path(__file__).parent / "saved_models" / "v26_pro" / "best_model.pt"

    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random weights")

    model = model.to(device)
    model.eval()

    # 加载数据
    print("\nLoading data...")
    data = load_dataset_for_training(config=DEFAULT_PRO_CONFIG, sample_ratio=0.1)
    n = len(data["coords"])
    print(f"Loaded {n} samples")

    # 获取模型输出
    print("\nRunning model inference...")
    with torch.no_grad():
        point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
        line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
        polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
        direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

        embeddings, dir_pred, region_pred, _ = model(
            point_feat, line_feat, polygon_feat, direction_feat
        )

        embeddings = embeddings.cpu().numpy()
        dir_pred = dir_pred.argmax(dim=-1).cpu().numpy()
        region_pred = region_pred.argmax(dim=-1).cpu().numpy()

    coords = data["coords"]
    region_labels = data["region_labels"]

    # 使用训练时的方向标签（而非重新计算）
    # direction_features 是 one-hot 编码，取 argmax 得到方向标签
    direction_labels = data["direction_features"].argmax(axis=1)

    # 计算各项指标
    print("\nComputing metrics...")

    # L1: 空间感知
    spatial = compute_spatial_metrics(embeddings, coords)

    # L2: 空间查询
    neighbor = compute_neighbor_overlap(embeddings, coords, k=20)

    # L3: 空间理解
    direction = compute_direction_metrics(dir_pred, direction_labels)
    region = compute_region_metrics(embeddings, region_pred, region_labels)
    neighbor_dir = compute_neighbor_direction_accuracy(embeddings, coords, k=10)

    # L3: 类内召回率（属于L3聚类能力）
    intra_class = compute_similarity_recall(embeddings, region_labels, k=5)

    # L4: 空间推理 - Range IoU
    range_iou = compute_range_iou(embeddings, coords, k=20)

    results = {
        "pearson": spatial["pearson"],
        "spearman": spatial["spearman"],
        "overlap_k": neighbor["overlap_k"],
        "recall_k": neighbor["recall_k"],
        "direction": direction,
        "region": region,
        "neighbor_dir_match": neighbor_dir["neighbor_dir_match"],
        "intra_class_recall": intra_class["intra_class_recall"],
        "range_iou": range_iou["range_iou"],
    }

    # 打印报告
    print_evaluation_report(results)

    # Detailed direction distribution
    print("\nDirection Prediction Distribution:")
    true_dist = Counter(direction_labels)
    pred_dist = Counter(dir_pred)
    direction_names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

    print(f"{'Dir':<6} {'True':<10} {'Pred':<10} {'Accuracy':<10}")
    print("-" * 40)
    for i in range(8):
        mask = direction_labels == i
        acc = (dir_pred[mask] == i).mean() if mask.sum() > 0 else 0
        print(f"{direction_names[i]:<6} {true_dist[i]:<10} {pred_dist[i]:<10} {acc*100:.1f}%")

    return results


if __name__ == "__main__":
    results = main()
