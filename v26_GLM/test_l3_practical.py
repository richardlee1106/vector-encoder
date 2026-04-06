# -*- coding: utf-8 -*-
"""
L3 实测脚本 - 验证空间编码器实际能力

测试场景：
1. 空间查询：给定坐标，找回空间邻居
2. 方向识别：给定两点，预测相对方向
3. 功能区分类：给定POI特征，预测功能区
4. 相似检索：给定POI，找语义相似的POI

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder


def load_model(model_path: str = None) -> Tuple[torch.nn.Module, torch.device]:
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DEFAULT_PRO_CONFIG
    model = build_mlp_encoder(config)

    # 尝试加载保存的模型
    if model_path is None:
        model_path = Path(__file__).parent / "saved_models" / "v26_pro" / "l3_trained_model.pt"

    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}, using random weights")

    model = model.to(device)
    model.eval()
    return model, device


def load_sample_data(n_samples: int = 1000) -> Dict:
    """加载样本数据用于测试"""
    try:
        from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training

        print(f"Loading {n_samples} sample cells...")
        data = load_dataset_for_training(config=DEFAULT_PRO_CONFIG, sample_ratio=0.01)

        # 限制样本数量
        n = min(n_samples, len(data["coords"]))
        indices = np.random.choice(len(data["coords"]), n, replace=False)

        return {
            "point_features": data["point_features"][indices],
            "line_features": data["line_features"][indices],
            "polygon_features": data["polygon_features"][indices],
            "direction_features": data["direction_features"][indices],
            "coords": data["coords"][indices],
            "region_labels": data["region_labels"][indices],
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using mock data for testing...")

        # 生成模拟数据
        np.random.seed(42)
        return {
            "point_features": np.random.randn(n_samples, 32).astype(np.float32),
            "line_features": np.random.randn(n_samples, 16).astype(np.float32),
            "polygon_features": np.random.randn(n_samples, 16).astype(np.float32),
            "direction_features": np.random.randn(n_samples, 8).astype(np.float32),
            "coords": np.random.rand(n_samples, 2).astype(np.float32) * 0.5 + np.array([114.0, 30.5]),
            "region_labels": np.random.randint(0, 7, n_samples).astype(np.int64),
        }


@torch.no_grad()
def test_spatial_query(model, data: Dict, device: torch.device, k: int = 10) -> Dict:
    """
    测试1：空间查询能力

    给定一个坐标，找回空间上最近的K个邻居
    验证：embedding空间中的邻居与真实空间邻居的重叠率
    """
    print("\n" + "="*60)
    print("测试1：空间查询能力")
    print("="*60)

    coords = data["coords"]
    n_samples = min(100, len(coords))

    # 获取所有embedding
    point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
    line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
    polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
    direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

    embeddings, _, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
    embeddings = embeddings.cpu().numpy()

    # 计算真实空间的K近邻
    from sklearn.neighbors import NearestNeighbors
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coords)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(embeddings)

    # 随机选择测试点
    test_indices = np.random.choice(len(coords), min(20, len(coords)), replace=False)

    overlaps = []
    for idx in test_indices:
        # 真实空间的邻居
        _, coord_neighbors = nbrs_coord.kneighbors([coords[idx]])
        coord_neighbors = set(coord_neighbors[0][1:])  # 排除自身

        # Embedding空间的邻居
        _, emb_neighbors = nbrs_emb.kneighbors([embeddings[idx]])
        emb_neighbors = set(emb_neighbors[0][1:])

        # 计算重叠率
        overlap = len(coord_neighbors & emb_neighbors) / k
        overlaps.append(overlap)

    avg_overlap = np.mean(overlaps) * 100

    print(f"测试样本数: {len(test_indices)}")
    print(f"K近邻数: {k}")
    print(f"平均重叠率: {avg_overlap:.2f}%")
    print(f"解释: 在Embedding空间找回的邻居中，平均{avg_overlap:.1f}%也是真实空间的邻居")

    return {
        "test": "spatial_query",
        "samples": len(test_indices),
        "k": k,
        "overlap_rate": avg_overlap,
    }


@torch.no_grad()
def test_direction_recognition(model, data: Dict, device: torch.device) -> Dict:
    """
    测试2：方向识别能力

    训练时使用邻居相对方向，这里测试全局中心方向作为替代指标。
    由于方向特征 (direction_features) 使用全局中心编码，
    模型应该能学习到全局位置感知能力。
    """
    print("\n" + "="*60)
    print("测试2：方向识别能力")
    print("="*60)

    coords = data["coords"]

    # 检查 direction_features 是否已经包含全局中心方向
    direction_features = data["direction_features"]
    true_directions = direction_features.argmax(axis=1)

    # 获取模型预测
    point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
    line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
    polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
    direction_feat = torch.tensor(direction_features, dtype=torch.float32).to(device)

    _, dir_pred, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
    pred_directions = dir_pred.argmax(dim=-1).cpu().numpy()

    # 计算准确率
    correct = (pred_directions == true_directions).sum()
    accuracy = correct / len(coords) * 100

    # 方向分布
    from collections import Counter
    true_dist = Counter(true_directions)
    pred_dist = Counter(pred_directions)

    # 方向名称（根据 direction_supervision.py 定义）
    # 0: 东(E), 1: 东北(NE), 2: 北(N), 3: 西北(NW), 4: 西(W), 5: 西南(SW), 6: 南(S), 7: 东南(SE)
    direction_names = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]

    print(f"测试样本数: {len(coords)}")
    print(f"方向准确率: {accuracy:.2f}%")
    print(f"\n方向分布对比:")
    print(f"{'方向':<8} {'真实':<10} {'预测':<10}")
    print("-" * 28)
    for i in range(8):
        print(f"{direction_names[i]:<8} {true_dist[i]:<10} {pred_dist[i]:<10}")

    return {
        "test": "direction_recognition",
        "samples": len(coords),
        "accuracy": accuracy,
    }


@torch.no_grad()
def test_region_classification(model, data: Dict, device: torch.device) -> Dict:
    """
    测试3：功能区分类能力

    给定POI特征，预测功能区类别
    """
    print("\n" + "="*60)
    print("测试3：功能区分类能力")
    print("="*60)

    region_labels = data["region_labels"]

    # 获取模型预测
    point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
    line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
    polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
    direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

    _, _, reg_pred, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
    pred_labels = reg_pred.argmax(dim=-1).cpu().numpy()

    # 过滤未知标签
    valid_mask = region_labels < 6
    true_labels = region_labels[valid_mask]
    pred_valid = pred_labels[valid_mask]

    # 计算准确率
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    accuracy = accuracy_score(true_labels, pred_valid) * 100
    f1 = f1_score(true_labels, pred_valid, average='macro') * 100

    # 类别分布
    region_names = ["居住类", "商业类", "工业类", "教育类", "公共类", "自然类"]

    print(f"有效样本数: {valid_mask.sum()}")
    print(f"标签覆盖率: {valid_mask.sum() / len(region_labels) * 100:.1f}%")
    print(f"分类准确率: {accuracy:.2f}%")
    print(f"宏平均F1: {f1:.2f}%")

    # 混淆矩阵简化版
    print(f"\n各类别预测分布:")
    for i in range(6):
        mask = true_labels == i
        if mask.sum() > 0:
            pred_dist = np.bincount(pred_valid[mask], minlength=6)
            print(f"  {region_names[i]}: 真实{mask.sum()}个, 预测正确{pred_dist[i]}个 ({pred_dist[i]/mask.sum()*100:.1f}%)")

    return {
        "test": "region_classification",
        "valid_samples": valid_mask.sum(),
        "coverage": valid_mask.sum() / len(region_labels) * 100,
        "accuracy": accuracy,
        "f1_score": f1,
    }


@torch.no_grad()
def test_similarity_search(model, data: Dict, device: torch.device, k: int = 5) -> Dict:
    """
    测试4：相似POI检索

    给定一个POI，找回语义最相似的K个POI
    验证：同类POI是否聚集在一起
    """
    print("\n" + "="*60)
    print("测试4：相似POI检索能力")
    print("="*60)

    region_labels = data["region_labels"]
    region_names = ["居住类", "商业类", "工业类", "教育类", "公共类", "自然类"]

    # 获取所有embedding
    point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
    line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
    polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
    direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

    embeddings, _, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
    embeddings = embeddings.cpu().numpy()

    # 只测试有标签的样本
    valid_mask = region_labels < 6
    valid_embeddings = embeddings[valid_mask]
    valid_labels = region_labels[valid_mask]

    if len(valid_embeddings) < k + 1:
        print("样本不足，跳过此测试")
        return {"test": "similarity_search", "status": "skipped"}

    # 计算K近邻
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(valid_embeddings)

    # 随机选择测试点
    n_tests = min(50, len(valid_embeddings))
    test_indices = np.random.choice(len(valid_embeddings), n_tests, replace=False)

    same_class_ratios = []

    for idx in test_indices:
        true_label = valid_labels[idx]
        _, neighbors = nbrs.kneighbors([valid_embeddings[idx]])
        neighbor_labels = valid_labels[neighbors[0][1:]]  # 排除自身

        same_class = (neighbor_labels == true_label).sum()
        same_class_ratios.append(same_class / k)

    avg_ratio = np.mean(same_class_ratios) * 100

    print(f"测试样本数: {n_tests}")
    print(f"K近邻数: {k}")
    print(f"同类召回率: {avg_ratio:.2f}%")
    print(f"解释: 在最相似的{k}个POI中，平均{avg_ratio:.1f}%与查询POI属于同一功能区")

    # 按类别分析
    print(f"\n各类别同类召回率:")
    for class_id in range(6):
        class_mask = valid_labels[test_indices] == class_id
        if class_mask.sum() > 0:
            class_ratio = np.mean([same_class_ratios[i] for i, m in enumerate(class_mask) if m]) * 100
            print(f"  {region_names[class_id]}: {class_ratio:.1f}%")

    return {
        "test": "similarity_search",
        "samples": n_tests,
        "k": k,
        "same_class_recall": avg_ratio,
    }


@torch.no_grad()
def test_embedding_quality(model, data: Dict, device: torch.device) -> Dict:
    """
    测试5：Embedding质量分析

    分析embedding空间的特性
    """
    print("\n" + "="*60)
    print("测试5：Embedding质量分析")
    print("="*60)

    coords = data["coords"]
    region_labels = data["region_labels"]

    # 获取所有embedding
    point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
    line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
    polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
    direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

    embeddings, _, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
    embeddings = embeddings.cpu().numpy()

    # 1. Pearson相关系数
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import pairwise_distances

    sample_size = min(500, len(embeddings))
    indices = np.random.choice(len(embeddings), sample_size, replace=False)

    emb_dist = pairwise_distances(embeddings[indices], metric='euclidean').flatten()
    coord_dist = pairwise_distances(coords[indices], metric='euclidean').flatten()

    pearson, _ = pearsonr(emb_dist, coord_dist)
    spearman, _ = spearmanr(emb_dist, coord_dist)

    # 2. Silhouette分数（功能区聚类质量）
    from sklearn.metrics import silhouette_score
    valid_mask = region_labels < 6
    if valid_mask.sum() > 100:
        sil_sample = np.random.choice(valid_mask.sum(), min(1000, valid_mask.sum()), replace=False)
        silhouette = silhouette_score(
            embeddings[valid_mask][sil_sample],
            region_labels[valid_mask][sil_sample]
        )
    else:
        silhouette = 0

    # 3. Embedding分布
    emb_norm = np.linalg.norm(embeddings, axis=1)

    print(f"Pearson相关: {pearson:.4f}")
    print(f"Spearman相关: {spearman:.4f}")
    print(f"Silhouette分数: {silhouette:.4f}")
    print(f"Embedding平均范数: {emb_norm.mean():.4f}")
    print(f"Embedding范数标准差: {emb_norm.std():.4f}")

    return {
        "test": "embedding_quality",
        "pearson": pearson,
        "spearman": spearman,
        "silhouette": silhouette,
        "emb_norm_mean": emb_norm.mean(),
        "emb_norm_std": emb_norm.std(),
    }


def main():
    print("="*60)
    print("L3 空间编码器实测")
    print("="*60)

    # 加载模型
    model, device = load_model()
    print(f"Device: {device}")

    # 加载数据
    data = load_sample_data(n_samples=5000)
    print(f"Loaded {len(data['coords'])} samples")

    # 运行所有测试
    results = []

    results.append(test_spatial_query(model, data, device, k=10))
    results.append(test_direction_recognition(model, data, device))
    results.append(test_region_classification(model, data, device))
    results.append(test_similarity_search(model, data, device, k=5))
    results.append(test_embedding_quality(model, data, device))

    # 汇总结果
    print("\n" + "="*60)
    print("实测结果汇总")
    print("="*60)

    print(f"\n{'测试项':<25} {'核心指标':<20} {'结果':<15}")
    print("-" * 60)

    for r in results:
        if r["test"] == "spatial_query":
            print(f"{'空间查询':<25} {'K近邻重叠率':<20} {r['overlap_rate']:.2f}%")
        elif r["test"] == "direction_recognition":
            print(f"{'方向识别':<25} {'准确率':<20} {r['accuracy']:.2f}%")
        elif r["test"] == "region_classification":
            print(f"{'功能区分类':<25} {'F1分数':<20} {r['f1_score']:.2f}%")
        elif r["test"] == "similarity_search":
            print(f"{'相似检索':<25} {'同类召回率':<20} {r['same_class_recall']:.2f}%")
        elif r["test"] == "embedding_quality":
            print(f"{'Embedding质量':<25} {'Pearson/Silhouette':<20} {r['pearson']:.4f}/{r['silhouette']:.4f}")

    print("\n" + "="*60)
    print("L3 能力评估完成")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
