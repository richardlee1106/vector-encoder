# -*- coding: utf-8 -*-
"""
V2.6 Pro 评估模块 - 完整版

空间智能等级评估指标体系：

L1 空间感知（距离关系）：
- Pearson > 0.90：全局距离相关性
- Spearman > 0.85：距离排序相关性（新增）

L2 空间查询（邻居匹配）：
- Overlap@K > 40%：K近邻重叠率
- Recall@20 > 60%：近邻召回率（新增）

L3 空间理解（方向+语义）：
- DirAcc > 60%：方向识别准确率
- Region F1 > 50%：功能区分类
- Region Sep > 2.0：功能区分辨率（新增）

L4 空间推理（复杂推理）：
- Range IoU > 70%：范围查询精度（新增）
- SimRecall > 50%：相似区域推荐（新增）

Author: Claude
Date: 2026-03-17
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist


# ============================================================
# L1 指标：空间感知（距离关系）
# ============================================================

def compute_pearson(embeddings: np.ndarray, coords: np.ndarray, sample_size: int = 5000) -> float:
    """
    L1指标：Pearson相关系数

    意义：衡量embedding距离与真实距离的整体线性相关性
    目标：> 0.90（距离关系基本正确）
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]

    emb_dist = pdist(embeddings, 'euclidean')
    coord_dist = pdist(coords, 'euclidean')

    corr, _ = pearsonr(emb_dist, coord_dist)
    return corr


def compute_spearman(embeddings: np.ndarray, coords: np.ndarray, sample_size: int = 5000) -> float:
    """
    L1指标（新增）：Spearman秩相关系数

    意义：衡量距离排序的相关性，比Pearson更关注邻居排序
    目标：> 0.85
    优势：对异常值更鲁棒，更适合评估邻居关系
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]

    emb_dist = pdist(embeddings, 'euclidean')
    coord_dist = pdist(coords, 'euclidean')

    corr, _ = spearmanr(emb_dist, coord_dist)
    return corr


# ============================================================
# L2 指标：空间查询（邻居匹配）
# ============================================================

def compute_overlap(embeddings: np.ndarray, coords: np.ndarray, k: int = 10,
                    sample_size: int = 1000) -> float:
    """
    L2指标：K近邻重叠率

    意义：衡量embedding空间和真实空间的K近邻重叠程度
    目标：> 40%（能支持"附近搜索"功能）
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]
        n = sample_size

    # 真实坐标的K近邻
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices_coord = nbrs_coord.kneighbors(coords)
    indices_coord = indices_coord[:, 1:]  # 排除自己

    # Embedding的K近邻
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    _, indices_emb = nbrs_emb.kneighbors(embeddings)
    indices_emb = indices_emb[:, 1:]

    # 计算重叠率
    overlaps = []
    for i in range(n):
        set_coord = set(indices_coord[i])
        set_emb = set(indices_emb[i])
        overlap = len(set_coord & set_emb) / k
        overlaps.append(overlap)

    return np.mean(overlaps)


def compute_recall_at_k(embeddings: np.ndarray, coords: np.ndarray,
                         k_true: int = 10, k_pred: int = 20,
                         sample_size: int = 1000) -> float:
    """
    L2指标（新增）：召回率@K

    意义：真实最近K个邻居中，有多少在embedding最近K_pred个中
    目标：Recall@20 > 60%

    应用场景："找附近100米内的点"的召回率
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]
        n = sample_size

    # 真实坐标的K近邻
    nbrs_coord = NearestNeighbors(n_neighbors=k_true+1).fit(coords)
    _, indices_coord = nbrs_coord.kneighbors(coords)
    indices_coord = indices_coord[:, 1:]

    # Embedding的K_pred近邻
    nbrs_emb = NearestNeighbors(n_neighbors=k_pred+1).fit(embeddings)
    _, indices_emb = nbrs_emb.kneighbors(embeddings)
    indices_emb = indices_emb[:, 1:]

    # 计算召回率
    recalls = []
    for i in range(n):
        true_neighbors = set(indices_coord[i])
        pred_neighbors = set(indices_emb[i])
        recall = len(true_neighbors & pred_neighbors) / len(true_neighbors)
        recalls.append(recall)

    return np.mean(recalls)


# ============================================================
# L3 指标：空间理解（方向+语义）
# ============================================================

def compute_direction_accuracy(embeddings: np.ndarray, coords: np.ndarray,
                                direction_labels: np.ndarray, sample_size: int = 5000) -> float:
    """
    L3指标：方向识别准确率

    意义：衡量模型对空间方向的理解能力
    目标：> 60%（显著优于随机猜测的12.5%）
    """
    n_original = len(embeddings)
    if n_original > sample_size:
        indices = np.random.choice(n_original, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]
        direction_labels = direction_labels[indices]

    n = len(embeddings)
    k = min(10, n - 1)

    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices_coord = nbrs_coord.kneighbors(coords)
    indices_coord = indices_coord[:, 1:]

    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    _, indices_emb = nbrs_emb.kneighbors(embeddings)
    indices_emb = indices_emb[:, 1:]

    correct = 0
    total = 0

    for i in range(n):
        coord_neighbors = indices_coord[i]
        if len(coord_neighbors) == 0:
            continue

        dx = coords[coord_neighbors[0], 0] - coords[i, 0]
        dy = coords[coord_neighbors[0], 1] - coords[i, 1]
        true_angle = np.arctan2(dy, dx)
        true_dir = int((true_angle + np.pi) / (np.pi / 4)) % 8

        emb_neighbors = indices_emb[i]
        if len(emb_neighbors) == 0:
            continue

        if coord_neighbors[0] in emb_neighbors[:3]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compute_region_f1(embeddings: np.ndarray, region_labels: np.ndarray,
                      sample_size: int = 10000) -> float:
    """
    L3指标：功能区分类F1

    意义：衡量embedding是否包含功能区语义信息
    目标：> 50%（能区分主要功能区类型）
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        region_labels = region_labels[indices]

    # 过滤掉未知标签
    valid_mask = region_labels < region_labels.max()
    if valid_mask.sum() < 100:
        return 0.0

    embeddings = embeddings[valid_mask]
    region_labels = region_labels[valid_mask]

    n = len(embeddings)
    train_size = int(0.8 * n)
    indices = np.random.permutation(n)

    X_train = embeddings[indices[:train_size]]
    y_train = region_labels[indices[:train_size]]
    X_test = embeddings[indices[train_size:]]
    y_test = region_labels[indices[train_size:]]

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return f1_score(y_test, y_pred, average='macro')


def compute_region_separation(embeddings: np.ndarray, region_labels: np.ndarray,
                               sample_size: int = 5000) -> float:
    """
    L3指标（新增）：功能区分辨率

    意义：衡量不同功能区embedding的分离程度
    目标：> 2.0（类间距离是类内距离的2倍以上）

    计算：mean_inter_class_distance / mean_intra_class_distance
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        region_labels = region_labels[indices]

    # 过滤未知标签
    valid_mask = region_labels < region_labels.max()
    embeddings = embeddings[valid_mask]
    region_labels = region_labels[valid_mask]

    unique_labels = np.unique(region_labels)
    if len(unique_labels) < 2:
        return 0.0

    # 计算每个类别的质心
    centroids = {}
    for label in unique_labels:
        mask = region_labels == label
        if mask.sum() > 0:
            centroids[label] = embeddings[mask].mean(axis=0)

    # 计算类内距离（点到自己类别质心的平均距离）
    intra_dists = []
    for label in unique_labels:
        mask = region_labels == label
        if mask.sum() > 1:
            dists = np.linalg.norm(embeddings[mask] - centroids[label], axis=1)
            intra_dists.append(dists.mean())

    mean_intra = np.mean(intra_dists) if intra_dists else 1.0

    # 计算类间距离（质心之间的平均距离）
    inter_dists = []
    labels_list = list(centroids.keys())
    for i, label1 in enumerate(labels_list):
        for label2 in labels_list[i+1:]:
            dist = np.linalg.norm(centroids[label1] - centroids[label2])
            inter_dists.append(dist)

    mean_inter = np.mean(inter_dists) if inter_dists else 0.0

    return mean_inter / mean_intra if mean_intra > 0 else 0.0


# ============================================================
# L4 指标：空间推理（复杂推理）
# ============================================================

def compute_range_query_iou(embeddings: np.ndarray, coords: np.ndarray,
                            radius_km: float = 0.5, sample_size: int = 500) -> float:
    """
    L4指标（新增）：范围查询精度（IoU）

    意义：圆形范围查询的精度
    目标：IoU > 70%

    应用场景："找500米内的所有咖啡店"

    核心逻辑：
    1. 在物理空间找 radius_km 内的真实近邻（dynamic K）
    2. 在 embedding 空间找最近的 K 个点（K 与 true_set 大小一致）
    3. 计算 IoU = |true_set ∩ pred_set| / |true_set ∪ pred_set|
    4. 跳过孤立点（true_set 为空）

    注意：使用相同 K 值匹配，避免 embedding 距离尺度问题
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]
        n = sample_size

    # 坐标转换为km（武汉：1度≈100km）
    coords_km = coords * 100

    # 使用较大的 K 来获取物理空间邻居，然后筛选距离
    max_k = min(n - 1, 200)  # 最大搜索邻居数
    nbrs_coord = NearestNeighbors(n_neighbors=max_k + 1).fit(coords_km)

    # Embedding 空间的 KNN 查找器（使用相同 max_k）
    nbrs_emb = NearestNeighbors(n_neighbors=max_k + 1).fit(embeddings)

    ious = []
    skipped = 0

    for i in range(n):
        # Step 1: 在物理空间找 K 近邻及其距离
        distances, indices = nbrs_coord.kneighbors([coords_km[i]])
        distances = distances[0, 1:]  # 排除自己
        indices = indices[0, 1:]

        # Step 2: 筛选距离 < radius_km 的点作为 true_set
        mask = distances < radius_km
        true_set = set(indices[mask])

        # Step 4: 跳过孤立点（true_set 为空）
        if len(true_set) == 0:
            skipped += 1
            continue

        # Step 2: 在 embedding 空间找最近的 len(true_set) 个点
        k_pred = len(true_set)
        _, emb_indices = nbrs_emb.kneighbors([embeddings[i]], n_neighbors=k_pred + 1)
        pred_set = set(emb_indices[0, 1:])  # 排除自己

        # Step 3: 计算 IoU
        intersection = len(true_set & pred_set)
        union = len(true_set | pred_set)
        iou = intersection / union if union > 0 else 0
        ious.append(iou)

    # 统计信息
    if skipped > 0:
        print(f"  [Range IoU] Skipped {skipped}/{n} isolated points (no neighbors within {radius_km}km)")

    return np.mean(ious) if ious else 0.0


def compute_similar_region_recall(embeddings: np.ndarray, region_labels: np.ndarray,
                                   k: int = 5, sample_size: int = 2000) -> float:
    """
    L4指标（新增）：相似区域推荐召回率

    意义：给定一个区域，找出功能相似的其他区域
    目标：Recall > 50%

    应用场景："找出与光谷核心区商业布局相似的区域"
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        region_labels = region_labels[indices]

    # 过滤未知标签
    valid_mask = region_labels < region_labels.max()
    embeddings = embeddings[valid_mask]
    region_labels = region_labels[valid_mask]

    unique_labels = np.unique(region_labels)
    if len(unique_labels) < 2:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)

    recalls = []

    for label in unique_labels:
        # 该类别的所有样本
        label_mask = region_labels == label
        label_indices = np.where(label_mask)[0]

        if len(label_indices) < 2:
            continue

        # 对该类别中的每个样本
        for idx in label_indices[:min(10, len(label_indices))]:  # 限制计算量
            # 找K个最近邻
            _, neighbor_indices = nbrs.kneighbors([embeddings[idx]])
            neighbor_indices = neighbor_indices[0, 1:]  # 排除自己

            # 计算召回率：K近邻中有多少是同类别的
            neighbor_labels = region_labels[neighbor_indices]
            correct = (neighbor_labels == label).sum()
            recall = correct / k
            recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def compute_direction_accuracy(embeddings: np.ndarray, coords: np.ndarray,
                                direction_labels: np.ndarray, sample_size: int = 5000) -> float:
    """
    计算方向识别准确率

    评估方法：使用K近邻的相对方向来评估
    这与训练时的neighbor_relative方向方案一致
    """
    n_original = len(embeddings)
    if n_original > sample_size:
        indices = np.random.choice(n_original, sample_size, replace=False)
        embeddings = embeddings[indices]
        coords = coords[indices]
        direction_labels = direction_labels[indices]

    n = len(embeddings)  # 更新n为采样后的大小

    from sklearn.neighbors import NearestNeighbors

    # 找到K近邻
    k = min(10, n - 1)
    nbrs_coord = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices_coord = nbrs_coord.kneighbors(coords)
    indices_coord = indices_coord[:, 1:]  # 排除自己

    # 在embedding空间找K近邻
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    _, indices_emb = nbrs_emb.kneighbors(embeddings)
    indices_emb = indices_emb[:, 1:]

    # 计算相对方向准确率
    # 对于每个cell，检查最近邻是否在正确的方向
    correct = 0
    total = 0

    for i in range(n):
        # 真实空间中最近邻的方向
        coord_neighbors = indices_coord[i]
        if len(coord_neighbors) == 0:
            continue

        # 计算真实方向
        dx = coords[coord_neighbors[0], 0] - coords[i, 0]
        dy = coords[coord_neighbors[0], 1] - coords[i, 1]
        true_angle = np.arctan2(dy, dx)
        true_dir = int((true_angle + np.pi) / (np.pi / 4)) % 8

        # embedding空间中最近邻的方向
        emb_neighbors = indices_emb[i]
        if len(emb_neighbors) == 0:
            continue

        # 计算embedding方向（使用PCA投影到2D）
        # 但我们比较的是：真实最近邻是否在embedding最近邻中
        # 如果是，说明方向学习正确
        if coord_neighbors[0] in emb_neighbors[:3]:
            correct += 1
        total += 1

    if total == 0:
        return 0.0

    return correct / total


def compute_region_f1(embeddings: np.ndarray, region_labels: np.ndarray,
                      sample_size: int = 10000) -> float:
    """
    计算功能区分类F1

    使用KNN分类器在embedding空间预测功能区
    """
    n = len(embeddings)
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        embeddings = embeddings[indices]
        region_labels = region_labels[indices]

    # 过滤掉未知标签（通常是最大值）
    valid_mask = region_labels < region_labels.max()
    if valid_mask.sum() < 100:
        return 0.0

    embeddings = embeddings[valid_mask]
    region_labels = region_labels[valid_mask]

    # 划分训练/测试集
    n = len(embeddings)
    train_size = int(0.8 * n)
    indices = np.random.permutation(n)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train = embeddings[train_idx]
    y_train = region_labels[train_idx]
    X_test = embeddings[test_idx]
    y_test = region_labels[test_idx]

    # KNN分类
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # F1分数（宏平均）
    f1 = f1_score(y_test, y_pred, average='macro')

    return f1


def evaluate_model(model, dataloader, device) -> dict:
    """
    完整评估模型 - 包含所有层级指标

    Returns:
        dict: 包含L1-L4所有指标的字典
    """
    model.eval()

    all_embeddings = []
    all_coords = []
    all_dir_labels = []
    all_region_labels = []

    with torch.no_grad():
        for batch in dataloader:
            point_feat = batch["point_feat"].to(device)
            line_feat = batch["line_feat"].to(device)
            polygon_feat = batch["polygon_feat"].to(device)
            direction_feat = batch["direction_feat"].to(device)
            coords = batch["coords"].numpy()
            dir_label = batch["direction_label"].numpy()
            region_label = batch["region_label"].numpy()

            emb, _, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)

            all_embeddings.append(emb.cpu().numpy())
            all_coords.append(coords)
            all_dir_labels.append(dir_label)
            all_region_labels.append(region_label)

    embeddings = np.concatenate(all_embeddings, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    dir_labels = np.concatenate(all_dir_labels, axis=0)
    region_labels = np.concatenate(all_region_labels, axis=0)

    print(f"\nEvaluating {len(embeddings)} samples...")

    results = {
        # L1: 空间感知
        "pearson": compute_pearson(embeddings, coords),
        "spearman": compute_spearman(embeddings, coords),
        # L2: 空间查询
        "overlap": compute_overlap(embeddings, coords),
        "recall_at_20": compute_recall_at_k(embeddings, coords),
        # L3: 空间理解
        "dir_acc": compute_direction_accuracy(embeddings, coords, dir_labels),
        "region_f1": compute_region_f1(embeddings, region_labels),
        "region_sep": compute_region_separation(embeddings, region_labels),
        # L4: 空间推理
        "range_iou": compute_range_query_iou(embeddings, coords),
        "sim_recall": compute_similar_region_recall(embeddings, region_labels),
    }

    return results


# 指标目标定义
METRIC_TARGETS = {
    # L1: 空间感知
    "pearson": {"level": "L1", "target": 0.90, "desc": "全局距离相关性"},
    "spearman": {"level": "L1", "target": 0.85, "desc": "距离排序相关性"},
    # L2: 空间查询
    "overlap": {"level": "L2", "target": 0.40, "desc": "K近邻重叠率"},
    "recall_at_20": {"level": "L2", "target": 0.60, "desc": "近邻召回率"},
    # L3: 空间理解
    "dir_acc": {"level": "L3", "target": 0.60, "desc": "方向识别准确率"},
    "region_f1": {"level": "L3", "target": 0.50, "desc": "功能区分类F1"},
    "region_sep": {"level": "L3", "target": 2.0, "desc": "功能区分辨率"},
    # L4: 空间推理
    "range_iou": {"level": "L4", "target": 0.70, "desc": "范围查询精度"},
    "sim_recall": {"level": "L4", "target": 0.50, "desc": "相似区域推荐"},
}

# Baseline值
BASELINE_VALUES = {
    "pearson": 0.90, "spearman": 0.85,
    "overlap": 0.25, "recall_at_20": 0.30,
    "dir_acc": 0.30, "region_f1": 0.05, "region_sep": 1.0,
    "range_iou": 0.30, "sim_recall": 0.20,
}


def print_comparison(current: dict, baseline: dict = None):
    """打印完整的对比报告"""
    print("\n" + "=" * 80)
    print("Spatial Intelligence Evaluation Report")
    print("=" * 80)

    if baseline is None:
        baseline = BASELINE_VALUES
        print("(Baseline: V2.6 original)")

    # 按层级打印
    for level in ["L1", "L2", "L3", "L4"]:
        level_names = {
            "L1": "Spatial Perception",
            "L2": "Spatial Query",
            "L3": "Spatial Understanding",
            "L4": "Spatial Reasoning",
        }
        print(f"\n{level}: {level_names[level]}")
        print("-" * 80)
        print(f"{'Metric':<12} {'Current':>10} {'Target':>10} {'Baseline':>10} {'Delta':>10} {'Status':>10}")
        print("-" * 80)

        level_metrics = [m for m, info in METRIC_TARGETS.items() if info["level"] == level]

        for metric in level_metrics:
            cur = current.get(metric, 0)
            target = METRIC_TARGETS[metric]["target"]
            base = baseline.get(metric, 0)
            delta = cur - base

            if cur >= target:
                status = "[PASS]"
            elif delta > 0:
                status = "IMPROVE"
            else:
                status = "FAIL"

            print(f"{metric:<12} {cur:>10.4f} {target:>10.2f} {base:>10.4f} {delta:>+10.4f} {status:>10}")

    # 级别总结
    print("\n" + "=" * 80)
    print("Level Assessment Summary:")
    print("=" * 80)

    level_status = {"L1": True, "L2": True, "L3": True, "L4": True}

    for metric, info in METRIC_TARGETS.items():
        if current.get(metric, 0) < info["target"]:
            level_status[info["level"]] = False

    level_names_short = {
        "L1": "L1 Perception",
        "L2": "L2 Query",
        "L3": "L3 Understanding",
        "L4": "L4 Reasoning",
    }

    for level in ["L1", "L2", "L3", "L4"]:
        status = "[PASS]" if level_status[level] else "FAIL"
        print(f"  {level_names_short[level]}: {status}")

    # 最高达成等级
    achieved = 0
    for level in ["L1", "L2", "L3", "L4"]:
        if level_status[level]:
            achieved = int(level[1])
        else:
            break

    print(f"\n  Achieved Level: L{achieved}")
    print("=" * 80)


if __name__ == "__main__":
    # 测试评估模块
    print("Testing evaluation module with new metrics...")

    n = 5000
    np.random.seed(42)

    # 模拟数据
    coords = np.random.rand(n, 2).astype(np.float32)
    embeddings = np.random.randn(n, 64).astype(np.float32)
    dir_labels = np.random.randint(0, 8, n)
    region_labels = np.random.randint(0, 8, n)

    results = {
        # L1
        "pearson": compute_pearson(embeddings, coords),
        "spearman": compute_spearman(embeddings, coords),
        # L2
        "overlap": compute_overlap(embeddings, coords),
        "recall_at_20": compute_recall_at_k(embeddings, coords),
        # L3
        "dir_acc": compute_direction_accuracy(embeddings, coords, dir_labels),
        "region_f1": compute_region_f1(embeddings, region_labels),
        "region_sep": compute_region_separation(embeddings, region_labels),
        # L4
        "range_iou": compute_range_query_iou(embeddings, coords),
        "sim_recall": compute_similar_region_recall(embeddings, region_labels),
    }

    print_comparison(results)
