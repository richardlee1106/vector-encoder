# -*- coding: utf-8 -*-
"""
P1E: 邻域特征融合 - 离线预计算

目标：扩展 MLP 视野，让模型看到空间上下文
输入：[72] → 输出：[72 + 40 = 112]

新增 40 维邻域特征：
1. 邻居 region_label 分布 (6 维) - 最重要！周围标签的分布
2. 邻居 POI 类别分布 (6 维) - 6大类比例
3. 邻居平均 POI 密度 (1 维)
4. 邻居平均路网密度 (1 维)
5. 自身 vs 邻居差异 (6 维) - 反映"与周围的差异"
6. 邻居数量和平均距离 (2 维)
7. 保留 padding (18 维)

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple
import time


def compute_neighbor_features(
    coords: np.ndarray,
    point_features: np.ndarray,
    line_features: np.ndarray,
    region_labels: np.ndarray,
    k_neighbors: int = 6,
) -> np.ndarray:
    """
    计算每个 cell 的邻域聚合特征

    Args:
        coords: 坐标 [N, 2] (lng, lat)
        point_features: 点特征 [N, 32]
        line_features: 线特征 [N, 16]
        region_labels: 功能区标签 [N], 值范围 0-6 (6=未知)
        k_neighbors: 物理近邻数量 (默认6个，对应H3一圈邻居)

    Returns:
        neighbor_feats: 邻域特征 [N, 40]
    """
    N = len(coords)
    print(f"P1E: Computing neighbor features for {N} cells (K={k_neighbors})")

    start_time = time.time()

    # 使用 KNN 找物理近邻
    print("  Finding K nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 排除自身（第一个是自身）
    neighbor_indices = indices[:, 1:]  # [N, K]
    neighbor_distances = distances[:, 1:]  # [N, K]

    # 初始化邻域特征
    neighbor_feats = np.zeros((N, 40), dtype=np.float32)

    # 1. 邻居 region_label 分布 (6 维)
    print("  Computing neighbor region label distribution...")
    for i in range(N):
        neighbor_ids = neighbor_indices[i]
        neighbor_labels = region_labels[neighbor_ids]
        # 只统计已知标签 (<6)
        valid_mask = neighbor_labels < 6
        if valid_mask.sum() > 0:
            valid_labels = neighbor_labels[valid_mask]
            for label in range(6):
                neighbor_feats[i, label] = (valid_labels == label).sum() / valid_mask.sum()

    # 2. 邻居 POI 类别分布 (6 维)
    # point_features 的第 3-18 维是 16 个 POI 类别的分布
    # 我们将其合并为 6 大类：餐饮/购物、生活服务、医疗/住宿、风景/商务、科教/交通、其他
    print("  Computing neighbor POI category distribution...")
    poi_category_features = point_features[:, 3:19]  # [N, 16] POI 类别分布

    # 合并为 6 大类
    # 餐饮服务(0) + 购物服务(1) → 商业类
    # 生活服务(2) + 体育休闲(3) → 生活类
    # 医疗保健(4) + 住宿服务(5) → 服务类
    # 风景名胜(6) + 商务住宅(7) → 休闲类
    # 政府机构(8) + 科教文化(9) + 交通设施(10) → 公共类
    # 其他(11-15) → 其他类
    poi_6cat = np.zeros((N, 6), dtype=np.float32)
    poi_6cat[:, 0] = poi_category_features[:, 0] + poi_category_features[:, 1]  # 商业
    poi_6cat[:, 1] = poi_category_features[:, 2] + poi_category_features[:, 3]  # 生活
    poi_6cat[:, 2] = poi_category_features[:, 4] + poi_category_features[:, 5]  # 服务
    poi_6cat[:, 3] = poi_category_features[:, 6] + poi_category_features[:, 7]  # 休闲
    poi_6cat[:, 4] = poi_category_features[:, 8] + poi_category_features[:, 9] + poi_category_features[:, 10]  # 公共
    poi_6cat[:, 5] = poi_category_features[:, 11:16].sum(axis=1)  # 其他

    for i in range(N):
        neighbor_ids = neighbor_indices[i]
        neighbor_poi = poi_6cat[neighbor_ids]  # [K, 6]
        neighbor_feats[i, 6:12] = neighbor_poi.mean(axis=0)

    # 3. 邻居平均 POI 密度 (1 维)
    # point_features[:, 2] 是 log(POI_count) / 10
    print("  Computing neighbor POI density...")
    poi_density = point_features[:, 2]  # [N]
    for i in range(N):
        neighbor_ids = neighbor_indices[i]
        neighbor_feats[i, 12] = poi_density[neighbor_ids].mean()

    # 4. 邻居平均路网密度 (1 维)
    # line_features[:, 2] 是 log(road_count) / 5
    print("  Computing neighbor road density...")
    road_density = line_features[:, 2]  # [N]
    for i in range(N):
        neighbor_ids = neighbor_indices[i]
        neighbor_feats[i, 13] = road_density[neighbor_ids].mean()

    # 5. 自身 vs 邻居差异 (6 维)
    # 使用合并后的 6 大类 POI 分布
    print("  Computing self vs neighbor difference...")
    for i in range(N):
        neighbor_ids = neighbor_indices[i]
        self_poi = poi_6cat[i]  # [6]
        neighbor_poi_mean = poi_6cat[neighbor_ids].mean(axis=0)  # [6]
        neighbor_feats[i, 14:20] = self_poi - neighbor_poi_mean

    # 6. 邻居数量和平均距离 (2 维)
    print("  Computing neighbor count and distance...")
    for i in range(N):
        neighbor_feats[i, 20] = k_neighbors  # 邻居数量（固定为 K）
        neighbor_feats[i, 21] = neighbor_distances[i].mean()  # 平均距离

    # 7. 保留 padding (18 维, 22-39)
    # 预留位置，暂填充 0

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # 打印统计信息
    print(f"\n  Neighbor feature statistics:")
    print(f"    Region label dist (dim 0-5): mean={neighbor_feats[:, :6].mean():.3f}")
    print(f"    POI category dist (dim 6-11): mean={neighbor_feats[:, 6:12].mean():.3f}")
    print(f"    POI density (dim 12): mean={neighbor_feats[:, 12].mean():.3f}")
    print(f"    Road density (dim 13): mean={neighbor_feats[:, 13].mean():.3f}")
    print(f"    Self-neighbor diff (dim 14-19): mean={neighbor_feats[:, 14:20].mean():.3f}")
    print(f"    Avg distance (dim 21): mean={neighbor_feats[:, 21].mean():.6f}")

    return neighbor_feats


def generate_and_save_neighbor_features(
    sample_ratio: float = 1.0,
    output_path: str = None,
) -> np.ndarray:
    """
    加载数据并生成邻域特征，保存到文件

    Args:
        sample_ratio: 数据采样比例
        output_path: 输出文件路径

    Returns:
        neighbor_feats: 邻域特征 [N, 40]
    """
    from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training

    print(f"\n{'='*60}")
    print(f"P1E: Generating neighbor features (sample_ratio={sample_ratio})")
    print(f"{'='*60}\n")

    # 加载基础数据
    config = DEFAULT_PRO_CONFIG
    data = load_dataset_for_training(config=config, sample_ratio=sample_ratio)

    coords = data["coords"]
    point_features = data["point_features"]
    line_features = data["line_features"]
    region_labels = data["region_labels"]

    # 计算邻域特征
    neighbor_feats = compute_neighbor_features(
        coords=coords,
        point_features=point_features,
        line_features=line_features,
        region_labels=region_labels,
        k_neighbors=6,
    )

    # 保存
    if output_path is None:
        output_path = Path(__file__).parent / "p1e_neighbor_feat.npy"
    else:
        output_path = Path(output_path)

    np.save(output_path, neighbor_feats)
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {neighbor_feats.shape}")

    return neighbor_feats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P1E: Generate neighbor features")
    parser.add_argument("--sample", type=float, default=1.0, help="Sample ratio (default: 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    generate_and_save_neighbor_features(
        sample_ratio=args.sample,
        output_path=args.output,
    )
