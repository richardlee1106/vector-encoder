# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 训练脚本

核心改进：让模型真正学到"空间拓扑关系"

损失函数组合：
1. 坐标重构损失：保持绝对位置信息
2. 距离保持损失：原始空间距离 ≈ embedding空间距离
3. 邻居一致性损失：KNN邻居的embedding应该相似

目标：Silhouette > 0.35（接近理论上限0.45）
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import SpatialEncoderConfig
from models.encoder import (
    SpatialTopologyEncoder,
    DistancePreserveLoss,
    NeighborConsistencyLoss,
    sample_distance_pairs
)


def load_area_data(area_name: str, data_dir: str) -> Dict:
    """加载区域数据"""
    area_dir = Path(data_dir) / area_name
    with open(area_dir / "pois.geojson", 'r', encoding='utf-8') as f:
        pois = json.load(f)

    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features = [], []

    for f in pois['features']:
        props = f['properties']
        coords = f['geometry']['coordinates']
        poi_coords.append(coords)

        cat = props.get('category_big', 'unknown') or 'unknown'
        if cat not in category_map:
            category_map[cat] = len(category_map)

        lu = props.get('land_use_type', 'unknown') or 'unknown'
        if lu not in landuse_map:
            landuse_map[lu] = len(landuse_map)

        aoi_type = props.get('aoi_type', 'unknown') or 'unknown'
        if aoi_type not in aoi_type_map:
            aoi_type_map[aoi_type] = len(aoi_type_map)

        rc = props.get('nearest_road_class', 'unknown') or 'unknown'
        if rc not in road_class_map:
            road_class_map[rc] = len(road_class_map)

        density = float(props.get('poi_density_500m', 0) or 0)
        entropy = float(props.get('category_entropy', 0) or 0)
        road_dist = float(props.get('nearest_road_dist_m', 0) or 0)

        poi_features.append([
            category_map[cat], landuse_map[lu], aoi_type_map[aoi_type],
            road_class_map[rc], density, entropy, road_dist
        ])

    coords = np.array(poi_coords, dtype=np.float32)

    # 预计算KNN邻居
    knn_k = 10
    adj = kneighbors_graph(coords, n_neighbors=knn_k, mode='connectivity', include_self=False)
    knn_neighbors = [adj[i].nonzero()[1] for i in range(len(coords))]

    return {
        'coords': coords,
        'features': np.array(poi_features, dtype=np.float32),
        'knn_neighbors': knn_neighbors,
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


def run_single_area(area_name: str, config: SpatialEncoderConfig, data_dir: str) -> Dict:
    """
    运行单区域实验

    Args:
        area_name: 区域名称
        config: 配置对象
        data_dir: 数据目录

    Returns:
        实验结果字典
    """
    print("=" * 70)
    print(f"V2.3 空间拓扑编码器: {area_name}")
    print("=" * 70)

    device = torch.device(config.device)
    print(f"设备: {device}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    data = load_area_data(area_name, data_dir)

    coords = data['coords']
    features = data['features']
    knn_neighbors = data['knn_neighbors']
    metadata = data['metadata']

    print(f"  POI数量: {metadata['num_pois']}")

    # 2. 生成标签
    print("\n[2] 生成空间聚类标签...")
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    sil_upper_bound = silhouette_score(coords, labels)
    print(f"  理论上限（原始坐标）: {sil_upper_bound:.4f}")

    # 3. 标准化坐标
    coords_mean = coords.mean(axis=0)
    coords_std = coords.std(axis=0)
    coords_norm = (coords - coords_mean) / coords_std

    # 4. 创建模型
    print("\n[3] 创建模型...")
    model = SpatialTopologyEncoder(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 5. 损失函数
    distance_loss_fn = DistancePreserveLoss()
    neighbor_loss_fn = NeighborConsistencyLoss()

    # 6. 训练
    print("\n[4] 开始训练...")
    print(f"  损失权重: coord_recon={config.coord_recon_weight}, "
          f"distance_preserve={config.distance_preserve_weight}, "
          f"neighbor_consistency={config.neighbor_consistency_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    best_sil = -1.0
    best_epoch = 0
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        z, coord_recon = model(features_t, coords_t)

        # 1. 坐标重构损失
        loss_recon = F.mse_loss(coord_recon, coords_norm_t)

        # 2. 距离保持损失
        pair_indices, spatial_dists = sample_distance_pairs(
            coords, config.num_distance_pairs, device
        )
        loss_distance = distance_loss_fn(z, pair_indices, spatial_dists)

        # 3. 邻居一致性损失
        loss_neighbor = neighbor_loss_fn(z, knn_neighbors)

        # 总损失
        loss = (
            config.coord_recon_weight * loss_recon +
            config.distance_preserve_weight * loss_distance +
            config.neighbor_consistency_weight * loss_neighbor
        )

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 评估
        if epoch % 20 == 0 or epoch == config.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z = model.encode(features_t, coords_t)
                z_np = z.cpu().numpy()
                sil = silhouette_score(z_np, labels)

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'loss_recon': loss_recon.item(),
                'loss_distance': loss_distance.item(),
                'loss_neighbor': loss_neighbor.item(),
                'silhouette': sil,
            })

            print(f"  Epoch {epoch:3d} | Loss={loss.item():.4f} "
                  f"(recon={loss_recon.item():.3f}, dist={loss_distance.item():.3f}, "
                  f"neighbor={loss_neighbor.item():.3f}) | Sil={sil:.4f} | Best={best_sil:.4f}")

    # 7. 最终结果
    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"最佳 Silhouette: {best_sil:.4f} (Epoch {best_epoch})")
    print(f"理论上限: {sil_upper_bound:.4f}")
    print(f"达成率: {best_sil / sil_upper_bound * 100:.1f}%")

    success = best_sil > 0.35
    if success:
        print("\n结论: V2.3 成功！模型学到了空间拓扑关系")
    else:
        print("\n结论: 效果有限，需要进一步优化")

    return {
        'area_name': area_name,
        'success': success,
        'silhouette': best_sil,
        'upper_bound': sil_upper_bound,
        'best_epoch': best_epoch,
        'num_pois': metadata['num_pois'],
        'num_params': total_params,
        'history': history,
    }


def run_multi_areas(areas: list, config: SpatialEncoderConfig, data_dir: str) -> Dict:
    """
    运行多区域实验

    Args:
        areas: 区域名称列表
        config: 配置对象
        data_dir: 数据目录

    Returns:
        汇总结果
    """
    all_results = {}

    for area in areas:
        print(f"\n{'#'*70}")
        print(f"# 区域: {area}")
        print(f"{'#'*70}")

        # 更新配置中的区域
        result = run_single_area(area, config, data_dir)
        all_results[area] = result

        # 如果效果太差，提前停止
        if result['silhouette'] < 0.1:
            print(f"\n警告: {area} 区域 Silhouette={result['silhouette']:.4f} 过低，停止后续实验")
            break

    # 汇总
    print("\n" + "=" * 70)
    print("多区域汇总结果")
    print("=" * 70)
    print(f"{'区域':<20} {'Silhouette':<12} {'理论上限':<12} {'达成率'}")
    print("-" * 60)

    for area, res in all_results.items():
        rate = res['silhouette'] / res['upper_bound'] * 100
        print(f"{area:<20} {res['silhouette']:.4f}       {res['upper_bound']:.4f}       {rate:.1f}%")

    if len(all_results) > 1:
        avg_sil = np.mean([r['silhouette'] for r in all_results.values()])
        avg_rate = np.mean([r['silhouette']/r['upper_bound'] for r in all_results.values()]) * 100
        print("-" * 60)
        print(f"{'平均':<20} {avg_sil:.4f}       -            {avg_rate:.1f}%")

        if avg_sil > 0.35:
            print("\n结论: V2.3 在多个区域均成功，具备普适性！")

    return all_results


def save_results(results: Dict, output_dir: str):
    """保存实验结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存JSON
    json_path = output_path / f"v23_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {json_path}")


if __name__ == "__main__":
    # 配置
    config = SpatialEncoderConfig(
        num_epochs=500,
        learning_rate=1e-3,
        weight_decay=1e-5,
        embed_dim=64,
        hidden_dim=128,
        dropout=0.1,
        coord_recon_weight=1.0,
        distance_preserve_weight=2.0,
        neighbor_consistency_weight=1.0,
        num_distance_pairs=5000,
        n_clusters=15,
    )

    data_dir = str(Path(__file__).resolve().parents[2] / "data" / "experiment_data")

    # 运行三个区域
    areas = ["guanggu_core", "wuda_area", "zhongjia_cun"]
    results = run_multi_areas(areas, config, data_dir)

    # 保存结果
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)
