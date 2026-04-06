# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 验证模块

验证模型是否真的学会了空间拓扑关系：
1. 距离保持：embedding距离 vs 真实距离的相关性
2. 空间相似性查询：找空间相似的POI
3. 区域相似性：哪些区域空间模式相似
4. 邻居预测：预测空间邻居
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(str(Path(__file__).parent))
from config import SpatialEncoderConfig
from models.encoder import SpatialTopologyEncoder, build_knn_neighbors


def load_area_data(area_name: str, data_dir: str):
    """加载数据"""
    area_dir = Path(data_dir) / area_name
    with open(area_dir / "pois.geojson", 'r', encoding='utf-8') as f:
        pois = json.load(f)

    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features, poi_names = [], [], []

    for f in pois['features']:
        props = f['properties']
        coords = f['geometry']['coordinates']
        poi_coords.append(coords)
        poi_names.append(props.get('name', 'Unknown'))

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

    return {
        'coords': np.array(poi_coords, dtype=np.float32),
        'features': np.array(poi_features, dtype=np.float32),
        'names': poi_names,
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


def validate_distance_preservation(coords, embeddings, sample_size=2000):
    """
    验证1：距离保持能力

    问题：embedding距离与真实距离是否相关？
    """
    N = len(coords)
    sample_size = min(sample_size, N * (N-1) // 2)

    # 随机采样点对
    idx_i = np.random.randint(0, N, sample_size)
    idx_j = np.random.randint(0, N, sample_size)

    # 真实距离
    real_dists = np.sqrt(((coords[idx_i] - coords[idx_j]) ** 2).sum(axis=1))

    # Embedding距离
    emb_i = embeddings[idx_i]
    emb_j = embeddings[idx_j]
    emb_dists = np.sqrt(((emb_i - emb_j) ** 2).sum(axis=1))

    # 相关性
    pearson_corr, _ = pearsonr(real_dists, emb_dists)
    spearman_corr, _ = spearmanr(real_dists, emb_dists)

    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'real_dists': real_dists,
        'emb_dists': emb_dists,
    }


def validate_spatial_query(coords, embeddings, poi_names, query_idx, top_k=10):
    """
    验证2：空间相似性查询

    问题：给定一个POI，能否找到空间上相似的其他POI？
    """
    query_coord = coords[query_idx]
    query_emb = embeddings[query_idx]

    # 真实最近邻
    nbrs_real = NearestNeighbors(n_neighbors=top_k+1).fit(coords)
    distances_real, indices_real = nbrs_real.kneighbors([query_coord])

    # Embedding最近邻
    nbrs_emb = NearestNeighbors(n_neighbors=top_k+1).fit(embeddings)
    distances_emb, indices_emb = nbrs_emb.kneighbors([query_emb])

    # 重叠率
    overlap = len(set(indices_real[0]) & set(indices_emb[0])) / top_k

    return {
        'query_name': poi_names[query_idx],
        'query_coord': query_coord,
        'real_neighbors': [poi_names[i] for i in indices_real[0]],
        'emb_neighbors': [poi_names[i] for i in indices_emb[0]],
        'overlap_rate': overlap,
    }


def validate_region_similarity(coords, embeddings, labels, n_clusters=15):
    """
    验证3：区域相似性

    问题：能否识别空间上相似的区域？
    """
    # 计算每个聚类的中心
    region_centers_real = []
    region_centers_emb = []

    for i in range(n_clusters):
        mask = labels == i
        if mask.sum() > 0:
            region_centers_real.append(coords[mask].mean(axis=0))
            region_centers_emb.append(embeddings[mask].mean(axis=0))

    region_centers_real = np.array(region_centers_real)
    region_centers_emb = np.array(region_centers_emb)

    # 找相似区域
    nbrs_real = NearestNeighbors(n_neighbors=4).fit(region_centers_real)
    _, indices_real = nbrs_real.kneighbors(region_centers_real)

    nbrs_emb = NearestNeighbors(n_neighbors=4).fit(region_centers_emb)
    _, indices_emb = nbrs_emb.kneighbors(region_centers_emb)

    # 计算一致性
    consistency_scores = []
    for i in range(len(indices_real)):
        overlap = len(set(indices_real[i][1:]) & set(indices_emb[i][1:])) / 3
        consistency_scores.append(overlap)

    return {
        'mean_consistency': np.mean(consistency_scores),
        'region_consistencies': consistency_scores,
    }


def validate_neighbor_prediction(coords, embeddings, knn_neighbors, sample_size=1000):
    """
    验证4：邻居预测

    问题：能否预测一个POI的空间邻居？
    """
    N = len(coords)
    sample_size = min(sample_size, N)

    sample_indices = np.random.choice(N, sample_size, replace=False)

    # 用embedding找邻居
    nbrs_emb = NearestNeighbors(n_neighbors=11).fit(embeddings)  # 11 = 自己 + 10邻居
    _, indices_emb = nbrs_emb.kneighbors(embeddings[sample_indices])

    # 计算准确率
    precisions = []
    for i, idx in enumerate(sample_indices):
        true_neighbors = set(knn_neighbors[idx])
        pred_neighbors = set(indices_emb[i][1:])  # 排除自己
        precision = len(true_neighbors & pred_neighbors) / len(true_neighbors) if true_neighbors else 0
        precisions.append(precision)

    return {
        'mean_precision': np.mean(precisions),
        'precisions': precisions,
    }


def run_validation(area_name: str, model_path: str = None):
    """运行完整验证"""
    print("=" * 70)
    print(f"V2.3 验证: {area_name}")
    print("=" * 70)

    # 加载数据
    data_dir = str(Path(__file__).resolve().parents[2] / "data" / "experiment_data")
    data = load_area_data(area_name, data_dir)

    coords = data['coords']
    features = data['features']
    poi_names = data['names']
    metadata = data['metadata']

    print(f"\n数据规模: {metadata['num_pois']} POI")

    # 构建配置
    config = SpatialEncoderConfig(
        num_categories=metadata['num_categories'],
        num_landuses=metadata['num_landuses'],
        num_aoi_types=metadata['num_aoi_types'],
        num_road_classes=metadata['num_road_classes'],
    )

    # 创建模型并加载权重（如果有）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatialTopologyEncoder(config).to(device)

    # 如果没有保存的模型，就重新训练一个简化版
    print("\n[训练模型...]")
    model.train()

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    coords_norm = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    knn_neighbors = build_knn_neighbors(coords, k=10)

    for epoch in range(200):
        optimizer.zero_grad()
        z, coord_recon = model(features_t, coords_t)
        loss = F.mse_loss(coord_recon, coords_norm_t)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}")

    # 获取embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features_t, coords_t).cpu().numpy()

    # 生成标签
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    print(f"\nSilhouette: {silhouette_score(embeddings, labels):.4f}")

    # ===== 验证1: 距离保持 =====
    print("\n" + "=" * 50)
    print("验证1: 距离保持能力")
    print("=" * 50)

    dist_result = validate_distance_preservation(coords, embeddings)
    print(f"  Pearson相关系数: {dist_result['pearson']:.4f}")
    print(f"  Spearman相关系数: {dist_result['spearman']:.4f}")

    if dist_result['pearson'] > 0.8:
        print("  结论: embedding距离与真实距离高度相关，模型学会了空间距离关系")
    elif dist_result['pearson'] > 0.5:
        print("  结论: embedding距离与真实距离中等相关，模型部分学会了空间距离关系")
    else:
        print("  结论: embedding距离与真实距离相关性弱，模型未能学会空间距离关系")

    # ===== 验证2: 空间相似性查询 =====
    print("\n" + "=" * 50)
    print("验证2: 空间相似性查询")
    print("=" * 50)

    # 随机选择一个POI
    query_idx = np.random.randint(0, len(coords))
    query_result = validate_spatial_query(coords, embeddings, poi_names, query_idx)

    print(f"  查询POI: {query_result['query_name']}")
    print(f"  真实空间邻居: {query_result['real_neighbors'][:5]}")
    print(f"  Embedding邻居: {query_result['emb_neighbors'][:5]}")
    print(f"  邻居重叠率: {query_result['overlap_rate']*100:.1f}%")

    if query_result['overlap_rate'] > 0.5:
        print("  结论: 模型能准确找到空间相似的POI")
    else:
        print("  结论: 模型在空间相似性查询上有改进空间")

    # ===== 验证3: 区域相似性 =====
    print("\n" + "=" * 50)
    print("验证3: 区域相似性识别")
    print("=" * 50)

    region_result = validate_region_similarity(coords, embeddings, labels)
    print(f"  平均区域一致性: {region_result['mean_consistency']*100:.1f}%")

    if region_result['mean_consistency'] > 0.5:
        print("  结论: 模型能识别空间上相似的区域")
    else:
        print("  结论: 模型在区域相似性识别上有改进空间")

    # ===== 验证4: 邻居预测 =====
    print("\n" + "=" * 50)
    print("验证4: 邻居预测能力")
    print("=" * 50)

    neighbor_result = validate_neighbor_prediction(coords, embeddings, knn_neighbors)
    print(f"  平均邻居预测准确率: {neighbor_result['mean_precision']*100:.1f}%")

    if neighbor_result['mean_precision'] > 0.3:
        print("  结论: 模型能预测空间邻居")
    else:
        print("  结论: 模型在邻居预测上有改进空间")

    # ===== 总结 =====
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    scores = {
        '距离保持': dist_result['pearson'],
        '邻居预测': neighbor_result['mean_precision'],
        '区域一致性': region_result['mean_consistency'],
    }

    print(f"{'验证项':<15} {'得分':<10} {'状态'}")
    print("-" * 40)
    for name, score in scores.items():
        status = "PASS" if score > 0.5 else "FAIL"
        print(f"{name:<15} {score:.4f}     {status}")

    avg_score = np.mean(list(scores.values()))
    print("-" * 40)
    print(f"{'平均':<15} {avg_score:.4f}")

    if avg_score > 0.6:
        print("\n结论: 模型成功学会了空间拓扑关系！")
    else:
        print("\n结论: 模型需要进一步优化")

    return scores


if __name__ == "__main__":
    run_validation("guanggu_core")
