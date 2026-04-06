# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 数据加载模块
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.neighbors import kneighbors_graph


def load_area_data(area_name: str, data_dir: str) -> Dict:
    """
    加载区域POI数据

    Args:
        area_name: 区域名称
        data_dir: 数据目录

    Returns:
        dict: {
            'coords': np.array [N, 2],
            'features': np.array [N, 7],
            'knn_neighbors': List[List[int]],
            'names': List[str],
            'metadata': dict
        }
    """
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

    coords = np.array(poi_coords, dtype=np.float32)

    # 预计算KNN邻居
    knn_k = 10
    adj = kneighbors_graph(coords, n_neighbors=knn_k, mode='connectivity', include_self=False)
    knn_neighbors = [adj[i].nonzero()[1] for i in range(len(coords))]

    return {
        'coords': coords,
        'features': np.array(poi_features, dtype=np.float32),
        'knn_neighbors': knn_neighbors,
        'names': poi_names,
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
            'category_map': category_map,
            'landuse_map': landuse_map,
            'aoi_type_map': aoi_type_map,
            'road_class_map': road_class_map,
        }
    }


def sample_distance_pairs(coords: np.ndarray, num_pairs: int, device) -> Tuple:
    """
    采样点对并计算空间距离

    Args:
        coords: [N, 2] 坐标
        num_pairs: 采样对数
        device: torch设备

    Returns:
        (pair_indices, spatial_dists): 点对索引和归一化距离
    """
    import torch

    N = len(coords)

    idx_i = np.random.randint(0, N, num_pairs)
    idx_j = np.random.randint(0, N, num_pairs)

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    coords_i = coords[idx_i]
    coords_j = coords[idx_j]
    spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))

    # 归一化到0-1
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    return (
        torch.from_numpy(np.stack([idx_i, idx_j], axis=1)).long().to(device),
        torch.from_numpy(spatial_dists).float().to(device),
    )


def build_knn_neighbors(coords: np.ndarray, k: int = 10) -> List:
    """构建KNN邻居列表"""
    adj = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    return [adj[i].nonzero()[1] for i in range(len(coords))]
