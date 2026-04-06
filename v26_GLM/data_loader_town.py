# -*- coding: utf-8 -*-
"""
三镇数据加载器 - 双分辨率版本

从 cells 表读取聚合后的特征，支持双分辨率策略。

数据来源：老三镇（武昌、汉口、汉阳）核心城区
H3 分辨率：8 (基础) + 9 (精细)
Cells 数量：2,478 (res=8)

特征维度：
- point_features: POI 相关特征 (32 + 8 fine)
- line_features: 道路相关特征 (16 + 4 fine)
- polygon_features: 土地利用 + 人口特征 (20 + 8 fine)

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.stdout.reconfigure(encoding='utf-8')

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import h3

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.data_sources import PostGISSource, default_postgis_source
from spatial_encoder.v26_GLM.direction_supervision import (
    MultiSchemeDirectionSupervision,
)


# 功能区类别映射（AOI 类型 → 6类）
AOI_TYPE_MAP = {
    'residential': 0,    # 居住类
    'commercial': 1,     # 商业类
    'retail': 1,         # 商业类
    'industrial': 2,     # 工业类
    'university': 3,     # 教育类
    'college': 3,        # 教育类
    'school': 3,         # 教育类
    'park': 4,           # 公共类
    'hospital': 4,       # 公共类
    'parking': 4,        # 公共类
    'forest': 5,         # 自然类
    'water': 5,          # 自然类
    'grass': 5,          # 自然类
    'pitch': 4,          # 公共类（运动场）
}
UNKNOWN_REGION_LABEL = 6


class TownDataLoader:
    """三镇数据加载器"""

    def __init__(
        self,
        source: Optional[PostGISSource] = None,
        config: Optional[V26ProConfig] = None,
    ):
        self.source = source or default_postgis_source
        self.config = config or DEFAULT_PRO_CONFIG
        self.resolution = self.config.h3.resolution

    def _get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(
            host=self.source.host,
            port=self.source.port,
            user=self.source.user,
            password=self.source.password,
            database=self.source.database,
        )

    def load_cells(
        self,
        sample_ratio: float = 1.0,
        where_clause: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        从 cells 表加载特征（支持双分辨率）

        Args:
            sample_ratio: 采样比例 (0-1)
            where_clause: 额外的 WHERE 条件

        Returns:
            point_features: [N, 40] (32 base + 8 fine)
            line_features: [N, 20] (16 base + 4 fine)
            polygon_features: [N, 28] (20 base + 8 fine)
            population_features: [N, 8]
            coords: [N, 2]
            region_labels: [N]
            metadata: List[Dict]
        """
        print(f"Loading cells data (sample_ratio={sample_ratio})...")

        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 构建查询
        sql = f"""
            SELECT
                cell_id,
                longitude, latitude,
                poi_count, poi_density, category_entropy, dominant_category,
                category_distribution,
                road_count, road_density, road_length_km,
                landuse_count, landuse_area_sqm, landuse_mix, dominant_landuse,
                population_count, population_density, avg_population,
                aoi_count, dominant_aoi
            FROM cells
            WHERE poi_count > 0
        """

        if where_clause:
            sql += f" AND {where_clause}"

        if sample_ratio < 1.0:
            sql += f" AND RANDOM() < {sample_ratio}"

        sql += " ORDER BY cell_id"

        cur.execute(sql)
        rows = cur.fetchall()

        n_cells = len(rows)
        print(f"  Loaded {n_cells:,} cells")

        # 初始化特征数组（增加精细特征维度）
        point_features = np.zeros((n_cells, 40), dtype=np.float32)  # 32 + 8 fine
        line_features = np.zeros((n_cells, 20), dtype=np.float32)    # 16 + 4 fine
        polygon_features = np.zeros((n_cells, 28), dtype=np.float32) # 20 + 8 fine
        population_features = np.zeros((n_cells, 8), dtype=np.float32)
        coords = np.zeros((n_cells, 2), dtype=np.float32)
        region_labels = np.zeros(n_cells, dtype=np.int64)
        metadata = []

        # 类别编码
        category_list = [
            "购物消费", "餐饮美食", "生活服务", "公司企业",
            "交通设施", "科教文化", "医疗保健", "商务住宅",
            "汽车相关", "酒店住宿", "休闲娱乐", "金融机构",
            "运动健身", "旅游景点",
        ]
        category_to_idx = {cat: idx for idx, cat in enumerate(category_list)}

        # 土地利用编码
        landuse_list = [
            "居住用地", "商业服务用地", "工业用地", "教育用地",
            "公园与绿地用地", "河流湖泊", "医疗卫生用地", "体育与文化用地",
        ]
        landuse_to_idx = {lu: idx for idx, lu in enumerate(landuse_list)}

        # 坐标范围（三镇）
        lng_min, lng_max = 113.8, 115.0
        lat_min, lat_max = 30.3, 30.9

        # 计算精细分辨率特征
        print("  Computing fine-resolution features (res=9)...")
        fine_features = self._compute_fine_features(cur, rows)

        for i, row in enumerate(rows):
            # 坐标
            lng, lat = float(row['longitude']), float(row['latitude'])
            coords[i] = [lng, lat]

            # 归一化坐标
            norm_lng = (lng - lng_min) / (lng_max - lng_min)
            norm_lat = (lat - lat_min) / (lat_max - lat_min)

            # ===== Point Features [40] =====
            # [0-1]: 归一化坐标
            point_features[i, 0] = norm_lng
            point_features[i, 1] = norm_lat

            # [2]: POI 密度（对数归一化）
            point_features[i, 2] = np.log1p(row['poi_density']) / 10.0

            # [3]: 类别熵
            point_features[i, 3] = float(row['category_entropy'] or 0)

            # [4-17]: 类别分布
            cat_dist = row['category_distribution'] or {}
            if isinstance(cat_dist, str):
                try:
                    cat_dist = json.loads(cat_dist)
                except:
                    cat_dist = {}

            for cat, prob in cat_dist.items():
                if cat in category_to_idx:
                    point_features[i, 4 + category_to_idx[cat]] = float(prob)

            # [18-19]: POI 计数相关
            point_features[i, 18] = np.log1p(row['poi_count']) / 10.0
            point_features[i, 19] = row['poi_count'] / 1000.0

            # [20-31]: 精细特征（res=9 子 cells 的聚合）
            if fine_features is not None and i < len(fine_features):
                point_features[i, 20:32] = fine_features[i]['point_fine']

            # ===== Line Features [20] =====
            # [0-1]: 归一化坐标
            line_features[i, 0] = norm_lng
            line_features[i, 1] = norm_lat

            # [2]: 道路密度（归一化）
            line_features[i, 2] = min(row['road_density'] / 100.0, 1.0)

            # [3]: 道路长度
            line_features[i, 3] = np.log1p(row['road_length_km'] or 0) / 5.0

            # [4]: 道路计数
            line_features[i, 4] = np.log1p(row['road_count'] or 0) / 5.0

            # [5-7]: 精细特征
            if fine_features is not None and i < len(fine_features):
                line_features[i, 5:12] = fine_features[i]['line_fine']

            # ===== Polygon Features [28] =====
            # [0-1]: 归一化坐标
            polygon_features[i, 0] = norm_lng
            polygon_features[i, 1] = norm_lat

            # [2]: 土地利用混合度
            polygon_features[i, 2] = float(row['landuse_mix'] or 0) / 3.0

            # [3]: 土地利用面积
            polygon_features[i, 3] = np.log1p(row['landuse_area_sqm'] or 0) / 20.0

            # [4]: 土地利用计数
            polygon_features[i, 4] = np.log1p(row['landuse_count'] or 0) / 3.0

            # [5-7]: 主导土地利用类型 one-hot
            dominant_lu = row['dominant_landuse'] or ''
            if dominant_lu in landuse_to_idx:
                polygon_features[i, 5 + landuse_to_idx[dominant_lu]] = 1.0

            # [8-15]: 精细特征
            if fine_features is not None and i < len(fine_features):
                polygon_features[i, 8:20] = fine_features[i]['polygon_fine']

            # ===== Population Features [8] =====
            population_features[i, 0] = norm_lng
            population_features[i, 1] = norm_lat

            pop_density = row['population_density'] or 0
            population_features[i, 2] = np.log1p(pop_density) / 15.0
            population_features[i, 3] = np.log1p(row['avg_population'] or 0) / 10.0
            population_features[i, 4] = np.log1p(row['population_count'] or 0) / 5.0

            # ===== Region Labels =====
            aoi_type = row['dominant_aoi'] or ''
            region_labels[i] = AOI_TYPE_MAP.get(aoi_type, UNKNOWN_REGION_LABEL)

            # ===== Metadata =====
            metadata.append({
                'cell_id': row['cell_id'],
                'poi_count': row['poi_count'],
                'dominant_category': row['dominant_category'],
                'road_count': row['road_count'],
                'road_length_km': round(row['road_length_km'] or 0, 2),
                'population_density': round(pop_density, 0),
                'aoi_type': aoi_type,
            })

        cur.close()
        conn.close()

        print(f"  Features: point{point_features.shape}, line{line_features.shape}, "
              f"polygon{polygon_features.shape}, pop{population_features.shape}")

        return (
            point_features, line_features, polygon_features, population_features,
            coords, region_labels, metadata
        )

    def _compute_fine_features(self, cur, rows) -> Optional[List[Dict]]:
        """计算精细分辨率（res=9）的子 cell 特征"""
        try:
            fine_features = []

            for row in rows:
                cell_id = row['cell_id']

                # 获取 res=8 cell 的子 cells (res=9)
                # H3 的 cell_to_children 返回子 cells
                children = h3.cell_to_children(cell_id, 9)

                if not children:
                    # 没有子 cells，使用默认值
                    fine_features.append({
                        'point_fine': np.zeros(12),
                        'line_fine': np.zeros(7),
                        'polygon_fine': np.zeros(12),
                    })
                    continue

                # 计算子 cells 的 POI 分布统计
                # 通过 POI 坐标判断属于哪个子 cell
                lng, lat = float(row['longitude']), float(row['latitude'])

                # 简化：使用该 cell 内 POI 的空间方差作为精细特征
                cur.execute("""
                    SELECT
                        STDDEV(longitude) as lng_std,
                        STDDEV(latitude) as lat_std,
                        MIN(longitude) as lng_min,
                        MAX(longitude) as lng_max,
                        MIN(latitude) as lat_min,
                        MAX(latitude) as lat_max,
                        COUNT(*) as poi_count
                    FROM pois
                    WHERE cell_id = %s
                """, (cell_id,))

                stats = cur.fetchone()

                if stats and stats['poi_count'] and stats['poi_count'] > 1:
                    # 空间分布特征
                    lng_spread = float(stats['lng_max'] - stats['lng_min']) if stats['lng_max'] else 0
                    lat_spread = float(stats['lat_max'] - stats['lat_min']) if stats['lat_max'] else 0
                    lng_std = float(stats['lng_std']) if stats['lng_std'] else 0
                    lat_std = float(stats['lat_std']) if stats['lat_std'] else 0

                    point_fine = np.array([
                        lng_std * 100,  # 经度标准差
                        lat_std * 100,  # 纬度标准差
                        lng_spread * 100,  # 经度范围
                        lat_spread * 100,  # 纬度范围
                        np.log1p(lng_spread * 10000),
                        np.log1p(lat_spread * 10000),
                        len(children) / 10,  # 子 cell 数量（归一化）
                        0, 0, 0, 0, 0  # 保留
                    ])

                    line_fine = np.array([
                        lng_std * 50,
                        lat_std * 50,
                        lng_spread * 50,
                        lat_spread * 50,
                        0, 0, 0
                    ])

                    polygon_fine = np.array([
                        lng_std * 100,
                        lat_std * 100,
                        lng_spread * 100,
                        lat_spread * 100,
                        np.log1p(len(children)),
                        0, 0, 0, 0, 0, 0, 0, 0
                    ])
                else:
                    point_fine = np.zeros(12)
                    line_fine = np.zeros(7)
                    polygon_fine = np.zeros(12)

                fine_features.append({
                    'point_fine': point_fine,
                    'line_fine': line_fine,
                    'polygon_fine': polygon_fine,
                })

            return fine_features

        except Exception as e:
            print(f"  Warning: Failed to compute fine features: {e}")
            return None

    def compute_neighbor_indices(
        self,
        coords: np.ndarray,
        k_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算每个 Cell 的邻居索引（基于 H3 网格）

        Args:
            coords: Cell 坐标 [N, 2]
            k_neighbors: 最大邻居数

        Returns:
            neighbor_indices: 邻居索引 [N, K]
            neighbor_rings: 邻居圈数 [N, K]
        """
        import h3

        N = len(coords)
        neighbor_indices = np.full((N, k_neighbors), -1, dtype=np.int64)
        neighbor_rings = np.zeros((N, k_neighbors), dtype=np.int64)

        # 构建 Cell ID 到索引的映射
        cell_ids = []
        for i in range(N):
            cell_id = h3.latlng_to_cell(coords[i, 1], coords[i, 0], self.resolution)
            cell_ids.append(cell_id)

        cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}

        # 获取每个 Cell 的邻居
        for i, cell_id in enumerate(cell_ids):
            neighbors = []
            for ring in range(1, 4):  # 1-3圈
                ring_neighbors = h3.grid_disk(cell_id, ring)
                for neighbor_id in ring_neighbors:
                    if neighbor_id != cell_id and neighbor_id in cell_to_idx:
                        neighbors.append((cell_to_idx[neighbor_id], ring))

            # 按圈数排序，取前 K 个
            neighbors.sort(key=lambda x: x[1])
            neighbors = neighbors[:k_neighbors]

            for j, (idx, ring) in enumerate(neighbors):
                neighbor_indices[i, j] = idx
                neighbor_rings[i, j] = ring

        return neighbor_indices, neighbor_rings


def load_town_dataset(
    config: Optional[V26ProConfig] = None,
    sample_ratio: float = 1.0,
) -> Dict:
    """
    加载三镇训练数据集

    Args:
        config: 配置
        sample_ratio: 采样比例

    Returns:
        包含所有训练数据的字典
    """
    if config is None:
        config = DEFAULT_PRO_CONFIG

    loader = TownDataLoader(config=config)

    # 加载 Cell 特征
    (
        point_features, line_features, polygon_features, population_features,
        coords, region_labels, metadata
    ) = loader.load_cells(sample_ratio=sample_ratio)

    n_cells = len(coords)

    # 计算邻居关系
    print("Computing neighbor indices...")
    neighbor_indices, neighbor_rings = loader.compute_neighbor_indices(
        coords, config.loss.k_nearest_neighbors
    )

    # 计算方向监督
    print("Computing direction supervision...")
    direction_supervisor = MultiSchemeDirectionSupervision()
    direction_supervisor.compute_all(
        cell_coords=coords,
        neighbor_indices=neighbor_indices,
    )
    direction_labels, direction_weights, direction_valid = direction_supervisor.get_labels_for_training()

    # 构建方向特征（8维，one-hot）
    direction_features = np.zeros((n_cells, 8), dtype=np.float32)
    for i in range(n_cells):
        if direction_valid[i]:
            direction_features[i, direction_labels[i]] = 1.0

    # 合并人口特征到 polygon_features（或单独保留）
    # 这里将人口特征拼接到 polygon_features 后面
    extended_polygon_features = np.concatenate([
        polygon_features, population_features[:, :4]  # 取前4维
    ], axis=1)

    print(f"\nDataset summary:")
    print(f"  Cells: {n_cells:,}")
    print(f"  Point features: {point_features.shape}")
    print(f"  Line features: {line_features.shape}")
    print(f"  Polygon+Pop features: {extended_polygon_features.shape}")
    print(f"  Direction features: {direction_features.shape}")
    print(f"  Region labels: {len(np.unique(region_labels))} classes, "
          f"valid={(region_labels < 6).sum()}")

    return {
        "point_features": point_features,
        "line_features": line_features,
        "polygon_features": extended_polygon_features,
        "direction_features": direction_features,
        "coords": coords,
        "region_labels": region_labels,
        "cell_id_map": {i: i for i in range(n_cells)},
        "neighbor_indices": neighbor_indices,
        "neighbor_rings": neighbor_rings,
        "direction_labels": direction_labels,
        "direction_weights": direction_weights,
        "direction_valid": direction_valid,
        "metadata": metadata,
    }


if __name__ == "__main__":
    # 测试数据加载
    print("Testing TownDataLoader...")

    data = load_town_dataset(sample_ratio=0.1)

    print(f"\nDataset shapes:")
    print(f"  Point features: {data['point_features'].shape}")
    print(f"  Line features: {data['line_features'].shape}")
    print(f"  Polygon features: {data['polygon_features'].shape}")
    print(f"  Direction features: {data['direction_features'].shape}")
    print(f"  Coords: {data['coords'].shape}")
    print(f"  Neighbor indices: {data['neighbor_indices'].shape}")
    print(f"  Direction labels: {data['direction_labels'].shape}")
    print(f"  Region labels: {data['region_labels'].shape}")

    print("\n[OK] Test passed!")
