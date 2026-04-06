# -*- coding: utf-8 -*-
"""
POI 级数据加载器

直接以单个 POI 为训练样本（565K），替代 Cell 级（1828）。
特征设计与 Cell 级保持相同的 72 维格式，模型架构零改动。

分层标注（已预计算写入 pois.region_label）：
  AOI fclass → Landuse land_type → POI 大类 → NULL

特征构建：
  point_features   [32]: 位置 + K 近邻类别分布 + 自身类别
  line_features    [16]: 位置 + 半径内道路密度/等级
  polygon_features [16]: 位置 + 所在地块类型
  direction_features[8]: 相对城市中心方向

Author: Sisyphus
Date: 2026-03-19
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.neighbors import BallTree

from spatial_encoder.v26_GLM.data_loader_v26 import (
    MERGED_REGION_NAMES,
    default_postgis_source,
)
from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.h3_projection import point_to_cell

# 三镇坐标范围（用于归一化）
LNG_MIN, LNG_MAX = 113.70, 114.65
LAT_MIN, LAT_MAX = 30.39, 30.79

# 高德 POI 大类列表（16 维，与 cell 级对齐）
POI_CATEGORY_LIST = [
    "购物服务", "生活服务", "科教文化服务", "医疗保健服务",
    "政府机构及社会团体", "交通设施服务", "体育休闲服务", "商务住宅",
    "住宿服务", "公司企业", "金融保险服务", "公共设施",
    "汽车服务", "汽车维修", "汽车销售", "风景名胜",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(POI_CATEGORY_LIST)}

# 道路等级（5 维）
ROAD_CLASS_LIST = ["primary", "secondary", "tertiary", "residential", "unclassified"]
ROAD_CLASS_TO_IDX = {c: i for i, c in enumerate(ROAD_CLASS_LIST)}

# 地块类型（11 维，与 cell 级对齐）
LANDUSE_TYPE_LIST = [
    "居住用地", "商业服务用地", "工业用地", "物流仓储用地",
    "道路交通设施用地", "公用设施用地", "绿地与广场用地",
    "医疗卫生用地", "教育用地", "河流湖泊", "公园与绿地用地",
]
LANDUSE_TO_IDX = {t: i for i, t in enumerate(LANDUSE_TYPE_LIST)}

# 城市中心（武汉）
CITY_CENTER_LNG = 114.305
CITY_CENTER_LAT = 30.593


def _norm_coord(lng: float, lat: float) -> Tuple[float, float]:
    return (
        (lng - LNG_MIN) / (LNG_MAX - LNG_MIN),
        (lat - LAT_MIN) / (LAT_MAX - LAT_MIN),
    )


def _direction_onehot(lng: float, lat: float) -> np.ndarray:
    dx = lng - CITY_CENTER_LNG
    dy = lat - CITY_CENTER_LAT
    angle = np.arctan2(dy, dx)
    direction = int((angle + np.pi) / (np.pi / 4)) % 8
    feat = np.zeros(8, dtype=np.float32)
    feat[direction] = 1.0
    return feat


class POIDataLoader:
    """
    POI 级数据加载器

    用法：
        loader = POIDataLoader()
        result = loader.load(sample_ratio=0.1)
        # result: (point_feat, line_feat, polygon_feat, dir_feat, coords, region_labels, metadata)
    """

    def __init__(
        self,
        k_neighbors: int = 50,
        road_radius_deg: float = 0.005,   # ~500m
        source=None,
    ):
        self.k_neighbors = k_neighbors
        self.road_radius_deg = road_radius_deg
        self.source = source or default_postgis_source

    def _get_conn(self):
        s = self.source
        return psycopg2.connect(
            host=s.host, port=s.port, user=s.user,
            password=s.password, database=s.database,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def load(
        self,
        sample_ratio: float = 1.0,
        limit: Optional[int] = None,
        city_filter: Optional[List[str]] = None,
        cell_model_path: Optional[str] = None,
        cell_resolution: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, List[Dict]]:
        """
        加载 POI 级特征。

        Args:
            cell_model_path: Cell 级模型路径（用于层次化多尺度，可选）
            cell_resolution: H3 分辨率（默认 8，与 Cell 级训练一致）

        Returns:
            point_features:    [N, 32]
            line_features:     [N, 16]
            polygon_features:  [N, 16]
            direction_features:[N, 8]
            coords:            [N, 2]  (lng, lat)
            region_labels:     [N]     (0-5 有效, 6=未知, -1=NULL)
            metadata:          List[Dict]
        """
        print("Loading POIs from DB...")
        pois = self._load_pois(sample_ratio, limit, city_filter)
        N = len(pois)
        print(f"  {N:,} POIs loaded")

        print("Loading roads from DB...")
        road_coords, road_classes = self._load_roads()
        print(f"  {len(road_coords):,} roads loaded")

        print("Loading landuse from DB...")
        landuse_map = self._load_landuse_for_pois(pois)
        print(f"  landuse mapped for {sum(1 for v in landuse_map.values() if v):,} POIs")

        print("Building BallTree for K-NN...")
        coords_rad = np.radians([[p["lat"], p["lng"]] for p in pois])
        tree = BallTree(coords_rad, metric="haversine")

        print("Building road spatial index...")
        road_tree = None
        if len(road_coords) > 0:
            road_tree = BallTree(np.radians(road_coords), metric="haversine")

        print("Computing features...")
        point_features    = np.zeros((N, 32), dtype=np.float32)
        line_features     = np.zeros((N, 16), dtype=np.float32)
        polygon_features  = np.zeros((N, 16), dtype=np.float32)
        direction_features= np.zeros((N, 8),  dtype=np.float32)
        coords            = np.zeros((N, 2),  dtype=np.float32)
        region_labels     = np.full(N, -1,    dtype=np.int32)
        metadata          = []

        # K-NN 批量查询（一次性）
        radius_m = 500.0
        radius_rad = radius_m / 6371000.0
        indices_list = tree.query_radius(coords_rad, r=radius_rad)

        road_radius_rad = self.road_radius_deg * np.pi / 180.0

        for i, poi in enumerate(pois):
            lng, lat = poi["lng"], poi["lat"]
            norm_lng, norm_lat = _norm_coord(lng, lat)
            coords[i] = [lng, lat]

            # region_label
            rl = poi["region_label"]
            region_labels[i] = rl if rl is not None else -1

            # ---- point_features [32] ----
            point_features[i, 0] = norm_lng
            point_features[i, 1] = norm_lat

            neighbor_idx = indices_list[i]
            k_count = max(len(neighbor_idx) - 1, 0)  # 排除自身
            point_features[i, 2] = np.log1p(k_count) / 10.0

            # K 近邻类别分布 [3:19]
            cat_counts = np.zeros(16, dtype=np.float32)
            for ni in neighbor_idx:
                if ni == i:
                    continue
                cat = pois[ni]["category_main"]
                if cat in CATEGORY_TO_IDX:
                    cat_counts[CATEGORY_TO_IDX[cat]] += 1
            if k_count > 0:
                cat_dist = cat_counts / k_count
            else:
                cat_dist = cat_counts
            point_features[i, 3:19] = cat_dist

            # 类别熵 [19]
            nz = cat_dist[cat_dist > 0]
            if len(nz) > 1:
                point_features[i, 19] = float(-np.sum(nz * np.log(nz + 1e-8)))

            # 自身类别 one-hot [20:32]（12 维，取前 12 类）
            own_cat = poi["category_main"]
            if own_cat in CATEGORY_TO_IDX and CATEGORY_TO_IDX[own_cat] < 12:
                point_features[i, 20 + CATEGORY_TO_IDX[own_cat]] = 1.0

            # ---- line_features [16] ----
            line_features[i, 0] = norm_lng
            line_features[i, 1] = norm_lat

            if road_tree is not None:
                q = np.radians([[lat, lng]])
                road_idx = road_tree.query_radius(q, r=road_radius_rad)[0]
                road_count = len(road_idx)
                line_features[i, 2] = np.log1p(road_count) / 5.0
                rc_counts = np.zeros(5, dtype=np.float32)
                for ri in road_idx:
                    rc = road_classes[ri]
                    if rc in ROAD_CLASS_TO_IDX:
                        rc_counts[ROAD_CLASS_TO_IDX[rc]] += 1
                if road_count > 0:
                    line_features[i, 3:8] = rc_counts / road_count

            # ---- polygon_features [16] ----
            polygon_features[i, 0] = norm_lng
            polygon_features[i, 1] = norm_lat

            lu_info = landuse_map.get(poi["id"])
            if lu_info:
                lu_type, lu_area = lu_info
                polygon_features[i, 2] = min(lu_area / 1e6, 10.0)  # km²
                if lu_type in LANDUSE_TO_IDX:
                    polygon_features[i, 3 + LANDUSE_TO_IDX[lu_type]] = 1.0

            # ---- direction_features [8] ----
            direction_features[i] = _direction_onehot(lng, lat)

            # ---- metadata ----
            metadata.append({
                "poi_id": poi["id"],
                "name": poi["name"],
                "category_main": poi["category_main"],
                "category_sub": poi["category_sub"],
                "region_label": MERGED_REGION_NAMES.get(
                    region_labels[i], "NULL" if region_labels[i] == -1 else "未知"
                ),
            })

            if (i + 1) % 50000 == 0:
                print(f"  {i+1:,}/{N:,}")

        print("Done.")

        # ---- 层次化多尺度：Cell embedding 条件注入 ----
        cell_embeddings = None
        cell_dist_to_center = None
        if cell_model_path is not None:
            cell_embeddings, cell_dist_to_center = self._compute_cell_embeddings(
                pois, coords, cell_model_path, cell_resolution,
                point_features, line_features, polygon_features, direction_features,
            )

        return (point_features, line_features, polygon_features,
                direction_features, coords, region_labels, metadata,
                cell_embeddings, cell_dist_to_center)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_pois(self, sample_ratio, limit, city_filter):
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT id, name, category_main, category_sub,
                       longitude AS lng, latitude AS lat, region_label
                FROM pois
                WHERE longitude IS NOT NULL AND latitude IS NOT NULL
            """
            if city_filter:
                placeholders = ",".join(["%s"] * len(city_filter))
                sql += f" AND city IN ({placeholders})"
            if sample_ratio < 1.0:
                sql += f" AND RANDOM() < {sample_ratio}"
            if limit:
                sql += f" LIMIT {limit}"

            params = city_filter if city_filter else []
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    def _load_roads(self):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT ST_Y(ST_Centroid(geom)) AS lat,
                       ST_X(ST_Centroid(geom)) AS lng,
                       fclass
                FROM roads
                WHERE geom IS NOT NULL
            """)
            rows = cur.fetchall()
            coords = [[r[0], r[1]] for r in rows]
            classes = [r[2] for r in rows]
            return coords, classes
        finally:
            conn.close()

    def _load_landuse_for_pois(self, pois) -> Dict[int, Optional[Tuple[str, float]]]:
        """返回 {poi_id: (land_type, area_sqm)} 或 None"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            # 批量空间连接：找每个 POI 所在的最小地块
            cur.execute("""
                SELECT p.id, l.land_type, l.area_sqm
                FROM pois p
                JOIN landuse l ON ST_Within(p.geom, l.geom)
                ORDER BY p.id, l.area_sqm ASC
            """)
            result: Dict[int, Optional[Tuple[str, float]]] = {}
            for poi_id, land_type, area in cur.fetchall():
                if poi_id not in result:
                    result[poi_id] = (land_type, area)
            return result
        finally:
            conn.close()

    def _compute_cell_embeddings(
        self,
        pois: List[Dict],
        coords: np.ndarray,
        cell_model_path: str,
        cell_resolution: int,
        point_features: np.ndarray,
        line_features: np.ndarray,
        polygon_features: np.ndarray,
        direction_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        用冻结的 Cell 级模型为每个 POI 生成宏观 Cell embedding。

        策略：
          1. 将每个 POI 映射到所属 H3 Cell（resolution=cell_resolution）
          2. 对同一 Cell 内的 POI 特征取均值，作为该 Cell 的输入特征
          3. 用冻结的 Cell 模型推理，得到 Cell embedding（352维）
          4. 每个 POI 的 cell_embedding = 所属 Cell 的 embedding

        Returns:
            cell_embeddings:    [N, 352]
            cell_dist_to_center:[N, 1]  POI 到 Cell 中心的归一化距离
        """
        import torch
        from spatial_encoder.v26_GLM.ultimate_encoder import build_ultimate_encoder

        print(f"  Loading cell model from {cell_model_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cell_model = build_ultimate_encoder(DEFAULT_PRO_CONFIG)
        sd = torch.load(cell_model_path, map_location="cpu", weights_only=False)
        inner_sd = sd.get("model_state_dict", sd)
        cell_model.load_state_dict(inner_sd, strict=False)
        cell_model.eval().to(device)
        for p in cell_model.parameters():
            p.requires_grad_(False)

        N = len(pois)
        # 将每个 POI 映射到 H3 Cell
        cell_ids = []
        for poi in pois:
            try:
                cid = point_to_cell(poi["lng"], poi["lat"], resolution=cell_resolution)
            except Exception:
                cid = None
            cell_ids.append(cid)

        # 按 Cell 分组，取均值特征
        from collections import defaultdict
        cell_to_indices: Dict = defaultdict(list)
        for i, cid in enumerate(cell_ids):
            cell_to_indices[cid].append(i)

        # 为每个唯一 Cell 计算 embedding
        all_features = np.concatenate([
            point_features, line_features, polygon_features, direction_features
        ], axis=1)  # [N, 72]

        cell_emb_map: Dict = {}
        cell_center_map: Dict = {}

        unique_cells = list(cell_to_indices.keys())
        batch_size = 512
        for start in range(0, len(unique_cells), batch_size):
            batch_cells = unique_cells[start:start + batch_size]
            batch_feats = []
            for cid in batch_cells:
                idx = cell_to_indices[cid]
                mean_feat = all_features[idx].mean(axis=0)
                batch_feats.append(mean_feat)
                # Cell 中心坐标（均值）
                cell_center_map[cid] = coords[idx].mean(axis=0)

            feat_tensor = torch.tensor(np.array(batch_feats), dtype=torch.float32).to(device)
            pt = feat_tensor[:, :32]
            ln = feat_tensor[:, 32:48]
            pg = feat_tensor[:, 48:64]
            dr = feat_tensor[:, 64:72]

            with torch.no_grad():
                emb, _, _, _ = cell_model.forward_simple(pt, ln, pg, dr)
            emb_np = emb.cpu().numpy()

            for j, cid in enumerate(batch_cells):
                cell_emb_map[cid] = emb_np[j]

        # 组装每个 POI 的 cell_embedding 和到 Cell 中心的距离
        cell_embeddings = np.zeros((N, DEFAULT_PRO_CONFIG.model.embedding_dim), dtype=np.float32)
        cell_dist_to_center = np.zeros((N, 1), dtype=np.float32)

        for i, cid in enumerate(cell_ids):
            if cid in cell_emb_map:
                cell_embeddings[i] = cell_emb_map[cid]
                center = cell_center_map[cid]
                dist = np.sqrt(((coords[i] - center) ** 2).sum())
                cell_dist_to_center[i, 0] = dist

        print(f"  Cell embeddings computed: {len(cell_emb_map)} unique cells → {N} POIs")
        return cell_embeddings, cell_dist_to_center
