# -*- coding: utf-8 -*-
"""
V3 双模型空间编码器服务

对外提供：
- POI 级编码（poi_encoder）
- 几何方向计算
- Cell / town 级上下文（town_encoder）
- POI 结果空间增强

说明：
1. `poi_encoder` 负责点位级 query anchor 编码；
2. `town_encoder` 负责真实 cells 数据的宏观 cell 上下文；
3. `/cell/context` 与 `/cell/context/batch` 会显式暴露 town/cell 路由结果；
4. 健康检查会同时返回两个模型的加载状态，供 Node 侧编排与诊断使用。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
    from spatial_encoder.v26_GLM.data_loader_poi import (
        CATEGORY_TO_IDX,
        LANDUSE_TO_IDX,
        ROAD_CLASS_TO_IDX,
    )
    from spatial_encoder.v26_GLM.data_sources import default_postgis_source
    from spatial_encoder.v26_GLM.data_loader_town import load_town_dataset
    from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder
    from spatial_encoder.v26_GLM.ultimate_encoder import build_ultimate_encoder

    ENCODER_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as error:  # pragma: no cover - import failure is handled via health payload
    ENCODER_AVAILABLE = False
    IMPORT_ERROR = str(error)


DIRECTION_NAMES = ['东', '东北', '北', '西北', '西', '西南', '南', '东南']
REGION_NAMES = ['居住类', '商业类', '工业类', '教育类', '公共类', '自然类']
IDX_TO_CATEGORY = {index: category for category, index in CATEGORY_TO_IDX.items()} if ENCODER_AVAILABLE else {}

RUNTIME_ASSET_ROOT = PROJECT_ROOT / 'runtime_assets'
POI_CHECKPOINT_PATH = Path(
    os.environ.get(
        'GEOLOOM_ENCODER_POI_CHECKPOINT',
        str(RUNTIME_ASSET_ROOT / 'saved_models' / 'poi_encoder' / 'best_model.pt'),
    )
)
TOWN_CHECKPOINT_PATH = Path(
    os.environ.get(
        'GEOLOOM_ENCODER_TOWN_CHECKPOINT',
        str(RUNTIME_ASSET_ROOT / 'saved_models' / 'town_encoder' / 'best_model.pt'),
    )
)
POI_FEATURE_CACHE_PATH = Path(
    os.environ.get(
        'GEOLOOM_ENCODER_POI_CACHE',
        str(RUNTIME_ASSET_ROOT / 'cache' / 'poi_feature_cache_v1.npz'),
    )
)
POI_FEATURE_CACHE_KEYS = (
    'poi_ids',
    'point_features',
    'line_features',
    'polygon_features',
    'direction_features',
)
POI_FEATURE_CACHE_DIMENSIONS = {
    'point_features': 32,
    'line_features': 16,
    'polygon_features': 16,
    'direction_features': 8,
}

POI_LNG_MIN, POI_LNG_MAX = 113.70, 114.65
POI_LAT_MIN, POI_LAT_MAX = 30.39, 30.79
CITY_CENTER_LON, CITY_CENTER_LAT = 114.305, 30.593

TOWN_BATCH_SIZE = 1024
ONLINE_POI_NEIGHBOR_RADIUS_M = 500.0
ONLINE_ROAD_RADIUS_M = 500.0
ONLINE_ANCHOR_CATEGORY_MAX_DISTANCE_M = 80.0
MACRO_TASK_SEARCH_RADIUS_M = {
    'support_gap_analysis': 1800.0,
    'site_suitability': 2200.0,
    'region_comparison': 3200.0,
    'area_overview': 2500.0,
}
MACRO_TASK_PER_CELL_RADIUS_M = {
    'support_gap_analysis': 700.0,
    'site_suitability': 850.0,
    'region_comparison': 1100.0,
    'area_overview': 900.0,
}
TOWN_CATEGORY_BUCKET_MAP = {
    '购物消费': '零售购物',
    '餐饮美食': '餐饮配套',
    '生活服务': '生活服务',
    '交通设施': '交通出行',
    '医疗保健': '医疗健康',
    '科教文化': '教育服务',
    '休闲娱乐': '休闲娱乐',
    '旅游景点': '休闲娱乐',
    '运动健身': '休闲娱乐',
    '酒店住宿': '生活服务',
    '金融机构': '生活服务',
    '商务住宅': '生活服务',
    '公司企业': '办公商务',
    '汽车相关': '交通出行',
}
AOI_SCENE_TAG_MAP = {
    'university': ['高校周边'],
    'college': ['高校周边'],
    'school': ['校园周边'],
    'commercial': ['商业活跃'],
    'retail': ['商业活跃'],
    'residential': ['居住社区'],
    'industrial': ['产业园区'],
    'hospital': ['医疗服务'],
    'parking': ['交通换乘'],
    'park': ['公共休闲'],
    'pitch': ['运动休闲'],
    'forest': ['自然休闲'],
    'water': ['滨水空间'],
    'grass': ['自然休闲'],
}
LANDUSE_SCENE_TAG_MAP = {
    '商业服务用地': ['商业混合'],
    '居住用地': ['居住社区'],
    '教育用地': ['教育片区'],
    '医疗卫生用地': ['医疗服务'],
    '公园与绿地用地': ['公共休闲'],
    '河流湖泊': ['滨水空间'],
    '体育与文化用地': ['文化休闲'],
    '工业用地': ['产业园区'],
}
DOMINANT_CATEGORY_SCENE_TAG_MAP = {
    '购物消费': ['零售密集'],
    '餐饮美食': ['餐饮活跃'],
    '交通设施': ['交通节点'],
    '科教文化': ['教育氛围'],
    '医疗保健': ['医疗服务'],
    '休闲娱乐': ['休闲活动'],
    '旅游景点': ['旅游休闲'],
}
TEXT_EMBEDDING_DIM = 32
TEXT_TOKEN_PATTERN = re.compile(r'[A-Za-z0-9\u4e00-\u9fff]+')
TEXT_EMBEDDING_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('高校', ('大学', '学院', '学校', '校区', '高校')),
    ('咖啡', ('咖啡', '咖啡店', '咖啡馆', 'coffee', 'cafe')),
    ('地铁', ('地铁', '地铁站', '站口', '出口', 'metro', 'subway')),
    ('餐饮', ('餐饮', '美食', '餐厅', '小吃', '轻食')),
    ('商圈', ('商圈', '商业', '广场', 'mall', '商场')),
    ('居住', ('社区', '小区', '住宅')),
    ('办公', ('办公', '写字楼', '公司', '商务')),
    ('公园', ('公园', '绿地', '滨水', '步道')),
)


class ModelRuntime:
    def __init__(self, name: str, checkpoint_path_obj: Path, expected_architecture: str) -> None:
        self.name = name
        self.checkpoint_path_obj = checkpoint_path_obj
        self.expected_architecture = expected_architecture
        self.model: Optional[torch.nn.Module] = None
        self.loaded: bool = False
        self.architecture: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.embedding_dim: Optional[int] = None
        self.startup_error: Optional[str] = None
        self.item_count: Optional[int] = None


class EncoderState:
    def __init__(self) -> None:
        self.device: str = 'cpu'
        self.loaded: bool = False
        self.startup_error: Optional[str] = None
        self.supported_features: Tuple[str, ...] = (
            'direction',
            'region',
            'encode',
            'encode_text',
            'enrich',
            'cell_context',
            'cell_search',
        )
        self.models: Dict[str, ModelRuntime] = {
            'poi': ModelRuntime(
                name='poi',
                checkpoint_path_obj=POI_CHECKPOINT_PATH,
                expected_architecture='ultimate',
            ),
            'town': ModelRuntime(
                name='town',
                checkpoint_path_obj=TOWN_CHECKPOINT_PATH,
                expected_architecture='mlp',
            ),
        }

        # 向后兼容的主模型字段：默认为 poi_encoder
        self.model: Optional[torch.nn.Module] = None
        self.architecture: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.embedding_dim: Optional[int] = None

        # town/cell 运行时索引
        self.town_coords: Optional[np.ndarray] = None
        self.town_embeddings: Optional[np.ndarray] = None
        self.town_region_probs: Optional[np.ndarray] = None
        self.town_cells: List[Dict[str, Any]] = []

        # POI 离线同构特征缓存（exact-anchor 直读）
        self.poi_feature_cache: Optional[Dict[str, np.ndarray]] = None
        self.poi_feature_index: Dict[int, int] = {}
        self.poi_feature_cache_loaded: bool = False
        self.poi_feature_cache_count: int = 0
        self.poi_feature_cache_path: Optional[str] = None
        self.poi_feature_cache_error: Optional[str] = None


state = EncoderState()


def reset_encoder_state() -> None:
    state.device = 'cpu'
    state.loaded = False
    state.startup_error = None
    state.model = None
    state.architecture = None
    state.checkpoint_path = None
    state.embedding_dim = None
    state.town_coords = None
    state.town_embeddings = None
    state.town_region_probs = None
    state.town_cells = []
    state.poi_feature_cache = None
    state.poi_feature_index = {}
    state.poi_feature_cache_loaded = False
    state.poi_feature_cache_count = 0
    state.poi_feature_cache_path = None
    state.poi_feature_cache_error = None

    for runtime in state.models.values():
        runtime.model = None
        runtime.loaded = False
        runtime.architecture = None
        runtime.checkpoint_path = None
        runtime.embedding_dim = None
        runtime.startup_error = None
        runtime.item_count = None


def extract_checkpoint_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if isinstance(checkpoint, (dict, OrderedDict)):
        return checkpoint
    raise TypeError(f'Unsupported checkpoint payload: {type(checkpoint).__name__}')


def infer_checkpoint_architecture(state_dict: Dict[str, Any]) -> str:
    keys = tuple(state_dict.keys())
    if any(key.startswith('spatial_attention') for key in keys):
        return 'ultimate'
    if any(key.startswith('prototype_learning') for key in keys):
        return 'ultimate'
    if any(key.startswith('coord_head') for key in keys):
        return 'ultimate'
    if any(key.startswith('coord_reconstruct_head') for key in keys):
        return 'mlp'
    return 'mlp'


def build_encoder_for_architecture(architecture: str):
    if architecture == 'ultimate':
        return build_ultimate_encoder(DEFAULT_PRO_CONFIG)
    return build_mlp_encoder(DEFAULT_PRO_CONFIG)


def build_model_payload(runtime: Any) -> Dict[str, Any]:
    item_count = getattr(runtime, 'item_count', None)
    startup_error = getattr(runtime, 'startup_error', None)
    is_town_runtime = getattr(runtime, 'name', None) == 'town'
    healthy_loaded = getattr(runtime, 'loaded', False) is True and not startup_error
    if is_town_runtime:
        healthy_loaded = healthy_loaded and item_count is not None and int(item_count or 0) > 0

    return {
        'loaded': healthy_loaded,
        'architecture': getattr(runtime, 'architecture', None),
        'checkpoint_path': getattr(runtime, 'checkpoint_path', None),
        'embedding_dim': getattr(runtime, 'embedding_dim', None),
        'startup_error': startup_error,
        'item_count': item_count,
    }


def build_health_payload() -> Dict[str, Any]:
    model_payloads = {
        name: build_model_payload(runtime)
        for name, runtime in state.models.items()
    }
    all_loaded = all(item['loaded'] for item in model_payloads.values())
    any_loaded = any(item['loaded'] for item in model_payloads.values())
    status = 'ok' if all_loaded else ('partial' if any_loaded else 'encoder_not_loaded')

    return {
        'status': status,
        'encoder_loaded': all_loaded,
        'device': state.device,
        'architecture': state.architecture,
        'checkpoint_path': state.checkpoint_path,
        'embedding_dim': state.embedding_dim,
        'supported_features': list(state.supported_features),
        'startup_error': state.startup_error,
        'models': model_payloads,
        'poi_feature_cache': {
            'loaded': state.poi_feature_cache_loaded,
            'count': state.poi_feature_cache_count,
            'path': state.poi_feature_cache_path,
            'error': state.poi_feature_cache_error,
        },
    }


def rebuild_town_index() -> bool:
    town_runtime = state.models['town']
    if not town_runtime.loaded:
        town_runtime.startup_error = town_runtime.startup_error or 'town_encoder_not_loaded'
        state.loaded = False
        return False

    town_index_ok = load_town_index()
    poi_runtime = state.models['poi']
    poi_ready = poi_runtime.loaded and not poi_runtime.startup_error
    state.loaded = poi_ready and town_index_ok

    if state.loaded:
        state.startup_error = None
        return True

    errors = []
    for name, runtime in state.models.items():
        if runtime.startup_error:
            errors.append(f'{name}:{runtime.startup_error}')
    if not town_index_ok:
        errors.append('town:cell_index_unavailable')
    state.startup_error = '; '.join(errors) if errors else 'dual_model_not_ready'
    return False


def load_single_model(runtime: ModelRuntime) -> bool:
    if not runtime.checkpoint_path_obj.exists():
        runtime.startup_error = f'checkpoint_not_found:{runtime.checkpoint_path_obj}'
        return False

    try:
        checkpoint = torch.load(runtime.checkpoint_path_obj, map_location=state.device, weights_only=False)
        state_dict = extract_checkpoint_state_dict(checkpoint)
        architecture = infer_checkpoint_architecture(state_dict)
        model = build_encoder_for_architecture(architecture)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(state.device)
        model.eval()

        runtime.model = model
        runtime.loaded = True
        runtime.architecture = architecture
        runtime.checkpoint_path = str(runtime.checkpoint_path_obj)
        runtime.embedding_dim = int(getattr(model, 'embedding_dim', DEFAULT_PRO_CONFIG.model.embedding_dim))
        runtime.startup_error = None
        return True
    except Exception as error:  # pragma: no cover - exercised via health and live boot
        runtime.startup_error = str(error)
        runtime.loaded = False
        runtime.model = None
        return False


def load_poi_feature_cache(cache_path: Optional[Path] = None) -> bool:
    path_obj = Path(cache_path or POI_FEATURE_CACHE_PATH)
    state.poi_feature_cache = None
    state.poi_feature_index = {}
    state.poi_feature_cache_loaded = False
    state.poi_feature_cache_count = 0
    state.poi_feature_cache_path = str(path_obj)
    state.poi_feature_cache_error = None

    if not path_obj.exists():
        state.poi_feature_cache_error = f'cache_not_found:{path_obj}'
        return False

    try:
        with np.load(path_obj, allow_pickle=False) as cache_data:
            missing_keys = [key for key in POI_FEATURE_CACHE_KEYS if key not in cache_data]
            if missing_keys:
                raise ValueError(f'cache_missing_keys:{",".join(missing_keys)}')

            poi_ids = np.asarray(cache_data['poi_ids'], dtype=np.int64)
            feature_cache = {
                'poi_ids': poi_ids,
                'point_features': np.asarray(cache_data['point_features'], dtype=np.float32),
                'line_features': np.asarray(cache_data['line_features'], dtype=np.float32),
                'polygon_features': np.asarray(cache_data['polygon_features'], dtype=np.float32),
                'direction_features': np.asarray(cache_data['direction_features'], dtype=np.float32),
            }

        if poi_ids.ndim != 1:
            raise ValueError(f'cache_invalid_poi_ids_shape:{poi_ids.shape}')

        total_count = int(poi_ids.shape[0])
        if total_count <= 0:
            raise ValueError('cache_empty')

        for feature_key, expected_dim in POI_FEATURE_CACHE_DIMENSIONS.items():
            feature_value = feature_cache[feature_key]
            if feature_value.ndim != 2:
                raise ValueError(f'cache_invalid_feature_rank:{feature_key}:{feature_value.ndim}')
            if feature_value.shape[0] != total_count:
                raise ValueError(
                    f'cache_length_mismatch:{feature_key}:{feature_value.shape[0]}!={total_count}'
                )
            if feature_value.shape[1] != expected_dim:
                raise ValueError(
                    f'cache_invalid_feature_dim:{feature_key}:{feature_value.shape[1]}!={expected_dim}'
                )

        poi_feature_index: Dict[int, int] = {}
        for row_index, raw_poi_id in enumerate(poi_ids.tolist()):
            poi_id = int(raw_poi_id)
            if poi_id in poi_feature_index:
                raise ValueError(f'cache_duplicate_poi_id:{poi_id}')
            poi_feature_index[poi_id] = row_index

        state.poi_feature_cache = feature_cache
        state.poi_feature_index = poi_feature_index
        state.poi_feature_cache_loaded = True
        state.poi_feature_cache_count = total_count
        state.poi_feature_cache_error = None
        return True
    except Exception as error:
        state.poi_feature_cache = None
        state.poi_feature_index = {}
        state.poi_feature_cache_loaded = False
        state.poi_feature_cache_count = 0
        state.poi_feature_cache_error = str(error)
        return False


def _norm_poi_coord(lon: float, lat: float) -> Tuple[float, float]:
    return (
        (lon - POI_LNG_MIN) / (POI_LNG_MAX - POI_LNG_MIN),
        (lat - POI_LAT_MIN) / (POI_LAT_MAX - POI_LAT_MIN),
    )


def _direction_onehot_for_coords(lon: float, lat: float) -> np.ndarray:
    dx = lon - CITY_CENTER_LON
    dy = lat - CITY_CENTER_LAT
    angle = np.arctan2(dy, dx)
    direction_idx = int((angle + np.pi) / (np.pi / 4)) % 8
    direction_feat = np.zeros(8, dtype=np.float32)
    direction_feat[direction_idx] = 1.0
    return direction_feat


def infer_anchor_category_from_point_feature(point_feat: np.ndarray) -> Optional[str]:
    own_category_slice = np.asarray(point_feat[20:32], dtype=np.float32)
    if own_category_slice.size == 0:
        return None
    max_value = float(np.max(own_category_slice))
    if max_value <= 0.0:
        return None
    return IDX_TO_CATEGORY.get(int(np.argmax(own_category_slice)))


def lookup_poi_feature_cache(poi_id: int) -> Optional[Dict[str, Any]]:
    if not state.poi_feature_cache_loaded or not state.poi_feature_cache:
        return None

    cache_index = state.poi_feature_index.get(int(poi_id))
    if cache_index is None:
        return None

    cache = state.poi_feature_cache
    return {
        'cache_index': cache_index,
        'point_features': np.array(cache['point_features'][cache_index], dtype=np.float32, copy=True),
        'line_features': np.array(cache['line_features'][cache_index], dtype=np.float32, copy=True),
        'polygon_features': np.array(cache['polygon_features'][cache_index], dtype=np.float32, copy=True),
        'direction_features': np.array(cache['direction_features'][cache_index], dtype=np.float32, copy=True),
    }


def build_cached_context_applied(
    point_feat: np.ndarray,
    line_feat: np.ndarray,
    polygon_feat: np.ndarray,
    anchor_category_main: Optional[str],
) -> Dict[str, bool]:
    return {
        'poi_neighbors': bool(float(point_feat[2]) > 0.0 or np.any(point_feat[3:19] > 0.0)),
        'roads': bool(float(line_feat[2]) > 0.0 or np.any(line_feat[3:8] > 0.0)),
        'landuse': bool(float(polygon_feat[2]) > 0.0 or np.any(polygon_feat[3:] > 0.0)),
        'anchor_category': anchor_category_main is not None,
    }


def get_postgis_connection():
    import psycopg2

    source = default_postgis_source
    return psycopg2.connect(
        host=source.host,
        port=source.port,
        user=source.user,
        password=source.password,
        database=source.database,
    )


def fetch_poi_anchor_row(poi_id: int) -> Optional[Dict[str, Any]]:
    try:
        from psycopg2.extras import RealDictCursor

        conn = get_postgis_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        id,
                        name,
                        category_main,
                        longitude AS lon,
                        latitude AS lat
                    FROM pois
                    WHERE id = %s
                    LIMIT 1
                """, (poi_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()
    except Exception:
        return None


def fetch_online_poi_feature_context(
    lon: float,
    lat: float,
    *,
    exclude_poi_id: Optional[int] = None,
) -> Dict[str, Any]:
    point_sql = 'ST_SetSRID(ST_MakePoint(%s, %s), 4326)'
    context = {
        'nearby_pois': [],
        'road_rows': [],
        'landuse_row': None,
        'query_error': None,
    }

    try:
        from psycopg2.extras import RealDictCursor

        conn = get_postgis_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                poi_query = f"""
                    SELECT
                        id,
                        category_main,
                        ST_Distance(geom::geography, {point_sql}::geography) AS distance_m
                    FROM pois
                    WHERE geom IS NOT NULL
                      AND geom && ST_Expand({point_sql}, 0.01)
                      AND ST_DWithin(geom::geography, {point_sql}::geography, %s)
                """
                poi_params: List[Any] = [lon, lat, lon, lat, lon, lat, ONLINE_POI_NEIGHBOR_RADIUS_M]
                if exclude_poi_id is not None:
                    poi_query += " AND id <> %s"
                    poi_params.append(exclude_poi_id)
                poi_query += """
                    ORDER BY distance_m ASC
                    LIMIT 256
                """
                cur.execute(poi_query, poi_params)
                context['nearby_pois'] = [dict(row) for row in cur.fetchall()]

                cur.execute(f"""
                    SELECT
                        COALESCE(fclass, 'unknown') AS fclass,
                        COUNT(*)::int AS count
                    FROM roads
                    WHERE geom IS NOT NULL
                      AND geom && ST_Expand({point_sql}, 0.01)
                      AND ST_DWithin(geom::geography, {point_sql}::geography, %s)
                    GROUP BY COALESCE(fclass, 'unknown')
                """, (lon, lat, lon, lat, ONLINE_ROAD_RADIUS_M))
                context['road_rows'] = [dict(row) for row in cur.fetchall()]

                cur.execute(f"""
                    SELECT land_type, area_sqm
                    FROM landuse
                    WHERE geom IS NOT NULL
                      AND ST_Within({point_sql}, geom)
                    ORDER BY area_sqm ASC
                    LIMIT 1
                """, (lon, lat))
                landuse_row = cur.fetchone()
                context['landuse_row'] = dict(landuse_row) if landuse_row else None
        finally:
            conn.close()
    except Exception as error:  # pragma: no cover - exercised in live runtime
        context['query_error'] = str(error)

    return context


def build_online_poi_features(
    lon: float,
    lat: float,
    nearby_pois: Optional[List[Dict[str, Any]]] = None,
    road_rows: Optional[List[Dict[str, Any]]] = None,
    landuse_row: Optional[Dict[str, Any]] = None,
    *,
    anchor_category_override: Optional[str] = None,
    anchor_category_distance_m_override: Optional[float] = None,
    feature_source_override: Optional[str] = None,
    feature_stats_overrides: Optional[Dict[str, Any]] = None,
    query_error: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    norm_lon, norm_lat = _norm_poi_coord(lon, lat)

    point_feat = np.zeros(32, dtype=np.float32)
    point_feat[0] = norm_lon
    point_feat[1] = norm_lat

    line_feat = np.zeros(16, dtype=np.float32)
    line_feat[0] = norm_lon
    line_feat[1] = norm_lat

    polygon_feat = np.zeros(16, dtype=np.float32)
    polygon_feat[0] = norm_lon
    polygon_feat[1] = norm_lat

    direction_feat = _direction_onehot_for_coords(lon, lat)

    safe_nearby_pois = [dict(item) for item in (nearby_pois or []) if item]
    safe_road_rows = [dict(item) for item in (road_rows or []) if item]
    safe_landuse_row = dict(landuse_row) if landuse_row else None

    neighbor_count = len(safe_nearby_pois)
    point_feat[2] = np.log1p(neighbor_count) / 10.0

    cat_counts = np.zeros(16, dtype=np.float32)
    for poi in safe_nearby_pois:
        category_main = str(poi.get('category_main') or '').strip()
        if category_main in CATEGORY_TO_IDX:
            cat_counts[CATEGORY_TO_IDX[category_main]] += 1
    cat_dist = cat_counts / neighbor_count if neighbor_count > 0 else cat_counts
    point_feat[3:19] = cat_dist

    nz = cat_dist[cat_dist > 0]
    if len(nz) > 1:
        point_feat[19] = float(-np.sum(nz * np.log(nz + 1e-8)))

    anchor_category_main = None
    anchor_category_distance_m = None
    if anchor_category_override and anchor_category_override in CATEGORY_TO_IDX:
        anchor_category_main = anchor_category_override
        anchor_category_distance_m = anchor_category_distance_m_override
        category_idx = CATEGORY_TO_IDX[anchor_category_override]
        if category_idx < 12:
            point_feat[20 + category_idx] = 1.0
    elif safe_nearby_pois:
        nearest_poi = min(
            safe_nearby_pois,
            key=lambda item: float(item.get('distance_m', float('inf'))),
        )
        nearest_distance = float(nearest_poi.get('distance_m', float('inf')))
        nearest_category = str(nearest_poi.get('category_main') or '').strip()
        if nearest_category in CATEGORY_TO_IDX and nearest_distance <= ONLINE_ANCHOR_CATEGORY_MAX_DISTANCE_M:
            anchor_category_main = nearest_category
            anchor_category_distance_m = nearest_distance
            category_idx = CATEGORY_TO_IDX[nearest_category]
            if category_idx < 12:
                point_feat[20 + category_idx] = 1.0

    road_count = 0
    rc_counts = np.zeros(5, dtype=np.float32)
    for row in safe_road_rows:
        row_count = max(0, int(row.get('count') or 0))
        road_count += row_count
        road_class = str(row.get('fclass') or '').strip()
        if road_class in ROAD_CLASS_TO_IDX:
            rc_counts[ROAD_CLASS_TO_IDX[road_class]] += row_count
    line_feat[2] = np.log1p(road_count) / 5.0
    if road_count > 0:
        line_feat[3:8] = rc_counts / road_count

    landuse_type = None
    landuse_area_sqm = None
    if safe_landuse_row:
        landuse_type = str(safe_landuse_row.get('land_type') or '').strip() or None
        landuse_area_sqm = float(safe_landuse_row.get('area_sqm') or 0.0)
        polygon_feat[2] = min(landuse_area_sqm / 1e6, 10.0)
        if landuse_type in LANDUSE_TO_IDX:
            polygon_feat[3 + LANDUSE_TO_IDX[landuse_type]] = 1.0

    context_applied = {
        'poi_neighbors': neighbor_count > 0,
        'roads': road_count > 0,
        'landuse': landuse_type is not None,
        'anchor_category': anchor_category_main is not None,
    }
    context_signal_count = sum(1 for applied in context_applied.values() if applied)
    feature_source = feature_source_override or (
        'poi_online_context_v2' if context_signal_count > 0 else 'coordinate_only_fallback_v1'
    )

    feature_stats = {
        'neighbor_poi_count': neighbor_count,
        'road_count': road_count,
        'landuse_type': landuse_type,
        'landuse_area_sqm': landuse_area_sqm,
        'anchor_category_main': anchor_category_main,
        'anchor_category_distance_m': anchor_category_distance_m,
        'context_applied': context_applied,
        'query_error': query_error,
    }
    if feature_stats_overrides:
        feature_stats.update(feature_stats_overrides)

    feature_meta = {
        'feature_source': feature_source,
        'feature_stats': feature_stats,
    }

    return point_feat, line_feat, polygon_feat, direction_feat, feature_meta


def build_poi_features_for_coords(
    lon: float,
    lat: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    context = fetch_online_poi_feature_context(lon, lat)
    return build_online_poi_features(
        lon,
        lat,
        nearby_pois=context.get('nearby_pois'),
        road_rows=context.get('road_rows'),
        landuse_row=context.get('landuse_row'),
        query_error=context.get('query_error'),
    )


def build_poi_features_for_anchor(
    lon: float,
    lat: float,
    *,
    poi_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if poi_id is None:
        return build_poi_features_for_coords(lon, lat)

    anchor_row = fetch_poi_anchor_row(poi_id)
    cached_features = lookup_poi_feature_cache(poi_id)

    if cached_features is not None:
        point_feat = cached_features['point_features']
        line_feat = cached_features['line_features']
        polygon_feat = cached_features['polygon_features']
        direction_feat = cached_features['direction_features']
        anchor_category_main = (
            str((anchor_row or {}).get('category_main') or '').strip()
            or infer_anchor_category_from_point_feature(point_feat)
        )
        feature_meta = {
            'feature_source': 'poi_offline_exact_v1',
            'feature_stats': {
                'neighbor_poi_count': None,
                'road_count': None,
                'landuse_type': None,
                'landuse_area_sqm': None,
                'anchor_category_main': anchor_category_main,
                'anchor_category_distance_m': 0.0 if anchor_category_main else None,
                'context_applied': build_cached_context_applied(
                    point_feat,
                    line_feat,
                    polygon_feat,
                    anchor_category_main,
                ),
                'query_error': None,
                'anchor_poi_id': int((anchor_row or {}).get('id') or poi_id),
                'anchor_poi_name': (anchor_row or {}).get('name'),
                'query_lon': lon,
                'query_lat': lat,
                'offline_cache_hit': True,
                'offline_cache_index': cached_features['cache_index'],
                'offline_cache_path': state.poi_feature_cache_path,
            },
        }
        return point_feat, line_feat, polygon_feat, direction_feat, feature_meta

    if not anchor_row:
        return build_poi_features_for_coords(lon, lat)

    anchor_lon = float(anchor_row.get('lon') or lon)
    anchor_lat = float(anchor_row.get('lat') or lat)
    anchor_category_main = str(anchor_row.get('category_main') or '').strip() or None

    context = fetch_online_poi_feature_context(
        anchor_lon,
        anchor_lat,
        exclude_poi_id=poi_id,
    )
    filtered_nearby_pois = [
        dict(item)
        for item in (context.get('nearby_pois') or [])
        if item and item.get('id') != poi_id
    ]

    return build_online_poi_features(
        anchor_lon,
        anchor_lat,
        nearby_pois=filtered_nearby_pois,
        road_rows=context.get('road_rows'),
        landuse_row=context.get('landuse_row'),
        anchor_category_override=anchor_category_main,
        anchor_category_distance_m_override=0.0 if anchor_category_main else None,
        feature_source_override='poi_exact_anchor_v1',
        feature_stats_overrides={
            'anchor_poi_id': anchor_row.get('id'),
            'anchor_poi_name': anchor_row.get('name'),
            'query_lon': lon,
            'query_lat': lat,
            'offline_cache_hit': False,
            'offline_cache_path': state.poi_feature_cache_path,
        },
        query_error=context.get('query_error'),
    )


def build_coordinate_only_poi_features(lon: float, lat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm_lon = (lon - POI_LNG_MIN) / (POI_LNG_MAX - POI_LNG_MIN)
    norm_lat = (lat - POI_LAT_MIN) / (POI_LAT_MAX - POI_LAT_MIN)

    point_feat = np.zeros(32, dtype=np.float32)
    point_feat[0] = norm_lon
    point_feat[1] = norm_lat

    line_feat = np.zeros(16, dtype=np.float32)
    line_feat[0] = norm_lon
    line_feat[1] = norm_lat

    polygon_feat = np.zeros(16, dtype=np.float32)
    polygon_feat[0] = norm_lon
    polygon_feat[1] = norm_lat

    dx = lon - CITY_CENTER_LON
    dy = lat - CITY_CENTER_LAT
    angle = np.arctan2(dy, dx)
    direction_idx = int((angle + np.pi) / (np.pi / 4)) % 8
    direction_feat = np.zeros(8, dtype=np.float32)
    direction_feat[direction_idx] = 1.0

    return point_feat, line_feat, polygon_feat, direction_feat


def run_model_inference(
    runtime: ModelRuntime,
    point_feat: torch.Tensor,
    line_feat: torch.Tensor,
    polygon_feat: torch.Tensor,
    direction_feat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if runtime.model is None:
        raise RuntimeError(f'{runtime.name}_encoder_not_loaded')

    if hasattr(runtime.model, 'forward_simple'):
        return runtime.model.forward_simple(point_feat, line_feat, polygon_feat, direction_feat)

    return runtime.model(point_feat, line_feat, polygon_feat, direction_feat)


def encode_coords(
    lon: float,
    lat: float,
    *,
    poi_id: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    runtime = state.models['poi']
    if not runtime.loaded:
        return None, {
            'feature_source': 'encoder_unavailable',
            'feature_stats': {}
        }

    try:
        point_feat, line_feat, polygon_feat, direction_feat, feature_meta = build_poi_features_for_anchor(
            lon,
            lat,
            poi_id=poi_id,
        )
        point_tensor = torch.tensor(point_feat, dtype=torch.float32).unsqueeze(0).to(state.device)
        line_tensor = torch.tensor(line_feat, dtype=torch.float32).unsqueeze(0).to(state.device)
        polygon_tensor = torch.tensor(polygon_feat, dtype=torch.float32).unsqueeze(0).to(state.device)
        direction_tensor = torch.tensor(direction_feat, dtype=torch.float32).unsqueeze(0).to(state.device)

        with torch.no_grad():
            embedding, _, _, _ = run_model_inference(
                runtime,
                point_tensor,
                line_tensor,
                polygon_tensor,
                direction_tensor,
            )
        return embedding.cpu().numpy()[0], feature_meta
    except Exception as error:  # pragma: no cover - runtime safety
        runtime.startup_error = str(error)
        fallback_point, fallback_line, fallback_polygon, fallback_direction = build_coordinate_only_poi_features(lon, lat)
        feature_meta = {
            'feature_source': 'coordinate_only_fallback_v1',
            'feature_stats': {
                'neighbor_poi_count': 0,
                'road_count': 0,
                'landuse_type': None,
                'landuse_area_sqm': None,
                'anchor_category_main': None,
                'anchor_category_distance_m': None,
                'context_applied': {
                    'poi_neighbors': False,
                    'roads': False,
                    'landuse': False,
                    'anchor_category': False,
                },
                'query_error': str(error),
            },
        }
        try:
            point_tensor = torch.tensor(fallback_point, dtype=torch.float32).unsqueeze(0).to(state.device)
            line_tensor = torch.tensor(fallback_line, dtype=torch.float32).unsqueeze(0).to(state.device)
            polygon_tensor = torch.tensor(fallback_polygon, dtype=torch.float32).unsqueeze(0).to(state.device)
            direction_tensor = torch.tensor(fallback_direction, dtype=torch.float32).unsqueeze(0).to(state.device)

            with torch.no_grad():
                embedding, _, _, _ = run_model_inference(
                    runtime,
                    point_tensor,
                    line_tensor,
                    polygon_tensor,
                    direction_tensor,
                )
            return embedding.cpu().numpy()[0], feature_meta
        except Exception:
            return None, feature_meta


def compute_direction(lon1: float, lat1: float, lon2: float, lat2: float) -> Tuple[int, str]:
    dx = lon2 - lon1
    dy = lat2 - lat1
    avg_lat = (lat1 + lat2) / 2
    dx_corrected = dx * np.cos(np.radians(avg_lat))
    angle_deg = np.degrees(np.arctan2(dy, dx_corrected))
    direction_idx = int((angle_deg + 180 + 22.5) // 45) % 8
    return direction_idx, DIRECTION_NAMES[direction_idx]


def compute_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371000.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(radius * c)


def _prepare_town_model_inputs(dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_features = np.asarray(dataset['point_features'], dtype=np.float32)[:, :32]
    line_features = np.asarray(dataset['line_features'], dtype=np.float32)[:, :16]
    polygon_features = np.asarray(dataset['polygon_features'], dtype=np.float32)[:, :16]
    direction_features = np.asarray(dataset['direction_features'], dtype=np.float32)[:, :8]
    return point_features, line_features, polygon_features, direction_features


def load_town_index() -> bool:
    runtime = state.models['town']
    if not runtime.loaded:
        runtime.startup_error = runtime.startup_error or 'town_encoder_not_loaded'
        return False

    try:
        dataset = load_town_dataset(sample_ratio=1.0)
        point_features, line_features, polygon_features, direction_features = _prepare_town_model_inputs(dataset)

        embeddings: List[np.ndarray] = []
        region_probs: List[np.ndarray] = []
        total_count = len(point_features)

        for start in range(0, total_count, TOWN_BATCH_SIZE):
            end = start + TOWN_BATCH_SIZE
            point_tensor = torch.tensor(point_features[start:end], dtype=torch.float32).to(state.device)
            line_tensor = torch.tensor(line_features[start:end], dtype=torch.float32).to(state.device)
            polygon_tensor = torch.tensor(polygon_features[start:end], dtype=torch.float32).to(state.device)
            direction_tensor = torch.tensor(direction_features[start:end], dtype=torch.float32).to(state.device)

            with torch.no_grad():
                batch_embedding, _, batch_region_pred, _ = run_model_inference(
                    runtime,
                    point_tensor,
                    line_tensor,
                    polygon_tensor,
                    direction_tensor,
                )
                batch_region_probs = torch.softmax(batch_region_pred, dim=-1)

            embeddings.append(batch_embedding.cpu().numpy())
            region_probs.append(batch_region_probs.cpu().numpy())

        state.town_coords = np.asarray(dataset['coords'], dtype=np.float32)
        state.town_embeddings = np.vstack(embeddings).astype(np.float32)
        state.town_region_probs = np.vstack(region_probs).astype(np.float32)
        state.town_cells = []

        metadata = dataset.get('metadata') or []
        for index, meta in enumerate(metadata):
            region_prob = state.town_region_probs[index]
            region_idx = int(np.argmax(region_prob))
            region_name = REGION_NAMES[region_idx] if region_idx < len(REGION_NAMES) else '未知'
            confidence = float(region_prob[region_idx]) if region_idx < len(region_prob) else 0.0
            lon, lat = state.town_coords[index]
            state.town_cells.append({
                'cell_id': meta.get('cell_id', str(index)),
                'lon': float(lon),
                'lat': float(lat),
                'region_idx': region_idx,
                'region_name': region_name,
                'region_confidence': confidence,
                'dominant_category': meta.get('dominant_category'),
                'category_distribution': meta.get('category_distribution') or {},
                'aoi_type': meta.get('aoi_type'),
                'poi_density': meta.get('poi_density'),
                'category_entropy': meta.get('category_entropy'),
                'poi_count': meta.get('poi_count'),
                'landuse_mix': meta.get('landuse_mix'),
                'dominant_landuse': meta.get('dominant_landuse'),
                'aoi_count': meta.get('aoi_count'),
                'road_count': meta.get('road_count'),
                'road_length_km': meta.get('road_length_km'),
                'population_density': meta.get('population_density'),
            })

        runtime.item_count = len(state.town_cells)
        runtime.startup_error = None
        return True
    except Exception as error:  # pragma: no cover - exercised in live boot
        runtime.startup_error = str(error)
        runtime.item_count = None
        state.town_coords = None
        state.town_embeddings = None
        state.town_region_probs = None
        state.town_cells = []
        return False


def load_encoder() -> bool:
    reset_encoder_state()

    if not ENCODER_AVAILABLE:
        state.startup_error = IMPORT_ERROR or 'encoder_module_not_available'
        return False

    state.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_poi_feature_cache()

    poi_ok = load_single_model(state.models['poi'])
    town_ok = load_single_model(state.models['town'])
    town_index_ok = load_town_index() if town_ok else False

    primary_runtime = state.models['poi']
    state.model = primary_runtime.model
    state.architecture = primary_runtime.architecture
    state.checkpoint_path = primary_runtime.checkpoint_path
    state.embedding_dim = primary_runtime.embedding_dim

    state.loaded = poi_ok and town_ok and town_index_ok

    if state.loaded:
        state.startup_error = None
        return True

    errors = []
    for name, runtime in state.models.items():
        if runtime.startup_error:
            errors.append(f'{name}:{runtime.startup_error}')
    if not town_index_ok:
        errors.append('town:cell_index_unavailable')
    state.startup_error = '; '.join(errors) if errors else 'dual_model_not_ready'
    return False


def _ensure_town_index_ready() -> None:
    if state.town_coords is None or state.town_embeddings is None or not state.town_cells:
        raise RuntimeError('town_cell_index_not_ready')


def _compute_cell_distances(lon: float, lat: float) -> np.ndarray:
    _ensure_town_index_ready()
    coords = state.town_coords
    lon1 = np.radians(lat)
    lon2 = np.radians(coords[:, 1])
    dlat = np.radians(coords[:, 1] - lat)
    dlon = np.radians(coords[:, 0] - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lon1) * np.cos(lon2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371000.0 * c


def find_nearest_cell_index(lon: float, lat: float) -> Tuple[int, float]:
    distances = _compute_cell_distances(lon, lat)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def build_cell_context(index: int, distance_m: float, anchor_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
    cell = state.town_cells[index]
    embedding = state.town_embeddings[index]
    similarity = None
    if anchor_embedding is not None:
        similarity = float(np.dot(anchor_embedding, embedding))

    return {
        'cell_id': cell['cell_id'],
        'lon': cell['lon'],
        'lat': cell['lat'],
        'distance_m': distance_m,
        'region_idx': cell['region_idx'],
        'region_name': cell['region_name'],
        'region_confidence': cell['region_confidence'],
        'dominant_category': cell.get('dominant_category'),
        'category_distribution': cell.get('category_distribution') or {},
        'aoi_type': cell.get('aoi_type'),
        'poi_density': cell.get('poi_density'),
        'category_entropy': cell.get('category_entropy'),
        'poi_count': cell.get('poi_count'),
        'landuse_mix': cell.get('landuse_mix'),
        'dominant_landuse': cell.get('dominant_landuse'),
        'aoi_count': cell.get('aoi_count'),
        'road_count': cell.get('road_count'),
        'road_length_km': cell.get('road_length_km'),
        'population_density': cell.get('population_density'),
        'similarity': similarity,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)

    if np.isnan(numeric) or np.isinf(numeric):
        return float(default)
    return numeric


def _normalize_category_distribution(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}

    normalized: Dict[str, float] = {}
    for key, raw_score in value.items():
        category_name = str(key or '').strip()
        if not category_name:
            continue
        score = _safe_float(raw_score, 0.0)
        if score <= 0:
            continue
        normalized[category_name] = score

    total_score = sum(normalized.values())
    if total_score <= 0:
        return {}

    return {
        key: float(score / total_score)
        for key, score in normalized.items()
    }


def _map_category_to_support_bucket(category_name: Any) -> str:
    return TOWN_CATEGORY_BUCKET_MAP.get(str(category_name or '').strip(), '其他配套')


def _build_support_bucket_distribution(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket_stats: Dict[str, Dict[str, Any]] = {}

    for cell in cells:
        search_score = max(0.0, min(1.0, _safe_float(cell.get('search_score') or cell.get('similarity'), 0.0)))
        poi_weight = min(max(_safe_float(cell.get('poi_count'), 0.0) / 20.0, 0.65), 2.5)
        cell_weight = max(0.45, 0.6 + search_score * 0.9) * poi_weight
        category_distribution = _normalize_category_distribution(cell.get('category_distribution'))

        bucket_weights: Dict[str, float] = {}
        if category_distribution:
            for category_name, probability in category_distribution.items():
                bucket_name = _map_category_to_support_bucket(category_name)
                if bucket_name == '其他配套':
                    continue
                bucket_weights[bucket_name] = bucket_weights.get(bucket_name, 0.0) + probability * cell_weight

        if not bucket_weights:
            bucket_name = _map_category_to_support_bucket(cell.get('dominant_category'))
            if bucket_name != '其他配套':
                bucket_weights[bucket_name] = cell_weight

        for bucket_name, weight in bucket_weights.items():
            if bucket_name not in bucket_stats:
                bucket_stats[bucket_name] = {
                    'bucket': bucket_name,
                    'weight': 0.0,
                    'contributors': 0,
                    'examples': [],
                    'representative_categories': [],
                }

            current = bucket_stats[bucket_name]
            current['weight'] += weight
            current['contributors'] += 1

            dominant_category = str(cell.get('dominant_category') or '').strip()
            if dominant_category and dominant_category not in current['representative_categories'] and len(current['representative_categories']) < 3:
                current['representative_categories'].append(dominant_category)

            example_label = dominant_category or str(cell.get('aoi_type') or '').strip() or str(cell.get('region_name') or '').strip()
            if example_label and example_label not in current['examples'] and len(current['examples']) < 3:
                current['examples'].append(example_label)

    aggregated: List[Dict[str, Any]] = []
    for item in bucket_stats.values():
        count = max(item['contributors'], int(round(item['weight'])))
        aggregated.append({
            'bucket': item['bucket'],
            'count': int(max(1, count)),
            'examples': item['examples'],
            'representative_categories': item['representative_categories'],
            'score': round(item['weight'], 4),
        })

    aggregated.sort(key=lambda item: (-item['count'], -_safe_float(item.get('score')), item['bucket']))
    return aggregated


def _build_scene_tags(cells: List[Dict[str, Any]]) -> List[str]:
    tag_scores: Dict[str, float] = {}

    for cell in cells:
        weight = max(0.35, 0.5 + max(0.0, min(1.0, _safe_float(cell.get('search_score') or cell.get('similarity'), 0.0))))

        for tag in AOI_SCENE_TAG_MAP.get(str(cell.get('aoi_type') or '').strip(), []):
            tag_scores[tag] = tag_scores.get(tag, 0.0) + weight

        for tag in LANDUSE_SCENE_TAG_MAP.get(str(cell.get('dominant_landuse') or '').strip(), []):
            tag_scores[tag] = tag_scores.get(tag, 0.0) + weight * 0.8

        for tag in DOMINANT_CATEGORY_SCENE_TAG_MAP.get(str(cell.get('dominant_category') or '').strip(), []):
            tag_scores[tag] = tag_scores.get(tag, 0.0) + weight * 0.7

        if _safe_float(cell.get('landuse_mix'), 0.0) >= 1.25:
            tag_scores['混合业态'] = tag_scores.get('混合业态', 0.0) + 0.6
        if _safe_float(cell.get('population_density'), 0.0) >= 22000:
            tag_scores['高密度活动'] = tag_scores.get('高密度活动', 0.0) + 0.5

    return [
        tag
        for tag, _ in sorted(tag_scores.items(), key=lambda item: (-item[1], item[0]))[:6]
    ]


def _build_cell_mix(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mix_map: Dict[str, int] = {}

    for cell in cells:
        label = str(cell.get('region_name') or '').strip()
        if not label:
            continue
        mix_map[label] = mix_map.get(label, 0) + 1

    total_count = sum(mix_map.values())
    mix = []
    for label, count in sorted(mix_map.items(), key=lambda item: (-item[1], item[0])):
        ratio = float(count / total_count) if total_count > 0 else 0.0
        mix.append({
            'label': label,
            'count': int(count),
            'ratio': round(ratio, 4),
        })

    return mix


def _build_macro_uncertainty(cells: List[Dict[str, Any]], support_buckets: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_size = len(cells)
    mean_similarity = float(np.mean([
        _safe_float(cell.get('similarity'), 0.0)
        for cell in cells
    ])) if cells else 0.0
    mean_region_confidence = float(np.mean([
        _safe_float(cell.get('region_confidence'), 0.0)
        for cell in cells
    ])) if cells else 0.0
    consistency_score = round(max(0.0, min(1.0, mean_similarity * 0.58 + mean_region_confidence * 0.42)), 4)

    if sample_size >= 4 and consistency_score >= 0.72:
        evidence_density = 'high'
    elif sample_size >= 2 and consistency_score >= 0.52:
        evidence_density = 'medium'
    else:
        evidence_density = 'low'

    return {
        'sample_size': sample_size,
        'support_bucket_count': len(support_buckets),
        'evidence_density': evidence_density,
        'low_sample_warning': sample_size < 2 or (sample_size < 3 and consistency_score < 0.55),
        'mean_similarity': round(mean_similarity, 4),
        'mean_region_confidence': round(mean_region_confidence, 4),
        'consistency_score': consistency_score,
    }


def build_macro_cell_summary(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    support_buckets = _build_support_bucket_distribution(cells)
    return {
        'support_bucket_distribution': support_buckets,
        'dominant_buckets': [item['bucket'] for item in support_buckets[:3]],
        'scene_tags': _build_scene_tags(cells),
        'cell_mix': _build_cell_mix(cells),
        'macro_uncertainty': _build_macro_uncertainty(cells, support_buckets),
    }


def get_cell_context_for_point(lon: float, lat: float, anchor_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
    index, distance_m = find_nearest_cell_index(lon, lat)
    return build_cell_context(index, distance_m, anchor_embedding=anchor_embedding)


def normalize_task_type(task_type: Optional[str]) -> str:
    return str(task_type or '').strip().lower()


def resolve_cell_search_radius(task_type: Optional[str]) -> float:
    normalized_task_type = normalize_task_type(task_type)
    return float(MACRO_TASK_SEARCH_RADIUS_M.get(normalized_task_type, 1600.0))


def resolve_per_cell_poi_radius(task_type: Optional[str]) -> float:
    normalized_task_type = normalize_task_type(task_type)
    return float(MACRO_TASK_PER_CELL_RADIUS_M.get(normalized_task_type, 750.0))


def compute_embedding_cosine(anchor_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    anchor_norm = float(np.linalg.norm(anchor_embedding))
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    safe_denominator = np.maximum(anchor_norm * embedding_norms, 1e-8)
    similarities = embeddings @ anchor_embedding
    return similarities / safe_denominator


def search_similar_cells(
    lon: float,
    lat: float,
    task_type: Optional[str] = None,
    top_k: int = 5,
    max_distance_m: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float, Dict[str, Any]]:
    _ensure_town_index_ready()

    safe_top_k = max(1, min(int(top_k or 5), 12))
    search_radius_m = float(max_distance_m or resolve_cell_search_radius(task_type))

    anchor_index, anchor_distance = find_nearest_cell_index(lon, lat)
    anchor_embedding = state.town_embeddings[anchor_index]
    anchor_context = build_cell_context(anchor_index, anchor_distance, anchor_embedding=anchor_embedding)
    anchor_context['similarity'] = 1.0
    anchor_context['search_score'] = 1.0
    anchor_context['rank'] = 1

    distances = _compute_cell_distances(lon, lat)
    similarities = compute_embedding_cosine(anchor_embedding, state.town_embeddings)

    candidate_indices = np.where(distances <= search_radius_m)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.argsort(distances)[:safe_top_k]

    candidate_distances = distances[candidate_indices]
    candidate_similarities = similarities[candidate_indices]
    similarity_scores = np.clip((candidate_similarities + 1.0) / 2.0, 0.0, 1.0)
    distance_scores = np.clip(
        1.0 - np.minimum(candidate_distances, search_radius_m) / max(search_radius_m, 1.0),
        0.0,
        1.0,
    )
    search_scores = similarity_scores * 0.62 + distance_scores * 0.38
    ranking = np.argsort(-search_scores)

    cells: List[Dict[str, Any]] = []
    for order_index in ranking[:safe_top_k]:
        cell_index = int(candidate_indices[order_index])
        context = build_cell_context(
            cell_index,
            float(distances[cell_index]),
            anchor_embedding=anchor_embedding,
        )
        context['similarity'] = float(candidate_similarities[order_index])
        context['search_score'] = float(search_scores[order_index])
        context['rank'] = len(cells) + 1
        cells.append(context)

    macro_summary = build_macro_cell_summary(cells)

    return anchor_context, cells, search_radius_m, macro_summary


def predict_region(lon: float, lat: float) -> Tuple[int, str, float]:
    context = get_cell_context_for_point(lon, lat)
    return context['region_idx'], context['region_name'], context['region_confidence']


def _stable_slot_hash(token: str, slot: int) -> int:
    digest = hashlib.md5(f'{token}#{slot}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little', signed=False)


def build_text_embedding(text: str) -> Tuple[np.ndarray, List[str]]:
    normalized_text = str(text or '').strip().lower()
    collapsed_text = re.sub(r'[\s,，。！？?;；:：、/()（）\[\]【】"\'“”‘’·\-—_]+', ' ', normalized_text)
    raw_tokens = [token for token in TEXT_TOKEN_PATTERN.findall(collapsed_text) if token]
    tokens: List[str] = []
    seen_tokens = set()

    def push_token(value: str) -> None:
        safe_value = str(value or '').strip()
        if safe_value and safe_value not in seen_tokens:
            seen_tokens.add(safe_value)
            tokens.append(safe_value)

    for concept, aliases in TEXT_EMBEDDING_HINTS:
        if any(alias in collapsed_text for alias in aliases):
            push_token(concept)

    for token in raw_tokens:
        if len(token) >= 2:
            push_token(token)
        if len(tokens) >= 12:
            break

    if not tokens:
        tokens = ['generic_query']

    vector = np.zeros(TEXT_EMBEDDING_DIM, dtype=np.float32)
    for token_index, token in enumerate(tokens):
        base_weight = 1.0 if token_index < len(TEXT_EMBEDDING_HINTS) else 0.42
        for slot in range(4):
            hashed = _stable_slot_hash(token, slot)
            dim = hashed % TEXT_EMBEDDING_DIM
            sign = 1.0 if ((hashed >> 7) & 1) == 0 else -1.0
            slot_weight = 1.0 if slot == 0 else 0.72 if slot == 1 else 0.46 if slot == 2 else 0.31
            vector[dim] += sign * base_weight * slot_weight

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        vector[0] = 1.0
        norm = 1.0

    vector = (vector / norm).astype(np.float32)
    return vector, tokens


class DirectionRequest(BaseModel):
    user_lon: float
    user_lat: float
    poi_lon: float
    poi_lat: float


class DirectionResponse(BaseModel):
    direction_idx: int
    direction_name: str
    distance_m: float
    angle_deg: float


class RegionRequest(BaseModel):
    lon: float
    lat: float


class RegionResponse(BaseModel):
    region_idx: int
    region_name: str
    confidence: float


class EncodeRequest(BaseModel):
    lon: float
    lat: float
    poi_id: Optional[int] = None


class EncodeResponse(BaseModel):
    embedding: List[float]
    embedding_dim: int
    feature_source: Optional[str] = None
    feature_stats: Dict[str, Any] = Field(default_factory=dict)


class EncodeTextRequest(BaseModel):
    text: str


class EncodeTextResponse(BaseModel):
    vector: List[float]
    tokens: List[str] = Field(default_factory=list)
    dimension: int


class BatchDirectionRequest(BaseModel):
    user_lon: float
    user_lat: float
    pois: List[Dict[str, float]]


class BatchDirectionResponse(BaseModel):
    results: List[Dict[str, Any]]


class EnrichPOIsRequest(BaseModel):
    user_lon: float
    user_lat: float
    pois: List[Dict[str, Any]]


class EnrichPOIsResponse(BaseModel):
    pois: List[Dict[str, Any]]
    models_used: List[str] = Field(default_factory=list)


class CellContextRequest(BaseModel):
    lon: float
    lat: float


class CellContextResponse(BaseModel):
    context: Dict[str, Any]
    models_used: List[str] = Field(default_factory=list)


class BatchCellContextRequest(BaseModel):
    anchor_lon: float
    anchor_lat: float
    user_query: str = ''
    task_type: Optional[str] = None
    pois: List[Dict[str, Any]]


class BatchCellContextResponse(BaseModel):
    anchor_cell_context: Dict[str, Any]
    results: List[Dict[str, Any]]
    model_route: str
    models_used: List[str] = Field(default_factory=list)


class CellSearchRequest(BaseModel):
    anchor_lon: float
    anchor_lat: float
    user_query: str = ''
    task_type: Optional[str] = None
    top_k: Optional[int] = 5
    max_distance_m: Optional[float] = None


class CellSearchResponse(BaseModel):
    anchor_cell_context: Dict[str, Any]
    cells: List[Dict[str, Any]]
    model_route: str
    models_used: List[str] = Field(default_factory=list)
    search_radius_m: float
    per_cell_radius_m: float
    support_bucket_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    dominant_buckets: List[str] = Field(default_factory=list)
    scene_tags: List[str] = Field(default_factory=list)
    cell_mix: List[Dict[str, Any]] = Field(default_factory=list)
    macro_uncertainty: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    encoder_loaded: bool
    device: str
    architecture: Optional[str] = None
    checkpoint_path: Optional[str] = None
    embedding_dim: Optional[int] = None
    supported_features: List[str] = Field(default_factory=list)
    startup_error: Optional[str] = None
    models: Dict[str, Any] = Field(default_factory=dict)
    poi_feature_cache: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title='V3 Spatial Encoder Service',
    description='双模型空间编码器服务：POI 编码 + Cell 上下文',
    version='1.2.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(**build_health_payload())


@app.get('/capabilities', response_model=HealthResponse)
async def capabilities():
    return HealthResponse(**build_health_payload())


@app.post('/admin/reload-town-index', response_model=HealthResponse)
async def reload_town_index_endpoint():
    rebuild_town_index()
    return HealthResponse(**build_health_payload())


@app.post('/direction', response_model=DirectionResponse)
async def predict_direction_endpoint(request: DirectionRequest):
    direction_idx, direction_name = compute_direction(
        request.user_lon,
        request.user_lat,
        request.poi_lon,
        request.poi_lat,
    )
    distance_m = compute_distance_m(
        request.user_lon,
        request.user_lat,
        request.poi_lon,
        request.poi_lat,
    )

    dx = request.poi_lon - request.user_lon
    dy = request.poi_lat - request.user_lat
    avg_lat = (request.user_lat + request.poi_lat) / 2
    dx_corrected = dx * np.cos(np.radians(avg_lat))
    angle_deg = np.degrees(np.arctan2(dy, dx_corrected))

    return DirectionResponse(
        direction_idx=direction_idx,
        direction_name=direction_name,
        distance_m=distance_m,
        angle_deg=float(angle_deg),
    )


@app.post('/region', response_model=RegionResponse)
async def predict_region_endpoint(request: RegionRequest):
    try:
        region_idx, region_name, confidence = predict_region(request.lon, request.lat)
        return RegionResponse(
            region_idx=region_idx,
            region_name=region_name,
            confidence=confidence,
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post('/encode', response_model=EncodeResponse)
async def encode_endpoint(request: EncodeRequest):
    embedding, feature_meta = encode_coords(request.lon, request.lat, poi_id=request.poi_id)
    if embedding is None:
        raise HTTPException(status_code=503, detail='POI encoder not available')
    return EncodeResponse(
        embedding=embedding.tolist(),
        embedding_dim=len(embedding),
        feature_source=feature_meta.get('feature_source'),
        feature_stats=feature_meta.get('feature_stats') or {},
    )


@app.post('/encode-text', response_model=EncodeTextResponse)
async def encode_text_endpoint(request: EncodeTextRequest):
    vector, tokens = build_text_embedding(request.text)
    return EncodeTextResponse(
        vector=vector.tolist(),
        tokens=tokens,
        dimension=len(vector),
    )


@app.post('/direction/batch', response_model=BatchDirectionResponse)
async def batch_direction_endpoint(request: BatchDirectionRequest):
    results = []
    for poi in request.pois:
        poi_lon = poi.get('lon', poi.get('longitude', 0))
        poi_lat = poi.get('lat', poi.get('latitude', 0))
        direction_idx, direction_name = compute_direction(
            request.user_lon,
            request.user_lat,
            poi_lon,
            poi_lat,
        )
        distance_m = compute_distance_m(
            request.user_lon,
            request.user_lat,
            poi_lon,
            poi_lat,
        )
        results.append({
            'direction_idx': direction_idx,
            'direction_name': direction_name,
            'distance_m': distance_m,
        })

    return BatchDirectionResponse(results=results)


@app.post('/enrich', response_model=EnrichPOIsResponse)
async def enrich_pois(request: EnrichPOIsRequest):
    enriched_pois = []
    for poi in request.pois:
        poi_lon = poi.get('lon', poi.get('longitude', 0))
        poi_lat = poi.get('lat', poi.get('latitude', 0))
        direction_idx, direction_name = compute_direction(
            request.user_lon,
            request.user_lat,
            poi_lon,
            poi_lat,
        )
        distance_m = compute_distance_m(
            request.user_lon,
            request.user_lat,
            poi_lon,
            poi_lat,
        )
        region_idx, region_name, confidence = predict_region(poi_lon, poi_lat)

        enriched_pois.append({
            **poi,
            'spatial_info': {
                'direction_idx': direction_idx,
                'direction_name': direction_name,
                'distance_m': distance_m,
                'region_idx': region_idx,
                'region_name': region_name,
                'region_confidence': confidence,
            },
        })

    return EnrichPOIsResponse(
        pois=enriched_pois,
        models_used=['poi_encoder'],
    )


@app.post('/cell/context', response_model=CellContextResponse)
async def cell_context_endpoint(request: CellContextRequest):
    try:
        context = get_cell_context_for_point(request.lon, request.lat)
        return CellContextResponse(
            context=context,
            models_used=['town_encoder'],
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post('/cell/context/batch', response_model=BatchCellContextResponse)
async def batch_cell_context_endpoint(request: BatchCellContextRequest):
    try:
        anchor_index, anchor_distance = find_nearest_cell_index(request.anchor_lon, request.anchor_lat)
        anchor_embedding = state.town_embeddings[anchor_index]
        anchor_context = build_cell_context(anchor_index, anchor_distance, anchor_embedding=anchor_embedding)
        anchor_context['similarity'] = 1.0

        results = []
        for poi in request.pois:
            poi_lon = poi.get('lon', poi.get('longitude'))
            poi_lat = poi.get('lat', poi.get('latitude'))
            if poi_lon is None or poi_lat is None:
                results.append({
                    **poi,
                    'cell_context': None,
                })
                continue

            context = get_cell_context_for_point(float(poi_lon), float(poi_lat), anchor_embedding=anchor_embedding)
            results.append({
                **poi,
                'cell_context': context,
            })

        return BatchCellContextResponse(
            anchor_cell_context=anchor_context,
            results=results,
            model_route='town_encoder',
            models_used=['town_encoder'],
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post('/cell/search', response_model=CellSearchResponse)
async def cell_search_endpoint(request: CellSearchRequest):
    try:
        anchor_context, cells, search_radius_m, macro_summary = search_similar_cells(
            request.anchor_lon,
            request.anchor_lat,
            task_type=request.task_type,
            top_k=request.top_k or 5,
            max_distance_m=request.max_distance_m,
        )
        return CellSearchResponse(
            anchor_cell_context=anchor_context,
            cells=cells,
            model_route='town_encoder',
            models_used=['town_encoder'],
            search_radius_m=search_radius_m,
            per_cell_radius_m=resolve_per_cell_poi_radius(request.task_type),
            support_bucket_distribution=macro_summary.get('support_bucket_distribution') or [],
            dominant_buckets=macro_summary.get('dominant_buckets') or [],
            scene_tags=macro_summary.get('scene_tags') or [],
            cell_mix=macro_summary.get('cell_mix') or [],
            macro_uncertainty=macro_summary.get('macro_uncertainty') or {},
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoLoom Dual Spatial Encoder Service')
    parser.add_argument('--port', type=int, default=8100, help='服务端口')
    parser.add_argument('--no-encoder', action='store_true', help='跳过编码器加载，仅启用几何能力')
    args = parser.parse_args()

    print('=' * 60)
    print('GeoLoom Dual Spatial Encoder Service')
    print('=' * 60)

    if not args.no_encoder:
        load_encoder()
    else:
        state.startup_error = 'encoder_loading_skipped'

    uvicorn.run(app, host='0.0.0.0', port=args.port)
