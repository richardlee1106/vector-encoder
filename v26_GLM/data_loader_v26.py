# -*- coding: utf-8 -*-
"""
V2.6 真实数据加载器

从PostgreSQL数据库加载：
- POI（点）
- 道路（线）
- 土地利用（面）

构建Cell级特征，支持H3投影和多尺度采样。

Author: GLM (Qianfan Code)
Date: 2026-03-15
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from spatial_encoder.v26_GLM.config_v26_pro import V26ProConfig, DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.data_sources import PostGISSource, default_postgis_source
from spatial_encoder.v26_GLM.h3_projection import (
    line_to_cells,
    point_to_cell,
    polygon_to_cells,
)
from spatial_encoder.v26_GLM.direction_supervision import (
    MultiSchemeDirectionSupervision,
    DirectionScheme,
)


# ============================================================
# 功能区标签体系（6类 + 未知）
#
# 分层标注策略（优先级从高到低）：
#   1. AOI fclass   → 自然/居住/工业最可靠（大面积多边形）
#   2. Landuse      → 补充 AOI 未覆盖区域
#   3. POI 大类     → 填充商业/生活/教育等 AOI 稀疏区域
#
# 类别定义：
#   0 居住类  — residential, 商务住宅
#   1 商业类  — commercial/retail, 购物/住宿/汽车/金融/生活服务
#   2 工业类  — industrial, 公司企业（工厂子类）
#   3 教育类  — school/university, 科教文化服务
#   4 公共类  — park/hospital/government, 医疗/政府/交通/体育/公共设施
#   5 自然类  — water/forest/riverbank/wetland（仅来自 AOI/Landuse，POI 无此类）
#   6 未知    — 排除训练
# ============================================================

# POI 大类 → 功能区标签（作为 AOI/Landuse 的补充，无自然类）
POI_CATEGORY_TO_REGION: dict[str, int] = {
    # 0 居住类
    "商务住宅": 0,

    # 1 商业类（购物零售 + 生活服务 + 汽车 + 金融 + 住宿）
    "购物服务": 1,
    "住宿服务": 1,
    "生活服务": 1,
    "汽车服务": 1,
    "汽车维修": 1,
    "汽车销售": 1,
    "摩托车服务": 1,
    "金融保险服务": 1,

    # 2 工业类（公司企业作为补充，AOI industrial 优先）
    "公司企业": 2,

    # 3 教育类
    "科教文化服务": 3,

    # 4 公共类
    "医疗保健服务": 4,
    "政府机构及社会团体": 4,
    "交通设施服务": 4,
    "体育休闲服务": 4,
    "公共设施": 4,
    "风景名胜": 4,

    # 5 自然类：POI 无此类，由 AOI/Landuse 提供
    # water, forest, riverbank, wetland, farmland, grass → 5

    # 6 未知（噪声，排除训练）
    # 地名地址信息、通行设施、道路附属设施、事件活动、室内设施、虚拟数据
}


AOI_TO_REGION_MAP = {
    # 居住类 (0)
    "residential": 0,
    "village": 0,
    "farmyard": 0,

    # 商业类 (1)
    "commercial": 1,
    "retail": 1,
    "mall": 1,
    "supermarket": 1,
    "market_place": 1,
    "shop": 1,
    "restaurant": 1,
    "hotel": 1,
    "fuel": 1,
    "cafe": 1,
    "fast_food": 1,
    "clothes": 1,
    "convenience": 1,
    "car_dealership": 1,
    "mobile_phone_shop": 1,
    "hairdresser": 1,
    "furniture_shop": 1,
    "bookshop": 1,
    "greengrocer": 1,
    "department_store": 1,
    "garden_centre": 1,
    "pharmacy": 1,
    "food_court": 1,

    # 工业类 (2)
    "industrial": 2,
    "quarry": 2,
    "wastewater_plant": 2,
    "water_works": 2,

    # 教育类 (3)
    "school": 3,
    "university": 3,
    "college": 3,
    "kindergarten": 3,
    "library": 3,

    # 公共类 (4)
    "park": 4,
    "parking": 4,
    "hospital": 4,
    "public_building": 4,
    "police": 4,
    "fire_station": 4,
    "courthouse": 4,
    "town_hall": 4,
    "community_centre": 4,
    "sports_centre": 4,
    "stadium": 4,
    "swimming_pool": 4,
    "playground": 4,
    "pitch": 4,
    "track": 4,
    "recreation_ground": 4,
    "bus_station": 4,
    "railway_station": 4,
    "ferry_terminal": 4,
    "helipad": 4,
    "post_office": 4,
    "bank": 4,
    "embassy": 4,
    "museum": 4,
    "theatre": 4,
    "arts_centre": 4,
    "attraction": 4,
    "memorial": 4,
    "monument": 4,
    "artwork": 4,
    "theme_park": 4,
    "zoo": 4,
    "viewpoint": 4,
    "beach": 4,
    "camp_site": 4,
    "bicycle_rental": 4,
    "shelter": 4,
    "toilet": 4,
    "fountain": 4,
    "tourist_info": 4,
    "cemetery": 4,
    "graveyard": 4,
    "clinic": 4,
    "veterinary": 4,
    "prison": 4,
    "military": 4,

    # 自然类 (5)
    "forest": 5,
    "water": 5,
    "riverbank": 5,
    "wetland": 5,
    "meadow": 5,
    "grass": 5,
    "scrub": 5,
    "orchard": 5,
    "farmland": 5,
    "nature_reserve": 5,
    "reservoir": 5,
    "dam": 5,
    "pier": 5,
    "marina": 5,
    "island": 5,
    "golf_course": 5,
    "ruins": 5,

    # 其他服务设施（归入商业类）
    "service": 1,
}

# AOI fclass 英文 → 中文翻译（为 NLP 准备）
AOI_FCLASS_CN = {
    # 居住类
    "residential": "居住区",
    "village": "村庄",
    "farmyard": "农家院",

    # 商业类
    "commercial": "商业区",
    "retail": "零售区",
    "mall": "购物中心",
    "supermarket": "超市",
    "market_place": "集市",
    "shop": "商店",
    "restaurant": "餐厅",
    "hotel": "酒店",
    "fuel": "加油站",
    "cafe": "咖啡馆",
    "fast_food": "快餐店",
    "clothes": "服装店",
    "convenience": "便利店",
    "car_dealership": "汽车销售",
    "mobile_phone_shop": "手机店",
    "hairdresser": "理发店",
    "furniture_shop": "家具店",
    "bookshop": "书店",
    "greengrocer": "果蔬店",
    "department_store": "百货商店",
    "garden_centre": "园艺中心",
    "pharmacy": "药店",
    "food_court": "美食城",

    # 工业类
    "industrial": "工业区",
    "quarry": "采石场",
    "wastewater_plant": "污水处理厂",
    "water_works": "自来水厂",

    # 教育类
    "school": "学校",
    "university": "大学",
    "college": "学院",
    "kindergarten": "幼儿园",
    "library": "图书馆",

    # 公共类
    "park": "公园",
    "parking": "停车场",
    "hospital": "医院",
    "public_building": "公共建筑",
    "police": "派出所",
    "fire_station": "消防站",
    "courthouse": "法院",
    "town_hall": "市政厅",
    "community_centre": "社区中心",
    "sports_centre": "体育中心",
    "stadium": "体育场",
    "swimming_pool": "游泳池",
    "playground": "游乐场",
    "pitch": "运动场",
    "track": "跑道",
    "recreation_ground": "娱乐场",
    "bus_station": "公交站",
    "railway_station": "火车站",
    "ferry_terminal": "轮渡码头",
    "helipad": "直升机坪",
    "post_office": "邮局",
    "bank": "银行",
    "embassy": "大使馆",
    "museum": "博物馆",
    "theatre": "剧院",
    "arts_centre": "艺术中心",
    "attraction": "景点",
    "memorial": "纪念馆",
    "monument": "纪念碑",
    "artwork": "艺术品",
    "theme_park": "主题公园",
    "zoo": "动物园",
    "viewpoint": "观景台",
    "beach": "海滩",
    "camp_site": "露营地",
    "bicycle_rental": "自行车租赁",
    "shelter": "庇护所",
    "toilet": "公厕",
    "fountain": "喷泉",
    "tourist_info": "游客中心",
    "cemetery": "墓地",
    "graveyard": "墓园",
    "clinic": "诊所",
    "veterinary": "兽医站",
    "prison": "监狱",
    "military": "军事区",

    # 自然类
    "forest": "森林",
    "water": "水域",
    "riverbank": "河岸",
    "wetland": "湿地",
    "meadow": "草地",
    "grass": "草坪",
    "scrub": "灌木丛",
    "orchard": "果园",
    "farmland": "农田",
    "nature_reserve": "自然保护区",
    "reservoir": "水库",
    "dam": "水坝",
    "pier": "码头",
    "marina": "游艇码头",
    "island": "岛屿",
    "golf_course": "高尔夫球场",
    "ruins": "遗址",

    # 其他
    "service": "服务区",
    "unknown": "未知",
}

# 功能区类别名称（中文）
MERGED_REGION_NAMES = {
    0: "居住类",
    1: "商业类",
    2: "工业类",
    3: "教育类",
    4: "公共类",
    5: "自然类",
    6: "未知",
}


def poi_category_to_region(category_main: str) -> int:
    """将 POI 大类映射到功能区标签（补充路径，AOI/Landuse 优先）"""
    return POI_CATEGORY_TO_REGION.get(category_main, 6)


def aoi_fclass_to_region(fclass: str) -> int:
    """将 AOI fclass 映射到功能区标签（第一优先级）"""
    return AOI_TO_REGION_MAP.get(fclass, 6)


def landuse_type_to_region(land_type: str) -> int:
    """将 Landuse land_type 映射到功能区标签（使用关键字匹配）"""
    if not land_type or land_type == "unknown":
        return 6

    # 使用关键字匹配（处理可能的编码问题）
    # 这些中文字符在 Python 内部是正确编码的
    # 对齐 6 类方案：0居住 1商业 2工业 3教育 4公共 5自然
    keywords = {
        "居住": 0,
        "商业": 1,
        "商务": 1,
        "零售": 1,
        "生活": 1,
        "工业": 2,
        "仓储": 2,
        "教育": 3,
        "科教": 3,
        "文化": 3,
        "体育": 4,
        "医疗": 4,
        "卫生": 4,
        "行政": 4,
        "公用": 4,
        "交通": 4,
        "枢纽": 4,
        "公园": 5,
        "绿地": 5,
        "河流": 5,
        "湖泊": 5,
    }

    for keyword, region_id in keywords.items():
        if keyword in land_type:
            return region_id

    return 6  # 未知


@dataclass
class POIRecord:
    """POI记录"""
    id: int
    lng: float
    lat: float
    name: str
    category_main: str = "unknown"
    category_sub: str = "unknown"
    aoi_type: str = "unknown"  # 功能区类型（原始英文）
    aoi_type_cn: str = "未知"  # 功能区类型（中文翻译，为NLP准备）


@dataclass
class RoadRecord:
    """道路记录"""
    id: int
    coords: List[Tuple[float, float]]
    road_class: str
    name: str


@dataclass
class LanduseRecord:
    """土地利用记录"""
    id: int
    coords: List[Tuple[float, float]]
    landuse_type: str
    area: float


class CellFeatureBuilder:
    """Cell特征构建器"""

    def __init__(self, resolution: int = 9):
        self.resolution = resolution
        self.cells: Dict[str, Dict] = {}

    def add_poi(self, poi: POIRecord) -> None:
        """添加POI到对应Cell"""
        cell_id = point_to_cell(poi.lng, poi.lat, self.resolution)

        if cell_id not in self.cells:
            self._init_cell(cell_id, poi.lng, poi.lat)

        cell = self.cells[cell_id]
        cell["poi_count"] += 1
        cell["poi_categories"].append(poi.category_main)

        # 添加功能区类型（原始英文和中文翻译）
        if poi.aoi_type and poi.aoi_type != "unknown":
            cell["aoi_types"].append(poi.aoi_type)
            cell["aoi_types_cn"].append(poi.aoi_type_cn)

        # 更新坐标范围
        cell["lng_min"] = min(cell["lng_min"], poi.lng)
        cell["lng_max"] = max(cell["lng_max"], poi.lng)
        cell["lat_min"] = min(cell["lat_min"], poi.lat)
        cell["lat_max"] = max(cell["lat_max"], poi.lat)

    def add_road(self, road: RoadRecord) -> None:
        """添加道路到覆盖的Cells"""
        if not road.coords:
            return

        cell_ids = line_to_cells(road.coords, self.resolution)

        for cell_id in cell_ids:
            if cell_id not in self.cells:
                # 使用道路中点初始化
                mid_idx = len(road.coords) // 2
                mid_lng, mid_lat = road.coords[mid_idx]
                self._init_cell(cell_id, mid_lng, mid_lat)

            cell = self.cells[cell_id]
            cell["road_count"] += 1
            cell["road_classes"].append(road.road_class)
            cell["road_length"] += 1  # 简化计数

    def add_landuse(self, landuse: LanduseRecord) -> None:
        """添加土地利用到覆盖的Cells"""
        if not landuse.coords:
            return

        cell_weights = polygon_to_cells(landuse.coords, self.resolution)

        for item in cell_weights:
            cell_id = item["cell"]
            weight = item["weight"]

            if cell_id not in self.cells:
                # 使用多边形中心初始化
                center_lng = np.mean([c[0] for c in landuse.coords])
                center_lat = np.mean([c[1] for c in landuse.coords])
                self._init_cell(cell_id, center_lng, center_lat)

            cell = self.cells[cell_id]
            cell["landuse_weight"] += weight
            cell["landuse_types"].append((landuse.landuse_type, weight))

    def _init_cell(self, cell_id: str, lng: float, lat: float) -> None:
        """初始化Cell"""
        self.cells[cell_id] = {
            "cell_id": cell_id,
            "center_lng": lng,
            "center_lat": lat,
            "lng_min": lng,
            "lng_max": lng,
            "lat_min": lat,
            "lat_max": lat,
            "poi_count": 0,
            "poi_categories": [],
            "aoi_types": [],  # 功能区类型列表（原始英文）
            "aoi_types_cn": [],  # 功能区类型列表（中文翻译）
            "road_count": 0,
            "road_classes": [],
            "road_length": 0.0,
            "landuse_weight": 0.0,
            "landuse_types": [],
        }

    def build_features(self, min_poi_count: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[Dict]]:
        """
        构建特征矩阵

        Args:
            min_poi_count: 最小 POI 数量阈值，低于此值的 cell 不包含在输出中

        Returns:
            point_features: 点特征 [N, 32]
            line_features: 线特征 [N, 16]
            polygon_features: 面特征 [N, 16]
            direction_features: 方向特征 [N, 8]
            coords: 坐标 [N, 2]
            region_labels: 功能区标签 [N]
            cell_id_map: Cell ID到索引的映射
            metadata: 元数据列表 [N]
        """
        # 过滤：只保留有 POI 的 cells
        valid_cell_ids = [cid for cid, c in self.cells.items() if c["poi_count"] >= min_poi_count]
        cell_ids = sorted(valid_cell_ids)
        cell_id_map = {cid: idx for idx, cid in enumerate(cell_ids)}
        n_cells = len(cell_ids)

        # 初始化特征数组
        point_features = np.zeros((n_cells, 32), dtype=np.float32)
        line_features = np.zeros((n_cells, 16), dtype=np.float32)
        polygon_features = np.zeros((n_cells, 16), dtype=np.float32)
        direction_features = np.zeros((n_cells, 8), dtype=np.float32)
        coords = np.zeros((n_cells, 2), dtype=np.float32)
        region_labels = np.zeros(n_cells, dtype=int)  # 功能区标签
        metadata = []  # 元数据列表

        # POI类别编码（简化版：使用前16个高频类别）
        poi_category_list = [
            "餐饮服务", "购物服务", "生活服务", "体育休闲服务",
            "医疗保健服务", "住宿服务", "风景名胜", "商务住宅",
            "政府机构及社会团体", "科教文化服务", "交通设施服务",
            "金融保险服务", "公司企业", "道路附属设施", "地名地址信息", "公共设施",
        ]
        category_to_idx = {cat: idx for idx, cat in enumerate(poi_category_list)}

        # 道路等级编码
        road_class_list = ["primary", "secondary", "tertiary", "residential", "unclassified"]
        road_class_to_idx = {cls: idx for idx, cls in enumerate(road_class_list)}

        # 土地利用类型编码
        landuse_type_list = [
            "居住用地", "商业服务业设施用地", "工业用地", "物流仓储用地",
            "道路交通设施用地", "公用设施用地", "绿地与广场用地",
            "医疗卫生用地", "教育科研用地", "河流湖泊", "公园与绿地用地",
        ]
        landuse_to_idx = {t: idx for idx, t in enumerate(landuse_type_list)}

        for idx, cell_id in enumerate(cell_ids):
            cell = self.cells[cell_id]

            # 坐标
            coords[idx] = [cell["center_lng"], cell["center_lat"]]

            # 计算归一化坐标（所有cell都有）
            # 武汉范围: lng [113.6, 115.0], lat [29.9, 31.3]
            norm_lng = (cell["center_lng"] - 113.6) / (115.0 - 113.6)
            norm_lat = (cell["center_lat"] - 29.9) / (31.3 - 29.9)

            # 点特征 [32维]
            # - 归一化坐标 (2维) - 确保所有cell都有
            # - POI计数归一化
            # - 类别分布 (16维)
            # - 密度特征
            point_features[idx, 0] = norm_lng
            point_features[idx, 1] = norm_lat

            poi_count = cell["poi_count"]
            point_features[idx, 2] = np.log1p(poi_count) / 10.0  # 归一化

            # 类别分布
            category_counts = {}
            for cat in cell["poi_categories"]:
                if cat in category_to_idx:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            for cat, count in category_counts.items():
                cat_idx = category_to_idx[cat]
                point_features[idx, 3 + cat_idx] = count / max(poi_count, 1)

            # 密度熵
            if len(category_counts) > 1:
                probs = np.array(list(category_counts.values())) / poi_count
                point_features[idx, 19] = -np.sum(probs * np.log(probs + 1e-8))

            # 线特征 [16维]
            # - 归一化坐标 (2维) - 确保所有cell都有
            # - 道路计数
            # - 道路等级分布 (5维)
            line_features[idx, 0] = norm_lng
            line_features[idx, 1] = norm_lat
            line_features[idx, 2] = np.log1p(cell["road_count"]) / 5.0

            road_class_counts = {}
            for cls in cell["road_classes"]:
                if cls in road_class_to_idx:
                    road_class_counts[cls] = road_class_counts.get(cls, 0) + 1

            for cls, count in road_class_counts.items():
                cls_idx = road_class_to_idx[cls]
                line_features[idx, 3 + cls_idx] = count / max(cell["road_count"], 1)

            # 面特征 [16维]
            # - 归一化坐标 (2维) - 确保所有cell都有
            # - 土地利用权重
            # - 土地利用类型分布 (11维)
            polygon_features[idx, 0] = norm_lng
            polygon_features[idx, 1] = norm_lat
            polygon_features[idx, 2] = min(cell["landuse_weight"], 10.0)

            landuse_weights = {}
            for landuse_type, weight in cell["landuse_types"]:
                if landuse_type in landuse_to_idx:
                    landuse_weights[landuse_type] = landuse_weights.get(landuse_type, 0) + weight

            for t, w in landuse_weights.items():
                t_idx = landuse_to_idx[t]
                polygon_features[idx, 3 + t_idx] = w / max(cell["landuse_weight"], 1)

            # 方向特征 [8维] - 初始化为0，后续由方向监督模块计算
            # 这里先计算一个简单的全局中心方向
            center_lng, center_lat = coords.mean(axis=0)
            dx = cell["center_lng"] - center_lng
            dy = cell["center_lat"] - center_lat
            angle = np.arctan2(dy, dx)
            direction = int((angle + np.pi) / (np.pi / 4)) % 8
            direction_features[idx, direction] = 1.0

            # 功能区标签：取最频繁的aoi_type，支持 AOI fclass 和 Landuse land_type
            region_label = 6  # 默认未知
            aoi_type_str = "unknown"
            aoi_type_cn = "未知"

            if cell["aoi_types"]:
                # 统计每个类型的出现次数
                from collections import Counter
                type_counts = Counter(cell["aoi_types"])
                most_common = type_counts.most_common(1)[0][0]

                # 尝试 AOI 映射，失败则尝试 Landuse 映射
                region_label = aoi_fclass_to_region(most_common)
                if region_label == 6:  # AOI 映射失败，尝试 Landuse
                    region_label = landuse_type_to_region(most_common)

                aoi_type_str = most_common
                # 获取中文翻译
                aoi_type_cn = AOI_FCLASS_CN.get(most_common, most_common)

            region_labels[idx] = region_label

            # 构建元数据（包含中文字段，为NLP准备）
            from collections import Counter
            cat_counter = Counter(cell["poi_categories"]) if cell["poi_categories"] else Counter()
            dominant_category = cat_counter.most_common(1)[0][0] if cat_counter else "unknown"

            cell_meta = {
                "cell_id": cell_id,
                "poi_count": poi_count,
                "dominant_category": dominant_category,
                "road_count": cell["road_count"],
                "road_length_km": round(cell["road_length"] / 1000, 2),
                "has_landuse": len(cell["landuse_types"]) > 0,
                "aoi_type": aoi_type_str,  # 原始类型（英文或中文）
                "aoi_type_cn": aoi_type_cn,  # 中文翻译（为NLP准备）
                "region_label": MERGED_REGION_NAMES.get(region_label, "未知"),  # 功能区大类（中文）
            }
            metadata.append(cell_meta)

        return point_features, line_features, polygon_features, direction_features, coords, region_labels, cell_id_map, metadata


class DataLoaderV26:
    """V2.6 数据加载器"""

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

    def load_pois(
        self,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        where_clause: Optional[str] = None,
    ) -> List[POIRecord]:
        """
        加载POI数据，通过空间关联获取功能区标签

        优先级：AOI > Landuse > Unknown

        Args:
            limit: 最大数量
            sample_ratio: 采样比例 (0-1)
            where_clause: 额外的WHERE条件

        Returns:
            POI记录列表
        """
        table = self.source.tables.get("point", "pois")
        aoi_table = "aois"
        landuse_table = "landuse"

        # 使用空间关联获取 POI 与 AOI、Landuse 的关系
        # 优先使用 AOI.fclass，如果不在 AOI 内则使用 Landuse.land_type
        sql = f"""
            SELECT p.id, ST_X(p.geom) as lng, ST_Y(p.geom) as lat,
                   p.name, p.category_main, p.category_sub,
                   COALESCE(a.fclass, l.land_type, 'unknown') as aoi_type,
                   CASE
                       WHEN a.fclass IS NOT NULL THEN 'aoi'
                       WHEN l.land_type IS NOT NULL THEN 'landuse'
                       ELSE 'unknown'
                   END as label_source
            FROM {table} p
            LEFT JOIN {aoi_table} a ON ST_Within(p.geom, a.geom)
            LEFT JOIN {landuse_table} l ON ST_Within(p.geom, l.geom) AND a.fclass IS NULL
            WHERE p.geom IS NOT NULL
        """

        if where_clause:
            sql += f" AND {where_clause}"

        if sample_ratio < 1.0:
            sql += f" AND RANDOM() < {sample_ratio}"

        if limit:
            sql += f" LIMIT {limit}"

        conn = self._get_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql)
            rows = cur.fetchall()

            pois = []
            for row in rows:
                aoi_type_raw = row.get("aoi_type") or "unknown"
                # 翻译成中文（为NLP准备）
                aoi_type_cn = AOI_FCLASS_CN.get(aoi_type_raw, aoi_type_raw)
                # 如果是 landuse 类型（中文），直接使用
                if aoi_type_cn == aoi_type_raw and aoi_type_raw != "unknown":
                    # 可能是 landuse 的中文类型，保持原样
                    aoi_type_cn = aoi_type_raw

                pois.append(POIRecord(
                    id=row["id"],
                    lng=float(row["lng"]),
                    lat=float(row["lat"]),
                    name=row["name"] or "",
                    category_main=row["category_main"] or "unknown",
                    category_sub=row["category_sub"] or "unknown",
                    aoi_type=aoi_type_raw,
                    aoi_type_cn=aoi_type_cn,
                ))

            cur.close()
            return pois
        finally:
            conn.close()

    def load_roads(
        self,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        where_clause: Optional[str] = None,
    ) -> List[RoadRecord]:
        """加载道路数据"""
        table = self.source.tables.get("line", "roads")

        sql = f"""
            SELECT id, ST_AsGeoJSON(geom) as geom_json,
                   fclass as road_class,
                   name
            FROM {table}
            WHERE geom IS NOT NULL
        """

        if where_clause:
            sql += f" AND {where_clause}"

        if sample_ratio < 1.0:
            sql += f" AND RANDOM() < {sample_ratio}"

        if limit:
            sql += f" LIMIT {limit}"

        conn = self._get_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql)
            rows = cur.fetchall()

            roads = []
            for row in rows:
                geom = json.loads(row["geom_json"])
                geom_type = geom.get("type", "")
                all_coords = geom.get("coordinates", [])

                # 处理不同几何类型
                if geom_type == "LineString":
                    coords = [(float(c[0]), float(c[1])) for c in all_coords]
                    if len(coords) >= 2:
                        roads.append(RoadRecord(
                            id=row["id"],
                            coords=coords,
                            road_class=row["road_class"] or "unclassified",
                            name=row["name"] or "",
                        ))
                elif geom_type == "MultiLineString":
                    # MultiLineString 有多个线段，分别处理
                    for segment_idx, segment in enumerate(all_coords):
                        coords = [(float(c[0]), float(c[1])) for c in segment]
                        if len(coords) >= 2:
                            roads.append(RoadRecord(
                                id=f"{row['id']}_{segment_idx}",
                                coords=coords,
                                road_class=row["road_class"] or "unclassified",
                                name=row["name"] or "",
                            ))

            cur.close()
            return roads
        finally:
            conn.close()

    def load_landuse(
        self,
        limit: Optional[int] = None,
        sample_ratio: float = 1.0,
        where_clause: Optional[str] = None,
    ) -> List[LanduseRecord]:
        """加载土地利用数据"""
        table = self.source.tables.get("polygon", "landuse")

        sql = f"""
            SELECT id, ST_AsGeoJSON(geom) as geom_json,
                   land_type as landuse_type,
                   area_sqm as area
            FROM {table}
            WHERE geom IS NOT NULL
        """

        if where_clause:
            sql += f" AND {where_clause}"

        if sample_ratio < 1.0:
            sql += f" AND RANDOM() < {sample_ratio}"

        if limit:
            sql += f" LIMIT {limit}"

        conn = self._get_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql)
            rows = cur.fetchall()

            landuses = []
            for row in rows:
                geom = json.loads(row["geom_json"])
                geom_type = geom.get("type", "")
                coords = []

                if geom_type == "Polygon":
                    coords = [(c[0], c[1]) for c in geom["coordinates"][0]]
                elif geom_type == "MultiPolygon":
                    coords = [(c[0], c[1]) for c in geom["coordinates"][0][0]]

                if len(coords) >= 3:
                    landuses.append(LanduseRecord(
                        id=row["id"],
                        coords=coords,
                        landuse_type=row["landuse_type"] or "unknown",
                        area=float(row["area"]) if row["area"] else 0.0,
                    ))

            cur.close()
            return landuses
        finally:
            conn.close()

    def build_cell_dataset(
        self,
        sample_ratio: float = 1.0,
        poi_limit: Optional[int] = None,
        road_limit: Optional[int] = None,
        landuse_limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        构建Cell级数据集

        Args:
            sample_ratio: 采样比例
            poi_limit: POI数量限制
            road_limit: 道路数量限制
            landuse_limit: 土地利用数量限制

        Returns:
            point_features, line_features, polygon_features, direction_features, coords, region_labels, cell_id_map, metadata
        """
        print(f"Loading data with sample_ratio={sample_ratio}...")

        # 加载原始数据
        pois = self.load_pois(limit=poi_limit, sample_ratio=sample_ratio)
        print(f"  Loaded {len(pois)} POIs")

        roads = self.load_roads(limit=road_limit, sample_ratio=sample_ratio)
        print(f"  Loaded {len(roads)} roads")

        landuses = self.load_landuse(limit=landuse_limit, sample_ratio=sample_ratio)
        print(f"  Loaded {len(landuses)} landuse polygons")

        # 构建Cell特征
        print("Building cell features...")
        builder = CellFeatureBuilder(resolution=self.resolution)

        for poi in pois:
            builder.add_poi(poi)

        for road in roads:
            builder.add_road(road)

        for landuse in landuses:
            builder.add_landuse(landuse)

        point_features, line_features, polygon_features, direction_features, coords, region_labels, cell_id_map, metadata = builder.build_features()

        print(f"  Built {len(cell_id_map)} cells")
        print(f"  Region labels: unique={len(np.unique(region_labels))}, with_label={(region_labels < 6).sum()}")

        return point_features, line_features, polygon_features, direction_features, coords, region_labels, cell_id_map, metadata


def compute_neighbor_indices(
    coords: np.ndarray,
    h3_resolution: int = 9,
    k_neighbors: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算每个Cell的邻居索引

    Args:
        coords: Cell坐标 [N, 2]
        h3_resolution: H3分辨率
        k_neighbors: 最大邻居数（默认50，从20提升）

    Returns:
        neighbor_indices: 邻居索引 [N, K]
        neighbor_rings: 邻居圈数 [N, K]
    """
    import h3

    N = len(coords)
    neighbor_indices = np.full((N, k_neighbors), -1, dtype=int)
    neighbor_rings = np.zeros((N, k_neighbors), dtype=int)

    # 构建Cell ID到索引的映射
    cell_ids = []
    for i in range(N):
        cell_id = h3.latlng_to_cell(coords[i, 1], coords[i, 0], h3_resolution)
        cell_ids.append(cell_id)

    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}

    # 获取每个Cell的邻居
    for i, cell_id in enumerate(cell_ids):
        # 获取多圈邻居
        neighbors = []
        for ring in range(1, 3):  # 1-2圈
            ring_neighbors = h3.grid_disk(cell_id, ring)
            for neighbor_id in ring_neighbors:
                if neighbor_id != cell_id and neighbor_id in cell_to_idx:
                    neighbors.append((cell_to_idx[neighbor_id], ring))

        # 按圈数排序，取前K个
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:k_neighbors]

        for j, (idx, ring) in enumerate(neighbors):
            neighbor_indices[i, j] = idx
            neighbor_rings[i, j] = ring

    return neighbor_indices, neighbor_rings


def load_dataset_for_training(
    config: Optional[V26ProConfig] = None,
    sample_ratio: float = 1.0,
) -> Dict:
    """
    加载训练数据的便捷函数

    Returns:
        包含所有训练所需数据的字典
    """
    if config is None:
        config = DEFAULT_PRO_CONFIG

    loader = DataLoaderV26(config=config)

    point_features, line_features, polygon_features, direction_features, coords, region_labels, cell_id_map, metadata = \
        loader.build_cell_dataset(sample_ratio=sample_ratio)

    # 双分辨率特征：res=8 基础 + res=9 精细
    if config.h3.use_dual_resolution and config.h3.resolution_fine != config.h3.resolution:
        import h3
        print(f"Building dual-resolution features: res={config.h3.resolution} + res={config.h3.resolution_fine}")

        # 为每个基础 cell 计算精细分辨率特征
        # 对于每个 cell 中心点，获取其 res=9 的邻居统计
        n_cells = len(coords)
        point_features_fine = np.zeros((n_cells, 32), dtype=np.float32)
        line_features_fine = np.zeros((n_cells, 16), dtype=np.float32)
        polygon_features_fine = np.zeros((n_cells, 16), dtype=np.float32)

        # 构建 res=9 的 cell 到索引的映射（用于精细特征查询）
        # 先收集所有 res=9 cell 的数据
        res9_cells = {}
        for idx, (lng, lat) in enumerate(coords):
            res9_id = h3.latlng_to_cell(lat, lng, config.h3.resolution_fine)
            if res9_id not in res9_cells:
                res9_cells[res9_id] = []
            res9_cells[res9_id].append(idx)

        # 为每个 res=9 cell 计算精细特征（局部密度、类别分布等）
        # 这里简化实现：使用坐标的局部归一化
        for idx in range(n_cells):
            lng, lat = coords[idx]

            # 计算局部坐标（相对于 res=9 cell 的中心）
            res9_id = h3.latlng_to_cell(lat, lng, config.h3.resolution_fine)
            res9_center = h3.cell_to_latlng(res9_id)
            res9_center_lng, res9_center_lat = res9_center[1], res9_center[0]

            # 局部归一化坐标（在 res=9 cell 内的相对位置）
            cell_size = h3.exact_edge_length(res9_id, unit='km') * 1000  # 米
            local_lng = (lng - res9_center_lng) / 0.001  # 归一化
            local_lat = (lat - res9_center_lat) / 0.001

            # 精细特征：局部坐标 + 基础特征的变换
            point_features_fine[idx, 0] = local_lng
            point_features_fine[idx, 1] = local_lat
            point_features_fine[idx, 2:] = point_features[idx, :30]  # 保留大部分基础特征

            line_features_fine[idx, 0] = local_lng
            line_features_fine[idx, 1] = local_lat
            line_features_fine[idx, 2:] = line_features[idx, :14]

            polygon_features_fine[idx, 0] = local_lng
            polygon_features_fine[idx, 1] = local_lat
            polygon_features_fine[idx, 2:] = polygon_features[idx, :14]

        # 拼接双分辨率特征：基础特征 + 精细特征
        point_features = np.concatenate([point_features, point_features_fine], axis=1)
        line_features = np.concatenate([line_features, line_features_fine], axis=1)
        polygon_features = np.concatenate([polygon_features, polygon_features_fine], axis=1)

        print(f"  Dual-resolution features: point={point_features.shape}, line={line_features.shape}, polygon={polygon_features.shape}")

    # 计算邻居关系 - 使用配置中的 K 值
    k_neighbors = config.loss.k_nearest_neighbors
    neighbor_indices, neighbor_rings = compute_neighbor_indices(
        coords, config.h3.resolution, k_neighbors=k_neighbors
    )

    # 计算方向监督（使用邻居相对方向）
    direction_supervisor = MultiSchemeDirectionSupervision()
    direction_supervisor.compute_all(
        cell_coords=coords,
        neighbor_indices=neighbor_indices,
    )
    direction_labels, direction_weights, direction_valid = direction_supervisor.get_labels_for_training()

    return {
        "point_features": point_features,
        "line_features": line_features,
        "polygon_features": polygon_features,
        "direction_features": direction_features,
        "coords": coords,
        "region_labels": region_labels,  # 真实功能区标签
        "cell_id_map": cell_id_map,
        "neighbor_indices": neighbor_indices,
        "neighbor_rings": neighbor_rings,
        "direction_labels": direction_labels,
        "direction_weights": direction_weights,
        "direction_valid": direction_valid,
        "metadata": metadata,  # 新增：POI元数据
    }


if __name__ == "__main__":
    # 测试数据加载
    print("Testing data loader...")

    data = load_dataset_for_training(sample_ratio=0.1)

    print(f"\nDataset shapes:")
    print(f"  Point features: {data['point_features'].shape}")
    print(f"  Line features: {data['line_features'].shape}")
    print(f"  Polygon features: {data['polygon_features'].shape}")
    print(f"  Direction features: {data['direction_features'].shape}")
    print(f"  Coords: {data['coords'].shape}")
    print(f"  Neighbor indices: {data['neighbor_indices'].shape}")
    print(f"  Direction labels: {data['direction_labels'].shape}")
