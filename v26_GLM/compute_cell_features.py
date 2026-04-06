# -*- coding: utf-8 -*-
"""
三镇多维度特征计算脚本

功能：
1. 创建 H3 Cells 表
2. 聚合 POI/道路/土地利用/人口到每个 Cell
3. 计算类别熵、道路密度、人口密度等特征
4. 存储到数据库 cells 表

新增特征维度：
- category_entropy: POI 类别熵（多样性）
- poi_density: POI 密度
- road_density: 道路密度
- population_density: 人口密度
- dominant_category: 主导类别
- landuse_mix: 土地利用混合度

Author: Sisyphus
Date: 2026-03-19
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import h3
from tqdm import tqdm

# 数据库连接
DB_PARAMS = {
    'host': 'localhost',
    'port': 15432,
    'database': 'geoloom',
    'user': 'postgres',
    'password': '123456'
}

# H3 分辨率（双分辨率策略）
H3_RESOLUTION = 8      # 基础分辨率
H3_RESOLUTION_FINE = 9  # 精细分辨率（可选）

# 是否生成双分辨率数据
DUAL_RESOLUTION = True

def get_connection():
    return psycopg2.connect(**DB_PARAMS)

def create_cells_table(conn):
    """创建 cells 表"""
    print("\n创建 cells 表...")

    cur = conn.cursor()

    # 删除旧表
    cur.execute("DROP TABLE IF EXISTS cells CASCADE;")

    # 创建新表
    cur.execute("""
        CREATE TABLE cells (
            id SERIAL PRIMARY KEY,
            cell_id TEXT UNIQUE NOT NULL,
            longitude DOUBLE PRECISION,
            latitude DOUBLE PRECISION,

            -- POI 统计
            poi_count INTEGER DEFAULT 0,
            poi_density DOUBLE PRECISION DEFAULT 0,
            category_entropy DOUBLE PRECISION DEFAULT 0,
            dominant_category TEXT DEFAULT '',
            category_distribution JSONB DEFAULT '{}',

            -- 道路统计
            road_count INTEGER DEFAULT 0,
            road_density DOUBLE PRECISION DEFAULT 0,
            road_length_km DOUBLE PRECISION DEFAULT 0,

            -- 土地利用统计
            landuse_count INTEGER DEFAULT 0,
            landuse_area_sqm DOUBLE PRECISION DEFAULT 0,
            landuse_mix DOUBLE PRECISION DEFAULT 0,
            dominant_landuse TEXT DEFAULT '',

            -- 人口统计
            population_count INTEGER DEFAULT 0,
            population_density DOUBLE PRECISION DEFAULT 0,
            avg_population DOUBLE PRECISION DEFAULT 0,

            -- AOI 统计
            aoi_count INTEGER DEFAULT 0,
            dominant_aoi TEXT DEFAULT '',

            -- 几何
            geom GEOMETRY(Polygon, 4326),

            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX idx_cells_cell_id ON cells(cell_id);
        CREATE INDEX idx_cells_geom ON cells USING GIST(geom);
        CREATE INDEX idx_cells_category ON cells(dominant_category);
    """)

    conn.commit()
    cur.close()
    print("  [OK] cells 表创建完成")

def aggregate_pois(conn) -> Dict[str, Dict]:
    """聚合 POI 到 H3 Cells"""
    print("\n聚合 POI 数据...")

    cur = conn.cursor()

    # 读取所有 POI
    cur.execute("""
        SELECT id, longitude, latitude, category_main
        FROM pois
        WHERE geom IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"  读取 {len(rows):,} 条 POI")

    # 聚合到 H3 Cells
    cells = {}
    for lng, lat, category in tqdm([(r[1], r[2], r[3]) for r in rows], desc="  处理 POI"):
        cell_id = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)

        if cell_id not in cells:
            cells[cell_id] = {
                'poi_count': 0,
                'categories': [],
                'longitude': lng,
                'latitude': lat,
            }

        cells[cell_id]['poi_count'] += 1
        cells[cell_id]['categories'].append(category)

    cur.close()
    print(f"  [OK] 聚合到 {len(cells):,} 个 Cells")
    return cells

def aggregate_roads(conn, cells: Dict) -> Dict:
    """聚合道路到 H3 Cells"""
    print("\n聚合道路数据...")

    cur = conn.cursor()

    # 读取道路几何
    cur.execute("""
        SELECT id, ST_AsGeoJSON(geom) as geom_json, length_km
        FROM roads
        WHERE geom IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"  读取 {len(rows):,} 条道路")

    import json

    for row in tqdm(rows, desc="  处理道路"):
        geom_json = json.loads(row[1])
        length_km = float(row[2]) if row[2] else 0

        geom_type = geom_json.get('type', '')
        coords = geom_json.get('coordinates', [])

        # 提取所有坐标点
        all_points = []
        if geom_type == 'LineString':
            all_points = coords
        elif geom_type == 'MultiLineString':
            for segment in coords:
                all_points.extend(segment)

        # 聚合到覆盖的 Cells
        cell_set = set()
        for coord in all_points:
            lng, lat = coord[0], coord[1]
            cell_id = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
            cell_set.add(cell_id)

        for cell_id in cell_set:
            if cell_id not in cells:
                cells[cell_id] = {
                    'poi_count': 0,
                    'categories': [],
                    'longitude': sum(c[0] for c in all_points) / len(all_points),
                    'latitude': sum(c[1] for c in all_points) / len(all_points),
                }

            cells[cell_id]['road_count'] = cells[cell_id].get('road_count', 0) + 1
            cells[cell_id]['road_length_km'] = cells[cell_id].get('road_length_km', 0) + length_km / len(cell_set)

    cur.close()
    print(f"  [OK] 道路聚合完成")
    return cells

def aggregate_landuse(conn, cells: Dict) -> Dict:
    """聚合土地利用到 H3 Cells"""
    print("\n聚合土地利用数据...")

    cur = conn.cursor()

    cur.execute("""
        SELECT id, ST_AsGeoJSON(geom) as geom_json, land_type, area_sqm
        FROM landuse
        WHERE geom IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"  读取 {len(rows):,} 条土地利用")

    import json

    for row in tqdm(rows, desc="  处理土地利用"):
        geom_json = json.loads(row[1])
        land_type = row[2] or 'unknown'
        area_sqm = float(row[3]) if row[3] else 0

        geom_type = geom_json.get('type', '')
        coords = geom_json.get('coordinates', [])

        # 提取外环坐标
        if geom_type == 'Polygon':
            outer_ring = coords[0]
        elif geom_type == 'MultiPolygon':
            outer_ring = coords[0][0]
        else:
            continue

        if len(outer_ring) < 3:
            continue

        # 计算中心点
        center_lng = sum(c[0] for c in outer_ring) / len(outer_ring)
        center_lat = sum(c[1] for c in outer_ring) / len(outer_ring)

        # 获取覆盖的 Cells
        try:
            from h3 import LatLngPoly
            # coords 格式: [(lat, lng), ...]
            poly = LatLngPoly([(c[1], c[0]) for c in outer_ring])
            cell_ids = h3.polygon_to_cells(poly, H3_RESOLUTION)
        except Exception as e:
            # 退化到中心点
            cell_id = h3.latlng_to_cell(center_lat, center_lng, H3_RESOLUTION)
            cell_ids = [cell_id]

        if not cell_ids:
            # 退化到中心点
            cell_id = h3.latlng_to_cell(center_lat, center_lng, H3_RESOLUTION)
            cell_ids = [cell_id]

        weight = 1.0 / len(cell_ids) if cell_ids else 1.0

        for cell_id in cell_ids:
            if cell_id not in cells:
                cells[cell_id] = {
                    'poi_count': 0,
                    'categories': [],
                    'longitude': center_lng,
                    'latitude': center_lat,
                }

            cells[cell_id]['landuse_count'] = cells[cell_id].get('landuse_count', 0) + 1
            cells[cell_id]['landuse_area_sqm'] = cells[cell_id].get('landuse_area_sqm', 0) + area_sqm * weight
            cells[cell_id]['landuse_types'] = cells[cell_id].get('landuse_types', [])
            cells[cell_id]['landuse_types'].append(land_type)

    cur.close()
    print(f"  [OK] 土地利用聚合完成")
    return cells

def aggregate_population(conn, cells: Dict) -> Dict:
    """聚合人口栅格到 H3 Cells"""
    print("\n聚合人口栅格数据...")

    cur = conn.cursor()

    cur.execute("""
        SELECT longitude, latitude, population
        FROM population_grid
        WHERE population > 0
    """)

    rows = cur.fetchall()
    print(f"  读取 {len(rows):,} 条人口栅格")

    for lng, lat, pop in tqdm(rows, desc="  处理人口"):
        cell_id = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)

        if cell_id not in cells:
            cells[cell_id] = {
                'poi_count': 0,
                'categories': [],
                'longitude': lng,
                'latitude': lat,
            }

        cells[cell_id]['population_count'] = cells[cell_id].get('population_count', 0) + 1
        cells[cell_id]['total_population'] = cells[cell_id].get('total_population', 0) + float(pop)

    cur.close()
    print(f"  [OK] 人口栅格聚合完成")
    return cells

def aggregate_aois(conn, cells: Dict) -> Dict:
    """聚合 AOI 到 H3 Cells"""
    print("\n聚合 AOI 数据...")

    cur = conn.cursor()

    cur.execute("""
        SELECT id, ST_X(ST_Centroid(geom)) as lng, ST_Y(ST_Centroid(geom)) as lat, fclass
        FROM aois
        WHERE geom IS NOT NULL
    """)

    rows = cur.fetchall()
    print(f"  读取 {len(rows):,} 条 AOI")

    for row in tqdm(rows, desc="  处理 AOI"):
        lng, lat, fclass = row[1], row[2], row[3]
        cell_id = h3.latlng_to_cell(lat, lng, H3_RESOLUTION)

        if cell_id not in cells:
            cells[cell_id] = {
                'poi_count': 0,
                'categories': [],
                'longitude': lng,
                'latitude': lat,
            }

        cells[cell_id]['aoi_count'] = cells[cell_id].get('aoi_count', 0) + 1
        cells[cell_id]['aoi_types'] = cells[cell_id].get('aoi_types', [])
        cells[cell_id]['aoi_types'].append(fclass or 'unknown')

    cur.close()
    print(f"  [OK] AOI 聚合完成")
    return cells

def compute_features(cells: Dict) -> Dict:
    """计算派生特征"""
    print("\n计算派生特征...")

    # H3 Cell 面积（res=8 约 0.053 km², res=9 约 0.01 km²）
    # h3 4.x 不再支持 cell_area，使用固定值
    cell_area_km2 = 0.053 if H3_RESOLUTION == 8 else 0.01

    for cell_id, cell in tqdm(cells.items(), desc="  计算特征"):
        # 1. POI 密度
        cell['poi_density'] = cell.get('poi_count', 0) / cell_area_km2

        # 2. 类别熵
        categories = cell.get('categories', [])
        if categories:
            counts = Counter(categories)
            total = len(categories)
            probs = np.array(list(counts.values())) / total
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(counts)) if len(counts) > 1 else 1
            cell['category_entropy'] = entropy / max_entropy if max_entropy > 0 else 0
            cell['dominant_category'] = counts.most_common(1)[0][0]
            cell['category_distribution'] = {k: v / total for k, v in counts.items()}
        else:
            cell['category_entropy'] = 0
            cell['dominant_category'] = 'unknown'
            cell['category_distribution'] = {}

        # 3. 道路密度
        cell['road_density'] = cell.get('road_length_km', 0) / cell_area_km2

        # 4. 土地利用混合度
        landuse_types = cell.get('landuse_types', [])
        if landuse_types:
            counts = Counter(landuse_types)
            total = len(landuse_types)
            probs = np.array(list(counts.values())) / total
            cell['landuse_mix'] = -np.sum(probs * np.log(probs + 1e-10))
            cell['dominant_landuse'] = counts.most_common(1)[0][0]
        else:
            cell['landuse_mix'] = 0
            cell['dominant_landuse'] = 'unknown'

        # 5. 人口密度
        total_pop = cell.get('total_population', 0)
        pop_count = cell.get('population_count', 0)
        cell['population_density'] = total_pop / cell_area_km2 if cell_area_km2 > 0 else 0
        cell['avg_population'] = total_pop / pop_count if pop_count > 0 else 0

        # 6. AOI 主导类型
        aoi_types = cell.get('aoi_types', [])
        if aoi_types:
            counts = Counter(aoi_types)
            cell['dominant_aoi'] = counts.most_common(1)[0][0]
        else:
            cell['dominant_aoi'] = 'unknown'

    print(f"  [OK] 特征计算完成")
    return cells

def save_to_database(conn, cells: Dict):
    """保存到数据库"""
    print("\n保存到数据库...")

    cur = conn.cursor()

    # 准备数据
    records = []
    for cell_id, cell in cells.items():
        # 获取 H3 Cell 边界
        boundary = h3.cell_to_boundary(cell_id)
        # boundary 是 [(lat, lng), ...] 格式

        # 创建 WKT Polygon
        polygon_coords = [(lng, lat) for lat, lng in boundary]
        polygon_coords.append(polygon_coords[0])  # 闭合
        wkt = f"POLYGON(({', '.join([f'{lng} {lat}' for lng, lat in polygon_coords])}))"

        records.append((
            cell_id,
            cell.get('longitude', 0),
            cell.get('latitude', 0),
            cell.get('poi_count', 0),
            cell.get('poi_density', 0),
            cell.get('category_entropy', 0),
            cell.get('dominant_category', 'unknown'),
            json.dumps(cell.get('category_distribution', {})),
            cell.get('road_count', 0),
            cell.get('road_density', 0),
            cell.get('road_length_km', 0),
            cell.get('landuse_count', 0),
            cell.get('landuse_area_sqm', 0),
            cell.get('landuse_mix', 0),
            cell.get('dominant_landuse', 'unknown'),
            cell.get('population_count', 0),
            cell.get('population_density', 0),
            cell.get('avg_population', 0),
            cell.get('aoi_count', 0),
            cell.get('dominant_aoi', 'unknown'),
            wkt,
        ))

    print(f"  写入 {len(records):,} 条记录...")

    # 批量插入（转换 numpy 类型为 Python 原生类型）
    clean_records = []
    for r in records:
        clean_records.append((
            str(r[0]), float(r[1]), float(r[2]),
            int(r[3]), float(r[4]), float(r[5]), str(r[6]), str(r[7]),
            int(r[8]), float(r[9]), float(r[10]),
            int(r[11]), float(r[12]), float(r[13]), str(r[14]),
            int(r[15]), float(r[16]), float(r[17]),
            int(r[18]), str(r[19]),
        ))

    for i, r in enumerate(clean_records):
        wkt = records[i][20]
        cur.execute("""
            INSERT INTO cells (
                cell_id, longitude, latitude,
                poi_count, poi_density, category_entropy, dominant_category, category_distribution,
                road_count, road_density, road_length_km,
                landuse_count, landuse_area_sqm, landuse_mix, dominant_landuse,
                population_count, population_density, avg_population,
                aoi_count, dominant_aoi,
                geom
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
        """, (*r, wkt))

        if i % 500 == 0 and i > 0:
            conn.commit()
            print(f"    已写入 {i:,} 条...")

    conn.commit()
    cur.close()
    print(f"  [OK] 保存完成")

def verify_results(conn):
    """验证结果"""
    print("\n验证结果...")

    cur = conn.cursor()

    # 统计
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(poi_count) as total_poi,
            AVG(poi_density) as avg_poi_density,
            AVG(category_entropy) as avg_entropy,
            AVG(road_density) as avg_road_density,
            AVG(population_density) as avg_pop_density
        FROM cells
    """)

    row = cur.fetchone()

    print(f"\nCells 统计:")
    print(f"  总 Cell 数: {row[0]:,}")
    print(f"  总 POI 数: {row[1]:,}")
    print(f"  平均 POI 密度: {row[2]:.1f} 个/km²")
    print(f"  平均类别熵: {row[3]:.3f}")
    print(f"  平均道路密度: {row[4]:.2f} km/km²")
    print(f"  平均人口密度: {row[5]:.1f} 人/km²")

    # 类别分布
    print(f"\n主导类别分布:")
    cur.execute("""
        SELECT dominant_category, COUNT(*) as cnt
        FROM cells
        WHERE dominant_category != 'unknown'
        GROUP BY dominant_category
        ORDER BY cnt DESC
        LIMIT 10
    """)
    for cat, cnt in cur.fetchall():
        print(f"  {cat}: {cnt:,}")

    cur.close()

def main():
    print("#" * 60)
    print("# 三镇多维度特征计算")
    print("# H3 分辨率:", H3_RESOLUTION)
    print("#" * 60)

    conn = get_connection()

    try:
        # 1. 创建表
        create_cells_table(conn)

        # 2. 聚合数据
        cells = {}
        cells = aggregate_pois(conn)
        cells = aggregate_roads(conn, cells)
        cells = aggregate_landuse(conn, cells)
        cells = aggregate_population(conn, cells)
        cells = aggregate_aois(conn, cells)

        # 3. 计算特征
        cells = compute_features(cells)

        # 4. 保存
        import json
        save_to_database(conn, cells)

        # 5. 验证
        verify_results(conn)

    finally:
        conn.close()

    print("\n" + "#" * 60)
    print("# 完成")
    print("#" * 60)

if __name__ == "__main__":
    main()
