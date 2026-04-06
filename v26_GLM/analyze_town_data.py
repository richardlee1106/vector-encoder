# -*- coding: utf-8 -*-
"""
三镇数据概况分析脚本

对比老三镇与武汉全市的数据特征差异，为模型参数调整提供依据。

分析内容：
1. POI 密度、分布、类别熵
2. 道路密度、等级分布
3. 土地利用类型分布
4. 人口分布（新增）
5. 空间聚集特征

Author: Sisyphus
Date: 2026-03-19
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import psycopg2
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 数据库连接
DB_PARAMS = {
    'host': 'localhost',
    'port': 15432,
    'database': 'geoloom',
    'user': 'postgres',
    'password': '123456'
}

def get_connection():
    return psycopg2.connect(**DB_PARAMS)

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def analyze_pois(conn):
    """分析 POI 数据"""
    print_section("1. POI 数据分析")

    cur = conn.cursor()

    # 基本统计
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(DISTINCT category_main) as main_cats,
            COUNT(DISTINCT category_sub) as sub_cats,
            MIN(longitude) as lng_min, MAX(longitude) as lng_max,
            MIN(latitude) as lat_min, MAX(latitude) as lat_max
        FROM pois
    """)
    row = cur.fetchone()

    print(f"\n基本统计:")
    print(f"  总数: {row[0]:,}")
    print(f"  主类别数: {row[1]}")
    print(f"  子类别数: {row[2]}")
    print(f"  经度范围: {row[3]:.4f} - {row[4]:.4f}")
    print(f"  纬度范围: {row[5]:.4f} - {row[6]:.4f}")

    # 计算面积（平方公里）
    lng_span = row[4] - row[3]
    lat_span = row[6] - row[5]
    # 粗略面积估算（1度约111km）
    area_km2 = lng_span * 111 * lat_span * 111 * np.cos(np.radians((row[5] + row[6]) / 2))
    density = row[0] / area_km2
    print(f"  覆盖面积（估算）: {area_km2:.1f} km²")
    print(f"  POI 密度: {density:.1f} 个/km²")

    # 主类别分布
    print(f"\n主类别分布 (Top 15):")
    cur.execute("""
        SELECT category_main, COUNT(*) as cnt
        FROM pois
        GROUP BY category_main
        ORDER BY cnt DESC
        LIMIT 15
    """)
    rows = cur.fetchall()
    total = sum(r[1] for r in rows)
    for cat, cnt in rows:
        pct = cnt / row[0] * 100
        print(f"  {cat}: {cnt:,} ({pct:.1f}%)")

    # 计算类别熵
    cur.execute("""
        SELECT category_main, COUNT(*) as cnt
        FROM pois
        GROUP BY category_main
    """)
    rows = cur.fetchall()
    counts = np.array([r[1] for r in rows])
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    print(f"\n类别多样性:")
    print(f"  类别数: {len(counts)}")
    print(f"  Shannon 熵: {entropy:.3f}")
    print(f"  归一化熵: {normalized_entropy:.3f} (1.0 = 完全均匀)")

    # 武汉全市对比（历史数据）
    print(f"\n与武汉全市对比:")
    wuhan_poi_count = 800000  # 历史数据
    wuhan_area_km2 = 8569  # 武汉全市面积
    wuhan_density = wuhan_poi_count / wuhan_area_km2

    print(f"  武汉全市 POI 密度: {wuhan_density:.1f} 个/km²")
    print(f"  三镇 POI 密度: {density:.1f} 个/km²")
    print(f"  密度倍数: {density / wuhan_density:.1f}x")

    cur.close()
    return {
        'total': row[0],
        'area_km2': area_km2,
        'density': density,
        'entropy': normalized_entropy,
        'main_categories': len(counts)
    }

def analyze_roads(conn):
    """分析道路数据"""
    print_section("2. 道路数据分析")

    cur = conn.cursor()

    # 基本统计
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(length_km) as total_length,
            AVG(length_km) as avg_length
        FROM roads
    """)
    row = cur.fetchone()

    print(f"\n基本统计:")
    print(f"  道路数: {row[0]:,}")
    print(f"  总长度: {row[1]:.1f} km")
    print(f"  平均长度: {row[2]:.2f} km")

    # 道路等级分布
    print(f"\n道路等级分布:")
    cur.execute("""
        SELECT fclass, COUNT(*) as cnt, SUM(length_km) as length
        FROM roads
        GROUP BY fclass
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    for fclass, cnt, length in rows[:10]:
        print(f"  {fclass}: {cnt:,} 条, {length:.1f} km")

    # 道路密度
    cur.execute("SELECT SUM(length_km) FROM roads")
    total_length = cur.fetchone()[0]

    # 估算面积（从 POI）
    cur.execute("""
        SELECT (MAX(longitude) - MIN(longitude)) * (MAX(latitude) - MIN(latitude))
        FROM pois
    """)
    area_deg2 = cur.fetchone()[0]
    area_km2 = area_deg2 * 111 * 111 * 0.85  # 修正系数

    road_density = total_length / area_km2 if area_km2 > 0 else 0
    print(f"\n道路密度: {road_density:.2f} km/km²")

    # 武汉全市对比
    wuhan_road_density = 1.2  # 历史数据估算
    print(f"武汉全市道路密度: ~{wuhan_road_density:.1f} km/km²")
    print(f"密度倍数: {road_density / wuhan_road_density:.1f}x")

    cur.close()
    return {
        'total': row[0],
        'total_length_km': row[1],
        'road_density': road_density
    }

def analyze_landuse(conn):
    """分析土地利用数据"""
    print_section("3. 土地利用数据分析")

    cur = conn.cursor()

    # 基本统计
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(area_sqm) / 1000000 as total_area_km2
        FROM landuse
    """)
    row = cur.fetchone()

    print(f"\n基本统计:")
    print(f"  地块数: {row[0]:,}")
    print(f"  总面积: {row[1]:.1f} km²")

    # 类型分布
    print(f"\n土地利用类型分布:")
    cur.execute("""
        SELECT land_type, COUNT(*) as cnt, SUM(area_sqm) / 1000000 as area_km2
        FROM landuse
        GROUP BY land_type
        ORDER BY area_km2 DESC
    """)
    rows = cur.fetchall()
    for land_type, cnt, area in rows:
        print(f"  {land_type}: {cnt:,} 块, {area:.2f} km²")

    cur.close()
    return {
        'total': row[0],
        'total_area_km2': row[1]
    }

def analyze_population(conn):
    """分析人口栅格数据"""
    print_section("4. 人口栅格数据分析 (100m)")

    cur = conn.cursor()

    # 基本统计
    cur.execute("""
        SELECT
            COUNT(*) as total_cells,
            SUM(population) as total_pop,
            AVG(population) as avg_pop,
            MIN(population) as min_pop,
            MAX(population) as max_pop,
            STDDEV(population) as std_pop
        FROM population_grid
    """)
    row = cur.fetchone()

    print(f"\n基本统计:")
    print(f"  栅格数: {row[0]:,}")
    print(f"  总人口（估算）: {row[1]:,.0f}")
    print(f"  平均人口/格: {row[2]:.1f}")
    print(f"  人口范围: {row[3]:.0f} - {row[4]:.0f}")
    print(f"  标准差: {row[5]:.1f}")

    # 人口密度分布
    print(f"\n人口密度分布:")
    cur.execute("""
        SELECT
            CASE
                WHEN population < 10 THEN '0-10'
                WHEN population < 50 THEN '10-50'
                WHEN population < 100 THEN '50-100'
                WHEN population < 200 THEN '100-200'
                ELSE '200+'
            END as pop_range,
            COUNT(*) as cnt
        FROM population_grid
        GROUP BY pop_range
        ORDER BY pop_range
    """)
    rows = cur.fetchall()
    for pop_range, cnt in rows:
        pct = cnt / row[0] * 100
        print(f"  {pop_range} 人/格: {cnt:,} ({pct:.1f}%)")

    # 人口密度计算
    # 100m x 100m = 0.01 km²
    cell_area_km2 = 0.01
    total_area_km2 = row[0] * cell_area_km2
    pop_density = row[1] / total_area_km2 if total_area_km2 > 0 else 0

    print(f"\n人口密度估算:")
    print(f"  覆盖面积: {total_area_km2:.1f} km²")
    print(f"  平均人口密度: {pop_density:,.0f} 人/km²")

    # 三镇人口密度对比
    print(f"\n参考对比:")
    print(f"  武汉全市平均: ~1,200 人/km²")
    print(f"  核心城区: ~15,000 人/km²")
    print(f"  三镇估算: {pop_density:,.0f} 人/km²")

    cur.close()
    return {
        'total_cells': row[0],
        'total_pop': row[1],
        'avg_density': pop_density
    }

def analyze_aois(conn):
    """分析 AOI 数据"""
    print_section("5. AOI 数据分析")

    cur = conn.cursor()

    # 基本统计
    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(area_sqm) / 1000000 as total_area_km2,
            AVG(area_sqm) as avg_area
        FROM aois
    """)
    row = cur.fetchone()

    print(f"\n基本统计:")
    print(f"  AOI 数: {row[0]:,}")
    print(f"  总面积: {row[1]:.1f} km²")
    print(f"  平均面积: {row[2]:.0f} m²")

    # AOI 类型分布
    print(f"\nAOI 类型分布 (Top 10):")
    cur.execute("""
        SELECT fclass, COUNT(*) as cnt, SUM(area_sqm) / 1000000 as area_km2
        FROM aois
        GROUP BY fclass
        ORDER BY cnt DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    for fclass, cnt, area in rows:
        print(f"  {fclass}: {cnt:,} 个, {area:.2f} km²")

    cur.close()
    return {
        'total': row[0],
        'total_area_km2': row[1]
    }

def analyze_spatial_distribution(conn):
    """分析空间分布特征"""
    print_section("6. 空间分布特征")

    cur = conn.cursor()

    # POI 空间聚集度
    cur.execute("""
        SELECT
            COUNT(DISTINCT ROUND(longitude * 100)) as lng_cells,
            COUNT(DISTINCT ROUND(latitude * 100)) as lat_cells,
            COUNT(*) as total
        FROM pois
    """)
    row = cur.fetchone()

    # 计算 Moran's I 近似（简化版）
    # 使用格网方法
    cur.execute("""
        WITH grid AS (
            SELECT
                FLOOR((longitude - 113.6) * 100) as grid_x,
                FLOOR((latitude - 30.5) * 100) as grid_y,
                COUNT(*) as poi_cnt
            FROM pois
            GROUP BY grid_x, grid_y
        )
        SELECT
            COUNT(*) as grid_count,
            AVG(poi_cnt) as avg_poi,
            STDDEV(poi_cnt) as std_poi,
            MAX(poi_cnt) as max_poi
        FROM grid
    """)
    row = cur.fetchone()

    print(f"\nPOI 空间分布（0.01度网格 ≈ 1km）:")
    print(f"  有 POI 的网格数: {row[0]:,}")
    print(f"  平均 POI/网格: {row[1]:.1f}")
    print(f"  标准差: {row[2]:.1f}")
    print(f"  最大值: {row[3]}")

    # 聚集度系数（变异系数）
    cv = row[2] / row[1] if row[1] > 0 else 0
    print(f"  变异系数: {cv:.2f} (越大越不均匀)")

    if cv > 2:
        print(f"  → POI 高度聚集在少数区域")
    elif cv > 1:
        print(f"  → POI 中等聚集")
    else:
        print(f"  → POI 分布较均匀")

    cur.close()
    return {
        'grid_count': row[0],
        'avg_poi_per_grid': row[1],
        'cv': cv
    }

def generate_summary(poi_stats, road_stats, landuse_stats, pop_stats, aoi_stats, spatial_stats):
    """生成总结报告"""
    print_section("7. 总结与建议")

    print("\n【数据概况】")
    print(f"  POI 总数: {poi_stats['total']:,}")
    print(f"  覆盖面积: {poi_stats['area_km2']:.1f} km²")
    print(f"  POI 密度: {poi_stats['density']:.0f} 个/km²")
    print(f"  道路密度: {road_stats['road_density']:.2f} km/km²")
    print(f"  人口密度: {pop_stats['avg_density']:,.0f} 人/km²")

    print("\n【与武汉全市对比】")
    print(f"  POI 密度: 三镇是全市的 ~{poi_stats['density'] / 93:.0f} 倍")
    print(f"  道路密度: 三镇是全市的 ~{road_stats['road_density'] / 1.2:.0f} 倍")
    print(f"  人口密度: 三镇是全市的 ~{pop_stats['avg_density'] / 1200:.0f} 倍")

    print("\n【模型参数建议】")

    # H3 分辨率建议
    poi_per_cell_9 = poi_stats['density'] * 0.01  # res=9 约 0.01 km²
    poi_per_cell_8 = poi_stats['density'] * 0.1    # res=8 约 0.1 km²

    print(f"\n  H3 分辨率:")
    print(f"    res=9 (约0.01km²): 预计 {poi_per_cell_9:.0f} POI/格")
    print(f"    res=8 (约0.1km²): 预计 {poi_per_cell_8:.0f} POI/格")

    if poi_per_cell_9 > 50:
        print(f"    → 建议: res=9 仍可保持足够样本量")
    else:
        print(f"    → 建议: 考虑使用 res=8 以获得更多邻居")

    # K_neighbors 建议
    print(f"\n  K_neighbors (邻居数):")
    if spatial_stats['cv'] > 2:
        print(f"    POI 高度聚集，建议 K=20-30 获取更多上下文")
    else:
        print(f"    POI 分布较均匀，K=15-20 足够")

    # 类别熵建议
    print(f"\n  类别多样性:")
    print(f"    归一化熵: {poi_stats['entropy']:.3f}")
    if poi_stats['entropy'] > 0.7:
        print(f"    类别非常多样，特征维度可能需要增加")

    # 人口栅格利用
    print(f"\n  人口栅格数据:")
    print(f"    可作为新的特征维度加入")
    print(f"    建议: 将人口密度归一化后加入 point_features")

    # 训练参数建议
    print(f"\n  训练参数:")
    print(f"    batch_size: 数据量较小，可降至 8192-12288")
    print(f"    epochs: 可增加至 100-120（过拟合风险较低）")
    print(f"    learning_rate: 保持 3e-4 或略降")

def main():
    print("#" * 60)
    print("# 三镇数据概况分析")
    print("# 数据源: geoloom (port 15432)")
    print("#" * 60)

    conn = get_connection()

    try:
        poi_stats = analyze_pois(conn)
        road_stats = analyze_roads(conn)
        landuse_stats = analyze_landuse(conn)
        pop_stats = analyze_population(conn)
        aoi_stats = analyze_aois(conn)
        spatial_stats = analyze_spatial_distribution(conn)

        generate_summary(poi_stats, road_stats, landuse_stats, pop_stats, aoi_stats, spatial_stats)

    finally:
        conn.close()

    print("\n" + "#" * 60)
    print("# 分析完成")
    print("#" * 60)

if __name__ == "__main__":
    main()
