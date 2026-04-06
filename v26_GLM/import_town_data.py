# -*- coding: utf-8 -*-
"""
三镇数据导入脚本 - 简化版

将老三镇（武昌、汉口、汉阳）的矢量数据导入 PostgreSQL/PostGIS

数据源：D:/AAA_Edu/TagCloud/三镇原始矢量数据/
目标数据库：geoloom-spatial-db (port 15432)

Author: Sisyphus
Date: 2026-03-19
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============

DATA_DIR = Path("D:/AAA_Edu/TagCloud/三镇原始矢量数据")
DB_PARAMS = {
    'host': 'localhost',
    'port': 15432,
    'database': 'geoloom',
    'user': 'postgres',
    'password': '123456'
}

def get_connection():
    """获取数据库连接"""
    return psycopg2.connect(**DB_PARAMS)

def execute_sql(conn, sql, description=""):
    """执行 SQL 语句"""
    with conn.cursor() as cur:
        try:
            cur.execute(sql)
            conn.commit()
            if description:
                print(f"  [OK] {description}")
        except Exception as e:
            conn.rollback()
            print(f"  [ERR] {e}")

def create_tables(conn):
    """创建表结构"""
    print("\n" + "=" * 60)
    print("创建表结构")
    print("=" * 60)

    # 删除所有表
    execute_sql(conn, "DROP TABLE IF EXISTS population_grid CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS road_blocks CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS districts CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS streets CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS landuse CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS aois CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS roads CASCADE;", "")
    execute_sql(conn, "DROP TABLE IF EXISTS pois CASCADE;", "")

    # POI 表
    sql_pois = """
    CREATE TABLE pois (
        id SERIAL PRIMARY KEY,
        name TEXT,
        category_main TEXT,
        category_sub TEXT,
        longitude DOUBLE PRECISION,
        latitude DOUBLE PRECISION,
        city TEXT,
        geom GEOMETRY(Point, 4326)
    );
    CREATE INDEX idx_pois_geom ON pois USING GIST(geom);
    CREATE INDEX idx_pois_category ON pois(category_main);
    """
    execute_sql(conn, sql_pois, "pois")

    # 道路表
    sql_roads = """
    CREATE TABLE roads (
        id SERIAL PRIMARY KEY,
        osm_id TEXT,
        code TEXT,
        fclass TEXT,
        name TEXT,
        oneway TEXT,
        maxspeed TEXT,
        length_km DOUBLE PRECISION,
        geom GEOMETRY(LineString, 4326)
    );
    CREATE INDEX idx_roads_geom ON roads USING GIST(geom);
    """
    execute_sql(conn, sql_roads, "roads")

    # AOI 表
    sql_aois = """
    CREATE TABLE aois (
        id SERIAL PRIMARY KEY,
        osm_id TEXT,
        code TEXT,
        fclass TEXT,
        name TEXT,
        population DOUBLE PRECISION,
        area_sqm DOUBLE PRECISION,
        geom GEOMETRY(Polygon, 4326)
    );
    CREATE INDEX idx_aois_geom ON aois USING GIST(geom);
    """
    execute_sql(conn, sql_aois, "aois")

    # 土地利用表
    sql_landuse = """
    CREATE TABLE landuse (
        id SERIAL PRIMARY KEY,
        land_type TEXT,
        area_sqm DOUBLE PRECISION,
        geom GEOMETRY(Polygon, 4326)
    );
    CREATE INDEX idx_landuse_geom ON landuse USING GIST(geom);
    """
    execute_sql(conn, sql_landuse, "landuse")

    # 街道表
    sql_streets = """
    CREATE TABLE streets (
        id SERIAL PRIMARY KEY,
        street_name TEXT,
        district_name TEXT,
        area_sqm DOUBLE PRECISION,
        geom GEOMETRY(Polygon, 4326)
    );
    CREATE INDEX idx_streets_geom ON streets USING GIST(geom);
    """
    execute_sql(conn, sql_streets, "streets")

    # 区级表
    sql_districts = """
    CREATE TABLE districts (
        id SERIAL PRIMARY KEY,
        name TEXT,
        eng_name TEXT,
        area_sqm DOUBLE PRECISION,
        geom GEOMETRY(Polygon, 4326)
    );
    CREATE INDEX idx_districts_geom ON districts USING GIST(geom);
    """
    execute_sql(conn, sql_districts, "districts")

    # 路网闭合地块表
    sql_road_blocks = """
    CREATE TABLE road_blocks (
        id SERIAL PRIMARY KEY,
        block_id TEXT,
        area_sqm DOUBLE PRECISION,
        perimeter_m DOUBLE PRECISION,
        geom GEOMETRY(Polygon, 4326)
    );
    CREATE INDEX idx_road_blocks_geom ON road_blocks USING GIST(geom);
    """
    execute_sql(conn, sql_road_blocks, "road_blocks")

    # 人口栅格点表
    sql_population = """
    CREATE TABLE population_grid (
        id SERIAL PRIMARY KEY,
        longitude DOUBLE PRECISION,
        latitude DOUBLE PRECISION,
        population DOUBLE PRECISION,
        geom GEOMETRY(Point, 4326)
    );
    CREATE INDEX idx_population_geom ON population_grid USING GIST(geom);
    """
    execute_sql(conn, sql_population, "population_grid")

    print("\n表结构创建完成！")

def import_pois(conn):
    """导入 POI 数据"""
    print("\n" + "=" * 60)
    print("导入 POI 数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇POI数据.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")

    # 重命名字段
    gdf.columns = ['name', 'category_main', 'category_sub', 'longitude', 'latitude', 'city', 'geometry']

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 写入数据库
    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO pois (name, category_main, category_sub, longitude, latitude, city, geom)
                VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (row['name'], row['category_main'], row['category_sub'],
                  row['longitude'], row['latitude'], row['city'], geom_wkt))

            if idx % 50000 == 0 and idx > 0:
                conn.commit()
                print(f"    已写入 {idx:,} 条...")

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条 POI 记录")

def import_roads(conn):
    """导入道路数据"""
    print("\n" + "=" * 60)
    print("导入道路数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇路网.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")
    print(f"  字段: {list(gdf.columns)}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算道路长度
    gdf['length_km'] = gdf.geometry.to_crs('EPSG:3857').length / 1000

    # 重命名字段
    col_map = {}
    for col in gdf.columns:
        if 'objectid' in col.lower():
            col_map[col] = 'osm_id'
    gdf = gdf.rename(columns=col_map)

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO roads (osm_id, code, fclass, name, oneway, maxspeed, length_km, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (
                row.get('osm_id', ''),
                row.get('code', ''),
                row.get('fclass', ''),
                row.get('name', ''),
                row.get('oneway', ''),
                row.get('maxspeed', ''),
                row['length_km'],
                geom_wkt
            ))

            if idx % 10000 == 0 and idx > 0:
                conn.commit()
                print(f"    已写入 {idx:,} 条...")

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条道路记录")

def import_aois(conn):
    """导入 AOI 数据"""
    print("\n" + "=" * 60)
    print("导入 AOI 数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇AOI.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算面积
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area

    # 重命名字段
    col_map = {}
    for col in gdf.columns:
        if 'osm_id' in col.lower():
            col_map[col] = 'osm_id'
        elif 'pop' in col.lower():
            col_map[col] = 'population'
    gdf = gdf.rename(columns=col_map)

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO aois (osm_id, code, fclass, name, population, area_sqm, geom)
                VALUES (%s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (
                row.get('osm_id', ''),
                row.get('code', ''),
                row.get('fclass', ''),
                row.get('name', ''),
                row.get('population', 0),
                row['area_sqm'],
                geom_wkt
            ))

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条 AOI 记录")

def import_landuse(conn):
    """导入土地利用数据"""
    print("\n" + "=" * 60)
    print("导入土地利用数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇土地利用.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")
    print(f"  字段: {list(gdf.columns)}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算面积
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area

    # 查找类型字段
    type_col = None
    for col in gdf.columns:
        if '类' in str(col) or 'type' in str(col).lower():
            type_col = col
            break

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            land_type = str(row.get(type_col, '')) if type_col else ''
            cur.execute("""
                INSERT INTO landuse (land_type, area_sqm, geom)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326))
            """, (land_type, row['area_sqm'], geom_wkt))

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条土地利用记录")

def import_streets(conn):
    """导入街道数据"""
    print("\n" + "=" * 60)
    print("导入街道数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇街道级面.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算面积
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area

    # 查找字段
    street_col = None
    district_col = None
    for col in gdf.columns:
        if '街道' in str(col):
            street_col = col
        elif '区' in str(col):
            district_col = col

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO streets (street_name, district_name, area_sqm, geom)
                VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (
                str(row.get(street_col, '')) if street_col else '',
                str(row.get(district_col, '')) if district_col else '',
                row['area_sqm'],
                geom_wkt
            ))

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条街道记录")

def import_districts(conn):
    """导入区级数据"""
    print("\n" + "=" * 60)
    print("导入区级数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇区级面.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")
    print(f"  字段: {list(gdf.columns)}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算面积
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area

    # 查找字段
    name_col = None
    eng_col = None
    for col in gdf.columns:
        if str(col) in ['名称', 'name']:
            name_col = col
        elif 'eng' in str(col).lower():
            eng_col = col

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO districts (name, eng_name, area_sqm, geom)
                VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (
                str(row.get(name_col, '')) if name_col else '',
                str(row.get(eng_col, '')) if eng_col else '',
                row['area_sqm'],
                geom_wkt
            ))

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条区级记录")

def import_road_blocks(conn):
    """导入路网闭合地块数据"""
    print("\n" + "=" * 60)
    print("导入路网闭合地块数据")
    print("=" * 60)

    shp_path = DATA_DIR / "三镇路网闭合地块.shp"
    print(f"读取: {shp_path}")

    gdf = gpd.read_file(shp_path, encoding='utf-8')
    print(f"  记录数: {len(gdf):,}")

    # 确保坐标系统
    gdf = gdf.to_crs('EPSG:4326')

    # 计算面积和周长
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area
    gdf['perimeter_m'] = gdf.geometry.to_crs('EPSG:3857').length

    # 查找 ID 字段
    id_col = None
    for col in gdf.columns:
        if 'objectid' in col.lower():
            id_col = col
            break

    print("  写入数据库...")
    with conn.cursor() as cur:
        for idx, row in gdf.iterrows():
            geom_wkt = row.geometry.wkt
            cur.execute("""
                INSERT INTO road_blocks (block_id, area_sqm, perimeter_m, geom)
                VALUES (%s, %s, %s, ST_GeomFromText(%s, 4326))
            """, (
                str(row.get(id_col, '')) if id_col else str(idx),
                row['area_sqm'],
                row['perimeter_m'],
                geom_wkt
            ))

        conn.commit()
    print(f"  [OK] 导入 {len(gdf):,} 条路网闭合地块记录")

def import_population_raster(conn):
    """导入人口栅格数据"""
    print("\n" + "=" * 60)
    print("导入人口栅格数据 (100m)")
    print("=" * 60)

    try:
        import rasterio
    except ImportError:
        print("  [SKIP] rasterio 未安装")
        return

    tif_path = DATA_DIR / "三镇人口栅格100米.tif"
    print(f"读取: {tif_path}")

    with rasterio.open(tif_path) as src:
        print(f"  CRS: {src.crs}")
        print(f"  尺寸: {src.width} x {src.height}")

        # 读取数据
        data = src.read(1)
        transform = src.transform

        # 获取有效数据的坐标
        rows, cols = np.where(data > 0)
        values = data[rows, cols]

        print(f"  有效栅格数: {len(values):,}")

        # 转换为经纬度
        coords = [transform * (col, row) for row, col in zip(rows, cols)]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

    print("  写入数据库...")
    with conn.cursor() as cur:
        for i, (lon, lat, pop) in enumerate(zip(lons, lats, values)):
            cur.execute("""
                INSERT INTO population_grid (longitude, latitude, population, geom)
                VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
            """, (lon, lat, float(pop), lon, lat))

            if i % 10000 == 0 and i > 0:
                conn.commit()
                print(f"    已写入 {i:,} 条...")

        conn.commit()
    print(f"  [OK] 导入 {len(values):,} 个人口栅格点")

def verify_import(conn):
    """验证导入结果"""
    print("\n" + "=" * 60)
    print("验证导入结果")
    print("=" * 60)

    tables = ['pois', 'roads', 'aois', 'landuse', 'streets', 'districts', 'road_blocks', 'population_grid']

    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  {table}: {count:,} records")

def main():
    print("\n" + "#" * 60)
    print("# 三镇数据导入脚本")
    print("# 数据源: D:/AAA_Edu/TagCloud/三镇原始矢量数据/")
    print("# 目标: PostgreSQL/PostGIS (port 15432)")
    print("#" * 60)

    conn = get_connection()

    # 1. 创建表结构
    create_tables(conn)

    # 2. 导入数据
    import_pois(conn)
    import_roads(conn)
    import_aois(conn)
    import_landuse(conn)
    import_streets(conn)
    import_districts(conn)
    import_road_blocks(conn)
    import_population_raster(conn)

    # 3. 验证
    verify_import(conn)

    conn.close()

    print("\n" + "#" * 60)
    print("# 数据导入完成！")
    print("#" * 60)

if __name__ == "__main__":
    main()
