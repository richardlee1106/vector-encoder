# -*- coding: utf-8 -*-
"""
测试空间编码器API

用法:
    1. 先启动API服务: python run_api.py
    2. 运行测试: python test_api.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8100"


def test_root():
    """测试根路由"""
    print("\n1. 测试根路由...")
    resp = requests.get(f"{BASE_URL}/")
    print(f"   状态: {resp.status_code}")
    print(f"   响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    return resp.status_code == 200


def test_stats():
    """测试统计接口"""
    print("\n2. 测试统计接口...")
    resp = requests.get(f"{BASE_URL}/stats")
    print(f"   状态: {resp.status_code}")
    data = resp.json()
    print(f"   POI数量: {data.get('num_pois', 0):,}")
    print(f"   Silhouette: {data.get('silhouette', 'N/A')}")
    print(f"   FAISS: {'可用' if data.get('faiss_available') else '不可用'}")
    return resp.status_code == 200


def test_spatial_search():
    """测试空间搜索"""
    print("\n3. 测试空间搜索...")

    # 先获取一个有效的POI ID
    stats_resp = requests.get(f"{BASE_URL}/stats")
    if stats_resp.status_code != 200:
        print("   跳过: 无法获取服务状态")
        return False

    # 测试坐标搜索
    payload = {
        "lng": 114.35,
        "lat": 30.55,
        "top_k": 5
    }

    resp = requests.post(f"{BASE_URL}/spatial_search", json=payload)
    print(f"   状态: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        results = data.get('results', [])
        print(f"   找到 {len(results)} 个相似POI:")
        for i, r in enumerate(results[:3]):
            print(f"     {i+1}. POI {r['poi_id']}, 距离={r['distance']:.4f}")
        return True
    else:
        print(f"   错误: {resp.text}")
        return False


def test_encode():
    """测试编码接口"""
    print("\n4. 测试编码接口...")

    payload = {
        "category": "餐饮服务",
        "landuse": "商业用地",
        "aoi_type": "商业区",
        "road_class": "主干道",
        "density": 50.0,
        "entropy": 2.5,
        "road_dist": 100.0,
        "lng": 114.35,
        "lat": 30.55
    }

    resp = requests.post(f"{BASE_URL}/encode", json=payload)
    print(f"   状态: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        emb = data.get('embedding', [])
        print(f"   Embedding维度: {len(emb)}")
        print(f"   前5维: {emb[:5]}")
        return True
    else:
        print(f"   错误: {resp.text}")
        return False


def main():
    print("=" * 60)
    print("V2.3 空间编码器 API 测试")
    print("=" * 60)

    # 检查服务是否运行
    try:
        requests.get(f"{BASE_URL}/", timeout=2)
    except:
        print("\n错误: API服务未运行")
        print("请先运行: python run_api.py")
        return

    results = []

    # 运行测试
    results.append(("根路由", test_root()))
    results.append(("统计接口", test_stats()))
    results.append(("空间搜索", test_spatial_search()))
    results.append(("编码接口", test_encode()))

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)")


if __name__ == "__main__":
    main()
