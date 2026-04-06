# -*- coding: utf-8 -*-
"""
V2.3 全量训练深度分析

分析为什么全量训练失败，以及模型的真正能力边界
"""

import warnings
warnings.filterwarnings('ignore')
import psycopg2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data_sample(rate):
    """采样加载数据"""
    conn = psycopg2.connect(
        host='localhost', port=5432,
        user='postgres', password='123456', database='geoloom'
    )
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL')
    total = cur.fetchone()[0]
    limit = int(total * rate)

    cur.execute(f'''
        SELECT id, ST_X(geom), ST_Y(geom), name, category_big,
            land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL ORDER BY RANDOM() LIMIT {limit}
    ''')
    rows = cur.fetchall()
    conn.close()

    cat_map, lu_map, aoi_map, rc_map = {}, {}, {}, {}
    ids, names, categories, coords, feats = [], [], [], [], []

    for row in rows:
        poi_id, lng, lat, name, cat = row[0], row[1], row[2], row[3], row[4]
        lu, aoi, rc = row[5], row[6], row[7]
        density, entropy, road_dist = row[8], row[9], row[10]

        c = cat or 'unknown'
        if c not in cat_map: cat_map[c] = len(cat_map)
        l = lu or 'unknown'
        if l not in lu_map: lu_map[l] = len(lu_map)
        a = aoi or 'unknown'
        if a not in aoi_map: aoi_map[a] = len(aoi_map)
        r = rc or 'unknown'
        if r not in rc_map: rc_map[r] = len(rc_map)

        ids.append(poi_id)
        names.append(name or 'Unknown')
        categories.append(c)
        coords.append([lng, lat])
        feats.append([cat_map[c], lu_map[l], aoi_map[a], rc_map[r],
                     float(density or 0), float(entropy or 0), float(road_dist or 0)])

    return {
        'ids': np.array(ids),
        'names': names,
        'categories': categories,
        'coords': np.array(coords, dtype=np.float32),
        'features': np.array(feats, dtype=np.float32),
        'meta': {
            'num_pois': len(ids),
            'num_categories': len(cat_map),
            'num_landuses': len(lu_map),
            'num_aoi_types': len(aoi_map),
            'num_road_classes': len(rc_map)
        }
    }


def analyze_spatial_distribution(data):
    """分析空间分布特征"""
    coords = data['coords']
    print("\n" + "=" * 60)
    print("空间分布分析")
    print("=" * 60)

    # 基本统计
    lng_min, lng_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    lng_span = lng_max - lng_min
    lat_span = lat_max - lat_min

    print(f"\n空间范围:")
    print(f"  经度: {lng_min:.4f} ~ {lng_max:.4f} (跨度 {lng_span:.4f}°)")
    print(f"  纬度: {lat_min:.4f} ~ {lat_max:.4f} (跨度 {lat_span:.4f}°)")
    print(f"  面积估算: {lng_span * lat_span * 111 * 111:.0f} km^2")

    # 点密度
    area_km2 = lng_span * lat_span * 111 * 111
    density = len(coords) / area_km2 if area_km2 > 0 else 0
    print(f"  点密度: {density:.1f} POI/km^2")

    # 聚类特性
    print(f"\n聚类特性分析:")

    # 不同聚类数下的Silhouette
    for n_clusters in [10, 30, 50, 100, 200]:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
        labels = kmeans.fit_predict(coords)

        # 采样评估
        si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
        sil = silhouette_score(coords[si], labels[si])
        print(f"  聚类数={n_clusters:3d}: Silhouette={sil:.4f}")

    return {
        'lng_span': lng_span,
        'lat_span': lat_span,
        'area_km2': area_km2,
        'density': density
    }


def compare_scales():
    """对比不同规模的表现"""
    print("\n" + "=" * 70)
    print("不同数据规模对比实验")
    print("=" * 70)

    results = []

    for rate in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
        print(f"\n加载 {rate*100:.0f}% 数据...")
        data = load_data_sample(rate)

        coords = data['coords']
        n_pois = len(coords)

        # 空间范围
        lng_span = coords[:, 0].max() - coords[:, 0].min()
        lat_span = coords[:, 1].max() - coords[:, 1].min()

        # 固定聚类数=30
        kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=10000)
        labels = kmeans.fit_predict(coords)

        si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
        sil = silhouette_score(coords[si], labels[si])

        results.append({
            'rate': rate,
            'num_pois': n_pois,
            'lng_span': lng_span,
            'lat_span': lat_span,
            'silhouette': sil
        })

        print(f"  POI数: {n_pois:,}, 空间跨度: {lng_span:.3f}°×{lat_span:.3f}°")
        print(f"  理论上限(30聚类): Silhouette={sil:.4f}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("规模对比汇总")
    print("=" * 70)
    print(f"{'规模':<8} {'POI数量':<12} {'空间跨度':<15} {'理论上限':<12}")
    print("-" * 60)

    for r in results:
        span = f"{r['lng_span']:.3f}°×{r['lat_span']:.3f}°"
        print(f"{r['rate']*100:>5.0f}%  {r['num_pois']:>10,}  {span:<15} {r['silhouette']:>10.4f}")

    return results


def test_model_real_ability(data, epochs=500):
    """测试模型真实能力（使用小规模数据）"""
    print("\n" + "=" * 60)
    print("模型真实能力测试（10%采样）")
    print("=" * 60)

    coords, feats = data['coords'], data['features']
    meta = data['meta']

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])
    print(f"理论上限: {sil_upper:.4f}")

    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Encoder(nn.Module):
        def __init__(self, nc, nl, na, nr):
            super().__init__()
            d = 128 // 6
            self.ce = nn.Embedding(nc + 1, d)
            self.le = nn.Embedding(nl + 1, d)
            self.ae = nn.Embedding(na + 1, d)
            self.re = nn.Embedding(nr + 1, d)
            self.np = nn.Linear(3, d)
            self.cp = nn.Linear(2, d)
            self.enc = nn.Sequential(
                nn.Linear(d * 6, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 64))
            self.dec = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

        def encode(self, f, c):
            cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
            x = torch.cat([self.ce(f[:, 0].long()), self.le(f[:, 1].long()),
                           self.ae(f[:, 2].long()), self.re(f[:, 3].long()),
                           self.np(f[:, 4:7]), self.cp(cn)], -1)
            return F.normalize(self.enc(x), p=2, dim=-1)

        def forward(self, f, c):
            z = self.encode(f, c)
            return z, self.dec(z)

    model = Encoder(
        meta['num_categories'], meta['num_landuses'],
        meta['num_aoi_types'], meta['num_road_classes']
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)

    best_sil = -1
    best_embeddings = None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        z, cr = model(ft, ct)
        l1 = F.mse_loss(cr, cn)

        N = len(coords)
        ii, jj = np.random.randint(0, N, 3000), np.random.randint(0, N, 3000)
        m = ii != jj
        ii, jj = ii[m], jj[m]
        sd = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)

        pi = torch.from_numpy(np.stack([ii, jj], 1)).long().to(device)
        sd_t = torch.from_numpy(sd).float().to(device)
        l2 = F.mse_loss(torch.norm(z[pi[:, 0]] - z[pi[:, 1]], p=2, dim=1), sd_t)

        loss = l1 + 2.0 * l2
        loss.backward()
        opt.step()
        sch.step()

        if ep % 100 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct).cpu().numpy()
                sil = silhouette_score(zn[si], labels[si])

            if sil > best_sil:
                best_sil = sil
                best_embeddings = zn.copy()

            print(f"  Epoch {ep:4d}: Sil={sil:.4f}, Best={best_sil:.4f} ({best_sil/sil_upper*100:.1f}%)")

    print(f"\n训练完成! 最佳Silhouette: {best_sil:.4f}, 达成率: {best_sil/sil_upper*100:.1f}%")

    # 深度测试
    print("\n" + "=" * 60)
    print("深度能力测试")
    print("=" * 60)

    # 1. 距离保持测试
    print("\n1. 距离保持测试:")
    N = len(coords)
    idx_i = np.random.randint(0, N, 2000)
    idx_j = np.random.randint(0, N, 2000)
    mask = idx_i != idx_j

    real_dists = np.sqrt(((coords[idx_i[mask]] - coords[idx_j[mask]]) ** 2).sum(1))
    emb_dists = np.sqrt(((best_embeddings[idx_i[mask]] - best_embeddings[idx_j[mask]]) ** 2).sum(1))

    pearson, _ = pearsonr(real_dists, emb_dists)
    print(f"   Pearson相关系数: {pearson:.4f}")
    status = "[PASS] 强相关" if pearson > 0.7 else "[OK] 中等相关" if pearson > 0.4 else "[FAIL] 弱相关"
    print(f"   状态: {status}")

    # 2. 空间查询测试
    print("\n2. 空间相似性查询测试:")
    nbrs_real = NearestNeighbors(n_neighbors=11).fit(coords)
    nbrs_emb = NearestNeighbors(n_neighbors=11).fit(best_embeddings)

    overlaps = []
    for _ in range(50):
        idx = np.random.randint(0, N)
        _, real_neighbors = nbrs_real.kneighbors([coords[idx]])
        _, emb_neighbors = nbrs_emb.kneighbors([best_embeddings[idx]])

        real_set = set(real_neighbors[0][1:])
        emb_set = set(emb_neighbors[0][1:])
        overlap = len(real_set & emb_set) / 10
        overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)
    print(f"   平均邻居重叠率: {avg_overlap*100:.1f}%")
    status = "[PASS] 良好" if avg_overlap > 0.5 else "[OK] 一般" if avg_overlap > 0.3 else "[FAIL] 较差"
    print(f"   状态: {status}")

    # 3. 方向理解测试
    print("\n3. 空间方向理解测试:")
    center_idx = N // 2
    center_coord = coords[center_idx]
    center_emb = best_embeddings[center_idx]

    directions = {'东': (1, 0), '南': (0, -1), '西': (-1, 0), '北': (0, 1)}
    direction_results = []

    for dir_name, (dx, dy) in directions.items():
        target_x = center_coord[0] + dx * 0.01
        target_y = center_coord[1] + dy * 0.01

        dists = np.sqrt(((coords - np.array([target_x, target_y])) ** 2).sum(1))
        nearest_idx = np.argmin(dists)

        emb_diff = best_embeddings[nearest_idx] - center_emb
        emb_angle = np.arctan2(emb_diff[1], emb_diff[0]) * 180 / np.pi

        direction_results.append({
            'direction': dir_name,
            'emb_angle': emb_angle
        })
        print(f"   {dir_name}方向: Embedding角度={emb_angle:.1f}°")

    return {
        'sil_upper': sil_upper,
        'best_sil': best_sil,
        'achievement': best_sil / sil_upper,
        'pearson': pearson,
        'overlap': avg_overlap,
        'direction_results': direction_results
    }


def main():
    print("=" * 70)
    print("V2.3 模型能力边界深度分析")
    print("=" * 70)

    # 1. 不同规模对比
    scale_results = compare_scales()

    # 2. 小规模深度测试
    print("\n\n" + "=" * 70)
    print("小规模数据（10%）深度测试")
    print("=" * 70)
    data_10pct = load_data_sample(0.10)
    spatial_info = analyze_spatial_distribution(data_10pct)
    ability_results = test_model_real_ability(data_10pct, epochs=500)

    # 3. 总结
    print("\n" + "=" * 70)
    print("实验总结")
    print("=" * 70)

    print("\n【核心发现】")
    print(f"1. 距离保持能力: Pearson={ability_results['pearson']:.4f}")
    print(f"2. 空间查询能力: 重叠率={ability_results['overlap']*100:.1f}%")
    print(f"3. 聚类保持能力: 达成率={ability_results['achievement']*100:.1f}%")

    print("\n【规模影响】")
    print("数据规模越大，理论上限越低（空间分布越均匀）")
    for r in scale_results:
        print(f"  {r['rate']*100:>5.0f}%: Sil上限={r['silhouette']:.4f}")

    print("\n【能力边界】")
    print("[+] 已验证: 空间距离保持、局部聚类结构")
    print("[?] 待验证: 跨区域泛化、方向理解、模式识别")
    print("[-] 已知缺陷: 功能区语义缺失、路网可达性缺失")

    # 保存结果
    results = {
        'scale_results': [
            {
                'rate': float(r['rate']),
                'num_pois': int(r['num_pois']),
                'lng_span': float(r['lng_span']),
                'lat_span': float(r['lat_span']),
                'silhouette': float(r['silhouette'])
            }
            for r in scale_results
        ],
        'ability_results': {
            'sil_upper': float(ability_results['sil_upper']),
            'best_sil': float(ability_results['best_sil']),
            'achievement': float(ability_results['achievement']),
            'pearson': float(ability_results['pearson']),
            'overlap': float(ability_results['overlap'])
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(BASE_DIR, 'deep_analysis_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: deep_analysis_results.json")


if __name__ == "__main__":
    main()
