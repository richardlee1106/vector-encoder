# -*- coding: utf-8 -*-
"""
V2.4 渐进式训练流程

实验策略：
1. Phase 1: 单实验区验证参数
2. Phase 2: 三实验区验证泛化性
3. Phase 3: 渐进式全量验证（10%→30%→60%→80%→100%）

每个阶段都可以手动停止或继续
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
import json
import os
import sys
from datetime import datetime

# ============ 固定随机种子 ============
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
SAVED_MODELS_DIR = os.path.join(PROJECT_DIR, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'docs', 'v24_results')

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============ 数据加载 ============

def load_saved_data(data_dir=None):
    """从已保存的数据加载"""
    if data_dir is None:
        # 找最新的模型目录
        import glob
        model_dirs = glob.glob(os.path.join(SAVED_MODELS_DIR, 'v23_*'))
        if not model_dirs:
            raise FileNotFoundError("没有找到已保存的数据")
        data_dir = sorted(model_dirs)[-1]

    print(f"从 {data_dir} 加载数据...")

    coords = np.load(os.path.join(data_dir, 'coords.npy'))
    embeddings = np.load(os.path.join(data_dir, 'embeddings.npy'))
    poi_ids = np.load(os.path.join(data_dir, 'poi_ids.npy'))

    with open(os.path.join(data_dir, 'mappings.json'), 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    # 构造特征（简化版，使用随机特征）
    N = len(coords)
    features = np.zeros((N, 7), dtype=np.float32)
    features[:, 4] = np.random.rand(N) * 100  # density
    features[:, 5] = np.random.rand(N) * 2     # entropy
    features[:, 6] = np.random.rand(N) * 500   # road_dist

    return {
        'ids': poi_ids,
        'coords': coords,
        'features': features,
        'embeddings': embeddings,
        'mappings': mappings,
        'meta': {
            'num_pois': len(coords),
            'num_categories': len(mappings.get('category', {})),
            'num_landuses': len(mappings.get('landuse', {})),
            'num_aoi_types': len(mappings.get('aoi_type', {})),
            'num_road_classes': len(mappings.get('road_class', {}))
        }
    }


def load_experiment_area(area_name):
    """加载指定实验区域数据"""
    conn = psycopg2.connect(
        host='localhost', port=5432,
        user='postgres', password='123456', database='geoloom'
    )
    cur = conn.cursor()

    # 实验区域边界
    areas = {
        'guanggu_core': (114.35, 114.42, 30.45, 30.52),  # 光谷核心区
        'wuda_area': (114.32, 114.38, 30.52, 30.58),     # 武大区域
        'zhongjia_cun': (114.25, 114.35, 30.55, 30.65),  # 中佳村区域
    }

    if area_name not in areas:
        raise ValueError(f"未知区域: {area_name}")

    lng_min, lng_max, lat_min, lat_max = areas[area_name]

    cur.execute(f'''
        SELECT id, ST_X(geom), ST_Y(geom), name, category_big,
            land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois
        WHERE geom IS NOT NULL
          AND ST_X(geom) BETWEEN {lng_min} AND {lng_max}
          AND ST_Y(geom) BETWEEN {lat_min} AND {lat_max}
    ''')
    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        raise ValueError(f"区域 {area_name} 没有数据")

    return _process_rows(rows)


def load_data_sample(rate, seed=42):
    """采样加载数据"""
    np.random.seed(seed)

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

    return _process_rows(rows)


def _process_rows(rows):
    """处理数据行"""
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


# ============ 特征计算 ============

def compute_knn_features(coords, k=10):
    """计算KNN邻域特征"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    neighbor_dists = distances[:, 1:]
    neighbor_indices = indices[:, 1:]

    knn_features = np.stack([
        neighbor_dists.mean(axis=1),
        neighbor_dists.std(axis=1),
        neighbor_dists[:, 0],
        neighbor_dists[:, -1],
    ], axis=1)

    return knn_features, neighbor_indices


# ============ 模型定义 ============

class EncoderV24(nn.Module):
    """V2.4 编码器"""

    def __init__(self, nc, nl, na, nr, embed_dim=64, hidden_dim=128, knn_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        d = hidden_dim // 6
        self.ce = nn.Embedding(nc + 1, d)
        self.le = nn.Embedding(nl + 1, d)
        self.ae = nn.Embedding(na + 1, d)
        self.re = nn.Embedding(nr + 1, d)
        self.np = nn.Linear(3, d)
        self.cp = nn.Linear(2, d)

        self.knn_enc = nn.Sequential(
            nn.Linear(knn_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.enc = nn.Sequential(
            nn.Linear(d * 6 + 16, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.dec = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def encode(self, f, c, knn_feat):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        knn_enc = self.knn_enc(knn_feat)
        x = torch.cat([
            self.ce(f[:, 0].long()), self.le(f[:, 1].long()),
            self.ae(f[:, 2].long()), self.re(f[:, 3].long()),
            self.np(f[:, 4:7]), self.cp(cn),
            knn_enc
        ], dim=-1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c, knn_feat):
        z = self.encode(f, c, knn_feat)
        return z, self.dec(z)


# ============ 训练函数 ============

def train_v24(data, epochs=500, seed=42, n_clusters=30, verbose=True):
    """V2.4训练"""
    set_seed(seed)

    coords, feats = data['coords'], data['features']
    meta = data['meta']

    # 计算KNN特征
    knn_features, neighbor_indices = compute_knn_features(coords, k=10)
    knn_t = torch.from_numpy(knn_features).float()

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    si = np.random.choice(len(coords), min(5000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])

    if verbose:
        print(f"  POI数量: {len(coords):,}")
        print(f"  理论上限: {sil_upper:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EncoderV24(
        meta['num_categories'], meta['num_landuses'],
        meta['num_aoi_types'], meta['num_road_classes']
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    knn_t = knn_t.to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)

    best_sil = -1
    best_embeddings = None
    history = []

    N = len(coords)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        z, cr = model(ft, ct, knn_t)
        l1 = F.mse_loss(cr, cn)

        # 距离保持
        ii, jj = np.random.randint(0, N, 3000), np.random.randint(0, N, 3000)
        m = ii != jj
        ii, jj = ii[m], jj[m]
        sd = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)

        pi = torch.from_numpy(np.stack([ii, jj], 1)).long().to(device)
        sd_t = torch.from_numpy(sd).float().to(device)
        l2 = F.mse_loss(torch.norm(z[pi[:, 0]] - z[pi[:, 1]], p=2, dim=1), sd_t)

        # 邻域一致性
        l3 = 0
        batch_size = min(500, N)
        for i in range(0, batch_size):
            neighbors = neighbor_indices[i]
            neighbor_embs = z[neighbors]
            center_emb = z[i]
            l3 += F.mse_loss(neighbor_embs.mean(0), center_emb)
        l3 = l3 / batch_size * 0.1

        loss = l1 + 2.0 * l2 + l3
        loss.backward()
        opt.step()
        sch.step()

        if ep % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct, knn_t).cpu().numpy()
                sil = silhouette_score(zn[si], labels[si])

            if sil > best_sil:
                best_sil = sil
                best_embeddings = zn.copy()

            pct = best_sil / sil_upper * 100
            history.append({'epoch': ep, 'sil': sil, 'best_sil': best_sil})

            if verbose:
                print(f"    Epoch {ep:4d}: Sil={sil:.4f}, Best={best_sil:.4f} ({pct:.1f}%)")

    # 最终评估
    nbrs_real = NearestNeighbors(n_neighbors=11).fit(coords)
    nbrs_emb = NearestNeighbors(n_neighbors=11).fit(best_embeddings)

    overlaps = []
    for _ in range(30):
        idx = np.random.randint(0, N)
        _, real_neighbors = nbrs_real.kneighbors([coords[idx]])
        _, emb_neighbors = nbrs_emb.kneighbors([best_embeddings[idx]])

        real_set = set(real_neighbors[0][1:])
        emb_set = set(emb_neighbors[0][1:])
        overlap = len(real_set & emb_set) / 10
        overlaps.append(overlap)

    avg_overlap = np.mean(overlaps)

    # Pearson
    idx_i = np.random.randint(0, N, 1000)
    idx_j = np.random.randint(0, N, 1000)
    mask = idx_i != idx_j
    real_dists = np.sqrt(((coords[idx_i[mask]] - coords[idx_j[mask]]) ** 2).sum(1))
    emb_dists = np.sqrt(((best_embeddings[idx_i[mask]] - best_embeddings[idx_j[mask]]) ** 2).sum(1))
    pearson, _ = pearsonr(real_dists, emb_dists)

    return {
        'num_pois': len(coords),
        'sil_upper': sil_upper,
        'best_sil': best_sil,
        'achievement': best_sil / sil_upper,
        'pearson': pearson,
        'overlap': avg_overlap,
        'history': history
    }


# ============ Phase 1: 单实验区验证 ============

def phase1_single_area():
    """Phase 1: 使用采样数据验证参数"""
    print("\n" + "=" * 70)
    print("Phase 1: 参数验证 (使用10%采样数据)")
    print("=" * 70)
    print("\n策略: 使用10%采样数据快速验证参数")
    print("目标: Silhouette达成率>70%, Pearson>0.7, 重叠率>20%")
    print("\n如果结果不理想，请按Ctrl+C停止，修改参数后重试")

    try:
        print("\n加载10%采样数据...")
        data = load_data_sample(0.1, seed=42)

        result = train_v24(data, epochs=800, seed=42, verbose=True)

        print("\n" + "-" * 50)
        print("Phase 1 结果:")
        print("-" * 50)
        print(f"  POI数量: {result['num_pois']:,}")
        print(f"  Silhouette: {result['best_sil']:.4f} / {result['sil_upper']:.4f}")
        print(f"  达成率: {result['achievement']*100:.1f}%")
        print(f"  Pearson: {result['pearson']:.4f}")
        print(f"  重叠率: {result['overlap']*100:.1f}%")

        # 判断是否继续
        passed = (
            result['achievement'] > 0.7 and
            result['pearson'] > 0.7 and
            result['overlap'] > 0.2
        )

        print("\n" + "=" * 50)
        if passed:
            print("[PASS] 参数验证通过，可以继续Phase 2")
        else:
            print("[FAIL] 参数验证失败，请修改参数后重试")
        print("=" * 50)

        return passed, result

    except KeyboardInterrupt:
        print("\n\n用户中断，请修改参数后重试")
        return False, None


# ============ Phase 2: 三采样比例验证 ============

def phase2_three_rates():
    """Phase 2: 三采样比例验证泛化性"""
    print("\n" + "=" * 70)
    print("Phase 2: 三采样比例验证泛化性")
    print("=" * 70)
    print("\n策略: 在不同采样比例下验证模型泛化性")
    print("目标: 三个比例达成率均>70%")

    rates = [0.05, 0.10, 0.20]
    results = {}

    for rate in rates:
        print(f"\n{'=' * 50}")
        print(f"采样比例: {rate*100:.0f}%")
        print('=' * 50)

        try:
            data = load_data_sample(rate, seed=42)
            result = train_v24(data, epochs=800, seed=42, verbose=True)
            results[rate] = result

            print(f"\n结果:")
            print(f"  POI数量: {result['num_pois']:,}")
            print(f"  Silhouette: {result['best_sil']:.4f} (达成率 {result['achievement']*100:.1f}%)")
            print(f"  Pearson: {result['pearson']:.4f}")
            print(f"  重叠率: {result['overlap']*100:.1f}%")

        except Exception as e:
            print(f"  错误: {e}")
            results[rate] = None

    # 汇总
    print("\n" + "=" * 70)
    print("Phase 2 汇总")
    print("=" * 70)

    passed_count = 0
    for rate, result in results.items():
        if result is None:
            print(f"  {rate*100:.0f}%: 失败")
        else:
            passed = result['achievement'] > 0.7 and result['pearson'] > 0.7
            status = "[PASS]" if passed else "[FAIL]"
            print(f"  {rate*100:>5.0f}%: {status} Sil={result['best_sil']:.4f}, 达成率={result['achievement']*100:.1f}%")
            if passed:
                passed_count += 1

    all_passed = passed_count == len(rates)
    print("\n" + "=" * 50)
    if all_passed:
        print("[PASS] 三比例验证通过，可以继续Phase 3")
    else:
        print(f"[FAIL] {passed_count}/{len(rates)}比例通过，请检查参数")
    print("=" * 50)

    return all_passed, results


# ============ Phase 3: 渐进式全量验证 ============

def phase3_progressive_full():
    """Phase 3: 渐进式全量验证"""
    print("\n" + "=" * 70)
    print("Phase 3: 渐进式全量验证")
    print("=" * 70)
    print("\n策略: 逐步增加数据规模")
    print("阶段: 10% -> 30% -> 60% -> 80% -> 100%")

    rates = [0.1, 0.3, 0.6, 0.8, 1.0]
    results = {}

    for rate in rates:
        print(f"\n{'=' * 50}")
        print(f"数据规模: {rate*100:.0f}%")
        print('=' * 50)

        try:
            data = load_data_sample(rate, seed=42)
            result = train_v24(data, epochs=500, seed=42, verbose=True)
            results[rate] = result

            print(f"\n结果:")
            print(f"  POI数量: {result['num_pois']:,}")
            print(f"  Silhouette: {result['best_sil']:.4f} (达成率 {result['achievement']*100:.1f}%)")
            print(f"  Pearson: {result['pearson']:.4f}")
            print(f"  重叠率: {result['overlap']*100:.1f}%")

            # 检查是否继续
            if result['achievement'] < 0.5 or result['pearson'] < 0.5:
                print("\n[WARNING] 性能下降明显，是否继续？(y/n)")
                # 在自动化流程中继续

        except Exception as e:
            print(f"  错误: {e}")
            results[rate] = None

    # 汇总
    print("\n" + "=" * 70)
    print("Phase 3 汇总")
    print("=" * 70)

    for rate, result in results.items():
        if result:
            print(f"  {rate*100:>5.0f}%: Sil={result['best_sil']:.4f}, 达成率={result['achievement']*100:.1f}%, 重叠率={result['overlap']*100:.1f}%")

    return results


# ============ 主流程 ============

def main():
    print("=" * 70)
    print("V2.4 渐进式训练流程")
    print("=" * 70)
    print("\n实验策略:")
    print("  1. Phase 1: 单实验区验证参数")
    print("  2. Phase 2: 三实验区验证泛化性")
    print("  3. Phase 3: 渐进式全量验证")
    print("\n每个阶段后可以按Ctrl+C停止")

    all_results = {}

    # Phase 1
    passed1, result1 = phase1_single_area()
    all_results['phase1'] = result1

    if not passed1:
        print("\nPhase 1未通过，流程结束")
        return

    # Phase 2
    passed2, results2 = phase2_three_rates()
    all_results['phase2'] = results2

    if not passed2:
        print("\nPhase 2未通过，流程结束")
        return

    # Phase 3
    results3 = phase3_progressive_full()
    all_results['phase3'] = results3

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(RESULTS_DIR, f'v24_progressive_{timestamp}.json')

    # 转换为可序列化格式
    serializable = {}
    for phase, data in all_results.items():
        if data is None:
            continue
        if isinstance(data, dict):
            serializable[phase] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    serializable[phase][k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                              for kk, vv in v.items() if kk != 'history'}
                elif isinstance(v, (np.floating, float)):
                    serializable[phase][k] = float(v)

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {result_path}")
    print("\n" + "=" * 70)
    print("V2.4 渐进式训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
