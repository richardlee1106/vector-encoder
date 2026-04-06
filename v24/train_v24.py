# -*- coding: utf-8 -*-
"""
V2.4 训练脚本 - 空间查询增强

改进点：
1. 固定随机种子，提升稳定性
2. 引入KNN邻域特征，提升空间查询能力
3. 多轮训练取最佳
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
from datetime import datetime

# ============ 固定随机种子 ============
def set_seed(seed=42):
    """固定所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'saved_models')


def load_data_sample(rate=0.1, seed=42):
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
        },
        'mappings': {
            'category': cat_map,
            'landuse': lu_map,
            'aoi_type': aoi_map,
            'road_class': rc_map
        }
    }


def compute_knn_features(coords, k=10):
    """计算KNN邻域特征"""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 排除自己
    neighbor_dists = distances[:, 1:]  # [N, k]
    neighbor_indices = indices[:, 1:]

    # 邻域统计特征
    knn_features = np.stack([
        neighbor_dists.mean(axis=1),      # 平均邻居距离
        neighbor_dists.std(axis=1),       # 邻居距离标准差
        neighbor_dists[:, 0],             # 最近邻距离
        neighbor_dists[:, -1],            # 最远邻居距离
    ], axis=1)

    return knn_features, neighbor_indices


class EncoderV24(nn.Module):
    """V2.4 编码器 - 增加邻域特征"""

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

        # KNN特征编码器
        self.knn_enc = nn.Sequential(
            nn.Linear(knn_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # 主编码器（输入维度增加）
        self.enc = nn.Sequential(
            nn.Linear(d * 6 + 16, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 解码器
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


def train_v24(data, epochs=800, seed=42, n_clusters=30):
    """V2.4训练"""
    set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"V2.4 训练 (seed={seed})")
    print(f"{'=' * 60}")

    coords, feats = data['coords'], data['features']
    meta = data['meta']

    # 计算KNN特征
    print("计算KNN邻域特征...")
    knn_features, neighbor_indices = compute_knn_features(coords, k=10)
    knn_t = torch.from_numpy(knn_features).float()

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])
    print(f"理论上限 Silhouette: {sil_upper:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = EncoderV24(
        meta['num_categories'], meta['num_landuses'],
        meta['num_aoi_types'], meta['num_road_classes']
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    knn_t = knn_t.to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)

    best_sil = -1
    best_embeddings = None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        z, cr = model(ft, ct, knn_t)
        l1 = F.mse_loss(cr, cn)

        # 距离保持损失
        N = len(coords)
        ii, jj = np.random.randint(0, N, 5000), np.random.randint(0, N, 5000)
        m = ii != jj
        ii, jj = ii[m], jj[m]
        sd = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)

        pi = torch.from_numpy(np.stack([ii, jj], 1)).long().to(device)
        sd_t = torch.from_numpy(sd).float().to(device)
        l2 = F.mse_loss(torch.norm(z[pi[:, 0]] - z[pi[:, 1]], p=2, dim=1), sd_t)

        # 邻域一致性损失
        l3 = 0
        for i in range(0, N, 1000):
            batch_neighbors = neighbor_indices[i:i+1000]
            for j, neighbors in enumerate(batch_neighbors):
                neighbor_embs = z[neighbors]
                center_emb = z[i + j]
                l3 += F.mse_loss(neighbor_embs.mean(0), center_emb)
        l3 = l3 / (N // 1000 + 1) * 0.1

        loss = l1 + 2.0 * l2 + l3
        loss.backward()
        opt.step()
        sch.step()

        if ep % 100 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct, knn_t).cpu().numpy()
                sil = silhouette_score(zn[si], labels[si])

            if sil > best_sil:
                best_sil = sil
                best_embeddings = zn.copy()

            pct = best_sil / sil_upper * 100
            print(f"  Epoch {ep:4d}: Sil={sil:.4f}, Best={best_sil:.4f} ({pct:.1f}%)")

    print(f"\n训练完成! 最佳Silhouette: {best_sil:.4f}")

    # 评估
    print("\n评估空间查询能力...")
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
    print(f"邻居重叠率: {avg_overlap*100:.1f}%")

    # 距离保持
    idx_i = np.random.randint(0, N, 2000)
    idx_j = np.random.randint(0, N, 2000)
    mask = idx_i != idx_j
    real_dists = np.sqrt(((coords[idx_i[mask]] - coords[idx_j[mask]]) ** 2).sum(1))
    emb_dists = np.sqrt(((best_embeddings[idx_i[mask]] - best_embeddings[idx_j[mask]]) ** 2).sum(1))
    pearson, _ = pearsonr(real_dists, emb_dists)
    print(f"距离保持 Pearson: {pearson:.4f}")

    return {
        'sil_upper': sil_upper,
        'best_sil': best_sil,
        'achievement': best_sil / sil_upper,
        'pearson': pearson,
        'overlap': avg_overlap,
        'embeddings': best_embeddings
    }


def multi_run_train(data, num_runs=3, epochs=800):
    """多轮训练取最佳"""
    print(f"\n{'=' * 60}")
    print(f"多轮训练 ({num_runs}次)")
    print(f"{'=' * 60}")

    results = []
    for run in range(num_runs):
        seed = 42 + run * 100
        result = train_v24(data, epochs=epochs, seed=seed)
        result['run'] = run + 1
        result['seed'] = seed
        results.append(result)

    # 选择最佳
    best = max(results, key=lambda x: x['best_sil'])

    print(f"\n{'=' * 60}")
    print("多轮训练结果汇总")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  Run {r['run']} (seed={r['seed']}): Sil={r['best_sil']:.4f}, 重叠率={r['overlap']*100:.1f}%")

    print(f"\n最佳结果: Run {best['run']}")
    print(f"  Silhouette: {best['best_sil']:.4f} (达成率 {best['achievement']*100:.1f}%)")
    print(f"  Pearson: {best['pearson']:.4f}")
    print(f"  重叠率: {best['overlap']*100:.1f}%")

    return best, results


if __name__ == "__main__":
    set_seed(42)

    print("加载数据...")
    data = load_data_sample(rate=0.1, seed=42)
    print(f"POI数量: {data['meta']['num_pois']:,}")

    best, results = multi_run_train(data, num_runs=3, epochs=800)

    # 保存结果
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(SAVED_MODELS_DIR, f'v24_results_{timestamp}.json')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best': {
                'sil_upper': float(best['sil_upper']),
                'best_sil': float(best['best_sil']),
                'achievement': float(best['achievement']),
                'pearson': float(best['pearson']),
                'overlap': float(best['overlap'])
            },
            'all_runs': [
                {
                    'run': r['run'],
                    'seed': r['seed'],
                    'best_sil': float(r['best_sil']),
                    'overlap': float(r['overlap'])
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {result_path}")
