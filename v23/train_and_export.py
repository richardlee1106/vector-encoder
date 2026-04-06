# -*- coding: utf-8 -*-
"""
V2.3 训练并导出模型

训练完成后保存：
- 模型权重
- 配置文件
- 特征映射表
- FAISS索引
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
import json
import os
from datetime import datetime

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')


def load_wuhan_full():
    """加载武汉全量POI数据"""
    print("加载武汉全量POI数据...")

    conn = psycopg2.connect(
        host='localhost', port=5432,
        user='postgres', password='123456', database='geoloom'
    )
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL')
    total = cur.fetchone()[0]
    print(f"总POI数量: {total:,}")

    cur.execute('''
        SELECT id, ST_X(geom), ST_Y(geom), name, category_big,
            land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL
    ''')
    rows = cur.fetchall()
    conn.close()

    # 构建特征映射
    cat_map, lu_map, aoi_map, rc_map = {}, {}, {}, {}
    ids, names, categories, coords, feats = [], [], [], [], []

    for row in rows:
        poi_id, lng, lat, name, cat = row[0], row[1], row[2], row[3], row[4]
        lu, aoi, rc = row[5], row[6], row[7]
        density, entropy, road_dist = row[8], row[9], row[10]

        # 映射
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

    data = {
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

    print(f"加载完成: {data['meta']['num_pois']:,} POI")
    return data


class Encoder(nn.Module):
    """V2.3 空间编码器"""

    def __init__(self, nc, nl, na, nr, embed_dim=64, hidden_dim=128):
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

        self.enc = nn.Sequential(
            nn.Linear(d * 6, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim))

        self.dec = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def encode(self, f, c):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        x = torch.cat([self.ce(f[:, 0].long()), self.le(f[:, 1].long()),
                       self.ae(f[:, 2].long()), self.re(f[:, 3].long()),
                       self.np(f[:, 4:7]), self.cp(cn)], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c):
        z = self.encode(f, c)
        return z, self.dec(z)


def train_and_export(data, epochs=800):
    """训练并导出模型"""
    print(f"\n{'=' * 60}")
    print("开始训练 V2.3 模型")
    print(f"{'=' * 60}")

    coords, feats = data['coords'], data['features']
    meta = data['meta']

    # 聚类评估
    n_clusters = max(30, len(coords) // 5000)
    print(f"聚类数: {n_clusters}")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])
    print(f"理论上限 Silhouette: {sil_upper:.4f}")

    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    model = Encoder(
        meta['num_categories'], meta['num_landuses'],
        meta['num_aoi_types'], meta['num_road_classes']
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")

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

        loss = l1 + 2.0 * l2
        loss.backward()
        opt.step()
        sch.step()

        if ep % 100 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct).cpu().numpy()
                es = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(zn[es], labels[es])

            if sil > best_sil:
                best_sil = sil
                best_embeddings = zn.copy()

            pct = best_sil / sil_upper * 100
            print(f"  Epoch {ep:4d}: Sil={sil:.4f}, Best={best_sil:.4f} ({pct:.1f}%)")

    print(f"\n训练完成! 最佳Silhouette: {best_sil:.4f}")

    # ========== 导出 ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(SAVED_MODELS_DIR, f'v23_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    # 1. 保存模型权重
    model_path = os.path.join(model_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_categories': meta['num_categories'],
            'num_landuses': meta['num_landuses'],
            'num_aoi_types': meta['num_aoi_types'],
            'num_road_classes': meta['num_road_classes'],
            'embed_dim': 64,
            'hidden_dim': 128
        },
        'metadata': {
            'silhouette': float(best_sil),
            'silhouette_upper': float(sil_upper),
            'achievement_rate': float(best_sil / sil_upper),
            'num_pois': meta['num_pois'],
            'n_clusters': n_clusters,
            'epochs': epochs,
            'timestamp': timestamp
        }
    }, model_path)
    print(f"模型已保存: {model_path}")

    # 2. 保存特征映射
    mappings_path = os.path.join(model_dir, 'mappings.json')
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(data['mappings'], f, ensure_ascii=False, indent=2)
    print(f"映射已保存: {mappings_path}")

    # 3. 保存POI ID列表
    ids_path = os.path.join(model_dir, 'poi_ids.npy')
    np.save(ids_path, data['ids'])
    print(f"POI ID已保存: {ids_path}")

    # 4. 保存embeddings
    emb_path = os.path.join(model_dir, 'embeddings.npy')
    np.save(emb_path, best_embeddings)
    print(f"Embeddings已保存: {emb_path}")

    # 5. 保存坐标
    coords_path = os.path.join(model_dir, 'coords.npy')
    np.save(coords_path, coords)
    print(f"坐标已保存: {coords_path}")

    print(f"\n导出完成! 目录: {model_dir}")

    return {
        'model_dir': model_dir,
        'best_sil': best_sil,
        'sil_upper': sil_upper,
        'embeddings': best_embeddings
    }


if __name__ == "__main__":
    # 加载数据
    data = load_wuhan_full()

    # 训练并导出
    result = train_and_export(data, epochs=800)

    print(f"\n{'=' * 60}")
    print("导出完成!")
    print(f"{'=' * 60}")
    print(f"模型目录: {result['model_dir']}")
    print(f"最佳Silhouette: {result['best_sil']:.4f}")
    print(f"达成率: {result['best_sil'] / result['sil_upper'] * 100:.1f}%")
