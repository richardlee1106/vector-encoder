# -*- coding: utf-8 -*-
"""单步训练脚本"""

import sys
import json
from pathlib import Path
from datetime import datetime
import time
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


def load_wuhan_sample(sample_rate: float):
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',
        'password': '123456',
        'database': 'geoloom'
    }
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL")
    total = cur.fetchone()[0]
    limit = int(total * sample_rate)
    cur.execute(f'''
        SELECT ST_X(geom), ST_Y(geom),
            category_big, land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL ORDER BY RANDOM() LIMIT {limit}
    ''')
    rows = cur.fetchall()
    conn.close()

    cat_map, lu_map, aoi_map, rc_map = {}, {}, {}, {}
    coords, features = [], []
    for row in rows:
        c = row[2] or 'unknown'
        if c not in cat_map: cat_map[c] = len(cat_map)
        l = row[3] or 'unknown'
        if l not in lu_map: lu_map[l] = len(lu_map)
        a = row[4] or 'unknown'
        if a not in aoi_map: aoi_map[a] = len(aoi_map)
        r = row[5] or 'unknown'
        if r not in rc_map: rc_map[r] = len(rc_map)
        coords.append([row[0], row[1]])
        features.append([cat_map[c], lu_map[l], aoi_map[a], rc_map[r],
                        float(row[6] or 0), float(row[7] or 0), float(row[8] or 0)])

    return {
        'coords': np.array(coords, dtype=np.float32),
        'features': np.array(features, dtype=np.float32),
        'meta': {'num_pois': len(coords), 'num_cat': len(cat_map), 'num_lu': len(lu_map),
                 'num_aoi': len(aoi_map), 'num_rc': len(rc_map)}
    }


class Encoder(nn.Module):
    def __init__(self, nc, nl, na, nr):
        super().__init__()
        ed = 128 // 6
        self.ce = nn.Embedding(nc + 1, ed)
        self.le = nn.Embedding(nl + 1, ed)
        self.ae = nn.Embedding(na + 1, ed)
        self.re = nn.Embedding(nr + 1, ed)
        self.np = nn.Linear(3, ed)
        self.cp = nn.Linear(2, ed)
        self.enc = nn.Sequential(
            nn.Linear(ed*6, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64))
        self.dec = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def encode(self, f, c):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        x = torch.cat([self.ce(f[:,0].long()), self.le(f[:,1].long()), self.ae(f[:,2].long()),
                       self.re(f[:,3].long()), self.np(f[:,4:7]), self.cp(cn)], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c):
        z = self.encode(f, c)
        return z, self.dec(z)


def train(data, rate, epochs=800):
    print(f"\n{'='*60}")
    print(f"训练: {rate*100:.0f}% 数据 ({data['meta']['num_pois']:,} POI)")
    print(f"{'='*60}")

    t0 = time.time()
    coords, features, meta = data['coords'], data['features'], data['meta']

    # 聚类
    nc = max(30, int(meta['num_pois'] / 3000))
    kmeans = MiniBatchKMeans(n_clusters=nc, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)
    si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_up = silhouette_score(coords[si], labels[si])
    print(f"聚类数: {nc}, 理论上限: {sil_up:.4f}")

    # 训练
    device = torch.device('cuda')
    model = Encoder(meta['num_cat'], meta['num_lu'], meta['num_aoi'], meta['num_rc']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cn = (coords - coords.mean(0)) / coords.std(0)
    ft = torch.from_numpy(features).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    cnt = torch.from_numpy(cn).float().to(device)

    best_sil = -1
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z, cr = model(ft, ct)
        l1 = F.mse_loss(cr, cnt)

        # L2
        N = len(coords)
        ii, jj = np.random.randint(0, N, 5000), np.random.randint(0, N, 5000)
        m = ii != jj
        sd = np.sqrt(((coords[ii[m]] - coords[jj[m]])**2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
        pi = torch.from_numpy(np.stack([ii[m], jj[m]], 1)).long().to(device)
        sd = torch.from_numpy(sd).float().to(device)
        ed = torch.norm(z[pi[:,0]] - z[pi[:,1]], p=2, dim=1)
        l2 = F.mse_loss(ed, sd)

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
            if sil > best_sil: best_sil = sil
            print(f"  Ep {ep}: Sil={sil:.4f}, Best={best_sil:.4f} ({best_sil/sil_up*100:.1f}%)")

    t = time.time() - t0
    rate_pct = best_sil / sil_up * 100
    print(f"\n结果: Sil={best_sil:.4f}, 达成率={rate_pct:.1f}%, 时间={t/60:.1f}min")

    return {'rate': rate, 'pois': meta['num_pois'], 'n_clusters': nc,
            'sil_up': sil_up, 'best_sil': best_sil, 'pct': rate_pct, 'time': t}


if __name__ == "__main__":
    results = []
    for r in [0.10, 0.30, 0.60, 0.80, 1.00]:
        data = load_wuhan_sample(r)
        res = train(data, r, epochs=800 if r < 1.0 else 600)
        results.append(res)

    print("\n" + "="*70)
    print("汇总报告")
    print("="*70)
    print(f"{'规模':<8} {'POI':<12} {'聚类':<8} {'上限':<8} {'Sil':<10} {'达成率':<10} {'时间'}")
    print("-"*70)
    for r in results:
        print(f"{r['rate']*100:>5.0f}%  {r['pois']:>10,}  {r['n_clusters']:>6}  {r['sil_up']:.4f}  "
              f"{r['best_sil']:.4f}    {r['pct']:.1f}%     {r['time']/60:.1f}min")
