# -*- coding: utf-8 -*-
"""渐进式训练 - 精简版"""

import warnings
warnings.filterwarnings('ignore')
import psycopg2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import time


def load_data(rate):
    conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='123456', database='geoloom')
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL')
    total = cur.fetchone()[0]
    limit = int(total * rate)
    cur.execute(f'''SELECT ST_X(geom), ST_Y(geom), category_big, land_use_type, aoi_type,
        nearest_road_class, poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL ORDER BY RANDOM() LIMIT {limit}''')
    rows = cur.fetchall()
    conn.close()

    cm, lm, am, rm = {}, {}, {}, {}
    coords, feats = [], []
    for row in rows:
        c = row[2] or 'x'
        if c not in cm: cm[c] = len(cm)
        l = row[3] or 'x'
        if l not in lm: lm[l] = len(lm)
        a = row[4] or 'x'
        if a not in am: am[a] = len(am)
        r = row[5] or 'x'
        if r not in rm: rm[r] = len(rm)
        coords.append([row[0], row[1]])
        feats.append([cm[c], lm[l], am[a], rm[r], float(row[6] or 0), float(row[7] or 0), float(row[8] or 0)])
    return np.array(coords, dtype=np.float32), np.array(feats, dtype=np.float32), len(cm), len(lm), len(am), len(rm)


class Encoder(nn.Module):
    def __init__(self, nc, nl, na, nr):
        super().__init__()
        d = 128 // 6
        self.ce = nn.Embedding(nc+1, d)
        self.le = nn.Embedding(nl+1, d)
        self.ae = nn.Embedding(na+1, d)
        self.re = nn.Embedding(nr+1, d)
        self.np = nn.Linear(3, d)
        self.cp = nn.Linear(2, d)
        self.enc = nn.Sequential(
            nn.Linear(d*6, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64))
        self.dec = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def encode(self, f, c):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        x = torch.cat([
            self.ce(f[:,0].long()), self.le(f[:,1].long()),
            self.ae(f[:,2].long()), self.re(f[:,3].long()),
            self.np(f[:,4:7]), self.cp(cn)
        ], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c):
        z = self.encode(f, c)
        return z, self.dec(z)


def train_rate(rate, epochs=600):
    print(f"\n{'='*60}")
    print(f"训练: {rate*100:.0f}% 数据")
    print('='*60)

    t0 = time.time()
    coords, feats, nc, nl, na, nr = load_data(rate)
    print(f"POI数量: {len(coords):,}")

    # 聚类
    n_clusters = max(30, len(coords) // 3000)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)
    si = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])
    print(f"聚类数: {n_clusters}, 理论上限: {sil_upper:.4f}")

    # 训练
    device = torch.device('cuda')
    model = Encoder(nc, nl, na, nr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)

    best_sil = -1

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        z, cr = model(ft, ct)
        l1 = F.mse_loss(cr, cn)

        # L2
        N = len(coords)
        ii, jj = np.random.randint(0, N, 5000), np.random.randint(0, N, 5000)
        m = ii != jj
        ii, jj = ii[m], jj[m]
        sd = np.sqrt(((coords[ii] - coords[jj])**2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)

        pi = torch.from_numpy(np.stack([ii, jj], 1)).long().to(device)
        sd_t = torch.from_numpy(sd).float().to(device)
        l2 = F.mse_loss(torch.norm(z[pi[:,0]] - z[pi[:,1]], p=2, dim=1), sd_t)

        loss = l1 + 2.0 * l2
        loss.backward()
        opt.step()
        sch.step()

        if ep % 100 == 0:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct).cpu().numpy()
                es = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(zn[es], labels[es])
            if sil > best_sil:
                best_sil = sil
            print(f"  Epoch {ep}: Sil={sil:.4f}, Best={best_sil:.4f} ({best_sil/sil_upper*100:.1f}%)")

    t = time.time() - t0
    pct = best_sil / sil_upper * 100
    print(f"\n结果: Sil={best_sil:.4f}, 达成率={pct:.1f}%, 时间={t/60:.1f}min")

    return {
        'rate': rate,
        'num_pois': len(coords),
        'n_clusters': n_clusters,
        'sil_upper': sil_upper,
        'best_sil': best_sil,
        'pct': pct,
        'time': t
    }


if __name__ == "__main__":
    results = []

    # 依次训练 10%, 30%, 60%, 80%, 100%
    for rate in [0.10, 0.30, 0.60, 0.80, 1.00]:
        epochs = 600 if rate >= 1.0 else 600
        result = train_rate(rate, epochs=epochs)
        results.append(result)

        # 清理GPU内存
        torch.cuda.empty_cache()

    # 汇总报告
    print("\n" + "="*70)
    print("渐进式验证报告")
    print("="*70)
    print(f"{'规模':<8} {'POI数量':<12} {'聚类':<8} {'理论上限':<10} {'Silhouette':<12} {'达成率':<10} {'时间'}")
    print("-"*70)

    for r in results:
        print(f"{r['rate']*100:>5.0f}%  {r['num_pois']:>10,}  {r['n_clusters']:>6}  "
              f"{r['sil_upper']:>8.4f}  {r['best_sil']:>10.4f}  {r['pct']:>6.1f}%    {r['time']/60:>5.1f}min")

    # 分析
    print("\n" + "="*70)
    print("分析")
    print("="*70)

    pcts = [r['pct'] for r in results]
    print(f"达成率范围: {min(pcts):.1f}% - {max(pcts):.1f}%")
    print(f"达成率均值: {np.mean(pcts):.1f}%")
    print(f"达成率标准差: {np.std(pcts):.1f}%")

    if np.std(pcts) < 5:
        print("[+] 模型在不同数据规模下表现稳定")
    else:
        print("[!] 模型表现随数据规模变化较大")
