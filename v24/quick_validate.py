# -*- coding: utf-8 -*-
"""
V2.4 快速验证脚本
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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(rate, seed=42):
    np.random.seed(seed)
    conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='123456', database='geoloom')
    cur = conn.cursor()
    limit = int(845676 * rate)
    cur.execute(f'''
        SELECT ST_X(geom), ST_Y(geom), category_big, land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL ORDER BY RANDOM() LIMIT {limit}
    ''')
    rows = cur.fetchall()
    conn.close()

    cat_map, lu_map, aoi_map, rc_map = {}, {}, {}, {}
    coords, feats = [], []
    for row in rows:
        lng, lat, cat, lu, aoi, rc, density, entropy, road_dist = row
        c = cat or 'unknown'
        if c not in cat_map: cat_map[c] = len(cat_map)
        l = lu or 'unknown'
        if l not in lu_map: lu_map[l] = len(lu_map)
        a = aoi or 'unknown'
        if a not in aoi_map: aoi_map[a] = len(aoi_map)
        r = rc or 'unknown'
        if r not in rc_map: rc_map[r] = len(rc_map)
        coords.append([lng, lat])
        feats.append([cat_map[c], lu_map[l], aoi_map[a], rc_map[r], float(density or 0), float(entropy or 0), float(road_dist or 0)])
    return np.array(coords, dtype=np.float32), np.array(feats, dtype=np.float32), len(cat_map), len(lu_map), len(aoi_map), len(rc_map)

class Encoder(nn.Module):
    def __init__(self, nc, nl, na, nr):
        super().__init__()
        d = 128 // 6
        self.ce, self.le, self.ae, self.re = nn.Embedding(nc+1, d), nn.Embedding(nl+1, d), nn.Embedding(na+1, d), nn.Embedding(nr+1, d)
        self.np, self.cp, self.knn = nn.Linear(3, d), nn.Linear(2, d), nn.Linear(4, 16)
        self.enc = nn.Sequential(nn.Linear(d*6+16, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64))
        self.dec = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def encode(self, f, c, knn):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        x = torch.cat([self.ce(f[:,0].long()), self.le(f[:,1].long()), self.ae(f[:,2].long()), self.re(f[:,3].long()), self.np(f[:,4:7]), self.cp(cn), self.knn(knn)], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)
    def forward(self, f, c, knn): return self.encode(f, c, knn), self.dec(self.encode(f, c, knn))

def train_and_eval(rate, epochs=800, seed=42):
    set_seed(seed)
    print(f'\n--- {rate*100:.0f}%采样 ---')
    coords, feats, nc, nl, na, nr = load_data(rate, seed)
    print(f'POI: {len(coords):,}')

    # KNN
    nbrs = NearestNeighbors(n_neighbors=11).fit(coords)
    dists, idxs = nbrs.kneighbors(coords)
    knn_feat = np.stack([dists[:,1:].mean(1), dists[:,1:].std(1), dists[:,1], dists[:,-1]], axis=1)
    neighbor_idx = idxs[:, 1:]

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=seed, batch_size=10000)
    labels = kmeans.fit_predict(coords)
    si = np.random.choice(len(coords), min(5000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[si], labels[si])
    print(f'理论上限: {sil_upper:.4f}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Encoder(nc, nl, na, nr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    knn_t = torch.from_numpy(knn_feat).float().to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)
    N = len(coords)
    best_sil = -1

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z, cr = model(ft, ct, knn_t)
        l1 = F.mse_loss(cr, cn)
        ii, jj = np.random.randint(0, N, 5000), np.random.randint(0, N, 5000)
        m = ii != jj
        ii, jj = ii[m], jj[m]
        sd = np.sqrt(((coords[ii] - coords[jj]) ** 2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
        l2 = F.mse_loss(torch.norm(z[torch.from_numpy(np.stack([ii,jj],1)).long().to(device)][:,0] - z[torch.from_numpy(np.stack([ii,jj],1)).long().to(device)][:,1], p=2, dim=1), torch.from_numpy(sd).float().to(device))
        (l1 + 2.0 * l2).backward()
        opt.step()
        sch.step()
        if ep % 200 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                zn = model.encode(ft, ct, knn_t).cpu().numpy()
                sil = silhouette_score(zn[si], labels[si])
            if sil > best_sil: best_sil = sil; best_emb = zn.copy()
            print(f'  Ep {ep}: Sil={sil:.4f}, Best={best_sil:.4f} ({best_sil/sil_upper*100:.1f}%)')

    # 评估
    idx_i, idx_j = np.random.randint(0, N, 2000), np.random.randint(0, N, 2000)
    m = idx_i != idx_j
    pearson, _ = pearsonr(np.sqrt(((coords[idx_i[m]] - coords[idx_j[m]])**2).sum(1)), np.sqrt(((best_emb[idx_i[m]] - best_emb[idx_j[m]])**2).sum(1)))
    nbrs_r, nbrs_e = NearestNeighbors(n_neighbors=11).fit(coords), NearestNeighbors(n_neighbors=11).fit(best_emb)
    overlaps = [len(set(nbrs_r.kneighbors([coords[i]])[1][0][1:]) & set(nbrs_e.kneighbors([best_emb[i]])[1][0][1:])) / 10 for i in np.random.randint(0, N, 30)]
    return {'num_pois': len(coords), 'sil_upper': sil_upper, 'best_sil': best_sil, 'achievement': best_sil/sil_upper, 'pearson': pearson, 'overlap': np.mean(overlaps)}

if __name__ == '__main__':
    print('=' * 70)
    print('V2.4 渐进式验证')
    print('=' * 70)
    results = {}
    for rate in [0.1, 0.3, 0.5]:
        results[rate] = train_and_eval(rate)

    print('\n' + '=' * 70)
    print('汇总')
    print('=' * 70)
    for rate, r in results.items():
        status = '[PASS]' if r['achievement'] > 0.7 and r['pearson'] > 0.7 else '[FAIL]'
        print(f"{rate*100:>5.0f}%: POI={r['num_pois']:>7,}, Sil={r['best_sil']:.4f}, 达成率={r['achievement']*100:.1f}%, Pearson={r['pearson']:.4f}, 重叠率={r['overlap']*100:.1f}% {status}")
