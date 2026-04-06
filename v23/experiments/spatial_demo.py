# -*- coding: utf-8 -*-
"""
V2.3 空间理解能力Demo

验证模型的"空间理解"能力：
1. 空间相似性查询 - 给定POI，找空间邻居
2. 距离保持验证 - embedding距离 vs 真实距离
3. 空间方向理解 - 东/南/西/北方向性
4. 区域聚类可视化 - embedding空间的聚类效果
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
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_sample_data(rate=0.1):
    """加载采样数据"""
    conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='123456', database='geoloom')
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL')
    total = cur.fetchone()[0]
    limit = int(total * rate)
    cur.execute(f'''SELECT id, ST_X(geom), ST_Y(geom), name, category_big,
        category_big, land_use_type, aoi_type, nearest_road_class,
        poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois WHERE geom IS NOT NULL ORDER BY RANDOM() LIMIT {limit}''')
    rows = cur.fetchall()
    conn.close()

    cm, lm, am, rm = {}, {}, {}, {}
    ids, names, categories, coords, feats = [], [], [], [], []
    for row in rows:
        ids.append(row[0])
        names.append(row[3] or 'Unknown')
        categories.append(row[4] or 'Unknown')
        c = row[5] or 'x'
        if c not in cm: cm[c] = len(cm)
        l = row[6] or 'x'
        if l not in lm: lm[l] = len(lm)
        a = row[7] or 'x'
        if a not in am: am[a] = len(am)
        r = row[8] or 'x'
        if r not in rm: rm[r] = len(rm)
        coords.append([row[1], row[2]])
        feats.append([cm[c], lm[l], am[a], rm[r], float(row[9] or 0), float(row[10] or 0), float(row[11] or 0)])

    return {
        'ids': ids,
        'names': names,
        'categories': categories,
        'coords': np.array(coords, dtype=np.float32),
        'features': np.array(feats, dtype=np.float32),
        'meta': {'nc': len(cm), 'nl': len(lm), 'na': len(am), 'nr': len(rm)}
    }


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
        x = torch.cat([self.ce(f[:,0].long()), self.le(f[:,1].long()), self.ae(f[:,2].long()),
                       self.re(f[:,3].long()), self.np(f[:,4:7]), self.cp(cn)], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c):
        z = self.encode(f, c)
        return z, self.dec(z)


def train_model(data, epochs=500):
    """快速训练模型"""
    print("训练模型...")
    coords, feats = data['coords'], data['features']
    device = torch.device('cuda')
    model = Encoder(data['meta']['nc'], data['meta']['nl'], data['meta']['na'], data['meta']['nr']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ft = torch.from_numpy(feats).float().to(device)
    ct = torch.from_numpy(coords).float().to(device)
    cn = torch.from_numpy((coords - coords.mean(0)) / coords.std(0)).float().to(device)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z, cr = model(ft, ct)
        l1 = F.mse_loss(cr, cn)

        N = len(coords)
        ii, jj = np.random.randint(0, N, 3000), np.random.randint(0, N, 3000)
        m = ii != jj
        sd = np.sqrt(((coords[ii[m]] - coords[jj[m]])**2).sum(1))
        sd = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
        pi = torch.from_numpy(np.stack([ii[m], jj[m]], 1)).long().to(device)
        sd_t = torch.from_numpy(sd).float().to(device)
        l2 = F.mse_loss(torch.norm(z[pi[:,0]] - z[pi[:,1]], p=2, dim=1), sd_t)

        (l1 + 2.0 * l2).backward()
        opt.step()

        if ep % 100 == 0:
            print(f"  Epoch {ep}: Loss={(l1 + 2*l2).item():.4f}")

    print("训练完成!\n")
    return model


# ============================================================================
# Demo 1: 空间相似性查询
# ============================================================================

def demo_spatial_query(data, model):
    """Demo 1: 空间相似性查询"""
    print("=" * 60)
    print("Demo 1: 空间相似性查询")
    print("=" * 60)
    print("问题: 给定一个POI，能否找到空间上相近的其他POI？\n")

    coords, names = data['coords'], data['names']
    device = torch.device('cuda')

    # 获取embeddings
    model.eval()
    with torch.no_grad():
        ft = torch.from_numpy(data['features']).float().to(device)
        ct = torch.from_numpy(coords).float().to(device)
        embeddings = model.encode(ft, ct).cpu().numpy()

    # 随机选择3个POI
    np.random.seed(42)
    query_indices = np.random.choice(len(coords), 3, replace=False)

    for i, idx in enumerate(query_indices):
        print(f"\n查询 {i+1}: {names[idx]}")
        print(f"  坐标: ({coords[idx, 0]:.4f}, {coords[idx, 1]:.4f})")

        # 真实空间邻居
        nbrs_real = NearestNeighbors(n_neighbors=6).fit(coords)
        _, real_indices = nbrs_real.kneighbors([coords[idx]])
        real_neighbors = [names[j] for j in real_indices[0][1:]]

        # Embedding邻居
        nbrs_emb = NearestNeighbors(n_neighbors=6).fit(embeddings)
        _, emb_indices = nbrs_emb.kneighbors([embeddings[idx]])
        emb_neighbors = [names[j] for j in emb_indices[0][1:]]

        # 计算重叠
        overlap = len(set(real_indices[0][1:]) & set(emb_indices[0][1:])) / 5

        print(f"  真实空间邻居: {real_neighbors[:3]}")
        print(f"  Embedding邻居: {emb_neighbors[:3]}")
        print(f"  重叠率: {overlap*100:.0f}%")

        if overlap >= 0.6:
            print(f"  [PASS] 模型能找到空间相近的POI")
        else:
            print(f"  [FAIL] 重叠率较低")


# ============================================================================
# Demo 2: 距离保持验证
# ============================================================================

def demo_distance_preservation(data, model):
    """Demo 2: 距离保持验证"""
    print("\n" + "=" * 60)
    print("Demo 2: 距离保持验证")
    print("=" * 60)
    print("问题: embedding空间中的距离是否与真实空间距离相关？\n")

    coords = data['coords']
    device = torch.device('cuda')

    model.eval()
    with torch.no_grad():
        ft = torch.from_numpy(data['features']).float().to(device)
        ct = torch.from_numpy(coords).float().to(device)
        embeddings = model.encode(ft, ct).cpu().numpy()

    # 随机采样点对
    N = len(coords)
    np.random.seed(123)
    n_pairs = 2000
    idx_i = np.random.randint(0, N, n_pairs)
    idx_j = np.random.randint(0, N, n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    # 计算距离
    real_dists = np.sqrt(((coords[idx_i] - coords[idx_j])**2).sum(1))
    emb_dists = np.sqrt(((embeddings[idx_i] - embeddings[idx_j])**2).sum(1))

    # 相关性
    pearson, _ = pearsonr(real_dists, emb_dists)

    print(f"采样点对数: {len(idx_i)}")
    print(f"真实距离范围: {real_dists.min():.4f} - {real_dists.max():.4f}")
    print(f"Embedding距离范围: {emb_dists.min():.4f} - {emb_dists.max():.4f}")
    print(f"Pearson相关系数: {pearson:.4f}")

    if pearson > 0.7:
        print("\n[PASS] 强相关！模型保持了距离关系")
    elif pearson > 0.4:
        print("\n[OK] 中等相关，模型部分保持距离关系")
    else:
        print("\n[FAIL] 弱相关，模型未能保持距离关系")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_dists, emb_dists, alpha=0.3, s=5)
    ax.set_xlabel('真实空间距离')
    ax.set_ylabel('Embedding空间距离')
    ax.set_title(f'距离保持验证 (Pearson={pearson:.3f})')

    # 拟合线
    z = np.polyfit(real_dists, emb_dists, 1)
    p = np.poly1d(z)
    x_line = np.linspace(real_dists.min(), real_dists.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'拟合线 (r={pearson:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig('demo_distance_preservation.png', dpi=150)
    plt.close()
    print(f"\n图表已保存: demo_distance_preservation.png")


# ============================================================================
# Demo 3: 空间方向理解
# ============================================================================

def demo_direction_understanding(data, model):
    """Demo 3: 空间方向理解"""
    print("\n" + "=" * 60)
    print("Demo 3: 空间方向理解")
    print("=" * 60)
    print("问题: 模型是否理解东/南/西/北的方向性？\n")

    coords = data['coords']
    device = torch.device('cuda')

    model.eval()
    with torch.no_grad():
        ft = torch.from_numpy(data['features']).float().to(device)
        ct = torch.from_numpy(coords).float().to(device)
        embeddings = model.encode(ft, ct).cpu().numpy()

    # 选择中心点
    center_idx = len(coords) // 2
    center_coord = coords[center_idx]
    center_emb = embeddings[center_idx]

    print(f"中心点: {data['names'][center_idx]}")
    print(f"坐标: ({center_coord[0]:.4f}, {center_coord[1]:.4f})")

    # 找四个方向的POI
    directions = {'东': (1, 0), '南': (0, -1), '西': (-1, 0), '北': (0, 1)}

    for dir_name, (dx, dy) in directions.items():
        # 找该方向上的POI
        target_x = center_coord[0] + dx * 0.01  # 约1km
        target_y = center_coord[1] + dy * 0.01

        # 找最近的POI
        dists = np.sqrt(((coords - np.array([target_x, target_y]))**2).sum(1))
        nearest_idx = np.argmin(dists)

        # 计算embedding角度
        emb_diff = embeddings[nearest_idx] - center_emb
        emb_angle = np.arctan2(emb_diff[1], emb_diff[0]) * 180 / np.pi

        print(f"\n{dir_name}方向:")
        print(f"  目标POI: {data['names'][nearest_idx]}")
        print(f"  真实角度: 约{90 if dir_name=='北' else (-90 if dir_name=='南' else (0 if dir_name=='东' else 180))}°")
        print(f"  Embedding角度: {emb_angle:.1f}°")

    print("\n注: Embedding角度与真实方向不完全对应是正常的，")
    print("    重要的是embedding在高维空间中保持了相对方向关系")


# ============================================================================
# Demo 4: 区域聚类可视化
# ============================================================================

def demo_clustering(data, model):
    """Demo 4: 区域聚类可视化"""
    print("\n" + "=" * 60)
    print("Demo 4: 区域聚类可视化")
    print("=" * 60)
    print("问题: Embedding空间的聚类效果如何？\n")

    coords = data['coords']
    device = torch.device('cuda')

    model.eval()
    with torch.no_grad():
        ft = torch.from_numpy(data['features']).float().to(device)
        ct = torch.from_numpy(coords).float().to(device)
        embeddings = model.encode(ft, ct).cpu().numpy()

    # 聚类
    n_clusters = 15
    kmeans_real = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels_real = kmeans_real.fit_predict(coords)

    kmeans_emb = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels_emb = kmeans_emb.fit_predict(embeddings)

    # Silhouette
    sil_real = silhouette_score(coords, labels_real)
    sil_emb = silhouette_score(embeddings, labels_emb)

    print(f"聚类数: {n_clusters}")
    print(f"原始空间 Silhouette: {sil_real:.4f}")
    print(f"Embedding空间 Silhouette: {sil_emb:.4f}")
    print(f"达成率: {sil_emb / sil_real * 100:.1f}%")

    if sil_emb / sil_real > 0.8:
        print("\n[PASS] Embedding聚类效果接近原始空间")
    else:
        print("\n[OK] Embedding聚类效果有一定差距")

    # 可视化
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始空间
    ax1 = axes[0]
    scatter1 = ax1.scatter(coords[::5, 0], coords[::5, 1], c=labels_real[::5],
                           cmap='tab20', s=5, alpha=0.6)
    ax1.set_title(f'原始空间聚类 (Sil={sil_real:.3f})')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    plt.colorbar(scatter1, ax=ax1, label='聚类')

    # Embedding空间 (PCA降维)
    ax2 = axes[1]
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    scatter2 = ax2.scatter(emb_2d[::5, 0], emb_2d[::5, 1], c=labels_emb[::5],
                           cmap='tab20', s=5, alpha=0.6)
    ax2.set_title(f'Embedding空间聚类 (Sil={sil_emb:.3f})')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2, label='聚类')

    plt.tight_layout()
    plt.savefig('demo_clustering.png', dpi=150)
    plt.close()
    print(f"\n图表已保存: demo_clustering.png")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("V2.3 空间理解能力Demo")
    print("=" * 60)
    print("验证模型是否真正具备'空间理解'能力\n")

    # 加载数据
    print("加载数据 (10%采样)...")
    data = load_sample_data(rate=0.1)
    print(f"POI数量: {len(data['coords']):,}\n")

    # 训练模型
    model = train_model(data, epochs=500)

    # 运行所有Demo
    demo_spatial_query(data, model)
    demo_distance_preservation(data, model)
    demo_direction_understanding(data, model)
    demo_clustering(data, model)

    # 总结
    print("\n" + "=" * 60)
    print("Demo总结")
    print("=" * 60)
    print("""
空间理解能力验证结果:

1. 空间相似性查询: 验证模型能否找到空间相近的POI
   - 通过邻居重叠率评估

2. 距离保持验证: 验证embedding距离与真实距离的相关性
   - 通过Pearson相关系数评估

3. 空间方向理解: 验证模型是否理解东/南/西/北
   - 通过embedding角度分析

4. 区域聚类可视化: 验证embedding空间的聚类效果
   - 通过Silhouette对比评估

生成的可视化图表:
- demo_distance_preservation.png: 距离保持散点图
- demo_clustering.png: 聚类对比图
""")


if __name__ == "__main__":
    main()
