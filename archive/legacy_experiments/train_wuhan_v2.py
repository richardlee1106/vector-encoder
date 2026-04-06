# -*- coding: utf-8 -*-
"""
V2.3 全量验证 - 武汉市84万+ POI (修正版)

修正：使用正确的聚类数(n=30)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
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
import tqdm


def load_wuhan_pois_sample(sample_rate: float = 0.1) -> Dict:
    """从数据库加载武汉POI采样"""
    print("=" * 60)
    print(f"加载武汉市POI数据 (采样率: {sample_rate*100:.0f}%)")
    print("=" * 60)

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
    total_count = cur.fetchone()[0]
    print(f"数据库总POI数: {total_count:,}")

    limit = int(total_count * sample_rate)
    cur.execute(f'''
        SELECT
            id, ST_X(geom), ST_Y(geom),
            category_big, land_use_type, aoi_type,
            nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois
        WHERE geom IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    ''')
    rows = cur.fetchall()
    print(f"加载了 {len(rows):,} 个POI")
    conn.close()

    # 解析
    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features = [], []

    for row in tqdm.tqdm(rows, desc="解析POI"):
        cat = row[3] or 'unknown'
        if cat not in category_map:
            category_map[cat] = len(category_map)

        lu = row[4] or 'unknown'
        if lu not in landuse_map:
            landuse_map[lu] = len(landuse_map)

        aoi_type = row[5] or 'unknown'
        if aoi_type not in aoi_type_map:
            aoi_type_map[aoi_type] = len(aoi_type_map)

        rc = row[6] or 'unknown'
        if rc not in road_class_map:
            road_class_map[rc] = len(road_class_map)

        poi_coords.append([row[1], row[2]])
        poi_features.append([
            category_map[cat], landuse_map[lu], aoi_type_map[aoi_type],
            road_class_map[rc],
            float(row[7] or 0), float(row[8] or 0), float(row[9] or 0)
        ])

    coords = np.array(poi_coords, dtype=np.float32)
    features = np.array(poi_features, dtype=np.float32)

    print(f"  类别数: {len(category_map)}")
    print(f"  空间范围: lng=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]")

    return {
        'coords': coords,
        'features': features,
        'metadata': {
            'num_pois': len(coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


class SpatialTopologyEncoder(nn.Module):
    def __init__(self, num_categories, num_landuses, num_aoi_types, num_road_classes,
                 embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        emb_dim = hidden_dim // 6

        self.category_emb = nn.Embedding(num_categories + 1, emb_dim)
        self.landuse_emb = nn.Embedding(num_landuses + 1, emb_dim)
        self.aoi_type_emb = nn.Embedding(num_aoi_types + 1, emb_dim)
        self.road_class_emb = nn.Embedding(num_road_classes + 1, emb_dim)
        self.num_proj = nn.Linear(3, emb_dim)
        self.coord_proj = nn.Linear(2, emb_dim)

        input_dim = emb_dim * 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.coord_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(self, features, coords):
        cat_emb = self.category_emb(features[:, 0].long())
        lu_emb = self.landuse_emb(features[:, 1].long())
        aoi_emb = self.aoi_type_emb(features[:, 2].long())
        rc_emb = self.road_class_emb(features[:, 3].long())
        num_emb = self.num_proj(features[:, 4:7])

        coords_norm = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-8)
        coord_emb = self.coord_proj(coords_norm)

        x = torch.cat([cat_emb, lu_emb, aoi_emb, rc_emb, num_emb, coord_emb], dim=-1)
        return F.normalize(self.encoder(x), p=2, dim=-1)

    def decode_coord(self, z):
        return self.coord_decoder(z)

    def forward(self, features, coords):
        z = self.encode(features, coords)
        return z, self.decode_coord(z)


def sample_distance_pairs(coords, num_pairs, device):
    N = len(coords)
    idx_i = np.random.randint(0, N, num_pairs)
    idx_j = np.random.randint(0, N, num_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    coords_i, coords_j = coords[idx_i], coords[idx_j]
    spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    return (
        torch.from_numpy(np.stack([idx_i, idx_j], axis=1)).long().to(device),
        torch.from_numpy(spatial_dists).float().to(device)
    )


def neighbor_loss(z, coords, k=10, sample_size=2000):
    sample_idx = np.random.choice(len(coords), min(sample_size, len(coords)), replace=False)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords[sample_idx])

    total_loss = 0.0
    count = 0

    for i, neighbors in enumerate(indices):
        if len(neighbors) > 1:
            z_center = z[sample_idx[i]]
            z_neighbors = z[neighbors[1:]]
            cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_neighbors, dim=1)
            total_loss += (1 - cos_sim.mean())
            count += 1

    return total_loss / max(count, 1)


def train_wuhan(data: Dict, num_epochs: int = 500, n_clusters: int = 30):
    """训练"""
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    coords = data['coords']
    features = data['features']
    metadata = data['metadata']

    # 聚类
    print(f"[1] 生成聚类标签 (n_clusters={n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    sample_idx = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[sample_idx], labels[sample_idx])
    print(f"  理论上限: {sil_upper:.4f}")

    # 标准化
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    # 模型
    print("\n[2] 创建模型...")
    device = torch.device('cuda')
    model = SpatialTopologyEncoder(
        metadata['num_categories'],
        metadata['num_landuses'],
        metadata['num_aoi_types'],
        metadata['num_road_classes']
    ).to(device)

    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 数据
    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    # 训练
    print(f"\n[3] 训练 ({num_epochs} epochs)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_sil = -1.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        # L1
        loss_l1 = F.mse_loss(coord_recon, coords_norm_t)

        # L2
        pair_idx, spatial_dists = sample_distance_pairs(coords, 5000, device)
        z_i, z_j = z[pair_idx[:, 0]], z[pair_idx[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        loss_l2 = F.mse_loss(emb_dists, spatial_dists)

        # L3 (每20个epoch计算一次)
        if epoch % 20 == 0:
            loss_l3 = neighbor_loss(z, coords)
        else:
            loss_l3 = torch.tensor(0.0, device=device)

        loss = loss_l1 + 2.0 * loss_l2 + loss_l3

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()
                eval_sample = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(z_np[eval_sample], labels[eval_sample])

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}, Sil={sil:.4f}, Best={best_sil:.4f}")

    # 结果
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"最佳 Silhouette: {best_sil:.4f} (Epoch {best_epoch})")
    print(f"理论上限: {sil_upper:.4f}")
    print(f"达成率: {best_sil / sil_upper * 100:.1f}%")

    print("\n对比实验区:")
    print(f"  实验区平均: 0.3994 (90.7%)")
    print(f"  武汉采样: {best_sil:.4f} ({best_sil / sil_upper * 100:.1f}%)")

    return best_sil, sil_upper


if __name__ == "__main__":
    # 10%采样验证
    data = load_wuhan_pois_sample(sample_rate=0.1)
    best_sil, sil_upper = train_wuhan(data, num_epochs=500, n_clusters=30)

    if best_sil / sil_upper > 0.70:  # 达成率超过70%
        print("\n" + "=" * 60)
        print("验证通过！准备全量训练...")
        print("=" * 60)
