# -*- coding: utf-8 -*-
"""
V2.3 优化实验 - 武汉全量

目标：达成率从76%提升到85%

优化方向：
1. 损失权重调优（消融实验显示L2是核心）
2. 学习率调整
3. 增加训练轮数
4. 聚类数优化
"""

import sys
import json
from pathlib import Path
from datetime import datetime
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


def load_wuhan_sample(sample_rate: float = 0.1):
    """加载武汉POI采样"""
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
    limit = int(total_count * sample_rate)

    cur.execute(f'''
        SELECT ST_X(geom), ST_Y(geom),
            category_big, land_use_type, aoi_type, nearest_road_class,
            poi_density_500m, category_entropy_500m, nearest_road_dist_m
        FROM pois
        WHERE geom IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
    ''')
    rows = cur.fetchall()
    conn.close()

    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features = [], []

    for row in rows:
        cat = row[2] or 'unknown'
        if cat not in category_map:
            category_map[cat] = len(category_map)
        lu = row[3] or 'unknown'
        if lu not in landuse_map:
            landuse_map[lu] = len(landuse_map)
        aoi_type = row[4] or 'unknown'
        if aoi_type not in aoi_type_map:
            aoi_type_map[aoi_type] = len(aoi_type_map)
        rc = row[5] or 'unknown'
        if rc not in road_class_map:
            road_class_map[rc] = len(road_class_map)

        poi_coords.append([row[0], row[1]])
        poi_features.append([
            category_map[cat], landuse_map[lu], aoi_type_map[aoi_type],
            road_class_map[rc],
            float(row[6] or 0), float(row[7] or 0), float(row[8] or 0)
        ])

    return {
        'coords': np.array(poi_coords, dtype=np.float32),
        'features': np.array(poi_features, dtype=np.float32),
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


class Encoder(nn.Module):
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
        return self.encode(features, coords), self.decode_coord(self.encode(features, coords))


def sample_pairs(coords, num_pairs, device):
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


def run_experiment(data, config, name):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"配置: {config}")
    print(f"{'='*60}")

    coords = data['coords']
    features = data['features']
    metadata = data['metadata']

    # 聚类
    kmeans = MiniBatchKMeans(n_clusters=config['n_clusters'], random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    sample_idx = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[sample_idx], labels[sample_idx])
    print(f"理论上限: {sil_upper:.4f}")

    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    device = torch.device('cuda')
    model = Encoder(
        metadata['num_categories'], metadata['num_landuses'],
        metadata['num_aoi_types'], metadata['num_road_classes']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    best_sil = -1
    best_epoch = 0

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        # 损失
        loss_l1 = F.mse_loss(coord_recon, coords_norm_t)

        pair_idx, spatial_dists = sample_pairs(coords, config['num_pairs'], device)
        z_i, z_j = z[pair_idx[:, 0]], z[pair_idx[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        loss_l2 = F.mse_loss(emb_dists, spatial_dists)

        if epoch % 10 == 0:
            loss_l3 = neighbor_loss(z, coords)
        else:
            loss_l3 = torch.tensor(0.0, device=device)

        loss = (config['w1'] * loss_l1 + config['w2'] * loss_l2 + config['w3'] * loss_l3)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()
                eval_sample = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(z_np[eval_sample], labels[eval_sample])

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            rate = best_sil / sil_upper * 100
            print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Sil={sil:.4f}, Best={best_sil:.4f} ({rate:.1f}%)")

    final_rate = best_sil / sil_upper * 100
    print(f"\n结果: Sil={best_sil:.4f}, 达成率={final_rate:.1f}%")

    return {
        'name': name,
        'best_sil': best_sil,
        'sil_upper': sil_upper,
        'rate': final_rate,
        'best_epoch': best_epoch,
        'config': config,
    }


def main():
    print("加载数据...")
    data = load_wuhan_sample(sample_rate=0.1)
    print(f"POI数量: {data['metadata']['num_pois']:,}")

    # 优化实验配置
    experiments = [
        # 基准（当前）
        {
            'name': 'Baseline',
            'config': {
                'epochs': 500, 'lr': 1e-3, 'weight_decay': 1e-5,
                'w1': 1.0, 'w2': 2.0, 'w3': 1.0,
                'n_clusters': 30, 'num_pairs': 5000
            }
        },
        # 提升L2权重
        {
            'name': 'L2_weight=3',
            'config': {
                'epochs': 500, 'lr': 1e-3, 'weight_decay': 1e-5,
                'w1': 1.0, 'w2': 3.0, 'w3': 1.0,
                'n_clusters': 30, 'num_pairs': 5000
            }
        },
        # 更多距离采样
        {
            'name': 'More_pairs',
            'config': {
                'epochs': 500, 'lr': 1e-3, 'weight_decay': 1e-5,
                'w1': 1.0, 'w2': 2.0, 'w3': 1.0,
                'n_clusters': 30, 'num_pairs': 10000
            }
        },
        # 更长训练
        {
            'name': 'More_epochs',
            'config': {
                'epochs': 800, 'lr': 1e-3, 'weight_decay': 1e-5,
                'w1': 1.0, 'w2': 2.0, 'w3': 1.0,
                'n_clusters': 30, 'num_pairs': 5000
            }
        },
        # 调整聚类数
        {
            'name': 'N_clusters=20',
            'config': {
                'epochs': 500, 'lr': 1e-3, 'weight_decay': 1e-5,
                'w1': 1.0, 'w2': 2.0, 'w3': 1.0,
                'n_clusters': 20, 'num_pairs': 5000
            }
        },
        # 综合优化
        {
            'name': 'Optimized',
            'config': {
                'epochs': 800, 'lr': 5e-4, 'weight_decay': 1e-5,
                'w1': 0.5, 'w2': 3.0, 'w3': 0.5,
                'n_clusters': 25, 'num_pairs': 8000
            }
        },
    ]

    results = []
    for exp in experiments:
        result = run_experiment(data, exp['config'], exp['name'])
        results.append(result)

        if result['rate'] >= 85:
            print(f"\n[SUCCESS] 达成目标！{exp['name']} 达成率 {result['rate']:.1f}%")
            break

    # 汇总
    print("\n" + "=" * 60)
    print("实验汇总")
    print("=" * 60)
    print(f"{'实验名':<15} {'Silhouette':<12} {'理论上限':<12} {'达成率'}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<15} {r['best_sil']:.4f}       {r['sil_upper']:.4f}       {r['rate']:.1f}%")

    best = max(results, key=lambda x: x['rate'])
    print("-" * 55)
    print(f"最佳: {best['name']} (达成率 {best['rate']:.1f}%)")


if __name__ == "__main__":
    main()
