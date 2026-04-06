# -*- coding: utf-8 -*-
"""快速消融实验"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

print('=' * 60)
print('快速消融实验 (150 epochs each)')
print('=' * 60)

# 加载数据
area_name = 'guanggu_core'
data_dir = str(Path(__file__).resolve().parents[2] / 'data' / 'experiment_data')
area_dir = Path(data_dir) / area_name

with open(area_dir / 'pois.geojson', 'r', encoding='utf-8') as f:
    pois = json.load(f)

category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
poi_coords, poi_features = [], []

for feat in pois['features']:
    props = feat['properties']
    coords = feat['geometry']['coordinates']
    poi_coords.append(coords)

    cat = props.get('category_big', 'unknown') or 'unknown'
    if cat not in category_map:
        category_map[cat] = len(category_map)

    lu = props.get('land_use_type', 'unknown') or 'unknown'
    if lu not in landuse_map:
        landuse_map[lu] = len(landuse_map)

    aoi_type = props.get('aoi_type', 'unknown') or 'unknown'
    if aoi_type not in aoi_type_map:
        aoi_type_map[aoi_type] = len(aoi_type_map)

    rc = props.get('nearest_road_class', 'unknown') or 'unknown'
    if rc not in road_class_map:
        road_class_map[rc] = len(road_class_map)

    density = float(props.get('poi_density_500m', 0) or 0)
    entropy = float(props.get('category_entropy', 0) or 0)
    road_dist = float(props.get('nearest_road_dist_m', 0) or 0)

    poi_features.append([
        category_map[cat], landuse_map[lu], aoi_type_map[aoi_type],
        road_class_map[rc], density, entropy, road_dist
    ])

coords = np.array(poi_coords, dtype=np.float32)
features = np.array(poi_features, dtype=np.float32)

adj = kneighbors_graph(coords, n_neighbors=10, mode='connectivity', include_self=False)
knn_neighbors = [adj[i].nonzero()[1] for i in range(len(coords))]

kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
labels = kmeans.fit_predict(coords)
sil_upper = silhouette_score(coords, labels)

print(f'POI数量: {len(coords)}')
print(f'理论上限: {sil_upper:.4f}')


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        emb_dim = 128 // 6
        self.category_emb = nn.Embedding(len(category_map) + 1, emb_dim)
        self.landuse_emb = nn.Embedding(len(landuse_map) + 1, emb_dim)
        self.aoi_type_emb = nn.Embedding(len(aoi_type_map) + 1, emb_dim)
        self.road_class_emb = nn.Embedding(len(road_class_map) + 1, emb_dim)
        self.num_proj = nn.Linear(3, emb_dim)
        self.coord_proj = nn.Linear(2, emb_dim)

        self.encoder = nn.Sequential(
            nn.Linear(emb_dim * 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        self.coord_decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
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


def neighbor_loss(z, knn_neighbors):
    total_loss = 0.0
    count = 0
    sample_size = min(500, len(knn_neighbors))
    for i in np.random.choice(len(knn_neighbors), sample_size, replace=False):
        neighbors = knn_neighbors[i]
        if len(neighbors) > 0:
            z_center = z[i]
            z_neighbors = z[neighbors]
            cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_neighbors, dim=1)
            total_loss += (1 - cos_sim.mean())
            count += 1
    return total_loss / max(count, 1)


device = torch.device('cuda')
coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)
features_t = torch.from_numpy(features).float().to(device)
coords_t = torch.from_numpy(coords).float().to(device)
coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

# 消融实验配置
experiments = [
    {'name': 'L1 only', 'l1': True, 'l2': False, 'l3': False, 'w1': 1.0, 'w2': 0.0, 'w3': 0.0},
    {'name': 'L2 only', 'l1': False, 'l2': True, 'l3': False, 'w1': 0.0, 'w2': 1.0, 'w3': 0.0},
    {'name': 'L3 only', 'l1': False, 'l2': False, 'l3': True, 'w1': 0.0, 'w2': 0.0, 'w3': 1.0},
    {'name': 'L1+L2', 'l1': True, 'l2': True, 'l3': False, 'w1': 1.0, 'w2': 2.0, 'w3': 0.0},
    {'name': 'L1+L3', 'l1': True, 'l2': False, 'l3': True, 'w1': 1.0, 'w2': 0.0, 'w3': 1.0},
    {'name': 'L2+L3', 'l1': False, 'l2': True, 'l3': True, 'w1': 0.0, 'w2': 2.0, 'w3': 1.0},
    {'name': 'Full (w2=2)', 'l1': True, 'l2': True, 'l3': True, 'w1': 1.0, 'w2': 2.0, 'w3': 1.0},
]

results = {}

for exp in experiments:
    print(f"\n--- {exp['name']} ---")
    model = SimpleEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    best_sil = -1

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        loss = 0.0
        if exp['l1']:
            loss += exp['w1'] * F.mse_loss(coord_recon, coords_norm_t)
        if exp['l2']:
            pair_idx, spatial_dists = sample_distance_pairs(coords, 3000, device)
            z_i, z_j = z[pair_idx[:, 0]], z[pair_idx[:, 1]]
            emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
            loss += exp['w2'] * F.mse_loss(emb_dists, spatial_dists)
        if exp['l3']:
            loss += exp['w3'] * neighbor_loss(z, knn_neighbors)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 149:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()
                sil = silhouette_score(z_np, labels)
                if sil > best_sil:
                    best_sil = sil
                print(f'  Epoch {epoch}: Loss={loss.item():.4f}, Sil={sil:.4f}, Best={best_sil:.4f}')

    results[exp['name']] = best_sil

# 汇总
print('\n' + '=' * 60)
print('消融实验汇总')
print('=' * 60)
print(f"{'配置':<15} {'Silhouette':<10} {'达成率'}")
print('-' * 45)
for name, sil in results.items():
    rate = sil / sil_upper * 100
    print(f'{name:<15} {sil:.4f}     {rate:.1f}%')
print('-' * 45)
print(f'理论上限: {sil_upper:.4f}')
