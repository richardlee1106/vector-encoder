# -*- coding: utf-8 -*-
"""
V2.3 消融实验

目的：理解三重损失各自的贡献

实验设计：
1. 只用坐标重构损失 (L1)
2. 只用距离保持损失 (L2)
3. 只用邻居一致性损失 (L3)
4. L1 + L2
5. L1 + L3
6. L2 + L3
7. L1 + L2 + L3 (完整V2.3)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph


# ============================================================================
# 数据加载
# ============================================================================

def load_area_data(area_name: str, data_dir: str) -> Dict:
    """加载区域数据"""
    area_dir = Path(data_dir) / area_name
    with open(area_dir / "pois.geojson", 'r', encoding='utf-8') as f:
        pois = json.load(f)

    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features = [], []

    for f in pois['features']:
        props = f['properties']
        coords = f['geometry']['coordinates']
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

    knn_k = 10
    adj = kneighbors_graph(coords, n_neighbors=knn_k, mode='connectivity', include_self=False)
    knn_neighbors = [adj[i].nonzero()[1] for i in range(len(coords))]

    return {
        'coords': coords,
        'features': np.array(poi_features, dtype=np.float32),
        'knn_neighbors': knn_neighbors,
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


# ============================================================================
# 模型
# ============================================================================

class SimpleEncoder(nn.Module):
    """简单MLP编码器"""
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
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=-1)

    def decode_coord(self, z):
        return self.coord_decoder(z)

    def forward(self, features, coords):
        z = self.encode(features, coords)
        coord_recon = self.decode_coord(z)
        return z, coord_recon


# ============================================================================
# 损失函数
# ============================================================================

def coord_recon_loss(coord_recon, coords_norm):
    """L1: 坐标重构损失"""
    return F.mse_loss(coord_recon, coords_norm)


def distance_preserve_loss(z, coords, num_pairs, device):
    """L2: 距离保持损失"""
    N = len(coords)
    idx_i = np.random.randint(0, N, num_pairs)
    idx_j = np.random.randint(0, N, num_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    coords_i, coords_j = coords[idx_i], coords[idx_j]
    spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    pair_indices = torch.from_numpy(np.stack([idx_i, idx_j], axis=1)).long().to(device)
    spatial_dists_t = torch.from_numpy(spatial_dists).float().to(device)

    z_i, z_j = z[pair_indices[:, 0]], z[pair_indices[:, 1]]
    emb_dists = torch.norm(z_i - z_j, p=2, dim=1)

    return F.mse_loss(emb_dists, spatial_dists_t)


def neighbor_consistency_loss(z, knn_neighbors):
    """L3: 邻居一致性损失"""
    total_loss = 0.0
    count = 0
    sample_size = min(1000, len(knn_neighbors))
    sample_indices = torch.randint(0, len(knn_neighbors), (sample_size,))

    for i in sample_indices:
        i = i.item()
        neighbors = knn_neighbors[i]
        if len(neighbors) > 0:
            z_center = z[i]
            z_neighbors = z[neighbors]
            cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_neighbors, dim=1)
            total_loss += (1 - cos_sim.mean())
            count += 1

    return total_loss / max(count, 1)


# ============================================================================
# 消融实验
# ============================================================================

def run_ablation(area_name: str, data_dir: str, loss_config: Dict, num_epochs: int = 500):
    """
    运行单个消融实验

    Args:
        loss_config: {
            'use_l1': bool,  # 坐标重构
            'use_l2': bool,  # 距离保持
            'use_l3': bool,  # 邻居一致性
            'w1': float,     # L1权重
            'w2': float,     # L2权重
            'w3': float,     # L3权重
        }
    """
    print(f"\n{'='*60}")
    config_str = f"L1={loss_config['use_l1']} L2={loss_config['use_l2']} L3={loss_config['use_l3']}"
    print(f"消融实验: {config_str}")
    print(f"权重: w1={loss_config['w1']}, w2={loss_config['w2']}, w3={loss_config['w3']}")
    print(f"{'='*60}")

    device = torch.device('cuda')

    # 加载数据
    data = load_area_data(area_name, data_dir)
    coords = data['coords']
    features = data['features']
    knn_neighbors = data['knn_neighbors']
    metadata = data['metadata']

    # 标签
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    sil_upper = silhouette_score(coords, labels)

    # 标准化
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    # 模型
    model = SimpleEncoder(
        metadata['num_categories'], metadata['num_landuses'],
        metadata['num_aoi_types'], metadata['num_road_classes']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    best_sil = -1.0
    best_epoch = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        # 计算损失
        loss = 0.0
        loss_components = {}

        if loss_config['use_l1']:
            l1 = coord_recon_loss(coord_recon, coords_norm_t)
            loss += loss_config['w1'] * l1
            loss_components['l1'] = l1.item()

        if loss_config['use_l2']:
            l2 = distance_preserve_loss(z, coords, 5000, device)
            loss += loss_config['w2'] * l2
            loss_components['l2'] = l2.item()

        if loss_config['use_l3']:
            l3 = neighbor_consistency_loss(z, knn_neighbors)
            loss += loss_config['w3'] * l3
            loss_components['l3'] = l3.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        # 评估
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()
                sil = silhouette_score(z_np, labels)

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'silhouette': sil,
                **loss_components
            })

            print(f"  Epoch {epoch:3d} | Loss={loss.item():.4f} | Sil={sil:.4f} | Best={best_sil:.4f}")

    return {
        'config': loss_config,
        'best_silhouette': best_sil,
        'best_epoch': best_epoch,
        'upper_bound': sil_upper,
        'history': history,
    }


def run_all_ablations(area_name: str = "guanggu_core", data_dir: str = None):
    """运行所有消融实验"""
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[2] / "data" / "experiment_data")

    # 实验配置
    experiments = [
        # 单损失
        {'name': 'L1 only', 'use_l1': True, 'use_l2': False, 'use_l3': False, 'w1': 1.0, 'w2': 0.0, 'w3': 0.0},
        {'name': 'L2 only', 'use_l1': False, 'use_l2': True, 'use_l3': False, 'w1': 0.0, 'w2': 1.0, 'w3': 0.0},
        {'name': 'L3 only', 'use_l1': False, 'use_l2': False, 'use_l3': True, 'w1': 0.0, 'w2': 0.0, 'w3': 1.0},

        # 两两组合
        {'name': 'L1+L2', 'use_l1': True, 'use_l2': True, 'use_l3': False, 'w1': 1.0, 'w2': 2.0, 'w3': 0.0},
        {'name': 'L1+L3', 'use_l1': True, 'use_l2': False, 'use_l3': True, 'w1': 1.0, 'w2': 0.0, 'w3': 1.0},
        {'name': 'L2+L3', 'use_l1': False, 'use_l2': True, 'use_l3': True, 'w1': 0.0, 'w2': 2.0, 'w3': 1.0},

        # 完整V2.3
        {'name': 'L1+L2+L3 (Full)', 'use_l1': True, 'use_l2': True, 'use_l3': True, 'w1': 1.0, 'w2': 2.0, 'w3': 1.0},

        # 权重消融
        {'name': 'L1+L2+L3 (w2=1)', 'use_l1': True, 'use_l2': True, 'use_l3': True, 'w1': 1.0, 'w2': 1.0, 'w3': 1.0},
        {'name': 'L1+L2+L3 (w2=3)', 'use_l1': True, 'use_l2': True, 'use_l3': True, 'w1': 1.0, 'w2': 3.0, 'w3': 1.0},
    ]

    results = {}

    for exp in experiments:
        name = exp['name']
        result = run_ablation(area_name, data_dir, exp, num_epochs=300)
        results[name] = result

    # 汇总
    print("\n" + "=" * 70)
    print("消融实验汇总")
    print("=" * 70)
    print(f"{'配置':<20} {'Silhouette':<12} {'理论上限':<12} {'达成率'}")
    print("-" * 60)

    for name, res in results.items():
        rate = res['best_silhouette'] / res['upper_bound'] * 100
        print(f"{name:<20} {res['best_silhouette']:.4f}       {res['upper_bound']:.4f}       {rate:.1f}%")

    # 保存结果
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 转换为可序列化格式
    results_json = {}
    for name, res in results.items():
        results_json[name] = {
            'config': res['config'],
            'best_silhouette': float(res['best_silhouette']),
            'best_epoch': int(res['best_epoch']),
            'upper_bound': float(res['upper_bound']),
        }

    with open(output_dir / f'ablation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_dir / f'ablation_results_{timestamp}.json'}")

    return results


if __name__ == "__main__":
    run_all_ablations()
