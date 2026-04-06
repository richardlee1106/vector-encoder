# -*- coding: utf-8 -*-
"""
V2.3 全量验证 - 武汉市84万+ POI

目的：验证模型在大规模数据上的泛化能力
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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import cKDTree
import tqdm


# ============================================================================
# 数据加载（从PostGIS）
# ============================================================================

def load_wuhan_pois(limit: int = None, sample_rate: float = None) -> Dict:
    """
    从数据库加载武汉市全量POI

    Args:
        limit: 限制数量（用于快速测试）
        sample_rate: 采样率（0-1，用于快速测试）
    """
    print("=" * 60)
    print("加载武汉市全量POI数据")
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

    # 查询总数
    cur.execute("SELECT COUNT(*) FROM pois WHERE geom IS NOT NULL")
    total_count = cur.fetchone()[0]
    print(f"数据库总POI数: {total_count:,}")

    # 构建查询
    sql = """
        SELECT
            id,
            ST_X(geom) as lng,
            ST_Y(geom) as lat,
            category_big,
            land_use_type,
            aoi_type,
            nearest_road_class,
            poi_density_500m,
            category_entropy_500m,
            nearest_road_dist_m
        FROM pois
        WHERE geom IS NOT NULL
    """

    if limit:
        sql += f" ORDER BY id LIMIT {limit}"

    cur.execute(sql)
    rows = cur.fetchall()

    print(f"加载了 {len(rows):,} 个POI")

    # 如果需要采样
    if sample_rate and sample_rate < 1.0:
        sample_size = int(len(rows) * sample_rate)
        indices = np.random.choice(len(rows), sample_size, replace=False)
        rows = [rows[i] for i in indices]
        print(f"采样后: {len(rows):,} 个POI")

    # 解析数据
    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features, poi_ids = [], [], []

    for row in tqdm.tqdm(rows, desc="解析POI"):
        poi_id, lng, lat = row[0], row[1], row[2]

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

        density = float(row[7] or 0)
        entropy = float(row[8] or 0)
        road_dist = float(row[9] or 0)

        poi_ids.append(poi_id)
        poi_coords.append([lng, lat])
        poi_features.append([
            category_map[cat], landuse_map[lu], aoi_type_map[aoi_type],
            road_class_map[rc], density, entropy, road_dist
        ])

    conn.close()

    coords = np.array(poi_coords, dtype=np.float32)
    features = np.array(poi_features, dtype=np.float32)

    print(f"\n数据统计:")
    print(f"  POI数量: {len(coords):,}")
    print(f"  类别数: {len(category_map)}")
    print(f"  土地利用类型数: {len(landuse_map)}")
    print(f"  AOI类型数: {len(aoi_type_map)}")
    print(f"  道路等级数: {len(road_class_map)}")
    print(f"  空间范围: lng=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]")
    print(f"             lat=[{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]")

    return {
        'coords': coords,
        'features': features,
        'ids': poi_ids,
        'metadata': {
            'num_pois': len(coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
        }
    }


# ============================================================================
# 模型定义
# ============================================================================

class SpatialTopologyEncoder(nn.Module):
    """V2.3 空间拓扑编码器"""

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

def sample_distance_pairs_batched(coords, num_pairs, device, batch_size=10000):
    """分批采样距离对（内存友好）"""
    N = len(coords)

    all_pair_indices = []
    all_spatial_dists = []

    remaining = num_pairs
    while remaining > 0:
        batch = min(batch_size, remaining)

        idx_i = np.random.randint(0, N, batch)
        idx_j = np.random.randint(0, N, batch)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        coords_i, coords_j = coords[idx_i], coords[idx_j]
        spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))

        all_pair_indices.append(np.stack([idx_i, idx_j], axis=1))
        all_spatial_dists.append(spatial_dists)

        remaining -= len(idx_i)

    pair_indices = np.vstack(all_pair_indices)
    spatial_dists = np.concatenate(all_spatial_dists)

    # 归一化
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    return (
        torch.from_numpy(pair_indices).long().to(device),
        torch.from_numpy(spatial_dists).float().to(device)
    )


def neighbor_loss_efficient(z, coords, k=10, sample_size=2000):
    """高效邻居损失（基于空间坐标动态计算）"""
    from sklearn.neighbors import NearestNeighbors

    sample_indices = np.random.choice(len(coords), min(sample_size, len(coords)), replace=False)
    sample_coords = coords[sample_indices]

    # 动态计算KNN
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(sample_coords)

    total_loss = 0.0
    count = 0

    z_np = z.detach().cpu().numpy()

    for i, neighbors in enumerate(indices):
        center_idx = sample_indices[i]
        neighbor_indices = [sample_indices[n] if n < len(sample_indices) else n for n in neighbors[1:]]

        if len(neighbor_indices) > 0:
            z_center = z[center_idx]
            z_neighbors = z[neighbor_indices]
            cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_neighbors, dim=1)
            total_loss += (1 - cos_sim.mean())
            count += 1

    return total_loss / max(count, 1)


# ============================================================================
# 训练
# ============================================================================

def train_full_wuhan(data: Dict, num_epochs: int = 500, n_clusters: int = 50):
    """
    训练全量武汉模型

    Args:
        data: 数据字典
        num_epochs: 训练轮数
        n_clusters: 聚类数（全量数据需要更多聚类）
    """
    print("\n" + "=" * 60)
    print("开始训练 - 武汉市全量")
    print("=" * 60)

    coords = data['coords']
    features = data['features']
    metadata = data['metadata']

    # 生成标签（使用MiniBatchKMeans处理大数据）
    print(f"\n[1] 生成空间聚类标签 (n_clusters={n_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    # 计算理论上限（采样计算，全量太慢）
    sample_idx = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[sample_idx], labels[sample_idx])
    print(f"  理论上限（采样估计）: {sil_upper:.4f}")

    # 标准化坐标
    coords_mean = coords.mean(axis=0)
    coords_std = coords.std(axis=0)
    coords_norm = (coords - coords_mean) / coords_std

    # 创建模型
    print("\n[2] 创建模型...")
    device = torch.device('cuda')
    model = SpatialTopologyEncoder(
        metadata['num_categories'],
        metadata['num_landuses'],
        metadata['num_aoi_types'],
        metadata['num_road_classes']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 准备数据
    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    # 训练
    print(f"\n[3] 开始训练 ({num_epochs} epochs)...")
    print("  损失配置: L1=1.0, L2=2.0, L3=1.0")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_sil = -1.0
    best_epoch = 0
    history = []

    for epoch in tqdm.tqdm(range(num_epochs), desc="训练中"):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        # L1: 坐标重构
        loss_l1 = F.mse_loss(coord_recon, coords_norm_t)

        # L2: 距离保持
        pair_idx, spatial_dists = sample_distance_pairs_batched(coords, 5000, device)
        z_i, z_j = z[pair_idx[:, 0]], z[pair_idx[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        loss_l2 = F.mse_loss(emb_dists, spatial_dists)

        # L3: 邻居一致性（每20个epoch计算一次，节省时间）
        if epoch % 20 == 0:
            loss_l3 = neighbor_loss_efficient(z, coords, k=10, sample_size=2000)
        else:
            loss_l3 = torch.tensor(0.0, device=device)

        # 总损失
        loss = loss_l1 + 2.0 * loss_l2 + loss_l3

        loss.backward()
        optimizer.step()
        scheduler.step()

        # 评估
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()

                # 采样评估Silhouette
                eval_sample = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(z_np[eval_sample], labels[eval_sample])

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'silhouette': sil,
            })

            print(f"\n  Epoch {epoch}: Loss={loss.item():.4f}, Sil={sil:.4f}, Best={best_sil:.4f}")

    # 最终结果
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"最佳 Silhouette: {best_sil:.4f} (Epoch {best_epoch})")
    print(f"理论上限: {sil_upper:.4f}")
    print(f"达成率: {best_sil / sil_upper * 100:.1f}%")

    # 与实验区对比
    print("\n对比实验区结果:")
    print(f"  guanggu_core: 0.4153 (92.4%)")
    print(f"  wuda_area: 0.4075 (89.6%)")
    print(f"  zhongjia_cun: 0.3756 (90.3%)")
    print(f"  平均: 0.3994 (90.7%)")
    print(f"  ---")
    print(f"  武汉全量: {best_sil:.4f} ({best_sil / sil_upper * 100:.1f}%)")

    return {
        'best_silhouette': best_sil,
        'best_epoch': best_epoch,
        'upper_bound': sil_upper,
        'history': history,
        'model': model,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 先用采样数据快速验证（10%采样）
    print("阶段1: 10%采样快速验证")
    data = load_wuhan_pois(sample_rate=0.1)
    result = train_full_wuhan(data, num_epochs=300, n_clusters=50)

    if result['best_silhouette'] > 0.30:
        print("\n采样验证通过，运行全量训练...")
        # 全量训练
        data_full = load_wuhan_pois()
        result_full = train_full_wuhan(data_full, num_epochs=500, n_clusters=100)
    else:
        print(f"\n采样验证效果不佳 (Sil={result['best_silhouette']:.4f})，需要分析原因")


if __name__ == "__main__":
    main()
