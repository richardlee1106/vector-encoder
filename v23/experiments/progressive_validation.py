# -*- coding: utf-8 -*-
"""
V2.3 渐进式全量验证

数据规模：10% → 30% → 60% → 80% → 100%
目标：观察模型性能随数据规模的变化

记录指标：
- Silhouette
- 理论上限
- 达成率
- 训练时间
- 内存使用
"""

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
import tqdm
import gc


# ============================================================================
# 全局结果存储
# ============================================================================

RESULTS = []


# ============================================================================
# 数据加载
# ============================================================================

def load_wuhan_sample(sample_rate: float):
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

    print(f"加载 {sample_rate*100:.0f}% 数据 (目标: {limit:,} 个POI)...")

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


# ============================================================================
# 模型
# ============================================================================

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


# ============================================================================
# 训练
# ============================================================================

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


def train_and_evaluate(data, sample_rate, epochs=800):
    """训练并评估"""
    print("\n" + "=" * 70)
    print(f"数据规模: {sample_rate*100:.0f}%")
    print("=" * 70)

    start_time = time.time()

    coords = data['coords']
    features = data['features']
    metadata = data['metadata']

    print(f"POI数量: {metadata['num_pois']:,}")
    print(f"类别数: {metadata['num_categories']}")

    # 空间范围
    print(f"空间范围: lng=[{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]")
    print(f"          lat=[{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]")

    # 计算空间跨度
    lng_span = coords[:,0].max() - coords[:,0].min()
    lat_span = coords[:,1].max() - coords[:,1].min()
    print(f"空间跨度: lng={lng_span:.4f}°, lat={lat_span:.4f}°")

    # 聚类
    n_clusters = max(30, int(metadata['num_pois'] / 3000))  # 动态调整聚类数
    print(f"\n聚类数: {n_clusters}")

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000)
    labels = kmeans.fit_predict(coords)

    sample_idx = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
    sil_upper = silhouette_score(coords[sample_idx], labels[sample_idx])
    print(f"理论上限: {sil_upper:.4f}")

    # 标准化
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    # 模型
    device = torch.device('cuda')
    model = Encoder(
        metadata['num_categories'], metadata['num_landuses'],
        metadata['num_aoi_types'], metadata['num_road_classes']
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    # 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    best_sil = -1
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        z, coord_recon = model(features_t, coords_t)

        loss_l1 = F.mse_loss(coord_recon, coords_norm_t)
        pair_idx, spatial_dists = sample_pairs(coords, 5000, device)
        z_i, z_j = z[pair_idx[:, 0]], z[pair_idx[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        loss_l2 = F.mse_loss(emb_dists, spatial_dists)

        if epoch % 10 == 0:
            loss_l3 = neighbor_loss(z, coords)
        else:
            loss_l3 = torch.tensor(0.0, device=device)

        loss = loss_l1 + 2.0 * loss_l2 + loss_l3

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(features_t, coords_t).cpu().numpy()
                eval_sample = np.random.choice(len(coords), min(10000, len(coords)), replace=False)
                sil = silhouette_score(z_np[eval_sample], labels[eval_sample])

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            rate = best_sil / sil_upper * 100
            print(f"  Epoch {epoch:3d}: Sil={sil:.4f}, Best={best_sil:.4f} ({rate:.1f}%)")

    train_time = time.time() - start_time

    # 清理
    del model, features_t, coords_t, coords_norm_t
    gc.collect()
    torch.cuda.empty_cache()

    # 结果
    result = {
        'sample_rate': sample_rate,
        'num_pois': metadata['num_pois'],
        'n_clusters': n_clusters,
        'sil_upper': sil_upper,
        'best_sil': best_sil,
        'best_epoch': best_epoch,
        'rate': best_sil / sil_upper,
        'train_time': train_time,
        'lng_span': lng_span,
        'lat_span': lat_span,
        'params': params,
    }

    print(f"\n结果:")
    print(f"  Silhouette: {best_sil:.4f}")
    print(f"  理论上限: {sil_upper:.4f}")
    print(f"  达成率: {best_sil / sil_upper * 100:.1f}%")
    print(f"  训练时间: {train_time/60:.1f} 分钟")

    return result


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(results):
    """生成对比报告"""
    print("\n" + "=" * 70)
    print("渐进式验证报告")
    print("=" * 70)

    # 表格
    print(f"\n{'数据规模':<10} {'POI数量':<12} {'聚类数':<8} {'理论上限':<10} {'Silhouette':<12} {'达成率':<10} {'训练时间'}")
    print("-" * 80)

    for r in results:
        print(f"{r['sample_rate']*100:>5.0f}%     {r['num_pois']:>10,}  {r['n_clusters']:>6}  "
              f"{r['sil_upper']:>8.4f}   {r['best_sil']:>10.4f}   {r['rate']*100:>6.1f}%    {r['train_time']/60:>5.1f}min")

    # 分析
    print("\n" + "=" * 70)
    print("分析")
    print("=" * 70)

    # 趋势分析
    print("\n1. 达成率趋势:")
    rates = [r['rate'] for r in results]
    print(f"   最小: {min(rates)*100:.1f}%")
    print(f"   最大: {max(rates)*100:.1f}%")
    print(f"   平均: {np.mean(rates)*100:.1f}%")
    print(f"   标准差: {np.std(rates)*100:.1f}%")

    print("\n2. 理论上限变化:")
    for r in results:
        print(f"   {r['sample_rate']*100:>5.0f}%: {r['sil_upper']:.4f}")

    print("\n3. 训练时间增长:")
    times = [r['train_time']/60 for r in results]
    scales = [r['sample_rate'] for r in results]
    print(f"   时间/数据比例: {times[-1]/times[0]:.1f}x (数据增长{scales[-1]/scales[0]:.1f}x)")

    # 关键发现
    print("\n" + "=" * 70)
    print("关键发现")
    print("=" * 70)

    # 达成率稳定性
    if np.std(rates) < 0.05:
        print("[+] 达成率稳定，模型泛化能力强")
    else:
        print("[!] 达成率波动较大，需要分析原因")

    # 理论上限趋势
    upper_bounds = [r['sil_upper'] for r in results]
    if upper_bounds[-1] < upper_bounds[0] * 0.9:
        print("[!] 理论上限随数据规模下降，空间分布更分散")
    else:
        print("[+] 理论上限稳定")

    # 训练效率
    time_per_poi = [r['train_time']/r['num_pois'] for r in results]
    if time_per_poi[-1] < time_per_poi[0] * 1.5:
        print("[+] 训练效率稳定")
    else:
        print("[!] 大规模数据训练效率下降")

    # 保存JSON
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f'progressive_validation_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_dir / f'progressive_validation_{timestamp}.json'}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    sample_rates = [0.10, 0.30, 0.60, 0.80, 1.00]
    epochs_per_scale = [800, 800, 800, 800, 600]  # 全量数据可能需要更少epoch

    print("=" * 70)
    print("V2.3 渐进式全量验证")
    print("=" * 70)
    print(f"数据规模: {[f'{r*100:.0f}%' for r in sample_rates]}")
    print(f"目标: 观察模型性能随数据规模的变化")

    results = []

    for i, rate in enumerate(sample_rates):
        print(f"\n\n{'#'*70}")
        print(f"# 阶段 {i+1}/{len(sample_rates)}: {rate*100:.0f}% 数据")
        print(f"{'#'*70}")

        # 加载数据
        data = load_wuhan_sample(rate)

        # 训练
        result = train_and_evaluate(data, rate, epochs=epochs_per_scale[i])
        results.append(result)

        # 中间报告
        if i < len(sample_rates) - 1:
            print(f"\n当前进度: {rate*100:.0f}% 完成")
            print(f"当前最佳达成率: {result['rate']*100:.1f}%")

        # 内存检查
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU内存使用: {mem_used:.1f}GB / {mem_total:.1f}GB")

            if mem_used > mem_total * 0.8:
                print("[!] GPU内存接近上限，可能需要调整参数")

    # 最终报告
    generate_report(results)

    # 对比实验区
    print("\n" + "=" * 70)
    print("与实验区对比")
    print("=" * 70)
    print(f"{'数据集':<20} {'POI数量':<12} {'Silhouette':<12} {'达成率'}")
    print("-" * 60)
    print(f"{'guanggu_core':<20} {'13,399':<12} {'0.4153':<12} {'92.4%'}")
    print(f"{'wuda_area':<20} {'6,847':<12} {'0.4075':<12} {'89.6%'}")
    print(f"{'zhongjia_cun':<20} {'17,407':<12} {'0.3756':<12} {'90.3%'}")
    print(f"{'实验区平均':<20} {'12,551':<12} {'0.3994':<12} {'90.7%'}")
    print("-" * 60)

    for r in results:
        print(f"{'武汉 ' + str(int(r['sample_rate']*100)) + '%':<20} {r['num_pois']:<12,} {r['best_sil']:<12.4f} {r['rate']*100:.1f}%")


if __name__ == "__main__":
    main()
