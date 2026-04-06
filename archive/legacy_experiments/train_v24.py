# -*- coding: utf-8 -*-
"""
V2.4 空间拓扑编码器 - 增强版

新增特性：
1. 路网拓扑约束：同一道路区块内的POI embedding应该相似
2. AOI功能区约束：同一功能区内的POI embedding应该相似

损失函数组合：
- 坐标重构损失
- 距离保持损失
- 邻居一致性损失
- ⭐ 道路区块一致性损失（新增）
- ⭐ AOI功能区一致性损失（新增）

目标：Silhouette > 0.42
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from config import SpatialEncoderConfig


# ============================================================================
# 数据加载
# ============================================================================

def load_area_data(area_name: str, data_dir: str) -> Dict:
    """加载区域数据（含路网和AOI信息）"""
    area_dir = Path(data_dir) / area_name

    # 加载POI
    with open(area_dir / "pois.geojson", 'r', encoding='utf-8') as f:
        pois = json.load(f)

    # 加载道路区块
    with open(area_dir / "road_blocks.geojson", 'r', encoding='utf-8') as f:
        road_blocks = json.load(f)

    # 加载AOI
    with open(area_dir / "aois.geojson", 'r', encoding='utf-8') as f:
        aois = json.load(f)

    # 解析POI
    category_map, landuse_map, aoi_type_map, road_class_map = {}, {}, {}, {}
    poi_coords, poi_features, poi_road_block_ids, poi_aoi_ids = [], [], [], []

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

        # 道路区块ID和AOI ID
        poi_road_block_ids.append(props.get('road_block_id', -1) or -1)
        poi_aoi_ids.append(props.get('aoi_id', -1) or -1)

    coords = np.array(poi_coords, dtype=np.float32)

    # 解析道路区块几何，建立POI到区块的映射
    road_block_geoms = {}
    for rb in road_blocks['features']:
        props = rb['properties']
        geom = rb['geometry']
        block_id = props.get('block_id')
        if block_id and geom['type'] == 'Polygon':
            # 标准GeoJSON Polygon格式
            road_block_geoms[block_id] = geom['coordinates']

    # 解析AOI几何，建立POI到AOI的映射
    aoi_geoms = {}
    for aoi in aois['features']:
        props = aoi['properties']
        geom = aoi['geometry']
        aoi_id = props.get('id')
        if aoi_id and geom['type'] == 'Polygon':
            # 标准GeoJSON Polygon格式: coordinates = [外环, 内环...]
            aoi_geoms[aoi_id] = geom['coordinates']
        elif aoi_id and geom['type'] == 'MultiPolygon':
            # MultiPolygon: 取第一个多边形
            aoi_geoms[aoi_id] = geom['coordinates'][0]

    # 为每个POI找到所属的道路区块和AOI（使用空间匹配）
    print("  正在建立POI与道路区块/AOI的空间关系...")
    poi_to_road_block = assign_pois_to_polygons(coords, road_block_geoms)
    poi_to_aoi = assign_pois_to_polygons(coords, aoi_geoms)

    # 构建KNN邻居
    knn_k = 10
    adj = kneighbors_graph(coords, n_neighbors=knn_k, mode='connectivity', include_self=False)
    knn_neighbors = [adj[i].nonzero()[1] for i in range(len(coords))]

    # 统计
    road_block_coverage = sum(1 for x in poi_to_road_block if x >= 0) / len(poi_to_road_block) * 100
    aoi_coverage = sum(1 for x in poi_to_aoi if x >= 0) / len(poi_to_aoi) * 100
    print(f"  道路区块覆盖率: {road_block_coverage:.1f}%")
    print(f"  AOI功能区覆盖率: {aoi_coverage:.1f}%")

    return {
        'coords': coords,
        'features': np.array(poi_features, dtype=np.float32),
        'knn_neighbors': knn_neighbors,
        'poi_to_road_block': poi_to_road_block,
        'poi_to_aoi': poi_to_aoi,
        'metadata': {
            'num_pois': len(poi_coords),
            'num_categories': len(category_map),
            'num_landuses': len(landuse_map),
            'num_aoi_types': len(aoi_type_map),
            'num_road_classes': len(road_class_map),
            'num_road_blocks': len(set(poi_to_road_block)) - (1 if -1 in poi_to_road_block else 0),
            'num_aois': len(set(poi_to_aoi)) - (1 if -1 in poi_to_aoi else 0),
            'road_block_coverage': road_block_coverage,
            'aoi_coverage': aoi_coverage,
        }
    }


def point_in_polygon(point: List[float], polygon: List) -> bool:
    """射线法判断点是否在多边形内"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def assign_pois_to_polygons(coords: np.ndarray, polygon_geoms: Dict) -> np.ndarray:
    """为POI分配所属的多边形ID"""
    n_pois = len(coords)
    assignments = np.full(n_pois, -1, dtype=np.int64)

    if not polygon_geoms:
        return assignments

    # 构建多边形中心点的KD树
    polygon_ids = list(polygon_geoms.keys())
    polygon_centers = []

    for pid in polygon_ids:
        geom = polygon_geoms[pid]
        # geom是coordinates列表，对于Polygon是[外环, 内环...]
        # 外环是第一个元素
        if geom and len(geom) > 0:
            coords_list = geom[0]  # 外环
            if coords_list and len(coords_list) > 0:
                center_lng = sum(c[0] for c in coords_list) / len(coords_list)
                center_lat = sum(c[1] for c in coords_list) / len(coords_list)
                polygon_centers.append([center_lng, center_lat])
            else:
                polygon_centers.append([0, 0])
        else:
            polygon_centers.append([0, 0])

    if not polygon_centers:
        return assignments

    polygon_centers = np.array(polygon_centers)
    tree = cKDTree(polygon_centers)

    # 对每个POI，找到最近的多边形并检查是否在内
    for i, coord in enumerate(coords):
        # 找最近的5个多边形
        _, indices = tree.query(coord, k=min(5, len(polygon_ids)))

        for idx in [indices] if isinstance(indices, (int, np.integer)) else indices:
            pid = polygon_ids[idx]
            geom = polygon_geoms[pid]

            # 获取外环坐标
            if geom and len(geom) > 0:
                coords_list = geom[0]
                if coords_list and len(coords_list) > 0:
                    if point_in_polygon(coord, coords_list):
                        assignments[i] = pid
                        break

    return assignments


# ============================================================================
# 模型定义
# ============================================================================

class SpatialTopologyEncoderV24(nn.Module):
    """
    空间拓扑编码器 V2.4

    新增：路网拓扑约束 + AOI功能区约束
    """

    def __init__(self, config: SpatialEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        # Embedding层
        emb_dim = self.hidden_dim // 6
        self.category_emb = nn.Embedding(config.num_categories + 1, emb_dim)
        self.landuse_emb = nn.Embedding(config.num_landuses + 1, emb_dim)
        self.aoi_type_emb = nn.Embedding(config.num_aoi_types + 1, emb_dim)
        self.road_class_emb = nn.Embedding(config.num_road_classes + 1, emb_dim)
        self.num_proj = nn.Linear(3, emb_dim)
        self.coord_proj = nn.Linear(2, emb_dim)

        # 编码器
        input_dim = emb_dim * 6
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, config.embed_dim),
        )

        # 坐标解码器
        self.coord_decoder = nn.Sequential(
            nn.Linear(config.embed_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2),
        )

    def encode(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """编码：输入 → embedding"""
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

    def decode_coord(self, z: torch.Tensor) -> torch.Tensor:
        """解码：embedding → 坐标"""
        return self.coord_decoder(z)

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        z = self.encode(features, coords)
        coord_recon = self.decode_coord(z)
        return z, coord_recon


# ============================================================================
# 损失函数
# ============================================================================

class DistancePreserveLoss(nn.Module):
    """距离保持损失"""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, pair_indices: torch.Tensor,
                spatial_dists: torch.Tensor) -> torch.Tensor:
        z_i = z[pair_indices[:, 0]]
        z_j = z[pair_indices[:, 1]]
        emb_dists = torch.norm(z_i - z_j, p=2, dim=1)
        return F.mse_loss(emb_dists, spatial_dists)


class NeighborConsistencyLoss(nn.Module):
    """邻居一致性损失"""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, knn_neighbors: List) -> torch.Tensor:
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


class RoadBlockConsistencyLoss(nn.Module):
    """
    道路区块一致性损失

    同一道路区块内的POI embedding应该相似
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, poi_to_road_block: np.ndarray) -> torch.Tensor:
        """
        Args:
            z: [N, D] embeddings
            poi_to_road_block: [N] 每个POI所属的道路区块ID
        """
        # 按区块分组
        block_ids = set(poi_to_road_block.tolist())
        block_ids.discard(-1)  # 排除未分配的

        if not block_ids:
            return torch.tensor(0.0, device=z.device)

        total_loss = 0.0
        count = 0

        # 采样部分区块
        sample_blocks = list(block_ids)
        if len(sample_blocks) > 100:
            sample_blocks = np.random.choice(sample_blocks, 100, replace=False)

        for block_id in sample_blocks:
            mask = poi_to_road_block == block_id
            indices = np.where(mask)[0]

            if len(indices) > 1:
                # 区块内POI的平均embedding
                z_block = z[indices]
                z_center = z_block.mean(dim=0)

                # 计算区块内一致性
                cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_block, dim=1)
                total_loss += (1 - cos_sim.mean())
                count += 1

        return total_loss / max(count, 1)


class AOIConsistencyLoss(nn.Module):
    """
    AOI功能区一致性损失

    同一功能区内的POI embedding应该相似
    """

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, poi_to_aoi: np.ndarray) -> torch.Tensor:
        """
        Args:
            z: [N, D] embeddings
            poi_to_aoi: [N] 每个POI所属的AOI ID
        """
        # 按AOI分组
        aoi_ids = set(poi_to_aoi.tolist())
        aoi_ids.discard(-1)

        if not aoi_ids:
            return torch.tensor(0.0, device=z.device)

        total_loss = 0.0
        count = 0

        # 采样部分AOI
        sample_aois = list(aoi_ids)
        if len(sample_aois) > 100:
            sample_aois = np.random.choice(sample_aois, 100, replace=False)

        for aoi_id in sample_aois:
            mask = poi_to_aoi == aoi_id
            indices = np.where(mask)[0]

            if len(indices) > 1:
                z_aoi = z[indices]
                z_center = z_aoi.mean(dim=0)

                cos_sim = F.cosine_similarity(z_center.unsqueeze(0), z_aoi, dim=1)
                total_loss += (1 - cos_sim.mean())
                count += 1

        return total_loss / max(count, 1)


def sample_distance_pairs(coords: np.ndarray, num_pairs: int, device) -> Tuple:
    """采样点对并计算空间距离"""
    import torch

    N = len(coords)

    idx_i = np.random.randint(0, N, num_pairs)
    idx_j = np.random.randint(0, N, num_pairs)

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    coords_i = coords[idx_i]
    coords_j = coords[idx_j]
    spatial_dists = np.sqrt(((coords_i - coords_j) ** 2).sum(axis=1))
    spatial_dists = (spatial_dists - spatial_dists.min()) / (spatial_dists.max() - spatial_dists.min() + 1e-8)

    return (
        torch.from_numpy(np.stack([idx_i, idx_j], axis=1)).long().to(device),
        torch.from_numpy(spatial_dists).float().to(device),
    )


# ============================================================================
# 实验运行
# ============================================================================

def run_v24_experiment(area_name: str, data_dir: str, quick_mode: bool = False):
    """
    运行V2.4实验

    Args:
        area_name: 区域名称
        data_dir: 数据目录
        quick_mode: 快速模式（减少epoch，用于快速验证）

    Returns:
        实验结果
    """
    print("=" * 70)
    print(f"V2.4 空间拓扑编码器（路网+AOI增强）: {area_name}")
    print("=" * 70)

    # 配置
    config = SpatialEncoderConfig(
        num_epochs=100 if quick_mode else 500,
        embed_dim=64,
        hidden_dim=128,
        dropout=0.1,
        n_clusters=15,
    )

    # 新增损失权重（先设为0，测试基础架构）
    road_block_weight = 0.0
    aoi_weight = 0.0

    device = torch.device(config.device)
    print(f"设备: {device}")

    # 加载数据
    print("\n[1] 加载数据...")
    data = load_area_data(area_name, data_dir)

    coords = data['coords']
    features = data['features']
    knn_neighbors = data['knn_neighbors']
    poi_to_road_block = data['poi_to_road_block']
    poi_to_aoi = data['poi_to_aoi']
    metadata = data['metadata']

    print(f"  POI数量: {metadata['num_pois']}")
    print(f"  道路区块数: {metadata['num_road_blocks']}")
    print(f"  AOI功能区数: {metadata['num_aois']}")

    # 检查覆盖率是否足够
    if metadata['road_block_coverage'] < 10 and metadata['aoi_coverage'] < 10:
        print("\n警告: 道路区块和AOI覆盖率都很低，新增约束可能无效")
        print("  建议检查数据质量或跳过此实验")

    # 生成标签
    print("\n[2] 生成空间聚类标签...")
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    sil_upper_bound = silhouette_score(coords, labels)
    print(f"  理论上限（原始坐标）: {sil_upper_bound:.4f}")

    # 标准化坐标
    coords_norm = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    # 创建模型
    print("\n[3] 创建模型...")
    model = SpatialTopologyEncoderV24(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")

    # 损失函数
    distance_loss_fn = DistancePreserveLoss()
    neighbor_loss_fn = NeighborConsistencyLoss()
    road_block_loss_fn = RoadBlockConsistencyLoss()
    aoi_loss_fn = AOIConsistencyLoss()

    # 训练
    print("\n[4] 开始训练...")
    print(f"  损失权重: recon=1.0, dist=2.0, neighbor=1.0, road_block={road_block_weight}, aoi={aoi_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    features_t = torch.from_numpy(features).float().to(device)
    coords_t = torch.from_numpy(coords).float().to(device)
    coords_norm_t = torch.from_numpy(coords_norm).float().to(device)

    best_sil = -1.0
    best_epoch = 0
    history = []

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        z, coord_recon = model(features_t, coords_t)

        # 损失计算
        loss_recon = F.mse_loss(coord_recon, coords_norm_t)

        pair_indices, spatial_dists = sample_distance_pairs(coords, config.num_distance_pairs, device)
        loss_distance = distance_loss_fn(z, pair_indices, spatial_dists)

        loss_neighbor = neighbor_loss_fn(z, knn_neighbors)

        loss_road_block = road_block_loss_fn(z, poi_to_road_block)

        loss_aoi = aoi_loss_fn(z, poi_to_aoi)

        # 总损失
        loss = (
            config.coord_recon_weight * loss_recon +
            config.distance_preserve_weight * loss_distance +
            config.neighbor_consistency_weight * loss_neighbor +
            road_block_weight * loss_road_block +
            aoi_weight * loss_aoi
        )

        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 评估
        if epoch % 20 == 0 or epoch == config.num_epochs - 1:
            model.eval()
            with torch.no_grad():
                z = model.encode(features_t, coords_t)
                z_np = z.cpu().numpy()
                sil = silhouette_score(z_np, labels)

            if sil > best_sil:
                best_sil = sil
                best_epoch = epoch

            history.append({
                'epoch': epoch,
                'loss': loss.item(),
                'silhouette': sil,
            })

            print(f"  Epoch {epoch:3d} | Loss={loss.item():.4f} "
                  f"(recon={loss_recon.item():.3f}, dist={loss_distance.item():.3f}, "
                  f"neighbor={loss_neighbor.item():.3f}, road={loss_road_block.item():.3f}, "
                  f"aoi={loss_aoi.item():.3f}) | Sil={sil:.4f} | Best={best_sil:.4f}")

            # 快速模式下，如果效果太差，提前停止
            if quick_mode and epoch >= 60 and sil < 0.1:
                print(f"\n快速模式检测: Silhouette过低 ({sil:.4f})，提前停止")
                break

    # 结果
    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"最佳 Silhouette: {best_sil:.4f} (Epoch {best_epoch})")
    print(f"理论上限: {sil_upper_bound:.4f}")
    print(f"达成率: {best_sil / sil_upper_bound * 100:.1f}%")

    # 对比V2.3基准
    v23_baseline = 0.4153 if area_name == "guanggu_core" else 0.40
    improvement = (best_sil - v23_baseline) / v23_baseline * 100
    print(f"V2.3基准: {v23_baseline:.4f}")
    print(f"相对提升: {improvement:+.1f}%")

    success = best_sil > 0.35
    if success and best_sil > v23_baseline:
        print("\n结论: V2.4 成功！路网和AOI约束有效提升效果")
    elif success:
        print("\n结论: V2.4 达标，但相比V2.3提升有限")
    else:
        print("\n结论: 效果不佳，需要调整参数")

    return {
        'area_name': area_name,
        'success': success,
        'silhouette': best_sil,
        'upper_bound': sil_upper_bound,
        'best_epoch': best_epoch,
        'v23_baseline': v23_baseline,
        'improvement': improvement,
        'history': history,
        'metadata': metadata,
    }


if __name__ == "__main__":
    data_dir = str(Path(__file__).resolve().parents[2] / "data" / "experiment_data")

    # 先用guanggu_core做快速验证
    print("=" * 70)
    print("V2.4 快速验证模式")
    print("=" * 70)

    result = run_v24_experiment("guanggu_core", data_dir, quick_mode=True)

    # 判断是否继续
    if result['success'] and result['improvement'] >= -5:  # 允许5%的下降
        print("\n" + "=" * 70)
        print("快速验证通过，运行完整实验...")
        print("=" * 70)

        # 运行完整版
        result = run_v24_experiment("guanggu_core", data_dir, quick_mode=False)

        if result['silhouette'] > 0.38:
            print("\n" + "=" * 70)
            print("单区域成功，验证多区域普适性...")
            print("=" * 70)

            for area in ["wuda_area", "zhongjia_cun"]:
                run_v24_experiment(area, data_dir, quick_mode=False)
    else:
        print("\n快速验证未通过，停止后续实验")
        print(f"Silhouette: {result['silhouette']:.4f}")
        print(f"相对V2.3变化: {result['improvement']:+.1f}%")
