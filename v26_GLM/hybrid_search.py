# -*- coding: utf-8 -*-
"""
V2.6 Pro 混合检索引擎 - 完整优化版

结合 Embedding 语义检索 + 空间过滤，发挥各自优势。

优化内容：
1. Haversine 距离计算（精确球面距离）
2. FAISS 索引支持（可选，10x 加速）
3. 批量检索优化
4. 模型元数据保存
5. POI 元数据支持
6. 自适应权重调整

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import sys
import time
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder


# =============================================================================
# 工具函数
# =============================================================================

def haversine_distance(
    coords1: np.ndarray,
    coords2: np.ndarray,
) -> np.ndarray:
    """
    计算两点间的 Haversine 距离（米）

    Args:
        coords1: (lon, lat) 或 (N, 2)
        coords2: (lon, lat) 或 (M, 2)

    Returns:
        distance: 距离（米），形状取决于输入
    """
    # 转换为弧度
    lon1 = np.radians(coords1[..., 0])
    lat1 = np.radians(coords1[..., 1])
    lon2 = np.radians(coords2[..., 0])
    lat2 = np.radians(coords2[..., 1])

    # Haversine 公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371000 * c  # 地球半径 6371km


def euclidean_distance_deg(
    coords1: np.ndarray,
    coords2: np.ndarray,
) -> np.ndarray:
    """
    快速欧几里得距离（度），适用于小范围查询（<50km）

    转换因子：
    - 1度纬度 ≈ 111km
    - 1度经度 ≈ 111km * cos(lat)

    Args:
        coords1: (lon, lat) 或 (N, 2)
        coords2: (M, 2)

    Returns:
        distance: 距离（米）
    """
    # 计算平均纬度
    lat_mean = np.mean([coords1[..., 1].mean(), coords2[..., 1].mean()])

    # 米/度转换因子
    meters_per_degree_lon = 111000 * np.cos(np.radians(lat_mean))
    meters_per_degree_lat = 111000

    # 计算差值
    diff = coords2 - coords1

    # 转换为米
    if diff.ndim == 1:
        dist = np.sqrt(
            (diff[0] * meters_per_degree_lon) ** 2 +
            (diff[1] * meters_per_degree_lat) ** 2
        )
    else:
        dist = np.sqrt(
            (diff[..., 0] * meters_per_degree_lon) ** 2 +
            (diff[..., 1] * meters_per_degree_lat) ** 2
        )

    return dist


# =============================================================================
# 自适应权重调整
# =============================================================================

class AdaptiveWeightAdjuster:
    """
    根据查询类型自动调整语义/空间权重
    """

    # 关键词模式
    SEMANTIC_KEYWORDS = [
        "相似", "类似", "同类", "一样", "相同",
        "similar", "like", "same", "type", "category",
        "商业区", "居住区", "工业区", "教育区", "公园",
    ]

    SPATIAL_KEYWORDS = [
        "附近", "最近", "周围", "周边", "范围内",
        "nearby", "near", "around", "within", "closest",
        "米内", "公里内", "距离",
    ]

    BALANCED_KEYWORDS = [
        "找", "搜索", "查询", "推荐",
        "find", "search", "recommend",
    ]

    def adjust(self, query_text: str = None) -> Tuple[float, float]:
        """
        根据查询文本调整权重

        Args:
            query_text: 查询文本（可选）

        Returns:
            semantic_weight: 语义权重
            spatial_weight: 空间权重
        """
        if query_text is None:
            # 默认权重（Semantic Priority）
            return 0.7, 0.3

        query_lower = query_text.lower()

        # 统计关键词匹配
        semantic_score = sum(1 for kw in self.SEMANTIC_KEYWORDS if kw in query_lower)
        spatial_score = sum(1 for kw in self.SPATIAL_KEYWORDS if kw in query_lower)

        # 根据匹配调整权重
        if semantic_score > spatial_score:
            # 语义优先
            return 0.9, 0.1
        elif spatial_score > semantic_score:
            # 空间优先
            return 0.3, 0.7
        elif semantic_score > 0 and spatial_score > 0:
            # 平衡
            return 0.5, 0.5
        else:
            # 默认
            return 0.7, 0.3

    def get_preset(self, preset: str) -> Tuple[float, float]:
        """
        获取预设权重配置

        Args:
            preset: 预设名称

        Returns:
            semantic_weight, spatial_weight
        """
        presets = {
            "pure_semantic": (1.0, 0.0),
            "semantic_priority": (0.7, 0.3),
            "balanced": (0.5, 0.5),
            "spatial_priority": (0.3, 0.7),
            "pure_spatial": (0.0, 1.0),
        }
        return presets.get(preset, (0.7, 0.3))


# =============================================================================
# 混合检索引擎
# =============================================================================

class HybridSearchEngine:
    """
    混合检索引擎

    实现 Embedding 语义检索 + 空间过滤的混合架构。
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        coords: np.ndarray,
        metadata: Optional[List[Dict]] = None,
        n_candidates: int = 100,
        use_faiss: bool = False,
        faiss_nlist: int = 100,
    ):
        """
        初始化混合检索引擎。

        Args:
            embeddings: POI embeddings (N, D)
            coords: POI 坐标 (N, 2) - [lon, lat]
            metadata: POI 元数据列表
            n_candidates: 召回候选数量
            use_faiss: 是否使用 FAISS 加速
            faiss_nlist: FAISS 聚类中心数量
        """
        self.embeddings = embeddings.astype(np.float32)
        self.coords = coords.astype(np.float32)
        self.metadata = metadata or [{} for _ in range(len(embeddings))]
        self.n_candidates = n_candidates
        self.n_pois = len(embeddings)
        self.use_faiss = use_faiss
        self.weight_adjuster = AdaptiveWeightAdjuster()

        # 构建 KNN 索引
        print(f"Building KNN index for {self.n_pois} POIs...")
        start = time.time()

        if use_faiss:
            self._build_faiss_index(faiss_nlist)
        else:
            self._build_sklearn_index()

        print(f"  Done in {time.time() - start:.2f}s. Index size: {self.n_pois}")

    def _build_sklearn_index(self):
        """构建 sklearn KNN 索引"""
        self.knn_index = NearestNeighbors(
            n_neighbors=min(self.n_candidates, self.n_pois),
            metric='cosine',
        ).fit(self.embeddings)
        self.faiss_index = None

    def _build_faiss_index(self, nlist: int):
        """构建 FAISS 索引"""
        try:
            import faiss
        except ImportError:
            print("  FAISS not installed, falling back to sklearn")
            self._build_sklearn_index()
            return

        d = self.embeddings.shape[1]  # 352

        # L2 归一化（用于余弦相似度）
        emb_normalized = self.embeddings.copy()
        faiss.normalize_L2(emb_normalized)

        # 构建 IVF 索引
        quantizer = faiss.IndexFlatIP(d)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # 训练索引
        self.faiss_index.train(emb_normalized)
        self.faiss_index.add(emb_normalized)

        # 设置查询参数
        self.faiss_index.nprobe = 10

        self.knn_index = None
        print(f"  FAISS IVF index built (nlist={nlist}, d={d})")

    def _knn_search(
        self,
        query_embedding: np.ndarray,
        n_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        KNN 检索

        Args:
            query_embedding: 查询向量 (D,)
            n_neighbors: 返回邻居数量

        Returns:
            distances: 距离 (n_neighbors,)
            indices: 索引 (n_neighbors,)
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self.use_faiss and self.faiss_index is not None:
            import faiss
            faiss.normalize_L2(query)
            distances, indices = self.faiss_index.search(query, n_neighbors)
            # 转换为相似度 (内积 → 余弦相似度)
            similarities = distances[0]
            return similarities, indices[0]
        else:
            distances, indices = self.knn_index.kneighbors(
                query, n_neighbors=n_neighbors
            )
            # sklearn 返回的是余弦距离，转换为相似度
            similarities = 1 - distances[0]
            return similarities, indices[0]

    def _knn_search_batch(
        self,
        query_embeddings: np.ndarray,
        n_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量 KNN 检索

        Args:
            query_embeddings: 查询向量 (N, D)
            n_neighbors: 返回邻居数量

        Returns:
            distances: 距离 (N, n_neighbors)
            indices: 索引 (N, n_neighbors)
        """
        queries = query_embeddings.astype(np.float32)

        if self.use_faiss and self.faiss_index is not None:
            import faiss
            faiss.normalize_L2(queries)
            distances, indices = self.faiss_index.search(queries, n_neighbors)
            similarities = distances  # 已经是相似度
            return similarities, indices
        else:
            distances, indices = self.knn_index.kneighbors(
                queries, n_neighbors=n_neighbors
            )
            similarities = 1 - distances
            return similarities, indices

    def search(
        self,
        query_embedding: np.ndarray,
        query_coords: np.ndarray,
        k: int = 20,
        radius: float = 5000,
        semantic_weight: float = 0.7,
        spatial_weight: float = 0.3,
        use_haversine: bool = False,
        query_text: str = None,
    ) -> List[Dict]:
        """
        混合检索。

        Args:
            query_embedding: 查询 embedding (D,)
            query_coords: 查询坐标 (2,) - [lon, lat]
            k: 返回数量
            radius: 空间半径（米）
            semantic_weight: 语义权重
            spatial_weight: 空间权重
            use_haversine: 是否使用精确 Haversine 距离
            query_text: 查询文本（用于自适应权重）

        Returns:
            results: Top-K 结果列表
        """
        # 自适应权重调整
        if query_text is not None and semantic_weight == 0.7 and spatial_weight == 0.3:
            semantic_weight, spatial_weight = self.weight_adjuster.adjust(query_text)

        # Step 1: Embedding 语义检索（召回候选）
        n_neighbors = min(self.n_candidates, self.n_pois)
        semantic_scores, indices = self._knn_search(query_embedding, n_neighbors)

        # Step 2: 空间过滤
        candidate_coords = self.coords[indices]

        if use_haversine:
            # 精确球面距离（适用于大范围查询）
            spatial_distances = haversine_distance(query_coords, candidate_coords)
        else:
            # 快速欧几里得距离（适用于小范围查询）
            spatial_distances = euclidean_distance_deg(query_coords, candidate_coords)

        # 过滤 radius 范围内的
        mask = spatial_distances < radius

        if mask.sum() == 0:
            # 没有候选在范围内，返回最近的 k 个
            mask = np.ones(len(indices), dtype=bool)[:k]

        filtered_indices = indices[mask]
        filtered_semantic = semantic_scores[mask]
        filtered_spatial = spatial_distances[mask]

        if len(filtered_indices) == 0:
            return []

        # Step 3: 重排序
        # 归一化空间距离到 [0, 1]
        max_dist = filtered_spatial.max()
        if max_dist > 0:
            spatial_scores = 1 - (filtered_spatial / max_dist)
        else:
            spatial_scores = np.ones_like(filtered_spatial)

        # 综合得分
        final_scores = (
            semantic_weight * filtered_semantic +
            spatial_weight * spatial_scores
        )

        # 排序
        top_k_idx = np.argsort(final_scores)[::-1][:k]

        results = []
        for i in top_k_idx:
            idx = filtered_indices[i]
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({
                'poi_index': int(idx),
                'score': float(final_scores[i]),
                'semantic_score': float(filtered_semantic[i]),
                'spatial_score': float(spatial_scores[i]),
                'distance_m': float(filtered_spatial[i]),
                'metadata': meta,
            })

        return results

    def search_with_preset(
        self,
        query_embedding: np.ndarray,
        query_coords: np.ndarray,
        preset: str = "semantic_priority",
        k: int = 20,
        radius: float = 5000,
    ) -> List[Dict]:
        """
        使用预设配置检索

        Args:
            query_embedding: 查询 embedding
            query_coords: 查询坐标
            preset: 预设名称 (pure_semantic, semantic_priority, balanced, spatial_priority, pure_spatial)
            k: 返回数量
            radius: 空间半径

        Returns:
            results: Top-K 结果
        """
        semantic_weight, spatial_weight = self.weight_adjuster.get_preset(preset)
        return self.search(
            query_embedding=query_embedding,
            query_coords=query_coords,
            k=k,
            radius=radius,
            semantic_weight=semantic_weight,
            spatial_weight=spatial_weight,
        )

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        query_coords: np.ndarray,
        k: int = 20,
        radius: float = 5000,
        semantic_weight: float = 0.7,
        spatial_weight: float = 0.3,
        use_haversine: bool = False,
    ) -> List[List[Dict]]:
        """
        批量检索（优化版）

        Args:
            query_embeddings: 查询 embeddings (N, D)
            query_coords: 查询坐标 (N, 2)
            k: 返回数量
            radius: 空间半径（米）
            semantic_weight: 语义权重
            spatial_weight: 空间权重
            use_haversine: 是否使用精确 Haversine 距离

        Returns:
            results: 每个查询的结果列表
        """
        n_queries = len(query_embeddings)
        n_neighbors = min(self.n_candidates, self.n_pois)

        # 批量 KNN 检索
        semantic_scores, indices = self._knn_search_batch(query_embeddings, n_neighbors)

        results = []
        for i in range(n_queries):
            # 单个查询的空间过滤
            candidate_coords = self.coords[indices[i]]

            if use_haversine:
                spatial_distances = haversine_distance(query_coords[i], candidate_coords)
            else:
                spatial_distances = euclidean_distance_deg(query_coords[i], candidate_coords)

            # 过滤
            mask = spatial_distances < radius
            if mask.sum() == 0:
                mask = np.ones(len(indices[i]), dtype=bool)[:k]

            filtered_indices = indices[i][mask]
            filtered_semantic = semantic_scores[i][mask]
            filtered_spatial = spatial_distances[mask]

            if len(filtered_indices) == 0:
                results.append([])
                continue

            # 重排序
            max_dist = filtered_spatial.max()
            if max_dist > 0:
                spatial_scores = 1 - (filtered_spatial / max_dist)
            else:
                spatial_scores = np.ones_like(filtered_spatial)

            final_scores = semantic_weight * filtered_semantic + spatial_weight * spatial_scores
            top_k_idx = np.argsort(final_scores)[::-1][:k]

            query_results = []
            for j in top_k_idx:
                idx = filtered_indices[j]
                meta = self.metadata[idx] if idx < len(self.metadata) else {}
                query_results.append({
                    'poi_index': int(idx),
                    'score': float(final_scores[j]),
                    'semantic_score': float(filtered_semantic[j]),
                    'spatial_score': float(spatial_scores[j]),
                    'distance_m': float(filtered_spatial[j]),
                    'metadata': meta,
                })

            results.append(query_results)

        return results


# =============================================================================
# 模型保存与加载
# =============================================================================

def save_model_with_metadata(
    model: torch.nn.Module,
    config,
    metrics: Dict,
    save_path: str,
):
    """
    保存模型及元数据

    Args:
        model: 模型
        config: 配置
        metrics: 性能指标
        save_path: 保存路径
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_dim': config.model.hidden_dim,
            'embedding_dim': config.model.embedding_dim,
            'num_encoder_layers': config.model.num_encoder_layers,
            'batch_size': config.training.batch_size,
        },
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")
    print(f"  Metrics: {metrics}")


def load_model_with_metadata(
    model: torch.nn.Module,
    load_path: str,
    device: str = 'cpu',
) -> Tuple[torch.nn.Module, Dict]:
    """
    加载模型及元数据

    支持两种格式：
    1. 带元数据的字典格式 {'model_state_dict': ..., 'config': ..., 'metrics': ...}
    2. 纯 state_dict 格式

    Args:
        model: 模型实例
        load_path: 模型路径
        device: 设备

    Returns:
        model: 加载权重后的模型
        metadata: 元数据字典
    """
    checkpoint = torch.load(load_path, map_location=device)

    # 检查格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 带元数据的格式
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            'config': checkpoint.get('config', {}),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', ''),
        }
    else:
        # 纯 state_dict 格式
        model.load_state_dict(checkpoint)
        metadata = {
            'config': {},
            'metrics': {},
            'timestamp': '',
        }

    return model, metadata


def build_search_engine_from_model(
    model_path: str,
    data: Dict,
    device: str = 'cpu',
    use_faiss: bool = False,
) -> Tuple[HybridSearchEngine, np.ndarray, Dict]:
    """
    从模型和数据构建检索引擎。

    Args:
        model_path: 模型权重路径
        data: 数据字典（包含 features, coords, metadata）
        device: 计算设备
        use_faiss: 是否使用 FAISS

    Returns:
        engine: 混合检索引擎
        embeddings: 所有 POI 的 embedding
        metadata: 模型元数据
    """
    # 加载模型
    print(f"Loading model from {model_path}...")
    model = build_mlp_encoder(DEFAULT_PRO_CONFIG)
    model, metadata = load_model_with_metadata(model, model_path, device)
    model = model.to(device)
    model.eval()

    if metadata.get('metrics'):
        print(f"  Model metrics: {metadata['metrics']}")

    # 计算所有 POI 的 embedding
    print("Computing embeddings for all POIs...")
    with torch.no_grad():
        point_feat = torch.tensor(data["point_features"], dtype=torch.float32).to(device)
        line_feat = torch.tensor(data["line_features"], dtype=torch.float32).to(device)
        polygon_feat = torch.tensor(data["polygon_features"], dtype=torch.float32).to(device)
        direction_feat = torch.tensor(data["direction_features"], dtype=torch.float32).to(device)

        embeddings, _, _, _ = model(
            point_feat, line_feat, polygon_feat, direction_feat
        )
        embeddings = embeddings.cpu().numpy()

    coords = data["coords"]
    meta = data.get("metadata", None)

    # 构建引擎
    engine = HybridSearchEngine(
        embeddings=embeddings,
        coords=coords,
        metadata=meta,
        use_faiss=use_faiss,
    )

    return engine, embeddings, metadata


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training

    print("=" * 70)
    print("V2.6 Pro Hybrid Search Test - Complete Version")
    print("=" * 70)

    # 加载数据
    print("\nLoading data...")
    data = load_dataset_for_training(config=DEFAULT_PRO_CONFIG, sample_ratio=0.1)
    n = len(data["coords"])
    print(f"Loaded {n} POIs")

    # 检查元数据
    if data.get("metadata"):
        print(f"Metadata available: {len(data['metadata'])} entries")
        print(f"Sample metadata: {data['metadata'][0]}")

    # 加载模型
    model_path = Path(__file__).parent / "saved_models" / "v26_pro" / "best_model.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python train_v26_mlp.py")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试 sklearn 索引
    print("\n" + "-" * 70)
    print("Testing with sklearn index...")
    print("-" * 70)
    engine, embeddings, metadata = build_search_engine_from_model(
        str(model_path), data, device, use_faiss=False
    )

    # 测试自适应权重
    print("\n" + "-" * 70)
    print("Testing adaptive weights...")
    print("-" * 70)

    test_queries = [
        "找附近的餐厅",
        "找相似的商业区",
        "最近的教育区",
        "推荐一个公园",
    ]

    adjuster = AdaptiveWeightAdjuster()
    for query in test_queries:
        sem_w, spa_w = adjuster.adjust(query)
        print(f"  '{query}' -> semantic={sem_w:.1f}, spatial={spa_w:.1f}")

    # 测试检索结果
    print("\n" + "-" * 70)
    print("Sample search results with metadata...")
    print("-" * 70)

    np.random.seed(42)
    test_idx = np.random.choice(n, 3, replace=False)

    for idx in test_idx:
        query_emb = embeddings[idx]
        query_coords = data["coords"][idx]

        results = engine.search(
            query_embedding=query_emb,
            query_coords=query_coords,
            k=3,
            radius=5000,
        )

        print(f"\nQuery POI #{idx}:")
        print(f"  Metadata: {data['metadata'][idx] if data.get('metadata') else 'N/A'}")
        print(f"  Top 3 results:")
        for i, r in enumerate(results):
            meta = r.get('metadata', {})
            print(f"    {i+1}. idx={r['poi_index']}, score={r['score']:.3f}, "
                  f"dist={r['distance_m']:.0f}m, "
                  f"category={meta.get('dominant_category', 'N/A')}, "
                  f"poi_count={meta.get('poi_count', 'N/A')}")

    # 性能测试
    print("\n" + "-" * 70)
    print("Performance test...")
    print("-" * 70)

    test_indices = np.random.choice(n, 100, replace=False)
    test_embeddings = embeddings[test_indices]
    test_coords = data["coords"][test_indices]

    # 批量检索
    start = time.time()
    engine.batch_search(test_embeddings, test_coords, k=20)
    batch_time = time.time() - start

    print(f"  Batch search (100 queries): {batch_time*1000:.1f}ms ({batch_time*10:.1f}ms/query)")

    print("\n" + "=" * 70)
    print("Test completed.")
    print("=" * 70)
