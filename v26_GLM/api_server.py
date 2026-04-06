# -*- coding: utf-8 -*-
"""
V2.6 Pro 空间编码器 FastAPI 服务 - 完整版

提供混合检索 API，支持：
- 语义检索 + 空间过滤
- Haversine 精确距离
- FAISS 加速（可选）
- POI 元数据返回
- 自适应权重调整

启动命令：
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder
from spatial_encoder.v26_GLM.hybrid_search import (
    HybridSearchEngine,
    load_model_with_metadata,
    haversine_distance,
    euclidean_distance_deg,
    AdaptiveWeightAdjuster,
)


# =============================================================================
# 全局状态
# =============================================================================

class AppState:
    """应用状态"""
    model: Optional[torch.nn.Module] = None
    engine: Optional[HybridSearchEngine] = None
    embeddings: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    device: str = 'cpu'
    data: Optional[Dict] = None
    weight_adjuster: Optional[AdaptiveWeightAdjuster] = None


state = AppState()


# =============================================================================
# Pydantic 模型
# =============================================================================

class SearchRequest(BaseModel):
    """检索请求"""
    query_coords: List[float] = Field(..., description="查询坐标 [lon, lat]")
    k: int = Field(20, ge=1, le=100, description="返回数量")
    radius: float = Field(5000, ge=100, le=50000, description="空间半径（米）")
    semantic_weight: float = Field(0.7, ge=0, le=1, description="语义权重（默认0.7，推荐值）")
    spatial_weight: float = Field(0.3, ge=0, le=1, description="空间权重（默认0.3，推荐值）")
    use_haversine: bool = Field(False, description="是否使用精确 Haversine 距离")
    query_text: Optional[str] = Field(None, description="查询文本（用于自适应权重）")


class SearchResult(BaseModel):
    """单个检索结果"""
    poi_index: int
    score: float
    semantic_score: float
    spatial_score: float
    distance_m: float
    coords: List[float]
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """检索响应"""
    results: List[SearchResult]
    query_time_ms: float
    total_candidates: int
    weights_used: Dict[str, float]


class BatchSearchRequest(BaseModel):
    """批量检索请求"""
    queries: list[list[float]] = Field(..., description="查询坐标列表 [[lon, lat], ...]")
    k: int = Field(20, ge=1, le=100)
    radius: float = Field(5000, ge=100, le=50000)
    semantic_weight: float = Field(0.7)
    spatial_weight: float = Field(0.3)


class BatchSearchResponse(BaseModel):
    """批量检索响应"""
    results: List[List[SearchResult]]
    query_time_ms: float


class AdaptiveWeightRequest(BaseModel):
    """自适应权重请求"""
    query_text: str = Field(..., description="查询文本")


class AdaptiveWeightResponse(BaseModel):
    """自适应权重响应"""
    query_text: str
    semantic_weight: float
    spatial_weight: float
    preset: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    engine_ready: bool
    num_pois: int
    embedding_dim: int
    use_faiss: bool
    metadata_available: bool


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    print("=" * 70)
    print("V2.6 Pro Spatial Encoder API Server")
    print("=" * 70)

    # 配置
    model_path = Path(__file__).parent / "saved_models" / "v26_pro" / "best_model.pt"
    sample_ratio = 0.1  # 默认加载 10% 数据
    use_faiss = False  # 默认不使用 FAISS

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("API will run in limited mode")
        yield
        return

    # 设置设备
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {state.device}")

    # 加载模型
    print(f"\nLoading model from {model_path}...")
    state.model = build_mlp_encoder(DEFAULT_PRO_CONFIG)
    state.model, state.metadata = load_model_with_metadata(
        state.model, str(model_path), state.device
    )
    state.model = state.model.to(state.device)
    state.model.eval()

    if state.metadata.get('metrics'):
        print(f"  Model metrics: {state.metadata['metrics']}")

    # 加载数据
    print(f"\nLoading data (sample_ratio={sample_ratio})...")
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
    state.data = load_dataset_for_training(
        config=DEFAULT_PRO_CONFIG,
        sample_ratio=sample_ratio,
    )
    n_pois = len(state.data["coords"])
    print(f"  Loaded {n_pois} POIs")

    # 检查元数据
    if state.data.get("metadata"):
        print(f"  Metadata available: {len(state.data['metadata'])} entries")

    # 计算 embeddings
    print("\nComputing embeddings...")
    with torch.no_grad():
        point_feat = torch.tensor(
            state.data["point_features"], dtype=torch.float32
        ).to(state.device)
        line_feat = torch.tensor(
            state.data["line_features"], dtype=torch.float32
        ).to(state.device)
        polygon_feat = torch.tensor(
            state.data["polygon_features"], dtype=torch.float32
        ).to(state.device)
        direction_feat = torch.tensor(
            state.data["direction_features"], dtype=torch.float32
        ).to(state.device)

        state.embeddings, _, _, _ = state.model(
            point_feat, line_feat, polygon_feat, direction_feat
        )
        state.embeddings = state.embeddings.cpu().numpy()

    print(f"  Embeddings shape: {state.embeddings.shape}")

    # 构建检索引擎
    print("\nBuilding search engine...")
    state.engine = HybridSearchEngine(
        embeddings=state.embeddings,
        coords=state.data["coords"],
        metadata=state.data.get("metadata"),
        use_faiss=use_faiss,
    )

    # 初始化权重调整器
    state.weight_adjuster = AdaptiveWeightAdjuster()

    print("\n" + "=" * 70)
    print("Server ready!")
    print(f"  POIs: {n_pois}")
    print(f"  FAISS: {'enabled' if use_faiss else 'disabled'}")
    print(f"  Metadata: {'available' if state.data.get('metadata') else 'not available'}")
    print("=" * 70)

    yield

    # 关闭时清理
    print("\nShutting down...")


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title="V2.6 Pro Spatial Encoder API",
    description="混合检索 API：Embedding 语义检索 + 空间过滤 + 元数据 + 自适应权重",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API 端点
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="ok" if state.engine is not None else "loading",
        model_loaded=state.model is not None,
        engine_ready=state.engine is not None,
        num_pois=len(state.data["coords"]) if state.data else 0,
        embedding_dim=state.embeddings.shape[1] if state.embeddings is not None else 0,
        use_faiss=state.engine.use_faiss if state.engine else False,
        metadata_available=bool(state.data.get("metadata")) if state.data else False,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """获取模型信息"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        config=state.metadata.get('config', {}),
        metrics=state.metadata.get('metrics', {}),
        timestamp=state.metadata.get('timestamp', ''),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    混合检索

    结合 Embedding 语义检索和空间过滤，返回 Top-K 结果。
    支持自适应权重调整。
    """
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    start = time.time()

    # 解析坐标
    query_coords = np.array(request.query_coords, dtype=np.float32)

    # 找最近的 POI 作为查询 embedding
    distances = euclidean_distance_deg(query_coords, state.data["coords"])
    nearest_idx = np.argmin(distances)
    query_embedding = state.embeddings[nearest_idx]

    # 执行检索
    results = state.engine.search(
        query_embedding=query_embedding,
        query_coords=query_coords,
        k=request.k,
        radius=request.radius,
        semantic_weight=request.semantic_weight,
        spatial_weight=request.spatial_weight,
        use_haversine=request.use_haversine,
        query_text=request.query_text,
    )

    query_time = (time.time() - start) * 1000

    # 构建响应
    search_results = []
    for r in results:
        idx = r['poi_index']
        search_results.append(SearchResult(
            poi_index=idx,
            score=r['score'],
            semantic_score=r['semantic_score'],
            spatial_score=r['spatial_score'],
            distance_m=r['distance_m'],
            coords=state.data["coords"][idx].tolist(),
            metadata=r.get('metadata', {}),
        ))

    return SearchResponse(
        results=search_results,
        query_time_ms=query_time,
        total_candidates=len(results),
        weights_used={
            "semantic_weight": request.semantic_weight,
            "spatial_weight": request.spatial_weight,
        },
    )


@app.post("/search/by-index", response_model=SearchResponse)
async def search_by_index(
    poi_index: int = Query(..., description="POI 索引"),
    k: int = Query(20, ge=1, le=100),
    radius: float = Query(5000, ge=100, le=50000),
    semantic_weight: float = Query(0.7),
    spatial_weight: float = Query(0.3),
    preset: Optional[str] = Query(None, description="预设权重配置"),
):
    """
    根据 POI 索引检索相似 POI

    可选预设：pure_semantic, semantic_priority, balanced, spatial_priority, pure_spatial
    """
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    if poi_index < 0 or poi_index >= len(state.embeddings):
        raise HTTPException(status_code=400, detail="Invalid POI index")

    start = time.time()

    query_embedding = state.embeddings[poi_index]
    query_coords = state.data["coords"][poi_index]

    if preset:
        results = state.engine.search_with_preset(
            query_embedding=query_embedding,
            query_coords=query_coords,
            preset=preset,
            k=k,
            radius=radius,
        )
    else:
        results = state.engine.search(
            query_embedding=query_embedding,
            query_coords=query_coords,
            k=k,
            radius=radius,
            semantic_weight=semantic_weight,
            spatial_weight=spatial_weight,
        )

    query_time = (time.time() - start) * 1000

    search_results = []
    for r in results:
        idx = r['poi_index']
        search_results.append(SearchResult(
            poi_index=idx,
            score=r['score'],
            semantic_score=r['semantic_score'],
            spatial_score=r['spatial_score'],
            distance_m=r['distance_m'],
            coords=state.data["coords"][idx].tolist(),
            metadata=r.get('metadata', {}),
        ))

    return SearchResponse(
        results=search_results,
        query_time_ms=query_time,
        total_candidates=len(results),
        weights_used={
            "semantic_weight": semantic_weight,
            "spatial_weight": spatial_weight,
        },
    )


@app.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest):
    """
    批量检索

    同时处理多个查询请求，性能更优。
    """
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    start = time.time()

    # 解析查询坐标
    query_coords = np.array(request.queries, dtype=np.float32)

    # 找最近的 POI 作为查询 embedding
    query_embeddings = []
    for coords in query_coords:
        distances = euclidean_distance_deg(coords, state.data["coords"])
        nearest_idx = np.argmin(distances)
        query_embeddings.append(state.embeddings[nearest_idx])
    query_embeddings = np.array(query_embeddings, dtype=np.float32)

    # 批量检索
    results = state.engine.batch_search(
        query_embeddings=query_embeddings,
        query_coords=query_coords,
        k=request.k,
        radius=request.radius,
        semantic_weight=request.semantic_weight,
        spatial_weight=request.spatial_weight,
    )

    query_time = (time.time() - start) * 1000

    # 构建响应
    batch_results = []
    for query_results in results:
        search_results = []
        for r in query_results:
            idx = r['poi_index']
            search_results.append(SearchResult(
                poi_index=idx,
                score=r['score'],
                semantic_score=r['semantic_score'],
                spatial_score=r['spatial_score'],
                distance_m=r['distance_m'],
                coords=state.data["coords"][idx].tolist(),
                metadata=r.get('metadata', {}),
            ))
        batch_results.append(search_results)

    return BatchSearchResponse(
        results=batch_results,
        query_time_ms=query_time,
    )


@app.post("/weights/adaptive", response_model=AdaptiveWeightResponse)
async def get_adaptive_weights(request: AdaptiveWeightRequest):
    """
    根据查询文本获取自适应权重

    返回推荐的语义/空间权重配置。
    """
    if state.weight_adjuster is None:
        raise HTTPException(status_code=503, detail="Weight adjuster not initialized")

    semantic_weight, spatial_weight = state.weight_adjuster.adjust(request.query_text)

    # 确定预设名称
    preset = None
    if semantic_weight == 1.0:
        preset = "pure_semantic"
    elif semantic_weight == 0.9:
        preset = "semantic_priority"
    elif semantic_weight == 0.5:
        preset = "balanced"
    elif semantic_weight == 0.3:
        preset = "spatial_priority"
    elif semantic_weight == 0.0:
        preset = "pure_spatial"

    return AdaptiveWeightResponse(
        query_text=request.query_text,
        semantic_weight=semantic_weight,
        spatial_weight=spatial_weight,
        preset=preset,
    )


@app.get("/weights/presets")
async def get_weight_presets():
    """获取所有预设权重配置"""
    return {
        "presets": {
            "pure_semantic": {"semantic": 1.0, "spatial": 0.0, "description": "纯语义检索"},
            "semantic_priority": {"semantic": 0.7, "spatial": 0.3, "description": "语义优先（推荐）"},
            "balanced": {"semantic": 0.5, "spatial": 0.5, "description": "平衡"},
            "spatial_priority": {"semantic": 0.3, "spatial": 0.7, "description": "空间优先"},
            "pure_spatial": {"semantic": 0.0, "spatial": 1.0, "description": "纯空间检索"},
        },
        "default": "semantic_priority",
    }


@app.get("/poi/{poi_index}")
async def get_poi(poi_index: int):
    """获取 POI 详情"""
    if state.data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if poi_index < 0 or poi_index >= len(state.data["coords"]):
        raise HTTPException(status_code=400, detail="Invalid POI index")

    # metadata 是列表，直接索引访问
    meta = state.data.get("metadata", [])
    poi_meta = meta[poi_index] if poi_index < len(meta) else {}

    return {
        "index": poi_index,
        "coords": state.data["coords"][poi_index].tolist(),
        "region_label": int(state.data["region_labels"][poi_index]),
        "metadata": poi_meta,
    }


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
