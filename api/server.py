# -*- coding: utf-8 -*-
"""
V2.3 空间编码器 API服务

提供REST API接口：
- POST /spatial_search: 空间相似性查询
- POST /encode: 编码新POI
- GET /stats: 获取服务状态
- POST /batch_search: 批量查询
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.vector_index import VectorIndexService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ Pydantic Models ============

class SpatialSearchRequest(BaseModel):
    """空间搜索请求"""
    poi_id: Optional[int] = Field(None, description="POI ID")
    lng: Optional[float] = Field(None, description="经度")
    lat: Optional[float] = Field(None, description="纬度")
    top_k: int = Field(10, ge=1, le=100, description="返回数量")

    class Config:
        json_schema_extra = {
            "example": {
                "poi_id": 12345,
                "top_k": 10
            }
        }


class EncodeRequest(BaseModel):
    """编码请求"""
    category: str = Field(..., description="POI类别")
    landuse: str = Field(..., description="土地利用类型")
    aoi_type: str = Field(..., description="AOI类型")
    road_class: str = Field(..., description="道路等级")
    density: float = Field(0.0, description="POI密度")
    entropy: float = Field(0.0, description="类别熵")
    road_dist: float = Field(0.0, description="道路距离")
    lng: float = Field(..., description="经度")
    lat: float = Field(..., description="纬度")


class SearchResult(BaseModel):
    """搜索结果"""
    poi_id: int
    distance: float
    lng: float
    lat: float


class StatsResponse(BaseModel):
    """统计响应"""
    status: str
    num_pois: int
    embed_dim: int
    silhouette: Optional[float]
    achievement_rate: Optional[float]
    faiss_available: bool
    timestamp: str


# ============ FastAPI App ============

app = FastAPI(
    title="V2.3 空间编码器 API",
    description="空间相似性查询服务",
    version="2.3.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
index_service: Optional[VectorIndexService] = None


def get_model_dir() -> str:
    """获取最新模型目录"""
    saved_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

    if not os.path.exists(saved_models_dir):
        raise FileNotFoundError(f"模型目录不存在: {saved_models_dir}")

    # 找最新的模型目录
    model_dirs = [d for d in os.listdir(saved_models_dir) if d.startswith('v23_')]
    if not model_dirs:
        raise FileNotFoundError("没有找到训练好的模型")

    latest_dir = sorted(model_dirs)[-1]
    return os.path.join(saved_models_dir, latest_dir)


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global index_service

    try:
        model_dir = get_model_dir()
        logger.info(f"加载模型: {model_dir}")
        index_service = VectorIndexService(model_dir)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


@app.get("/")
async def root():
    """根路由"""
    return {
        "service": "V2.3 空间编码器 API",
        "version": "2.3.0",
        "status": "running",
        "endpoints": [
            "/spatial_search",
            "/encode",
            "/stats",
            "/batch_search"
        ]
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取服务统计信息"""
    if index_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    stats = index_service.get_stats()

    return StatsResponse(
        status="ok",
        num_pois=stats['num_pois'],
        embed_dim=stats['embed_dim'],
        silhouette=stats['metadata'].get('silhouette'),
        achievement_rate=stats['metadata'].get('achievement_rate'),
        faiss_available=stats['faiss_available'],
        timestamp=datetime.now().isoformat()
    )


@app.post("/spatial_search")
async def spatial_search(request: SpatialSearchRequest):
    """
    空间相似性查询

    根据POI ID或坐标查找空间相似的POI

    - 使用POI ID: 提供 poi_id
    - 使用坐标: 提供 lng 和 lat
    """
    if index_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        if request.poi_id:
            # 使用POI ID搜索
            results = index_service.search_by_poi_id(request.poi_id, request.top_k)
        elif request.lng is not None and request.lat is not None:
            # 使用坐标搜索
            results = index_service.search_by_coord(request.lng, request.lat, request.top_k)
        else:
            raise HTTPException(status_code=400, detail="需要提供poi_id或坐标(lng, lat)")

        return {
            "success": True,
            "query": {
                "poi_id": request.poi_id,
                "lng": request.lng,
                "lat": request.lat
            },
            "results": results,
            "count": len(results)
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.post("/encode")
async def encode_poi(request: EncodeRequest):
    """
    编码新的POI

    返回64维空间embedding向量
    """
    if index_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        # 映射特征
        category_map = index_service.mappings.get('category', {})
        landuse_map = index_service.mappings.get('landuse', {})
        aoi_map = index_service.mappings.get('aoi_type', {})
        road_map = index_service.mappings.get('road_class', {})

        # 未知特征使用0
        cat_id = category_map.get(request.category, 0)
        lu_id = landuse_map.get(request.landuse, 0)
        aoi_id = aoi_map.get(request.aoi_type, 0)
        rc_id = road_map.get(request.road_class, 0)

        features = [cat_id, lu_id, aoi_id, rc_id, request.density, request.entropy, request.road_dist]
        coords = [request.lng, request.lat]

        embedding = index_service.encode_new_poi(features, coords)

        return {
            "success": True,
            "embedding": embedding.tolist(),
            "dim": len(embedding),
            "features": {
                "category": request.category,
                "landuse": request.landuse,
                "aoi_type": request.aoi_type,
                "road_class": request.road_class
            },
            "coords": coords
        }

    except Exception as e:
        logger.error(f"编码失败: {e}")
        raise HTTPException(status_code=500, detail=f"编码失败: {str(e)}")


@app.post("/batch_search")
async def batch_search(poi_ids: List[int], top_k: int = Query(10, ge=1, le=50)):
    """
    批量搜索

    一次查询多个POI的相似POI
    """
    if index_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    if len(poi_ids) > 100:
        raise HTTPException(status_code=400, detail="最多支持100个POI")

    results = {}
    for poi_id in poi_ids:
        try:
            similar = index_service.search_by_poi_id(poi_id, top_k)
            results[poi_id] = similar
        except ValueError:
            results[poi_id] = {"error": f"POI ID {poi_id} 不存在"}
        except Exception as e:
            results[poi_id] = {"error": str(e)}

    return {
        "success": True,
        "results": results,
        "count": len(results)
    }


@app.get("/poi/{poi_id}")
async def get_poi(poi_id: int):
    """获取POI的embedding和坐标"""
    if index_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        embedding = index_service.get_poi_embedding(poi_id)
        coord = index_service.get_poi_coord(poi_id)

        return {
            "success": True,
            "poi_id": poi_id,
            "embedding": embedding.tolist(),
            "coord": {
                "lng": coord[0],
                "lat": coord[1]
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============ Main ============

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info"
    )
