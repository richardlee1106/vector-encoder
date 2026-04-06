# -*- coding: utf-8 -*-
"""
V2.3 向量索引服务

使用FAISS构建高效向量索引，支持：
- 空间相似性查询
- 范围查询
- 批量编码
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入faiss
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS已加载")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS未安装，使用numpy进行向量搜索")


class Encoder(nn.Module):
    """V2.3 空间编码器"""

    def __init__(self, nc, nl, na, nr, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        d = hidden_dim // 6
        self.ce = nn.Embedding(nc + 1, d)
        self.le = nn.Embedding(nl + 1, d)
        self.ae = nn.Embedding(na + 1, d)
        self.re = nn.Embedding(nr + 1, d)
        self.np = nn.Linear(3, d)
        self.cp = nn.Linear(2, d)

        self.enc = nn.Sequential(
            nn.Linear(d * 6, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim))

        self.dec = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def encode(self, f, c):
        cn = (c - c.mean(0)) / (c.std(0) + 1e-8)
        x = torch.cat([self.ce(f[:, 0].long()), self.le(f[:, 1].long()),
                       self.ae(f[:, 2].long()), self.re(f[:, 3].long()),
                       self.np(f[:, 4:7]), self.cp(cn)], -1)
        return F.normalize(self.enc(x), p=2, dim=-1)

    def forward(self, f, c):
        z = self.encode(f, c)
        return z, self.dec(z)


class VectorIndexService:
    """向量索引服务"""

    def __init__(self, model_dir: str):
        """
        初始化向量索引服务

        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self._load_model()

        # 加载数据
        self._load_data()

        # 构建索引
        self._build_index()

        logger.info(f"向量索引服务初始化完成, 共 {self.num_pois:,} POI")

    def _load_model(self):
        """加载模型"""
        model_path = os.path.join(self.model_dir, 'model.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        self.model = Encoder(
            config['num_categories'],
            config['num_landuses'],
            config['num_aoi_types'],
            config['num_road_classes'],
            config['embed_dim'],
            config['hidden_dim']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.embed_dim = config['embed_dim']
        self.metadata = checkpoint.get('metadata', {})

        logger.info(f"模型加载完成: {self.metadata}")

    def _load_data(self):
        """加载数据"""
        # 加载映射
        mappings_path = os.path.join(self.model_dir, 'mappings.json')
        with open(mappings_path, 'r', encoding='utf-8') as f:
            self.mappings = json.load(f)

        # 构建反向映射
        self.reverse_mappings = {
            key: {v: k for k, v in mapping.items()}
            for key, mapping in self.mappings.items()
        }

        # 加载POI ID
        ids_path = os.path.join(self.model_dir, 'poi_ids.npy')
        self.poi_ids = np.load(ids_path)

        # 构建ID到索引的映射
        self.id_to_idx = {int(poi_id): idx for idx, poi_id in enumerate(self.poi_ids)}

        # 加载embeddings
        emb_path = os.path.join(self.model_dir, 'embeddings.npy')
        self.embeddings = np.load(emb_path)

        # 加载坐标
        coords_path = os.path.join(self.model_dir, 'coords.npy')
        self.coords = np.load(coords_path)

        self.num_pois = len(self.poi_ids)

        logger.info(f"数据加载完成: {self.num_pois:,} POI")

    def _build_index(self):
        """构建FAISS索引"""
        if FAISS_AVAILABLE:
            # 使用FAISS IVF索引
            nlist = min(100, self.num_pois // 1000)  # 聚类中心数
            quantizer = faiss.IndexFlatL2(self.embed_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist)

            # 训练索引
            train_size = min(100000, self.num_pois)
            train_data = self.embeddings[np.random.choice(self.num_pois, train_size, replace=False)]
            self.index.train(train_data)

            # 添加向量
            self.index.add(self.embeddings)

            logger.info(f"FAISS IVF索引构建完成: nlist={nlist}")
        else:
            # 使用numpy
            self.index = None
            logger.info("使用numpy进行向量搜索")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        向量搜索

        Args:
            query_vec: 查询向量 [D] 或 [N, D]
            top_k: 返回数量

        Returns:
            distances: 距离
            indices: 索引
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(query_vec, top_k)
            return distances, indices
        else:
            # numpy实现
            similarities = np.dot(query_vec, self.embeddings.T)
            indices = np.argsort(-similarities, axis=1)[:, :top_k]
            distances = np.take_along_axis(similarities, indices, axis=1)
            return distances, indices

    def search_by_poi_id(self, poi_id: int, top_k: int = 10) -> List[Dict]:
        """
        根据POI ID搜索相似POI

        Args:
            poi_id: POI ID
            top_k: 返回数量

        Returns:
            相似POI列表
        """
        if poi_id not in self.id_to_idx:
            raise ValueError(f"POI ID {poi_id} 不存在")

        idx = self.id_to_idx[poi_id]
        query_vec = self.embeddings[idx]

        distances, indices = self.search(query_vec, top_k + 1)  # +1 因为包含自己

        results = []
        for i, (dist, neighbor_idx) in enumerate(zip(distances[0], indices[0])):
            if neighbor_idx == idx:
                continue  # 跳过自己

            results.append({
                'poi_id': int(self.poi_ids[neighbor_idx]),
                'index': int(neighbor_idx),
                'distance': float(dist),
                'coord': self.coords[neighbor_idx].tolist()
            })

            if len(results) >= top_k:
                break

        return results

    def search_by_coord(self, lng: float, lat: float, top_k: int = 10) -> List[Dict]:
        """
        根据坐标搜索附近POI（空间相似）

        Args:
            lng: 经度
            lat: 纬度
            top_k: 返回数量

        Returns:
            相似POI列表
        """
        # 计算与所有POI的空间距离
        dists = np.sqrt((self.coords[:, 0] - lng) ** 2 + (self.coords[:, 1] - lat) ** 2)
        nearest_idx = np.argmin(dists)

        # 使用最近POI的embedding搜索
        return self.search_by_poi_id(int(self.poi_ids[nearest_idx]), top_k)

    def encode_new_poi(self, features: List, coords: List[float]) -> np.ndarray:
        """
        编码新的POI

        Args:
            features: [category_id, landuse_id, aoi_type_id, road_class_id, density, entropy, road_dist]
            coords: [lng, lat]

        Returns:
            embedding向量
        """
        with torch.no_grad():
            f = torch.tensor([features], dtype=torch.float32).to(self.device)
            c = torch.tensor([coords], dtype=torch.float32).to(self.device)
            emb = self.model.encode(f, c).cpu().numpy()[0]

        return emb

    def get_poi_embedding(self, poi_id: int) -> np.ndarray:
        """获取POI的embedding"""
        if poi_id not in self.id_to_idx:
            raise ValueError(f"POI ID {poi_id} 不存在")

        idx = self.id_to_idx[poi_id]
        return self.embeddings[idx]

    def get_poi_coord(self, poi_id: int) -> Tuple[float, float]:
        """获取POI的坐标"""
        if poi_id not in self.id_to_idx:
            raise ValueError(f"POI ID {poi_id} 不存在")

        idx = self.id_to_idx[poi_id]
        return tuple(self.coords[idx])

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'num_pois': self.num_pois,
            'embed_dim': self.embed_dim,
            'metadata': self.metadata,
            'faiss_available': FAISS_AVAILABLE
        }
