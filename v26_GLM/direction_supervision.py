# -*- coding: utf-8 -*-
"""
V2.6 方向监督信号模块

支持多种方向监督方案并行：
1. 相邻Cell相对方向 (neighbor_relative) - 学习局部方向关系
2. 道路/线路方向 (road_direction) - 交通方向语义
3. 功能区中心方向 (region_center) - 功能语义相关
4. 全局中心方向 (global_center) - 城市位置感知

根据用户语义和需求路由选择最合适的方向头。

Author: GLM (Qianfan Code)
Date: 2026-03-15
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DirectionScheme(Enum):
    """方向监督方案"""
    NEIGHBOR_RELATIVE = "neighbor_relative"  # 相邻Cell相对方向
    ROAD_DIRECTION = "road_direction"          # 道路方向
    REGION_CENTER = "region_center"            # 功能区中心方向
    GLOBAL_CENTER = "global_center"            # 全局中心方向


@dataclass
class DirectionLabel:
    """方向标签结构"""
    labels: np.ndarray        # 方向标签 [N] 或 [N, K]
    weights: np.ndarray       # 样本权重 [N] 或 [N, K]
    valid_mask: np.ndarray    # 有效掩码 [N] 或 [N, K]
    scheme: DirectionScheme   # 使用的方案


def angle_to_direction(angle: float) -> int:
    """
    角度转8方向类别

    方向定义：
    - 0: 东 (E)
    - 1: 东北 (NE)
    - 2: 北 (N)
    - 3: 西北 (NW)
    - 4: 西 (W)
    - 5: 西南 (SW)
    - 6: 南 (S)
    - 7: 东南 (SE)
    """
    return int((angle + np.pi) / (np.pi / 4)) % 8


def direction_to_angle(direction: int) -> float:
    """方向类别转角度（弧度）"""
    return direction * (np.pi / 4) - np.pi


def compute_neighbor_relative_direction(
    cell_coords: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: Optional[np.ndarray] = None,
) -> DirectionLabel:
    """
    方案1：计算相邻Cell的相对方向

    这是最核心的方向监督，让模型学会"东/南/西/北"的相对方向概念。

    Args:
        cell_coords: Cell中心坐标 [N, 2] (lng, lat)
        neighbor_indices: 邻居索引 [N, K]，-1表示无效
        neighbor_weights: 邻居权重 [N, K]，可选

    Returns:
        DirectionLabel: 包含方向标签和有效掩码
    """
    N, K = neighbor_indices.shape
    direction_labels = np.zeros((N, K), dtype=int)
    valid_mask = np.zeros((N, K), dtype=bool)

    if neighbor_weights is None:
        neighbor_weights = np.ones((N, K), dtype=np.float32)

    for i in range(N):
        for j, neighbor_idx in enumerate(neighbor_indices[i]):
            if neighbor_idx < 0 or neighbor_idx >= N:
                continue

            # 计算从Cell i到邻居j的方向
            dx = cell_coords[neighbor_idx, 0] - cell_coords[i, 0]
            dy = cell_coords[neighbor_idx, 1] - cell_coords[i, 1]

            # 距离太近则跳过（避免噪声）
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                continue

            angle = np.arctan2(dy, dx)
            direction_labels[i, j] = angle_to_direction(angle)
            valid_mask[i, j] = True

    return DirectionLabel(
        labels=direction_labels,
        weights=neighbor_weights.astype(np.float32),
        valid_mask=valid_mask,
        scheme=DirectionScheme.NEIGHBOR_RELATIVE,
    )


def compute_road_direction(
    cell_coords: np.ndarray,
    road_segments: List[Dict],
    h3_resolution: int = 9,
) -> DirectionLabel:
    """
    方案2：基于道路方向的方向监督

    道路方向代表真实的交通流向，对可达性分析很有价值。

    Args:
        cell_coords: Cell中心坐标 [N, 2]
        road_segments: 道路段列表，每个包含:
            - cells: 覆盖的Cell索引列表
            - direction: 道路方向（角度或"EW"/"NS"等）
            - road_class: 道路等级
        h3_resolution: H3分辨率

    Returns:
        DirectionLabel: 道路方向标签
    """
    N = len(cell_coords)
    direction_labels = np.zeros(N, dtype=int)
    weights = np.zeros(N, dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    for road in road_segments:
        cells = road.get("cells", [])
        road_direction = road.get("direction")
        road_class = road.get("road_class", "unclassified")

        # 道路等级权重
        class_weights = {
            "primary": 1.0,
            "secondary": 0.8,
            "tertiary": 0.6,
            "residential": 0.4,
            "unclassified": 0.2,
        }
        weight = class_weights.get(road_class, 0.2)

        # 解析方向
        if isinstance(road_direction, (int, float)):
            angle = road_direction
        elif isinstance(road_direction, str):
            direction_map = {
                "EW": 0,       # 东西向
                "WE": np.pi,   # 西东向
                "NS": np.pi/2, # 南北向
                "SN": -np.pi/2,# 北南向
            }
            angle = direction_map.get(road_direction.upper(), 0)
        else:
            continue

        dir_label = angle_to_direction(angle)

        for cell_idx in cells:
            if 0 <= cell_idx < N:
                # 取最大权重的方向
                if weight > weights[cell_idx]:
                    direction_labels[cell_idx] = dir_label
                    weights[cell_idx] = weight
                    valid_mask[cell_idx] = True

    return DirectionLabel(
        labels=direction_labels,
        weights=weights,
        valid_mask=valid_mask,
        scheme=DirectionScheme.ROAD_DIRECTION,
    )


def compute_region_center_direction(
    cell_coords: np.ndarray,
    region_labels: np.ndarray,
    region_centers: Optional[Dict[int, np.ndarray]] = None,
) -> DirectionLabel:
    """
    方案3：到功能区中心的方向

    适用于功能语义相关的问题，如"找出与光谷相似的区域"。

    Args:
        cell_coords: Cell中心坐标 [N, 2]
        region_labels: 功能区标签 [N]
        region_centers: 功能区中心坐标 {region_id: [lng, lat]}
                       如果为None，自动计算为各功能区质心

    Returns:
        DirectionLabel: 功能区中心方向标签
    """
    N = len(cell_coords)
    direction_labels = np.zeros(N, dtype=int)
    weights = np.ones(N, dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    unique_regions = np.unique(region_labels)

    # 自动计算功能区中心
    if region_centers is None:
        region_centers = {}
        for region_id in unique_regions:
            mask = region_labels == region_id
            if mask.sum() > 0:
                region_centers[region_id] = cell_coords[mask].mean(axis=0)

    # 计算每个Cell到其功能区中心的方向
    for i in range(N):
        region_id = region_labels[i]
        if region_id not in region_centers:
            continue

        center = region_centers[region_id]
        dx = center[0] - cell_coords[i, 0]
        dy = center[1] - cell_coords[i, 1]

        dist = np.sqrt(dx**2 + dy**2)
        if dist < 1e-6:
            continue

        angle = np.arctan2(dy, dx)
        direction_labels[i] = angle_to_direction(angle)
        valid_mask[i] = True

        # 距离中心越远，权重越高（边界Cell更需要方向信息）
        weights[i] = min(dist * 1000, 1.0)  # 归一化

    return DirectionLabel(
        labels=direction_labels,
        weights=weights,
        valid_mask=valid_mask,
        scheme=DirectionScheme.REGION_CENTER,
    )


def compute_global_center_direction(
    cell_coords: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> DirectionLabel:
    """
    方案4：到全局中心的方向

    让模型感知"在城市哪个位置"，适用于大范围位置感知。

    Args:
        cell_coords: Cell中心坐标 [N, 2]
        center: 全局中心坐标 [2]，如果为None使用质心

    Returns:
        DirectionLabel: 全局中心方向标签
    """
    N = len(cell_coords)

    if center is None:
        center = cell_coords.mean(axis=0)

    direction_labels = np.zeros(N, dtype=int)
    weights = np.ones(N, dtype=np.float32)
    valid_mask = np.ones(N, dtype=bool)

    for i in range(N):
        dx = center[0] - cell_coords[i, 0]
        dy = center[1] - cell_coords[i, 1]

        angle = np.arctan2(dy, dx)
        direction_labels[i] = angle_to_direction(angle)

        # 距离中心越远，权重越高
        dist = np.sqrt(dx**2 + dy**2)
        weights[i] = min(dist * 100, 1.0)

    return DirectionLabel(
        labels=direction_labels,
        weights=weights,
        valid_mask=valid_mask,
        scheme=DirectionScheme.GLOBAL_CENTER,
    )


class MultiSchemeDirectionSupervision:
    """
    多方案方向监督管理器

    支持：
    1. 并行训练多个方向头
    2. 根据用户语义路由选择
    3. 方案组合与加权
    """

    def __init__(
        self,
        enabled_schemes: Optional[List[DirectionScheme]] = None,
    ):
        """
        Args:
            enabled_schemes: 启用的方案列表，None表示全部启用
        """
        if enabled_schemes is None:
            enabled_schemes = list(DirectionScheme)
        self.enabled_schemes = enabled_schemes
        self.labels: Dict[DirectionScheme, DirectionLabel] = {}

    def compute_all(
        self,
        cell_coords: np.ndarray,
        neighbor_indices: Optional[np.ndarray] = None,
        road_segments: Optional[List[Dict]] = None,
        region_labels: Optional[np.ndarray] = None,
        region_centers: Optional[Dict[int, np.ndarray]] = None,
        global_center: Optional[np.ndarray] = None,
    ) -> Dict[DirectionScheme, DirectionLabel]:
        """
        计算所有启用的方向监督

        Args:
            cell_coords: Cell中心坐标 [N, 2]
            neighbor_indices: 邻居索引 [N, K]
            road_segments: 道路段列表
            region_labels: 功能区标签 [N]
            region_centers: 功能区中心坐标
            global_center: 全局中心坐标

        Returns:
            各方案的DirectionLabel字典
        """
        self.labels = {}

        for scheme in self.enabled_schemes:
            if scheme == DirectionScheme.NEIGHBOR_RELATIVE:
                if neighbor_indices is not None:
                    self.labels[scheme] = compute_neighbor_relative_direction(
                        cell_coords, neighbor_indices
                    )

            elif scheme == DirectionScheme.ROAD_DIRECTION:
                if road_segments is not None:
                    self.labels[scheme] = compute_road_direction(
                        cell_coords, road_segments
                    )

            elif scheme == DirectionScheme.REGION_CENTER:
                if region_labels is not None:
                    self.labels[scheme] = compute_region_center_direction(
                        cell_coords, region_labels, region_centers
                    )

            elif scheme == DirectionScheme.GLOBAL_CENTER:
                self.labels[scheme] = compute_global_center_direction(
                    cell_coords, global_center
                )

        return self.labels

    def get_labels_for_training(
        self,
        scheme: Optional[DirectionScheme] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取训练用的方向标签

        Args:
            scheme: 指定方案，None返回组合方案

        Returns:
            labels: 方向标签
            weights: 样本权重
            valid_mask: 有效掩码
        """
        if scheme is not None:
            if scheme in self.labels:
                label = self.labels[scheme]
                return label.labels, label.weights, label.valid_mask
            else:
                raise ValueError(f"Scheme {scheme} not computed")

        # 组合多个方案
        # 优先级：neighbor_relative > road > region_center > global_center
        priority = [
            DirectionScheme.NEIGHBOR_RELATIVE,
            DirectionScheme.ROAD_DIRECTION,
            DirectionScheme.REGION_CENTER,
            DirectionScheme.GLOBAL_CENTER,
        ]

        # 初始化
        N = None
        for scheme in priority:
            if scheme in self.labels:
                N = len(self.labels[scheme].labels)
                break

        if N is None:
            raise ValueError("No direction labels computed")

        combined_labels = np.zeros(N, dtype=int)
        combined_weights = np.zeros(N, dtype=np.float32)
        combined_mask = np.zeros(N, dtype=bool)

        for scheme in priority:
            if scheme not in self.labels:
                continue

            label = self.labels[scheme]

            # 只填充尚未有标签的位置
            if label.labels.ndim == 1:
                new_mask = ~combined_mask & label.valid_mask
                combined_labels[new_mask] = label.labels[new_mask]
                combined_weights[new_mask] = label.weights[new_mask]
                combined_mask[new_mask] = True

        return combined_labels, combined_weights, combined_mask

    def route_by_query_type(self, query_type: str) -> DirectionScheme:
        """
        根据查询类型路由到最合适的方向方案

        Args:
            query_type: 查询类型

        Returns:
            推荐的方向方案
        """
        routing_map = {
            "nearby": DirectionScheme.NEIGHBOR_RELATIVE,      # 附近推荐
            "navigation": DirectionScheme.ROAD_DIRECTION,      # 导航路径
            "region_similarity": DirectionScheme.REGION_CENTER, # 区域相似
            "location": DirectionScheme.GLOBAL_CENTER,         # 位置感知
        }

        return routing_map.get(query_type, DirectionScheme.NEIGHBOR_RELATIVE)


def build_direction_supervisor(
    enabled_schemes: Optional[List[str]] = None,
) -> MultiSchemeDirectionSupervision:
    """
    构建方向监督器

    Args:
        enabled_schemes: 启用的方案名称列表
            - "neighbor_relative": 相邻Cell相对方向
            - "road_direction": 道路方向
            - "region_center": 功能区中心方向
            - "global_center": 全局中心方向

    Returns:
        MultiSchemeDirectionSupervision实例
    """
    if enabled_schemes is None:
        schemes = list(DirectionScheme)
    else:
        scheme_map = {s.value: s for s in DirectionScheme}
        schemes = [scheme_map[name] for name in enabled_schemes if name in scheme_map]

    return MultiSchemeDirectionSupervision(schemes)
