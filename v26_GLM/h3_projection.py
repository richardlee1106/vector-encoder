# -*- coding: utf-8 -*-
"""
H3 投影模块 - 简化版

提供点、线、面到 H3 Cell 的投影功能

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations
from typing import List, Tuple, Set, Dict
import numpy as np

def point_to_cell(lng: float, lat: float, resolution: int = 10) -> str:
    """
    将点投影到 H3 Cell

    Args:
        lng: 经度
        lat: 纬度
        resolution: H3 分辨率 (0-15)

    Returns:
        H3 Cell ID (字符串)
    """
    try:
        import h3
        return h3.latlng_to_cell(lat, lng, resolution)
    except ImportError:
        # h3 未安装，使用简化版本
        # 使用网格划分模拟 H3
        grid_size = 0.001 * (2 ** (15 - resolution))
        return f"cell_{int(lng / grid_size)}_{int(lat / grid_size)}"


def cell_to_point(cell_id: str) -> Tuple[float, float]:
    """
    将 H3 Cell 转换为中心点坐标

    Args:
        cell_id: H3 Cell ID

    Returns:
        (lng, lat) 中心点坐标
    """
    try:
        import h3
        lat, lng = h3.cell_to_latlng(cell_id)
        return lng, lat
    except ImportError:
        # 解析简化版本
        if cell_id.startswith("cell_"):
            parts = cell_id.split("_")
            if len(parts) == 3:
                return float(parts[1]) * 0.001, float(parts[2]) * 0.001
        return 0.0, 0.0


def line_to_cells(
    coords: List[Tuple[float, float]],
    resolution: int = 10,
) -> Set[str]:
    """
    将线投影到 H3 Cell 集合

    Args:
        coords: 线坐标列表 [(lng, lat), ...]
        resolution: H3 分辨率

    Returns:
        H3 Cell ID 集合
    """
    cells = set()
    for lng, lat in coords:
        cell = point_to_cell(lng, lat, resolution)
        cells.add(cell)
    return cells


def polygon_to_cells(
    exterior: List[Tuple[float, float]],
    holes: List[List[Tuple[float, float]]] = None,
    resolution: int = 10,
) -> List[Dict]:
    """
    将面投影到 H3 Cell 集合（带权重）

    Args:
        exterior: 外环坐标 [(lng, lat), ...]
        holes: 内环列表
        resolution: H3 分辨率

    Returns:
        List[{"cell": cell_id, "weight": weight}, ...]
    """
    from collections import defaultdict

    cell_counts = defaultdict(int)
    total_points = len(exterior)

    # 简化版本：遍历所有点，统计每个cell的出现次数
    for lng, lat in exterior:
        cell = point_to_cell(lng, lat, resolution)
        cell_counts[cell] += 1

    # 转换为带权重的列表
    result = []
    for cell_id, count in cell_counts.items():
        weight = count / total_points if total_points > 0 else 1.0
        result.append({"cell": cell_id, "weight": weight})

    return result


def get_cell_neighbors(cell_id: str) -> List[str]:
    """
    获取 H3 Cell 的邻居

    Args:
        cell_id: H3 Cell ID

    Returns:
        邻居 Cell ID 列表
    """
    try:
        import h3
        return list(h3.grid_disk(cell_id, 1))
    except ImportError:
        return []


def cell_to_boundary(cell_id: str) -> List[Tuple[float, float]]:
    """
    获取 H3 Cell 的边界坐标

    Args:
        cell_id: H3 Cell ID

    Returns:
        边界坐标列表 [(lng, lat), ...]
    """
    try:
        import h3
        boundary = h3.cell_to_boundary(cell_id)
        return [(lng, lat) for lat, lng in boundary]
    except ImportError:
        return []
