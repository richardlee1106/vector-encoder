# -*- coding: utf-8 -*-
"""
V2.6 H3 投影工具。
"""

from __future__ import annotations

from math import ceil
from typing import Iterable

import h3


def point_to_cell(lng: float, lat: float, resolution: int) -> str:
    """将点投影到单个 H3 cell。"""
    return h3.latlng_to_cell(lat, lng, resolution)


def _iter_line_sample_points(
    coords: list[tuple[float, float]], samples_per_segment: int = 8
) -> Iterable[tuple[float, float]]:
    if len(coords) == 1:
        yield coords[0]
        return

    for start, end in zip(coords, coords[1:]):
        lng1, lat1 = start
        lng2, lat2 = end
        steps = max(
            samples_per_segment, ceil(max(abs(lng2 - lng1), abs(lat2 - lat1)) * 10_000)
        )
        for i in range(steps + 1):
            ratio = i / steps
            yield (
                lng1 + (lng2 - lng1) * ratio,
                lat1 + (lat2 - lat1) * ratio,
            )


def line_to_cells(coords: list[tuple[float, float]], resolution: int) -> list[str]:
    """将线按采样点覆盖映射到多个 H3 cell。"""
    ordered_cells: list[str] = []
    seen: set[str] = set()

    for lng, lat in _iter_line_sample_points(coords):
        cell = point_to_cell(lng, lat, resolution)
        if cell not in seen:
            ordered_cells.append(cell)
            seen.add(cell)

    return ordered_cells


def polygon_to_cells(
    coords: list[tuple[float, float]], resolution: int
) -> list[dict[str, float | str]]:
    """将面映射到多个 H3 cell，并返回简单权重。"""
    shell = [(lat, lng) for lng, lat in coords]
    poly = h3.LatLngPoly(shell)
    cells = list(h3.polygon_to_cells(poly, resolution))

    if not cells:
        fallback = point_to_cell(coords[0][0], coords[0][1], resolution)
        return [{"cell": fallback, "weight": 1.0}]

    weight = 1.0 / len(cells)
    return [{"cell": cell, "weight": weight} for cell in cells]
