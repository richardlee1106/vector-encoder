# -*- coding: utf-8 -*-
"""
V2.6 多几何 agent 构建器。
"""

from __future__ import annotations

from spatial_encoder.v26.agent_records import build_agent_record
from spatial_encoder.v26.h3_projection import (
    line_to_cells,
    point_to_cell,
    polygon_to_cells,
)


def build_point_agent(row: dict, resolution: int) -> dict:
    """由点对象行构造 point agent。"""
    cell = point_to_cell(row["lng"], row["lat"], resolution)
    metadata = {
        "name": row.get("name"),
        "category": row.get("category"),
        "lng": row["lng"],
        "lat": row["lat"],
    }
    return build_agent_record(
        agent_id=f"point:{row['id']}",
        agent_type="point",
        cells=[{"cell": cell, "weight": 1.0}],
        metadata=metadata,
    )


def build_line_agent(row: dict, resolution: int) -> dict:
    """由线对象行构造 line agent。"""
    cells = [
        {"cell": cell, "weight": 1.0}
        for cell in line_to_cells(row["coords"], resolution)
    ]
    metadata = {
        "road_class": row.get("road_class"),
        "coord_count": len(row.get("coords", [])),
        "cell_count": len(cells),
    }
    return build_agent_record(
        agent_id=f"line:{row['id']}",
        agent_type="line",
        cells=cells,
        metadata=metadata,
    )


def build_polygon_agent(row: dict, resolution: int) -> dict:
    """由面对象行构造 polygon agent。"""
    cells = polygon_to_cells(row["coords"], resolution)
    metadata = {
        "landuse": row.get("landuse"),
        "coord_count": len(row.get("coords", [])),
        "cell_count": len(cells),
    }
    return build_agent_record(
        agent_id=f"polygon:{row['id']}",
        agent_type="polygon",
        cells=cells,
        metadata=metadata,
    )
