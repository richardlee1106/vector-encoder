# -*- coding: utf-8 -*-
"""
V2.6 cell 特征聚合。
"""

from __future__ import annotations


def _ensure_cell_bucket(result: dict, cell_id: str) -> dict:
    if cell_id not in result:
        result[cell_id] = {
            "point_count": 0,
            "line_weight_sum": 0.0,
            "polygon_weight_sum": 0.0,
            "point_categories": {},
            "road_classes": {},
            "landuse_types": {},
        }
    return result[cell_id]


def aggregate_cell_features(records: list[dict]) -> dict[str, dict]:
    """按 cell 聚合点、线、面信号。"""
    result: dict[str, dict] = {}

    for record in records:
        agent_type = record["agent_type"]
        metadata = record.get("metadata", {})

        for cell_item in record.get("cells", []):
            cell_id = cell_item["cell"]
            weight = float(cell_item.get("weight", 0.0))
            bucket = _ensure_cell_bucket(result, cell_id)

            if agent_type == "point":
                bucket["point_count"] += 1
                category = metadata.get("category")
                if category:
                    bucket["point_categories"][category] = (
                        bucket["point_categories"].get(category, 0) + 1
                    )
            elif agent_type == "line":
                bucket["line_weight_sum"] += weight
                road_class = metadata.get("road_class")
                if road_class:
                    bucket["road_classes"][road_class] = (
                        bucket["road_classes"].get(road_class, 0.0) + weight
                    )
            elif agent_type == "polygon":
                bucket["polygon_weight_sum"] += weight
                landuse = metadata.get("landuse")
                if landuse:
                    bucket["landuse_types"][landuse] = (
                        bucket["landuse_types"].get(landuse, 0.0) + weight
                    )

    return result
