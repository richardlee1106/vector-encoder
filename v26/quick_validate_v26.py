# -*- coding: utf-8 -*-
"""
V2.6 小样本快速验证脚本。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import psycopg2

from spatial_encoder.v26.cell_features import aggregate_cell_features
from spatial_encoder.v26.data_sources import resolve_sample_source, sanitize_data_source
from spatial_encoder.v26.dataset_builder import (
    build_line_agent,
    build_point_agent,
    build_polygon_agent,
)
from spatial_encoder.v26.output_manager import save_v26_json
from spatial_encoder.v26.relation_graph import build_relation_edges


def _format_metric_value(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 4)


def _summarize_feature_distribution(
    cell_features: dict[str, dict],
    feature_key: str,
    metric_key: str,
    limit: int = 5,
) -> list[dict]:
    totals: defaultdict[str, float] = defaultdict(float)

    for cell_payload in cell_features.values():
        for name, value in cell_payload.get(feature_key, {}).items():
            totals[name] += float(value)

    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [
        {"name": name, metric_key: _format_metric_value(value)}
        for name, value in ranked
    ]


def build_quick_validate_report(
    records: list[dict],
    cell_features: dict,
    relation_edges: list[dict],
    sample_config: dict | None = None,
    sample_source: dict | None = None,
) -> dict:
    """根据采样后的中间结果生成快速验证摘要。"""
    agent_counts = Counter(record["agent_type"] for record in records)
    relation_counts = Counter(edge["relation_type"] for edge in relation_edges)

    return {
        "agent_counts": {
            "point": agent_counts.get("point", 0),
            "line": agent_counts.get("line", 0),
            "polygon": agent_counts.get("polygon", 0),
        },
        "unique_cells": len(cell_features),
        "relation_counts": dict(relation_counts),
        "sample_config": sample_config or {},
        "sample_source": sample_source or {},
        "top_point_categories": _summarize_feature_distribution(
            cell_features,
            feature_key="point_categories",
            metric_key="count",
        ),
        "top_road_classes": _summarize_feature_distribution(
            cell_features,
            feature_key="road_classes",
            metric_key="weight",
        ),
        "top_landuse_types": _summarize_feature_distribution(
            cell_features,
            feature_key="landuse_types",
            metric_key="weight",
        ),
    }


def build_quick_validate_metrics(report: dict) -> dict:
    """从详细报告中提炼稳定的指标快照。"""
    agent_counts = report["agent_counts"]
    relation_counts = report["relation_counts"]

    return {
        "total_agents": sum(agent_counts.values()),
        "unique_cells": report["unique_cells"],
        "total_relations": sum(relation_counts.values()),
        "dominant_point_category": (
            report["top_point_categories"][0]["name"]
            if report["top_point_categories"]
            else None
        ),
        "dominant_road_class": (
            report["top_road_classes"][0]["name"]
            if report["top_road_classes"]
            else None
        ),
        "dominant_landuse_type": (
            report["top_landuse_types"][0]["name"]
            if report["top_landuse_types"]
            else None
        ),
    }


def _extract_coords(geom_json: str) -> list[tuple[float, float]]:
    geom = json.loads(geom_json)
    geom_type = geom["type"]
    coords = geom["coordinates"]

    if geom_type == "LineString":
        return [tuple(item) for item in coords]
    if geom_type == "MultiLineString":
        return [tuple(item) for segment in coords for item in segment]
    if geom_type == "Polygon":
        return [tuple(item) for item in coords[0]]
    if geom_type == "MultiPolygon":
        return [tuple(item) for item in coords[0][0]]

    raise ValueError(f"不支持的几何类型: {geom_type}")


def _build_sample_relation_edges(cell_features: dict[str, dict]) -> list[dict]:
    cells = sorted(cell_features.keys())
    adjacency_pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]

    functional_pairs = []
    cooccurrence_pairs = []
    for i in range(len(cells) - 1):
        left = cell_features[cells[i]]
        right = cell_features[cells[i + 1]]
        if left["point_categories"] and right["point_categories"]:
            functional_pairs.append((cells[i], cells[i + 1], 0.8))
        if left["line_weight_sum"] > 0 and right["polygon_weight_sum"] > 0:
            cooccurrence_pairs.append((cells[i], cells[i + 1], 0.5))

    return build_relation_edges(
        adjacency_pairs=adjacency_pairs,
        functional_pairs=functional_pairs,
        cooccurrence_pairs=cooccurrence_pairs,
    )


def run_quick_validate_v26(
    resolution: int = 9,
    point_limit: int = 200,
    line_limit: int = 60,
    polygon_limit: int = 60,
) -> dict:
    """从数据库采样点线面并生成快速验证报告。"""
    source = resolve_sample_source()
    conn = psycopg2.connect(
        host=source["host"],
        port=source["port"],
        user=source["user"],
        password=source["password"],
        database=source["database"],
    )
    cur = conn.cursor()

    records: list[dict] = []

    cur.execute(
        f"""
        SELECT id, ST_X(geom), ST_Y(geom), name, category_big
        FROM {source["tables"]["point"]}
        WHERE geom IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {int(point_limit)}
        """
    )
    for poi_id, lng, lat, name, category in cur.fetchall():
        records.append(
            build_point_agent(
                {
                    "id": poi_id,
                    "lng": float(lng),
                    "lat": float(lat),
                    "name": name or "Unknown",
                    "category": category or "unknown",
                },
                resolution=resolution,
            )
        )

    cur.execute(
        f"""
        SELECT id, ST_AsGeoJSON(geom), properties->>'fclass'
        FROM {source["tables"]["line"]}
        WHERE geom IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {int(line_limit)}
        """
    )
    for road_id, geom_json, road_class in cur.fetchall():
        records.append(
            build_line_agent(
                {
                    "id": road_id,
                    "coords": _extract_coords(geom_json),
                    "road_class": road_class or "unknown",
                },
                resolution=resolution,
            )
        )

    cur.execute(
        f"""
        SELECT id, ST_AsGeoJSON(geom), properties->>'类别'
        FROM {source["tables"]["polygon"]}
        WHERE geom IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {int(polygon_limit)}
        """
    )
    for landuse_id, geom_json, landuse in cur.fetchall():
        records.append(
            build_polygon_agent(
                {
                    "id": landuse_id,
                    "coords": _extract_coords(geom_json),
                    "landuse": landuse or "unknown",
                },
                resolution=resolution,
            )
        )

    cur.close()
    conn.close()

    sample_config = {
        "resolution": resolution,
        "point_limit": point_limit,
        "line_limit": line_limit,
        "polygon_limit": polygon_limit,
    }
    cell_features = aggregate_cell_features(records)
    relation_edges = _build_sample_relation_edges(cell_features)
    report = build_quick_validate_report(
        records,
        cell_features,
        relation_edges,
        sample_config=sample_config,
        sample_source=sanitize_data_source(source),
    )
    metrics = build_quick_validate_metrics(report)

    report_path = save_v26_json("quick_validate_v26", report)
    metrics_path = save_v26_json(
        "quick_validate_v26_metrics",
        metrics,
        artifact_type="metrics",
    )

    report["output_path"] = str(report_path)
    report["report_path"] = str(report_path)
    report["metrics_path"] = str(metrics_path)
    return report


if __name__ == "__main__":
    print(run_quick_validate_v26())
