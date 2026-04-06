# -*- coding: utf-8 -*-

from spatial_encoder.v26.quick_validate_v26 import build_quick_validate_report


def test_build_quick_validate_report_returns_core_summary():
    records = [
        {"agent_type": "point", "primary_cell": "c1"},
        {"agent_type": "line", "primary_cell": "c2"},
        {"agent_type": "polygon", "primary_cell": "c1"},
    ]
    cell_features = {"c1": {"point_count": 1}, "c2": {"point_count": 0}}
    relation_edges = [
        {"relation_type": "adjacency"},
        {"relation_type": "cooccurrence"},
        {"relation_type": "functional_similarity"},
    ]

    report = build_quick_validate_report(records, cell_features, relation_edges)

    assert report["agent_counts"]["point"] == 1
    assert report["agent_counts"]["line"] == 1
    assert report["agent_counts"]["polygon"] == 1
    assert report["unique_cells"] == 2
    assert report["relation_counts"]["adjacency"] == 1


def test_build_quick_validate_report_includes_sample_and_semantic_summary():
    records = [
        {"agent_type": "point", "primary_cell": "c1"},
        {"agent_type": "point", "primary_cell": "c2"},
        {"agent_type": "line", "primary_cell": "c2"},
    ]
    cell_features = {
        "c1": {
            "point_count": 1,
            "line_weight_sum": 0.0,
            "polygon_weight_sum": 0.0,
            "point_categories": {"cafe": 2, "bakery": 1},
            "road_classes": {"secondary": 0.5},
            "landuse_types": {"commercial": 0.4},
        },
        "c2": {
            "point_count": 1,
            "line_weight_sum": 1.0,
            "polygon_weight_sum": 0.0,
            "point_categories": {"cafe": 1, "restaurant": 3},
            "road_classes": {"primary": 1.5},
            "landuse_types": {"mixed_use": 0.8},
        },
    }
    relation_edges = [
        {"relation_type": "adjacency"},
        {"relation_type": "adjacency"},
        {"relation_type": "functional_similarity"},
    ]

    report = build_quick_validate_report(
        records,
        cell_features,
        relation_edges,
        sample_config={"resolution": 9, "point_limit": 200},
    )

    assert report["sample_config"]["resolution"] == 9
    assert report["top_point_categories"][0]["name"] == "cafe"
    assert report["top_point_categories"][0]["count"] == 3
    assert report["top_road_classes"][0]["name"] == "primary"
    assert report["top_landuse_types"][0]["name"] == "mixed_use"
