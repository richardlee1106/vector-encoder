# -*- coding: utf-8 -*-

from spatial_encoder.v26.dataset_builder import (
    build_line_agent,
    build_point_agent,
    build_polygon_agent,
)


def test_build_point_agent_generates_primary_cell():
    row = {
        "id": 1,
        "lng": 114.364,
        "lat": 30.532,
        "name": "测试咖啡馆",
        "category": "cafe",
    }

    agent = build_point_agent(row, resolution=9)

    assert agent["agent_type"] == "point"
    assert agent["primary_cell"]
    assert agent["metadata"]["name"] == "测试咖啡馆"


def test_build_line_agent_generates_multiple_cells():
    row = {
        "id": 2,
        "coords": [(114.360, 30.530), (114.365, 30.535)],
        "road_class": "primary",
    }

    agent = build_line_agent(row, resolution=9)

    assert agent["agent_type"] == "line"
    assert len(agent["cells"]) >= 1


def test_build_polygon_agent_generates_weighted_cells():
    row = {
        "id": 3,
        "coords": [
            (114.360, 30.530),
            (114.366, 30.530),
            (114.366, 30.536),
            (114.360, 30.536),
            (114.360, 30.530),
        ],
        "landuse": "commercial",
    }

    agent = build_polygon_agent(row, resolution=9)

    assert agent["agent_type"] == "polygon"
    assert len(agent["cells"]) >= 1
    assert all("weight" in item for item in agent["cells"])
