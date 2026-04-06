# -*- coding: utf-8 -*-

from spatial_encoder.v26.cell_features import aggregate_cell_features


def test_aggregate_cell_features_collects_point_line_polygon_signals():
    records = [
        {
            "agent_type": "point",
            "cells": [{"cell": "c1", "weight": 1.0}],
            "metadata": {"category": "cafe"},
        },
        {
            "agent_type": "line",
            "cells": [{"cell": "c1", "weight": 0.5}, {"cell": "c2", "weight": 0.5}],
            "metadata": {"road_class": "primary"},
        },
        {
            "agent_type": "polygon",
            "cells": [{"cell": "c1", "weight": 0.8}],
            "metadata": {"landuse": "commercial"},
        },
    ]
    result = aggregate_cell_features(records)

    assert "c1" in result
    assert result["c1"]["point_count"] == 1
    assert result["c1"]["line_weight_sum"] > 0
    assert result["c1"]["polygon_weight_sum"] > 0
