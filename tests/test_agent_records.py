# -*- coding: utf-8 -*-

from spatial_encoder.v26.agent_records import build_agent_record


def test_build_agent_record_for_point_agent():
    record = build_agent_record(
        agent_id="poi:1",
        agent_type="point",
        cells=[{"cell": "89283082813ffff", "weight": 1.0}],
        metadata={"name": "test poi", "category": "cafe"},
    )

    assert record["agent_type"] == "point"
    assert record["primary_cell"] == "89283082813ffff"
    assert record["metadata"]["category"] == "cafe"
