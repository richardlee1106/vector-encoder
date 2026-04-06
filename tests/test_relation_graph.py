# -*- coding: utf-8 -*-

from spatial_encoder.v26.relation_graph import build_relation_edges


def test_build_relation_edges_supports_multiple_relation_types():
    edges = build_relation_edges(
        adjacency_pairs=[("c1", "c2")],
        functional_pairs=[("c1", "c3", 0.8)],
        cooccurrence_pairs=[("c2", "c3", 0.4)],
    )

    relation_types = {edge["relation_type"] for edge in edges}

    assert "adjacency" in relation_types
    assert "functional_similarity" in relation_types
    assert "cooccurrence" in relation_types
