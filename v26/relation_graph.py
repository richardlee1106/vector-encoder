# -*- coding: utf-8 -*-
"""
V2.6 多关系图构建。
"""

from __future__ import annotations


def build_relation_edges(
    adjacency_pairs: list[tuple[str, str]] | None = None,
    functional_pairs: list[tuple[str, str, float]] | None = None,
    cooccurrence_pairs: list[tuple[str, str, float]] | None = None,
    od_pairs: list[tuple[str, str, float]] | None = None,
) -> list[dict]:
    """构建统一关系边。"""
    edges: list[dict] = []

    for source, target in adjacency_pairs or []:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": 1.0,
                "relation_type": "adjacency",
            }
        )

    for source, target, weight in functional_pairs or []:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": float(weight),
                "relation_type": "functional_similarity",
            }
        )

    for source, target, weight in cooccurrence_pairs or []:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": float(weight),
                "relation_type": "cooccurrence",
            }
        )

    for source, target, weight in od_pairs or []:
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": float(weight),
                "relation_type": "od_flow",
            }
        )

    return edges
