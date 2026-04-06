# -*- coding: utf-8 -*-
"""
V2.6 评估指标结构定义。
"""

from __future__ import annotations


def build_v26_evaluation_schema() -> dict:
    return {
        "metrics": {
            "spatial_retrieval": {
                "status": "pending",
                "required": ["recall_at_k", "overlap_at_k", "pearson"],
            },
            "direction_semantics": {
                "status": "pending",
                "required": ["direction_accuracy", "angle_error_mean"],
            },
            "region_semantics": {
                "status": "pending",
                "required": ["region_purity", "region_recall_at_k"],
            },
            "topology_reasoning": {
                "status": "pending",
                "required": ["path_consistency", "od_consistency"],
            },
        },
        "notes": "指标定义完成后，在正式实验阶段补齐数值。",
    }
