# -*- coding: utf-8 -*-
"""
V2.6 统一 agent 记录结构。
"""

from __future__ import annotations


def build_agent_record(
    agent_id: str,
    agent_type: str,
    cells: list[dict[str, float | str]],
    metadata: dict,
) -> dict:
    """构造统一 agent 记录。"""
    if not cells:
        raise ValueError("cells 不能为空")

    primary = max(cells, key=lambda item: float(item.get("weight", 0.0)))

    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "cells": cells,
        "primary_cell": primary["cell"],
        "metadata": metadata,
    }
