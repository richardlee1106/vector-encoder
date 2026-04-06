# -*- coding: utf-8 -*-
"""
V2.6 运行清单。
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from uuid import uuid4


def build_run_manifest(
    experiment: str,
    stage: str,
    config: dict,
    data_source: dict,
    sample_config: dict | None = None,
    notes: str | None = None,
) -> dict:
    return {
        "run_id": uuid4().hex,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "stage": stage,
        "config": config,
        "data_source": data_source,
        "sample_config": sample_config or {},
        "notes": notes,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
