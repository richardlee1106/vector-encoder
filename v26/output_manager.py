# -*- coding: utf-8 -*-
"""
V2.6 实验产出目录管理。
"""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_DIRS = {
    "reports": "reports",
    "exports": "exports",
    "metrics": "metrics",
}


def ensure_v26_output_dirs(base_dir: str | Path | None = None) -> dict[str, Path]:
    """确保 V2.6 产出目录存在。"""
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent / "outputs"
    )

    paths = {"root": root}
    root.mkdir(parents=True, exist_ok=True)

    for artifact_type, dir_name in ARTIFACT_DIRS.items():
        path = root / dir_name
        path.mkdir(parents=True, exist_ok=True)
        paths[artifact_type] = path

    return paths


def save_v26_json(
    name: str,
    payload: dict,
    base_dir: str | Path | None = None,
    artifact_type: str = "reports",
) -> Path:
    """以 UTF-8 保存 V2.6 JSON 产出。"""
    paths = ensure_v26_output_dirs(base_dir)
    if artifact_type not in ARTIFACT_DIRS:
        raise ValueError(f"不支持的产出类型: {artifact_type}")

    output_path = paths[artifact_type] / f"{name}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path
