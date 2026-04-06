# -*- coding: utf-8 -*-
"""
P1E+ 伪标签策略：解决标签稀疏问题

问题：只有 17.2% 的 cell 有有效标签，Region Loss 无法有效学习
方案：使用 POI 类别分布生成伪标签

策略：
1. 已有标签的 cell → 使用真实标签
2. 无标签但有 POI 的 cell → 使用 POI 类别推断伪标签
3. 无 POI 的 cell → 保持未知

Author: Claude
Date: 2026-03-18
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# POI 大类到功能区映射
POI_TO_REGION_MAP = {
    # 餐饮服务(0) → 商业类
    "餐饮服务": 1,
    # 购物服务(1) → 商业类
    "购物服务": 1,
    # 生活服务(2) → 居住类（配套服务）
    "生活服务": 0,
    # 体育休闲(3) → 公共类
    "体育休闲服务": 4,
    # 医疗保健(4) → 公共类
    "医疗保健服务": 4,
    # 住宿服务(5) → 商业类
    "住宿服务": 1,
    # 风景名胜(6) → 自然类
    "风景名胜": 5,
    # 商务住宅(7) → 居住类
    "商务住宅": 0,
    # 政府机构(8) → 公共类
    "政府机构及社会团体": 4,
    # 科教文化(9) → 教育类
    "科教文化服务": 3,
    # 交通设施(10) → 公共类
    "交通设施服务": 4,
    # 金融保险(11) → 商业类
    "金融保险服务": 1,
    # 公司企业(12) → 工业类（或商业类）
    "公司企业": 2,
}

# POI 类别索引映射（与 data_loader_v26.py 一致）
POI_CATEGORY_LIST = [
    "餐饮服务", "购物服务", "生活服务", "体育休闲服务",
    "医疗保健服务", "住宿服务", "风景名胜", "商务住宅",
    "政府机构及社会团体", "科教文化服务", "交通设施服务",
    "金融保险服务", "公司企业", "道路附属设施", "地名地址信息", "公共设施",
]


def infer_region_from_poi(poi_distribution: np.ndarray) -> int:
    """
    根据 POI 类别分布推断功能区

    Args:
        poi_distribution: [16] POI 类别分布

    Returns:
        region_label: 0-5 的功能区标签
    """
    # 找到主要 POI 类别
    top_categories = np.argsort(poi_distribution)[::-1]

    for cat_idx in top_categories:
        if poi_distribution[cat_idx] < 0.1:  # 忽略占比 < 10% 的类别
            continue

        cat_name = POI_CATEGORY_LIST[cat_idx]
        if cat_name in POI_TO_REGION_MAP:
            return POI_TO_REGION_MAP[cat_name]

    # 默认返回居住类
    return 0


def generate_pseudo_labels(
    point_features: np.ndarray,
    region_labels: np.ndarray,
    poi_threshold: float = 0.0,  # POI 数量阈值
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成伪标签

    Args:
        point_features: [N, 32+] 点特征（前 32 维是原始 point_features）
        region_labels: [N] 原始功能区标签

    Returns:
        pseudo_labels: [N] 伪标签
        pseudo_mask: [N] 是否使用伪标签
    """
    N = len(region_labels)
    pseudo_labels = region_labels.copy()
    pseudo_mask = np.zeros(N, dtype=bool)

    # POI 类别分布在 point_features 的第 3-18 维
    poi_distribution = point_features[:, 3:19]  # [N, 16]
    poi_count = point_features[:, 2]  # log(POI_count) / 10

    for i in range(N):
        # 已有有效标签 → 保持
        if region_labels[i] < 6:
            continue

        # 无 POI → 保持未知
        if poi_count[i] < poi_threshold:
            continue

        # 有 POI 但无标签 → 推断伪标签
        pseudo_labels[i] = infer_region_from_poi(poi_distribution[i])
        pseudo_mask[i] = True

    return pseudo_labels, pseudo_mask


def analyze_pseudo_labels(
    region_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    pseudo_mask: np.ndarray,
) -> Dict:
    """分析伪标签分布"""
    original_valid = (region_labels < 6).sum()
    pseudo_valid = (pseudo_labels < 6).sum()
    pseudo_generated = pseudo_mask.sum()

    return {
        "original_valid": original_valid,
        "original_ratio": original_valid / len(region_labels),
        "pseudo_valid": pseudo_valid,
        "pseudo_ratio": pseudo_valid / len(region_labels),
        "pseudo_generated": pseudo_generated,
        "improvement": (pseudo_valid - original_valid) / len(region_labels),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
    from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training

    print("Testing pseudo-label generation...")

    data = load_dataset_for_training(config=DEFAULT_PRO_CONFIG, sample_ratio=1.0)
    point_features = data["point_features"]
    region_labels = data["region_labels"]

    pseudo_labels, pseudo_mask = generate_pseudo_labels(point_features, region_labels)
    stats = analyze_pseudo_labels(region_labels, pseudo_labels, pseudo_mask)

    print(f"\nPseudo-label Statistics:")
    print(f"  Original valid labels: {stats['original_valid']:,} ({stats['original_ratio']:.1%})")
    print(f"  Pseudo valid labels: {stats['pseudo_valid']:,} ({stats['pseudo_ratio']:.1%})")
    print(f"  Pseudo labels generated: {stats['pseudo_generated']:,}")
    print(f"  Improvement: +{stats['improvement']:.1%}")

    print(f"\nPseudo-label distribution:")
    for i in range(6):
        orig_count = (region_labels == i).sum()
        pseudo_count = (pseudo_labels == i).sum()
        print(f"  Label {i}: {orig_count:,} → {pseudo_count:,}")
