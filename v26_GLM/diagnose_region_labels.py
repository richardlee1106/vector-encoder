# -*- coding: utf-8 -*-
"""
P0.5-DiagA: 功能区标签覆盖率诊断脚本

目标：
1. 统计 region_labels 各类别的样本数量和比例
2. 计算"未知"类别的比例
3. 输出清晰的统计表格

Author: GLM (Qianfan Code)
Date: 2026-03-17
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from collections import Counter


# 功能区类型名称映射（与 data_loader_v26.py 一致）
AOI_TYPE_NAMES = {
    0: "居住区",
    1: "商业区",
    2: "零售区",
    3: "工业区",
    4: "大学",
    5: "购物中心",
    6: "集市",
    7: "公园",
    8: "学院",
    9: "学校",
    10: "医院",
    11: "森林",
    12: "超市",
    13: "停车场",
    14: "水域",
    15: "未知",
}

# P1B: 合并后的类别名称
MERGED_TYPE_NAMES = {
    0: "居住类",
    1: "商业类",
    2: "工业类",
    3: "教育类",
    4: "公共类",
    5: "自然类",
    6: "未知",
}

# P1B: 合并映射
REGION_MERGE_MAP = {
    0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 1, 7: 4,
    8: 3, 9: 3, 10: 4, 11: 5, 12: 1, 13: 4, 14: 5, 15: 6,
}


def diagnose_region_labels(region_labels: np.ndarray, use_merged: bool = True) -> dict:
    """
    诊断 region_labels 的分布

    Args:
        region_labels: 功能区标签数组 [N]
        use_merged: 是否显示合并后的类别（P1B）。如果标签已合并则自动检测

    Returns:
        诊断结果字典
    """
    total = len(region_labels)

    # 检测标签是否已合并（最大值 <= 6 表示已合并）
    is_already_merged = region_labels.max() <= 6

    if use_merged and not is_already_merged:
        # 需要应用合并
        merged_labels = np.zeros_like(region_labels)
        for old_id, new_id in REGION_MERGE_MAP.items():
            merged_labels[region_labels == old_id] = new_id
        region_labels = merged_labels

    # 根据标签范围确定类别名称
    if is_already_merged or use_merged:
        type_names = MERGED_TYPE_NAMES
        unknown_label = 6  # 合并后未知类为6
    else:
        type_names = AOI_TYPE_NAMES
        unknown_label = 15  # 原始未知类为15

    counts = Counter(region_labels)

    # 统计各类别
    stats = []
    for label in sorted(counts.keys()):
        count = counts[label]
        pct = count / total * 100
        name = type_names.get(label, f"Unknown({label})")
        stats.append({
            "label": label,
            "name": name,
            "count": count,
            "pct": pct,
        })

    # 已知标签统计
    known_count = (region_labels < unknown_label).sum()
    known_pct = known_count / total * 100

    # 未知标签统计
    unknown_count = (region_labels == unknown_label).sum()
    unknown_pct = unknown_count / total * 100

    # 有效标签统计
    valid_labels = region_labels[region_labels < unknown_label]
    valid_unique = len(np.unique(valid_labels)) if len(valid_labels) > 0 else 0

    return {
        "total": total,
        "stats": stats,
        "known_count": known_count,
        "known_pct": known_pct,
        "unknown_count": unknown_count,
        "unknown_pct": unknown_pct,
        "valid_unique": valid_unique,
        "num_classes": unknown_label,  # 类别数（不含未知）
    }


def print_diagnosis(result: dict, use_merged: bool = True) -> None:
    """打印诊断结果"""
    # 使用 ASCII 符号避免编码问题
    print("\n" + "=" * 60)
    if use_merged:
        print("Region Label Coverage Diagnosis Report (P1B Merged)")
    else:
        print("Region Label Coverage Diagnosis Report")
    print("=" * 60)

    print(f"\n总样本数: {result['total']:,}")

    print("\n┌─────────┬────────────┬──────────┬──────────┐")
    print("│ 标签ID  │ 类型名称   │ 样本数   │ 占比(%)  │")
    print("├─────────┼────────────┼──────────┼──────────┤")

    for s in result["stats"]:
        # 格式化输出，对齐列
        label_str = f"{s['label']:>7}"
        name_str = f"{s['name']:<10}"
        count_str = f"{s['count']:>8,}"
        pct_str = f"{s['pct']:>8.2f}"
        print(f"│ {label_str} │ {name_str} │ {count_str} │ {pct_str} │")

    print("└─────────┴────────────┴──────────┴──────────┘")

    print("\n" + "-" * 60)
    print("[Key Metrics]")
    print("-" * 60)

    # 已知标签统计
    print(f"\nValid Label Coverage:")
    print(f"  Known labels: {result['known_count']:,} ({result['known_pct']:.2f}%)")
    print(f"  Unknown labels: {result['unknown_count']:,} ({result['unknown_pct']:.2f}%)")
    print(f"  Known categories: {result['valid_unique']} / 15")

    # 质量评估
    print("\n[Quality Assessment]")

    if result['known_pct'] >= 50:
        quality = "[OK] Good - Sufficient coverage for contrastive learning"
    elif result['known_pct'] >= 20:
        quality = "[WARN] Moderate - Consider label propagation"
    elif result['known_pct'] >= 5:
        quality = "[WARN] Low - Need label propagation + pseudo-labels"
    else:
        quality = "[CRIT] Very Low - Recommend re-annotation or merge categories"

    print(f"  Label coverage: {result['known_pct']:.2f}%")
    print(f"  Assessment: {quality}")

    # 类别均衡性检查
    print("\n[Class Balance]")
    num_classes = result.get("num_classes", 6 if use_merged else 15)
    valid_stats = [s for s in result["stats"] if s["label"] < num_classes and s["count"] > 0]

    if valid_stats:
        counts = [s["count"] for s in valid_stats]
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / max(min_count, 1)

        print(f"  Max class count: {max_count:,}")
        print(f"  Min class count: {min_count:,}")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")

        if imbalance_ratio > 100:
            print(f"  Assessment: [WARN] Severely imbalanced - merge classes or use class weights")
        elif imbalance_ratio > 10:
            print(f"  Assessment: [WARN] Imbalanced - use class weights or oversampling")
        else:
            print(f"  Assessment: [OK] Relatively balanced")

    print("\n" + "=" * 60)

    # 决策建议
    print("\n[P1 Strategy Recommendations]")
    print("-" * 60)

    if use_merged:
        # P1B合并后的策略
        print("P1B COMPLETED - Classes merged 16->6")
        print("")
        print("NEXT: P1C Enable contrastive learning")
        print("  1. Set region_weight = 0.3 in config_v26_pro.py")
        print("  2. Use class weights: {0: 0.3, 1: 1.5, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}")
        print("  3. Run 10% data validation first")
        print("  4. Expected Region F1: 30-45% (after P1C)")
    else:
        if result['known_pct'] >= 50:
            print("1. Enable RegionContrastiveLoss directly (region_weight=0.5~1.0)")
            print("2. Use class weights to balance imbalance")
            print("3. Expected Region F1: 35-50%")
        elif result['known_pct'] >= 20:
            print("1. First use label propagation to expand labeled samples")
            print("2. Then enable RegionContrastiveLoss")
            print("3. Expected Region F1: 30-45%")
        else:
            print("1. Need to merge small classes (e.g. 16 classes -> 5-6 major classes)")
            print("2. Use semi-supervised learning + pseudo-label strategy")
            print("3. Consider external region data augmentation")

    print("=" * 60 + "\n")


def main(sample_ratio: float = 1.0):
    """主函数"""
    print("P0.5-DiagA: Region Label Coverage Diagnosis")
    print(f"Sample ratio: {sample_ratio * 100:.0f}%")

    try:
        # 尝试从数据库加载真实数据
        from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
        from spatial_encoder.v26_GLM.config_v26 import DEFAULT_CONFIG

        print("\nLoading data from database...")
        data = load_dataset_for_training(config=DEFAULT_CONFIG, sample_ratio=sample_ratio)
        region_labels = data["region_labels"]
        print(f"Data loaded! {len(region_labels):,} cells")

        # 检测是否已合并（最大值 <= 6）
        is_merged = region_labels.max() <= 6
        if is_merged:
            print("Labels already merged (P1B applied in data_loader)")
        else:
            print("Original labels detected (not merged)")

    except Exception as e:
        print(f"\nCannot load from database: {e}")
        print("\nUsing mock data for demo...")

        # 使用模拟数据
        np.random.seed(42)
        n = 67138
        # 模拟已合并的标签分布
        labels = np.random.choice(
            list(range(7)),  # 0-6
            size=n,
            p=[0.09, 0.01, 0.03, 0.01, 0.02, 0.02, 0.82]  # 模拟合并后的分布
        )
        region_labels = labels
        is_merged = True

    # 执行诊断（如果是已合并的标签，use_merged=True 会正确显示）
    result = diagnose_region_labels(region_labels, use_merged=is_merged)
    print_diagnosis(result, use_merged=is_merged)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="功能区标签覆盖率诊断")
    parser.add_argument("--sample", type=float, default=1.0, help="采样比例 (0-1)")
    args = parser.parse_args()

    main(args.sample)
