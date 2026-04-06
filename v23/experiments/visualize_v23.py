# -*- coding: utf-8 -*-
"""
V2.3 空间拓扑编码器 - 可视化脚本

生成科研性质的图表：
1. 模型架构流程图
2. 训练曲线图（Loss、Silhouette）
3. 三区域对比图
4. 版本对比图（V1/V6/V61/V2.3）
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'new computer modern'
plt.rcParams['mathtext.fontset'] = 'cm'


def plot_model_architecture(output_dir: str):
    """
    绘制V2.3模型架构流程图
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 颜色定义
    colors = {
        'input': '#E3F2FD',
        'embedding': '#BBDEFB',
        'encoder': '#90CAF9',
        'output': '#64B5F6',
        'loss': '#FFCDD2',
    }

    # 输入层
    input_box = FancyBboxPatch((0.5, 5), 2, 2, boxstyle="round,pad=0.1",
                                facecolor=colors['input'], edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.5, 6, '输入层\n\nPOI特征 (7维)\n坐标 (2维)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Embedding层
    emb_boxes = [
        ('Category\nEmb', (3.5, 7)),
        ('Landuse\nEmb', (3.5, 5.5)),
        ('AOI Type\nEmb', (3.5, 4)),
        ('Road Class\nEmb', (3.5, 2.5)),
        ('Numerical\nProj', (5.5, 7)),
        ('Coord\nProj', (5.5, 4)),
    ]

    for i, (text, pos) in enumerate(emb_boxes):
        box = FancyBboxPatch((pos[0]-0.8, pos[1]-0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['embedding'], edgecolor='black')
        ax.add_patch(box)
        ax.text(pos[0], pos[1], text, ha='center', va='center', fontsize=8)

    # 编码器层
    encoder_box = FancyBboxPatch((7.5, 3.5), 2.5, 3, boxstyle="round,pad=0.1",
                                  facecolor=colors['encoder'], edgecolor='black', linewidth=1.5)
    ax.add_patch(encoder_box)
    ax.text(8.75, 5.5, 'MLP 编码器\n\nLinear(128→128)\nLayerNorm + GELU\nDropout\nLinear(128→64)',
            ha='center', va='center', fontsize=9)

    # 输出层
    output_box = FancyBboxPatch((11, 4.5), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(12, 5.25, '空间Embedding\n(64维, L2归一化)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # 坐标解码器
    decoder_box = FancyBboxPatch((11, 2), 2, 1.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['loss'], edgecolor='black', linewidth=1.5)
    ax.add_patch(decoder_box)
    ax.text(12, 2.75, '坐标解码器\n重构坐标 (2维)',
            ha='center', va='center', fontsize=9)

    # 箭头连接
    arrows = [
        ((2.5, 6), (2.7, 7)),      # 输入到Category
        ((2.5, 6), (2.7, 5.5)),    # 输入到Landuse
        ((2.5, 6), (2.7, 4)),      # 输入到AOI
        ((2.5, 6), (2.7, 2.5)),    # 输入到Road
        ((2.5, 5), (4.7, 7)),      # 输入到Numerical
        ((2.5, 5), (4.7, 4)),      # 输入到Coord
        ((6.3, 4.75), (7.5, 5)),   # Embedding到Encoder
        ((10, 5.25), (11, 5.25)),  # Encoder到Output
        ((12, 4.5), (12, 3.5)),    # Output到Decoder
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 标题
    ax.text(7, 0.8, 'V2.3 空间拓扑编码器架构', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / 'model_architecture_v23.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"模型架构图已保存: {output_path}")


def plot_training_curves(history: list, output_dir: str, title: str = "训练曲线"):
    """
    绘制训练曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    sils = [h['silhouette'] for h in history]

    # Loss曲线
    axes[0].plot(epochs, losses, 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('损失函数变化', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Silhouette曲线
    axes[1].plot(epochs, sils, 'g-', linewidth=2, label='Silhouette')
    axes[1].axhline(y=0.35, color='r', linestyle='--', label='目标线 (0.35)')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('聚类质量变化', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f'training_curves_{timestamp}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"训练曲线图已保存: {output_path}")


def plot_multi_area_comparison(results: dict, output_dir: str):
    """
    绘制三区域对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    areas = list(results.keys())
    silhouettes = [results[a]['silhouette'] for a in areas]
    upper_bounds = [results[a]['upper_bound'] for a in areas]
    rates = [results[a]['silhouette'] / results[a]['upper_bound'] * 100 for a in areas]

    x = np.arange(len(areas))
    width = 0.35

    # Silhouette对比
    bars1 = axes[0].bar(x - width/2, silhouettes, width, label='实际值', color='#2196F3')
    bars2 = axes[0].bar(x + width/2, upper_bounds, width, label='理论上限', color='#4CAF50')

    axes[0].set_xlabel('区域', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette得分对比', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(areas, rotation=15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 达成率
    bars3 = axes[1].bar(x, rates, width=0.5, color='#FF9800')

    axes[1].set_xlabel('区域', fontsize=11)
    axes[1].set_ylabel('达成率 (%)', fontsize=11)
    axes[1].set_title('理论上限达成率', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(areas, rotation=15)
    axes[1].axhline(y=90, color='r', linestyle='--', label='90%基准线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar in bars3:
        height = bar.get_height()
        axes[1].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('V2.3 三区域实验结果', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / 'multi_area_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"三区域对比图已保存: {output_path}")


def plot_version_comparison(output_dir: str):
    """
    绘制版本对比图 (V1/V6/V61/V2.3)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    versions = ['V1\n(防作弊)', 'V6\n(Triplet)', 'V61\n(Memory Bank)', 'V2.3\n(重构)']
    silhouettes = [0.01, 0.80, 0.98, 0.40]
    params = [1.5, 1.5, 1.2, 0.047]  # 参数量(M)
    status = ['失败', '作弊', '作弊', '成功']
    colors = ['#F44336', '#FF9800', '#FF9800', '#4CAF50']

    # Silhouette对比
    bars1 = axes[0].bar(versions, silhouettes, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('版本', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('聚类质量对比', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar, sil, st in zip(bars1, silhouettes, status):
        height = bar.get_height()
        axes[0].annotate(f'{sil:.2f}\n({st})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # 参数量对比
    bars2 = axes[1].bar(versions, params, color=colors, edgecolor='black', linewidth=1)
    axes[1].set_xlabel('版本', fontsize=11)
    axes[1].set_ylabel('参数量 (M)', fontsize=11)
    axes[1].set_title('模型复杂度对比', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')

    for bar, p in zip(bars2, params):
        height = bar.get_height()
        label = f'{p:.3f}M' if p < 0.1 else f'{p:.1f}M'
        axes[1].annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('各版本模型对比分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / 'version_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"版本对比图已保存: {output_path}")


def plot_loss_function_diagram(output_dir: str):
    """
    绘制损失函数组合示意图
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 颜色
    colors = {
        'coord': '#E8F5E9',
        'dist': '#E3F2FD',
        'neighbor': '#FFF3E0',
        'total': '#FCE4EC',
    }

    # 三个损失分支
    losses = [
        ('坐标重构损失\nλ₁=1.0', (1, 3), colors['coord'], 'MSE(重构坐标, 真实坐标)'),
        ('距离保持损失\nλ₂=2.0', (5, 3), colors['dist'], 'MSE(emb距离, 空间距离)'),
        ('邻居一致性损失\nλ₃=1.0', (9, 3), colors['neighbor'], '1 - cos(邻居emb)'),
    ]

    for text, pos, color, formula in losses:
        box = FancyBboxPatch((pos[0]-1.5, pos[1]-1), 3, 2,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1]+0.3, text, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(pos[0], pos[1]-0.5, formula, ha='center', va='center', fontsize=8, style='italic')

    # 总损失
    total_box = FancyBboxPatch((4, 0.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['total'], edgecolor='black', linewidth=2)
    ax.add_patch(total_box)
    ax.text(6, 1.25, '总损失 = λ₁L₁ + λ₂L₂ + λ₃L₃', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # 连接线
    for pos in [(1, 3), (5, 3), (9, 3)]:
        ax.plot([pos[0], pos[0]], [pos[1]-1, 2], 'k-', linewidth=1)
    ax.plot([1, 9], [2, 2], 'k-', linewidth=1)
    ax.plot([6, 6], [2, 2], 'k-', linewidth=1)
    ax.plot([6, 6], [2, 2], 'k-', linewidth=1)
    ax.annotate('', xy=(6, 2), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 标题
    ax.text(6, 5.5, 'V2.3 损失函数组合', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / 'loss_function_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"损失函数图已保存: {output_path}")


def generate_all_plots(results_dir: str = None):
    """生成所有图表"""
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("生成V2.3可视化图表")
    print("=" * 50)

    # 1. 模型架构图
    plot_model_architecture(output_dir)

    # 2. 损失函数图
    plot_loss_function_diagram(output_dir)

    # 3. 版本对比图
    plot_version_comparison(output_dir)

    # 4. 三区域对比图（如果有结果）
    if results_dir:
        results_path = Path(results_dir)
        json_files = list(results_path.glob('v23_results_*.json'))
        if json_files:
            with open(json_files[-1], 'r', encoding='utf-8') as f:
                results = json.load(f)
            plot_multi_area_comparison(results, output_dir)

            # 训练曲线
            for area, data in results.items():
                if 'history' in data:
                    plot_training_curves(data['history'], output_dir, f"训练曲线 - {area}")

    print("\n所有图表已生成完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    generate_all_plots(str(Path(__file__).resolve().parent / "results"))
