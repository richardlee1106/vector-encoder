# -*- coding: utf-8 -*-
"""
GPU显存测试脚本 - 找出最优配置

目标：在8GB显存限制下，最大化GPU利用率
测试变量：batch_size, hidden_dim, embedding_dim, K_neighbors

Author: Claude
Date: 2026-03-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc

# 显存限制 (GB)
VRAM_LIMIT = 7.5  # 留0.5GB余量


def get_memory_mb():
    """获取当前显存使用(MB)"""
    return torch.cuda.max_memory_allocated() / 1024**2


def clear_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class SimpleEncoder(nn.Module):
    """简化编码器用于测试"""
    def __init__(self, input_dim=72, hidden_dim=512, embed_dim=256, num_layers=6):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def test_config(batch_size, hidden_dim, embed_dim, num_layers, k_neighbors):
    """
    测试单个配置

    Returns:
        dict: 包含显存使用、速度等信息
    """
    clear_memory()

    try:
        # 创建模型
        model = SimpleEncoder(
            input_dim=72,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_layers=num_layers
        ).cuda()

        # 参数量
        params = sum(p.numel() for p in model.parameters())

        # 输入数据
        x = torch.randn(batch_size, 72, device='cuda')
        coords = torch.randn(batch_size, 2, device='cuda')
        knn_idx = torch.randint(0, batch_size, (batch_size, k_neighbors), device='cuda')

        # 预热
        for _ in range(3):
            emb = model(x)
            neighbor_emb = emb[knn_idx]
            pred_dist = torch.norm(emb.unsqueeze(1) - neighbor_emb, p=2, dim=-1)
            neighbor_coords = coords[knn_idx]
            true_dist = torch.norm(coords.unsqueeze(1) - neighbor_coords, p=2, dim=-1)
            loss = ((pred_dist - true_dist) ** 2).mean()
            loss.backward()

        clear_memory()

        # 正式测试
        start = time.time()
        for _ in range(10):
            emb = model(x)
            neighbor_emb = emb[knn_idx]
            pred_dist = torch.norm(emb.unsqueeze(1) - neighbor_emb, p=2, dim=-1)
            neighbor_coords = coords[knn_idx]
            true_dist = torch.norm(coords.unsqueeze(1) - neighbor_coords, p=2, dim=-1)
            loss = ((pred_dist - true_dist) ** 2).mean()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        mem_mb = get_memory_mb()
        mem_gb = mem_mb / 1024

        # 检查是否安全
        safe = mem_gb < VRAM_LIMIT

        # 清理
        del model, x, coords, knn_idx, emb, neighbor_emb, pred_dist, neighbor_coords, true_dist, loss
        clear_memory()

        return {
            'success': True,
            'params_m': params / 1e6,
            'mem_gb': mem_gb,
            'safe': safe,
            'ms_per_iter': elapsed / 10 * 1000,
            'samples_per_sec': batch_size * 10 / elapsed,
            'utilization': mem_gb / 8.0 * 100,
        }

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            clear_memory()
            return {
                'success': False,
                'error': 'OOM',
            }
        raise


def main():
    print("=" * 70)
    print("GPU显存测试 - 寻找最优配置")
    print("=" * 70)
    print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"限制: {VRAM_LIMIT} GB")
    print()

    # 测试配置网格
    # 关键约束：K近邻张量 [batch, K, embed_dim] 的显存占用
    # 估算: batch * K * embed_dim * 4 bytes (float32)

    configs = []

    # 直接遍历所有组合，不过滤
    for hidden_dim in [512, 640, 768]:
        for embed_dim in [256, 320, 384]:
            for num_layers in [6, 8, 10]:
                for k in [32, 48, 64]:
                    for batch in [4096, 8192, 12288, 16384]:
                        configs.append({
                            'batch': batch,
                            'hidden': hidden_dim,
                            'embed': embed_dim,
                            'layers': num_layers,
                            'k': k,
                        })

    print(f"测试配置数量: {len(configs)}")
    print()

    # 按预估显存排序
    configs.sort(key=lambda c: c['batch'] * c['k'] * c['embed'], reverse=True)

    results = []

    print(f"{'Batch':>8} {'Hidden':>8} {'Embed':>8} {'Layers':>8} {'K':>4} {'Params':>8} {'Mem':>8} {'Util':>6} {'Status':>8}")
    print("-" * 90)

    for cfg in configs:
        result = test_config(
            batch_size=cfg['batch'],
            hidden_dim=cfg['hidden'],
            embed_dim=cfg['embed'],
            num_layers=cfg['layers'],
            k_neighbors=cfg['k'],
        )

        if result['success']:
            status = 'OK' if result['safe'] else 'WARN'
            print(f"{cfg['batch']:>8} {cfg['hidden']:>8} {cfg['embed']:>8} {cfg['layers']:>8} {cfg['k']:>4} "
                  f"{result['params_m']:>7.1f}M {result['mem_gb']:>7.2f}G {result['utilization']:>5.1f}% [{status}]")

            results.append({**cfg, **result})
        else:
            print(f"{cfg['batch']:>8} {cfg['hidden']:>8} {cfg['embed']:>8} {cfg['layers']:>8} {cfg['k']:>4} OOM")

    # 找出最优配置
    print()
    print("=" * 70)
    print("最优配置推荐 (GPU利用率 85-95%)")
    print("=" * 70)

    # 筛选安全且利用率高的配置
    good_configs = [r for r in results if r['safe'] and r['utilization'] >= 80]

    if good_configs:
        # 按参数量排序，找最大模型
        good_configs.sort(key=lambda x: x['params_m'], reverse=True)

        best = good_configs[0]
        print(f"\n推荐配置:")
        print(f"  batch_size: {best['batch']}")
        print(f"  hidden_dim: {best['hidden']}")
        print(f"  embedding_dim: {best['embed']}")
        print(f"  num_layers: {best['layers']}")
        print(f"  K_neighbors: {best['k']}")
        print(f"\n性能指标:")
        print(f"  参数量: {best['params_m']:.1f}M")
        print(f"  显存占用: {best['mem_gb']:.2f} GB")
        print(f"  GPU利用率: {best['utilization']:.1f}%")
        print(f"  速度: {best['ms_per_iter']:.1f} ms/iter")
        print(f"  吞吐量: {best['samples_per_sec']:.0f} samples/s")
    else:
        print("未找到合适的配置，尝试降低参数...")
        # 找利用率最高的安全配置
        safe_configs = [r for r in results if r['safe']]
        if safe_configs:
            safe_configs.sort(key=lambda x: x['utilization'], reverse=True)
            best = safe_configs[0]
            print(f"\n降级推荐:")
            print(f"  batch_size: {best['batch']}")
            print(f"  hidden_dim: {best['hidden']}")
            print(f"  embedding_dim: {best['embed']}")
            print(f"  num_layers: {best['layers']}")
            print(f"  K_neighbors: {best['k']}")
            print(f"  显存: {best['mem_gb']:.2f} GB ({best['utilization']:.1f}%)")


if __name__ == "__main__":
    main()
