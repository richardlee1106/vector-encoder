# -*- coding: utf-8 -*-
"""
Phase 2: Weight Tuning Experiments

Compare different semantic/spatial weight configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import json
from collections import Counter
import time

from spatial_encoder.v26_GLM.data_loader_v26 import load_dataset_for_training
from spatial_encoder.v26_GLM.config_v26_pro import DEFAULT_PRO_CONFIG
from spatial_encoder.v26_GLM.encoder_v26_mlp import build_mlp_encoder
from spatial_encoder.v26_GLM.hybrid_search import HybridSearchEngine


def main():
    print('='*70)
    print('Phase 2: Weight Tuning Experiments')
    print('='*70)

    # Load data
    print('\nLoading data...')
    data = load_dataset_for_training(config=DEFAULT_PRO_CONFIG, sample_ratio=0.1)
    n = len(data['coords'])
    print(f'  Total POIs: {n}')

    region_labels = data['region_labels']
    region_names = ['Residential', 'Commercial', 'Industrial', 'Education', 'Public', 'Nature', 'Unknown']

    # Load model
    print('\nLoading model...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_mlp_encoder(DEFAULT_PRO_CONFIG)
    ckpt = torch.load(Path(__file__).parent / 'saved_models' / 'v26_pro' / 'best_model.pt', map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Compute embeddings
    print('Computing embeddings...')
    with torch.no_grad():
        point_feat = torch.tensor(data['point_features'], dtype=torch.float32).to(device)
        line_feat = torch.tensor(data['line_features'], dtype=torch.float32).to(device)
        polygon_feat = torch.tensor(data['polygon_features'], dtype=torch.float32).to(device)
        direction_feat = torch.tensor(data['direction_features'], dtype=torch.float32).to(device)
        embeddings, _, _, _ = model(point_feat, line_feat, polygon_feat, direction_feat)
        embeddings = embeddings.cpu().numpy()

    coords = data['coords']

    # Build search engine
    print('Building search engine...')
    engine = HybridSearchEngine(embeddings, coords, n_candidates=100)

    print('\n' + '='*70)
    print('Experiment Configuration')
    print('='*70)

    # Weight configurations
    weight_configs = [
        {'name': 'Pure Semantic', 'semantic': 1.0, 'spatial': 0.0},
        {'name': 'Semantic Priority', 'semantic': 0.7, 'spatial': 0.3},
        {'name': 'Balanced', 'semantic': 0.5, 'spatial': 0.5},
        {'name': 'Spatial Priority', 'semantic': 0.3, 'spatial': 0.7},
        {'name': 'Pure Spatial', 'semantic': 0.0, 'spatial': 1.0},
    ]

    print('\nWeight configurations:')
    for cfg in weight_configs:
        print(f'  {cfg["name"]}: semantic={cfg["semantic"]}, spatial={cfg["spatial"]}')

    # Test parameters
    radius = 5000  # 5km
    k = 10
    n_test = 200  # samples per class

    print(f'\nTest parameters:')
    print(f'  radius: {radius}m')
    print(f'  k: {k}')
    print(f'  samples per class: {n_test}')

    print('\n' + '='*70)
    print('Running Experiments')
    print('='*70)

    np.random.seed(42)

    results = []

    for cfg in weight_configs:
        print(f'\n--- {cfg["name"]} (semantic={cfg["semantic"]}, spatial={cfg["spatial"]}) ---')
        start_time = time.time()

        # Metrics
        intra_class_recalls = []
        spatial_precisions = []
        query_times = []

        for class_id in range(6):
            class_indices = np.where(region_labels == class_id)[0]
            if len(class_indices) < 50:
                continue

            test_indices = np.random.choice(class_indices, min(n_test, len(class_indices)), replace=False)

            class_recalls = []
            class_precisions = []

            for idx in test_indices:
                query_start = time.time()

                results_k = engine.search(
                    query_embedding=embeddings[idx],
                    query_coords=coords[idx],
                    k=k,
                    radius=radius,
                    semantic_weight=cfg['semantic'],
                    spatial_weight=cfg['spatial'],
                )

                query_time = (time.time() - query_start) * 1000
                query_times.append(query_time)

                if len(results_k) == 0:
                    continue

                # Intra-class recall
                result_labels = [region_labels[r['poi_index']] for r in results_k if r['poi_index'] != idx]
                if result_labels:
                    same_class = sum(1 for l in result_labels if l == class_id)
                    recall = same_class / len(result_labels)
                    class_recalls.append(recall)

                # Spatial precision (within radius)
                within_radius = sum(1 for r in results_k if r['distance_m'] < radius)
                if results_k:
                    precision = within_radius / len(results_k)
                    class_precisions.append(precision)

            if class_recalls:
                avg_recall = np.mean(class_recalls) * 100
                intra_class_recalls.append(avg_recall)
                print(f'    {region_names[class_id]}: Recall={avg_recall:.1f}%')

            if class_precisions:
                avg_precision = np.mean(class_precisions) * 100
                spatial_precisions.append(avg_precision)

        elapsed = time.time() - start_time
        avg_recall = np.mean(intra_class_recalls) if intra_class_recalls else 0
        avg_precision = np.mean(spatial_precisions) if spatial_precisions else 0
        avg_query_time = np.mean(query_times) if query_times else 0

        print(f'  Average: Recall={avg_recall:.1f}%, Precision={avg_precision:.1f}%, Time={avg_query_time:.2f}ms')

        results.append({
            'config': cfg,
            'intra_class_recall': avg_recall,
            'spatial_precision': avg_precision,
            'query_time_ms': avg_query_time,
            'elapsed_s': elapsed,
        })

    print('\n' + '='*70)
    print('RESULTS SUMMARY')
    print('='*70)

    print('\n| Config | Semantic | Spatial | Intra-class Recall | Spatial Precision | Query Time |')
    print('|--------|----------|---------|-------------------|-------------------|------------|')
    for r in results:
        cfg = r['config']
        print(f'| {cfg["name"]:<20} | {cfg["semantic"]:.1f} | {cfg["spatial"]:.1f} | {r["intra_class_recall"]:.1f}% | {r["spatial_precision"]:.1f}% | {r["query_time_ms"]:.2f}ms |')

    print('\n' + '='*70)
    print('ANALYSIS')
    print('='*70)

    # Find best config
    best_recall = max(results, key=lambda x: x['intra_class_recall'])
    best_precision = max(results, key=lambda x: x['spatial_precision'])

    print(f'\nBest Intra-class Recall: {best_recall["config"]["name"]} ({best_recall["intra_class_recall"]:.1f}%)')
    print(f'Best Spatial Precision: {best_precision["config"]["name"]} ({best_precision["spatial_precision"]:.1f}%)')

    # Trade-off analysis
    print('\nTrade-off Analysis:')
    print('  As semantic weight increases:')
    print(f'    - Recall: {results[-1]["intra_class_recall"]:.1f}% -> {results[0]["intra_class_recall"]:.1f}%')
    print(f'    - Precision: {results[-1]["spatial_precision"]:.1f}% -> {results[0]["spatial_precision"]:.1f}%')

    print('\n' + '='*70)
    print('RECOMMENDATION')
    print('='*70)

    # Calculate F1-like score
    for r in results:
        r['f1'] = 2 * r['intra_class_recall'] * r['spatial_precision'] / (r['intra_class_recall'] + r['spatial_precision'] + 1e-8)

    best_f1 = max(results, key=lambda x: x['f1'])
    print(f'\nBest F1 (Recall * Precision): {best_f1["config"]["name"]}')
    print(f'  Semantic Weight: {best_f1["config"]["semantic"]}')
    print(f'  Spatial Weight: {best_f1["config"]["spatial"]}')
    print(f'  Intra-class Recall: {best_f1["intra_class_recall"]:.1f}%')
    print(f'  Spatial Precision: {best_f1["spatial_precision"]:.1f}%')
    print(f'  F1 Score: {best_f1["f1"]:.1f}')

    print('\nRecommended Default: Semantic Priority (0.7, 0.3)')
    print('  Rationale: Good balance of semantic quality and spatial relevance')

    # Save results
    output = {
        'experiments': [
            {
                'name': r['config']['name'],
                'semantic_weight': r['config']['semantic'],
                'spatial_weight': r['config']['spatial'],
                'intra_class_recall': r['intra_class_recall'],
                'spatial_precision': r['spatial_precision'],
                'query_time_ms': r['query_time_ms'],
                'f1': r['f1'],
            }
            for r in results
        ],
        'best_f1': best_f1['config']['name'],
        'recommendation': 'Semantic Priority (0.7, 0.3)',
    }

    output_path = Path(__file__).parent / 'weight_tuning_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
