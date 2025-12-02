"""
Validate and analyze discovered semantic axes.

This script performs comprehensive validation and analysis of axes
discovered by cluster_antonym_axes.py, including:
- Quantitative metrics (coherence, separation, coverage)
- Qualitative analysis (axis naming, interpretability)
- Comparison to manual seed clusters
- Visualizations (t-SNE, axis distributions)

Usage:
    python scripts/validate_discovered_axes.py [options]

Author: NAOMI-II Development Team
Date: 2025-11-30
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


def load_results(results_dir: str) -> Dict:
    """Load clustering results from disk."""
    print(f"Loading results from {results_dir}...")

    # Load axes
    axes_path = os.path.join(results_dir, 'semantic_axes.json')
    with open(axes_path, 'r', encoding='utf-8') as f:
        axes_data = json.load(f)

    # Load statistics
    stats_path = os.path.join(results_dir, 'clustering_stats.json')
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    # Load similarity matrix
    sim_path = os.path.join(results_dir, 'similarity_matrix.npy')
    similarity_matrix = np.load(sim_path)

    print(f"  [OK] Loaded {len(axes_data['axes'])} axes")
    print(f"  [OK] Loaded similarity matrix ({similarity_matrix.shape})")

    return {
        'axes': axes_data['axes'],
        'metadata': axes_data['metadata'],
        'stats': stats,
        'similarity_matrix': similarity_matrix
    }


def compute_validation_metrics(results: Dict) -> Dict:
    """Compute comprehensive validation metrics."""
    print("\nComputing validation metrics...")

    axes = results['axes']
    stats = results['stats']

    metrics = {
        'n_axes': len(axes),
        'n_total_pairs': stats['total_antonym_pairs'],
        'n_clusters': stats['n_clusters'],
        'n_singletons': stats['n_singletons'],
        'singleton_rate': stats['singleton_rate'],
        'silhouette_score': stats.get('silhouette_score'),

        # Axis size distribution
        'axis_size_min': min(a['size'] for a in axes) if axes else 0,
        'axis_size_max': max(a['size'] for a in axes) if axes else 0,
        'axis_size_mean': np.mean([a['size'] for a in axes]) if axes else 0,
        'axis_size_median': np.median([a['size'] for a in axes]) if axes else 0,

        # Coherence distribution
        'coherence_min': min(a['coherence_score'] for a in axes) if axes else 0,
        'coherence_max': max(a['coherence_score'] for a in axes) if axes else 0,
        'coherence_mean': np.mean([a['coherence_score'] for a in axes]) if axes else 0,
        'coherence_median': np.median([a['coherence_score'] for a in axes]) if axes else 0,

        # Separation distribution
        'separation_min': min(a['separation_score'] for a in axes) if axes else 0,
        'separation_max': max(a['separation_score'] for a in axes) if axes else 0,
        'separation_mean': np.mean([a['separation_score'] for a in axes]) if axes else 0,
        'separation_median': np.median([a['separation_score'] for a in axes]) if axes else 0,

        # Coverage
        'pairs_in_axes': sum(a['size'] for a in axes),
        'coverage_rate': sum(a['size'] for a in axes) / stats['total_antonym_pairs']
                         if stats['total_antonym_pairs'] > 0 else 0,
    }

    # Axis name diversity
    axis_names = [a['name'] for a in axes]
    name_counts = Counter(axis_names)
    metrics['unique_axis_names'] = len(name_counts)
    metrics['axis_name_diversity'] = len(name_counts) / len(axes) if axes else 0

    # Most common axis names (may indicate naming issues)
    metrics['most_common_names'] = name_counts.most_common(10)

    return metrics


def print_validation_report(metrics: Dict, axes: List[Dict]):
    """Print comprehensive validation report."""
    print("\n" + "=" * 100)
    print("VALIDATION REPORT")
    print("=" * 100)

    print("\n--- CLUSTERING QUALITY ---")
    print(f"Silhouette Score:      {metrics['silhouette_score']:.4f}")
    print(f"Total Clusters:        {metrics['n_clusters']}")
    print(f"Valid Axes:            {metrics['n_axes']}")
    print(f"Singletons:            {metrics['n_singletons']} ({metrics['singleton_rate']:.1%})")

    print("\n--- COVERAGE ---")
    print(f"Total Antonym Pairs:   {metrics['n_total_pairs']}")
    print(f"Pairs in Axes:         {metrics['pairs_in_axes']}")
    print(f"Coverage Rate:         {metrics['coverage_rate']:.1%}")

    print("\n--- AXIS SIZE DISTRIBUTION ---")
    print(f"Min:                   {metrics['axis_size_min']}")
    print(f"Max:                   {metrics['axis_size_max']}")
    print(f"Mean:                  {metrics['axis_size_mean']:.1f}")
    print(f"Median:                {metrics['axis_size_median']:.1f}")

    print("\n--- COHERENCE DISTRIBUTION ---")
    print(f"Min:                   {metrics['coherence_min']:.4f}")
    print(f"Max:                   {metrics['coherence_max']:.4f}")
    print(f"Mean:                  {metrics['coherence_mean']:.4f}")
    print(f"Median:                {metrics['coherence_median']:.4f}")

    print("\n--- SEPARATION DISTRIBUTION ---")
    print(f"Min:                   {metrics['separation_min']:.4f}")
    print(f"Max:                   {metrics['separation_max']:.4f}")
    print(f"Mean:                  {metrics['separation_mean']:.4f}")
    print(f"Median:                {metrics['separation_median']:.4f}")

    print("\n--- AXIS NAMING ---")
    print(f"Unique Names:          {metrics['unique_axis_names']}")
    print(f"Name Diversity:        {metrics['axis_name_diversity']:.1%}")
    print(f"\nMost Common Names:")
    for name, count in metrics['most_common_names']:
        print(f"  {name:<20} ({count} axes)")

    print("\n--- TOP 20 AXES BY SIZE ---")
    top_axes = sorted(axes, key=lambda x: x['size'], reverse=True)[:20]
    for i, axis in enumerate(top_axes, 1):
        pos_sample = ", ".join(axis['positive_pole'][:3])
        neg_sample = ", ".join(axis['negative_pole'][:3])
        print(f"{i:2}. {axis['name']:<15} (size={axis['size']:>3}, "
              f"coh={axis['coherence_score']:.3f}, sep={axis['separation_score']:.3f})")
        print(f"    + {pos_sample}")
        print(f"    - {neg_sample}")

    print("\n" + "=" * 100)


def plot_distribution_histograms(axes: List[Dict], output_dir: str):
    """Plot distribution histograms for axis metrics."""
    print("\nGenerating distribution histograms...")

    fig, axes_plt = plt.subplots(2, 3, figsize=(18, 12))

    # Extract metrics
    sizes = [a['size'] for a in axes]
    coherences = [a['coherence_score'] for a in axes]
    separations = [a['separation_score'] for a in axes]
    pos_pole_sizes = [len(a['positive_pole']) for a in axes]
    neg_pole_sizes = [len(a['negative_pole']) for a in axes]

    # 1. Axis size distribution
    axes_plt[0, 0].hist(sizes, bins=30, color='steelblue', edgecolor='black')
    axes_plt[0, 0].set_xlabel('Axis Size (# pairs)', fontsize=12)
    axes_plt[0, 0].set_ylabel('Frequency', fontsize=12)
    axes_plt[0, 0].set_title('Axis Size Distribution', fontsize=14)
    axes_plt[0, 0].axvline(np.mean(sizes), color='red', linestyle='--',
                           label=f'Mean={np.mean(sizes):.1f}')
    axes_plt[0, 0].legend()

    # 2. Coherence distribution
    axes_plt[0, 1].hist(coherences, bins=30, color='green', edgecolor='black')
    axes_plt[0, 1].set_xlabel('Coherence Score', fontsize=12)
    axes_plt[0, 1].set_ylabel('Frequency', fontsize=12)
    axes_plt[0, 1].set_title('Pole Coherence Distribution', fontsize=14)
    axes_plt[0, 1].axvline(np.mean(coherences), color='red', linestyle='--',
                           label=f'Mean={np.mean(coherences):.3f}')
    axes_plt[0, 1].legend()

    # 3. Separation distribution
    axes_plt[0, 2].hist(separations, bins=30, color='orange', edgecolor='black')
    axes_plt[0, 2].set_xlabel('Separation Score', fontsize=12)
    axes_plt[0, 2].set_ylabel('Frequency', fontsize=12)
    axes_plt[0, 2].set_title('Pole Separation Distribution', fontsize=14)
    axes_plt[0, 2].axvline(np.mean(separations), color='red', linestyle='--',
                           label=f'Mean={np.mean(separations):.3f}')
    axes_plt[0, 2].legend()

    # 4. Positive pole size
    axes_plt[1, 0].hist(pos_pole_sizes, bins=30, color='lightblue', edgecolor='black')
    axes_plt[1, 0].set_xlabel('Positive Pole Size (# words)', fontsize=12)
    axes_plt[1, 0].set_ylabel('Frequency', fontsize=12)
    axes_plt[1, 0].set_title('Positive Pole Size Distribution', fontsize=14)

    # 5. Negative pole size
    axes_plt[1, 1].hist(neg_pole_sizes, bins=30, color='lightcoral', edgecolor='black')
    axes_plt[1, 1].set_xlabel('Negative Pole Size (# words)', fontsize=12)
    axes_plt[1, 1].set_ylabel('Frequency', fontsize=12)
    axes_plt[1, 1].set_title('Negative Pole Size Distribution', fontsize=14)

    # 6. Coherence vs Separation scatter
    axes_plt[1, 2].scatter(coherences, separations, alpha=0.5, color='purple')
    axes_plt[1, 2].set_xlabel('Coherence Score', fontsize=12)
    axes_plt[1, 2].set_ylabel('Separation Score', fontsize=12)
    axes_plt[1, 2].set_title('Coherence vs Separation', fontsize=14)
    axes_plt[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'distribution_histograms.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved to {output_path}")

    plt.close()


def compare_to_manual_clusters(axes: List[Dict], output_dir: str):
    """Compare discovered axes to known manual seed clusters."""
    print("\nComparing to manual seed clusters...")

    # Known semantic categories from bootstrap_semantic_clusters.py
    manual_categories = {
        'morality': {'good', 'virtuous', 'moral', 'righteous', 'ethical'},
        'temperature': {'hot', 'warm', 'cold', 'cool', 'freezing'},
        'size': {'big', 'large', 'small', 'tiny', 'huge'},
        'emotion': {'happy', 'sad', 'joyful', 'sorrowful', 'cheerful'},
        'strength': {'strong', 'weak', 'powerful', 'feeble', 'mighty'},
        'light': {'bright', 'dark', 'light', 'dim', 'luminous'}
    }

    # Find which axes match known categories
    matches = {}

    for category, seed_words in manual_categories.items():
        best_match = None
        best_overlap = 0

        for axis in axes:
            # Combine both poles
            axis_words = set(axis['positive_pole']) | set(axis['negative_pole'])

            # Compute overlap
            overlap = len(seed_words & axis_words)

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = axis

        if best_match:
            matches[category] = {
                'axis_id': best_match['axis_id'],
                'axis_name': best_match['name'],
                'overlap': best_overlap,
                'seed_size': len(seed_words),
                'axis_size': best_match['size'],
                'coherence': best_match['coherence_score'],
                'separation': best_match['separation_score']
            }

    # Print comparison
    print(f"\nFound {len(matches)}/{len(manual_categories)} known categories:")
    for category, match in matches.items():
        print(f"  [OK] {category:<12} -> Axis {match['axis_id']:2} '{match['axis_name']}' "
              f"(overlap={match['overlap']}/{match['seed_size']}, "
              f"size={match['axis_size']})")

    missing = set(manual_categories.keys()) - set(matches.keys())
    if missing:
        print(f"\nMissing categories: {', '.join(missing)}")

    # Save comparison
    comparison_path = os.path.join(output_dir, 'manual_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2)
    print(f"\n  [OK] Saved comparison to {comparison_path}")

    return matches


def save_validation_results(metrics: Dict, output_dir: str):
    """Save validation metrics to JSON."""
    print("\nSaving validation metrics...")

    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value

    output_path = os.path.join(output_dir, 'validation_metrics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2)

    print(f"  [OK] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate discovered semantic axes'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/discovered_axes',
        help='Directory with clustering results (default: data/discovered_axes)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("SEMANTIC AXIS VALIDATION")
    print("=" * 100)
    print(f"Results directory: {args.results_dir}")
    print()

    # Load results
    results = load_results(args.results_dir)

    # Compute validation metrics
    metrics = compute_validation_metrics(results)

    # Print report
    print_validation_report(metrics, results['axes'])

    # Generate visualizations
    if not args.no_plots:
        plot_distribution_histograms(results['axes'], args.results_dir)

    # Compare to manual clusters
    compare_to_manual_clusters(results['axes'], args.results_dir)

    # Save validation metrics
    save_validation_results(metrics, args.results_dir)

    print("\n" + "=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()
