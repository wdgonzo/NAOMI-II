"""
Cluster WordNet antonym pairs into semantic axes.

This script runs the complete antonym clustering pipeline:
1. Load antonym pairs from WordNet
2. Build pairwise similarity matrix (4 signals)
3. Hierarchical agglomerative clustering (complete linkage)
4. Optimize cut height via silhouette score
5. Extract and validate semantic axes
6. Save results and visualizations

Usage:
    python scripts/cluster_antonym_axes.py [options]

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
matplotlib.use('Agg')  # Non-interactive backend

# Add src to path - import directly to avoid __init__.py issues
src_path = Path(__file__).parent.parent / 'src' / 'embeddings'
sys.path.insert(0, str(src_path))

from antonym_clustering import AntonymClusterer
from axis_extraction import AxisExtractor


def load_antonym_pairs(input_path: str) -> list:
    """Load antonym pairs from JSON file."""
    print(f"Loading antonym pairs from {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} antonym pairs")
    return pairs


def save_results(
    output_dir: str,
    similarity_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    cluster_assignments: np.ndarray,
    axes: list,
    stats: dict
):
    """Save clustering results to disk."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to {output_dir}...")

    # Save similarity matrix
    sim_path = os.path.join(output_dir, 'similarity_matrix.npy')
    np.save(sim_path, similarity_matrix)
    print(f"  [OK] Similarity matrix: {sim_path}")

    # Save linkage matrix
    link_path = os.path.join(output_dir, 'linkage_matrix.npy')
    np.save(link_path, linkage_matrix)
    print(f"  [OK] Linkage matrix: {link_path}")

    # Save cluster assignments
    cluster_path = os.path.join(output_dir, 'cluster_assignments.npy')
    np.save(cluster_path, cluster_assignments)
    print(f"  [OK] Cluster assignments: {cluster_path}")

    # Save semantic axes (JSON)
    axes_path = os.path.join(output_dir, 'semantic_axes.json')
    extractor = AxisExtractor([], cluster_assignments)  # Dummy for export
    axes_json = extractor.export_axes_to_json(axes)
    with open(axes_path, 'w', encoding='utf-8') as f:
        json.dump(axes_json, f, indent=2)
    print(f"  [OK] Semantic axes: {axes_path}")

    # Save statistics (convert numpy types to Python types)
    stats_path = os.path.join(output_dir, 'clustering_stats.json')
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    stats_serializable = convert_numpy(stats)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"  [OK] Statistics: {stats_path}")

    # Save human-readable report
    report_path = os.path.join(output_dir, 'axis_report.txt')
    extractor = AxisExtractor([], cluster_assignments)
    report = extractor.generate_axis_report(axes)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [OK] Axis report: {report_path}")

    print(f"\nAll results saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster WordNet antonym pairs into semantic axes'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/full_wordnet/antonym_pairs.json',
        help='Path to antonym pairs JSON file (default: data/full_wordnet/antonym_pairs.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/discovered_axes',
        help='Output directory for results (default: data/discovered_axes)'
    )

    parser.add_argument(
        '--min-clusters',
        type=int,
        default=10,
        help='Minimum number of clusters (default: 10)'
    )

    parser.add_argument(
        '--max-clusters',
        type=int,
        default=200,
        help='Maximum number of clusters (default: 200)'
    )

    parser.add_argument(
        '--min-axis-size',
        type=int,
        default=3,
        help='Minimum pairs per axis (default: 3)'
    )

    parser.add_argument(
        '--min-coherence',
        type=float,
        default=0.3,
        help='Minimum pole coherence score (default: 0.3)'
    )

    parser.add_argument(
        '--min-separation',
        type=float,
        default=0.5,
        help='Minimum pole separation score (default: 0.5)'
    )

    parser.add_argument(
        '--plot-dendrogram',
        action='store_true',
        help='Generate dendrogram visualization'
    )

    parser.add_argument(
        '--plot-silhouette',
        action='store_true',
        help='Generate silhouette optimization plot'
    )

    parser.add_argument(
        '--max-pairs',
        type=int,
        default=None,
        help='Maximum pairs to use (for testing, default: use all)'
    )

    args = parser.parse_args()

    print("=" * 100)
    print("ANTONYM CLUSTERING FOR SEMANTIC AXIS DISCOVERY")
    print("=" * 100)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Cluster range: {args.min_clusters} - {args.max_clusters}")
    print(f"Validation: min_size={args.min_axis_size}, "
          f"min_coherence={args.min_coherence}, min_separation={args.min_separation}")
    print()

    # Step 1: Load antonym pairs
    antonym_pairs = load_antonym_pairs(args.input)

    # Limit for testing if requested
    if args.max_pairs and args.max_pairs < len(antonym_pairs):
        print(f"WARNING: Limiting to {args.max_pairs} pairs for testing")
        antonym_pairs = antonym_pairs[:args.max_pairs]

    # Step 2: Run clustering pipeline
    clusterer = AntonymClusterer(antonym_pairs)

    def progress_callback(current, total):
        """Print progress during similarity matrix computation."""
        percent = (current / total) * 100
        if current % 50000 == 0:
            print(f"  Progress: {current:,}/{total:,} ({percent:.1f}%)")

    # Build similarity matrix
    clusterer.build_similarity_matrix(progress_callback=progress_callback)

    # Perform clustering
    clusterer.perform_hierarchical_clustering(method='complete')

    # Optimize cut height
    clusterer.optimize_cut_height(
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        n_steps=100,
        verbose=True
    )

    # Extract clusters
    cluster_assignments = clusterer.extract_clusters()

    # Get statistics
    stats = clusterer.get_cluster_statistics()

    # Step 3: Extract semantic axes
    extractor = AxisExtractor(antonym_pairs, cluster_assignments)
    axes = extractor.extract_all_axes(
        min_size=args.min_axis_size,
        min_coherence=args.min_coherence,
        min_separation=args.min_separation,
        verbose=True
    )

    # Step 4: Save results
    save_results(
        output_dir=args.output_dir,
        similarity_matrix=clusterer.similarity_matrix,
        linkage_matrix=clusterer.linkage_matrix,
        cluster_assignments=cluster_assignments,
        axes=axes,
        stats=stats
    )

    # Step 5: Generate visualizations (optional)
    if args.plot_dendrogram:
        print("\nGenerating dendrogram...")
        dendrogram_path = os.path.join(args.output_dir, 'dendrogram.png')
        clusterer.plot_dendrogram(output_path=dendrogram_path)

    if args.plot_silhouette:
        print("\nGenerating silhouette optimization plot...")
        silhouette_path = os.path.join(args.output_dir, 'silhouette_optimization.png')
        clusterer.plot_silhouette_optimization(output_path=silhouette_path)

    # Step 6: Print summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"[OK] Processed {len(antonym_pairs)} antonym pairs")
    print(f"[OK] Discovered {len(axes)} interpretable semantic axes")
    print(f"[OK] Silhouette score: {stats.get('silhouette_score', 'N/A'):.4f}")
    print(f"[OK] Mean axis size: {np.mean([a['size'] for a in axes]):.1f} pairs")
    print(f"[OK] Mean coherence: {np.mean([a['coherence_score'] for a in axes]):.4f}")
    print(f"[OK] Mean separation: {np.mean([a['separation_score'] for a in axes]):.4f}")
    print()
    print("Top 10 largest axes:")
    for i, axis in enumerate(axes[:10], 1):
        print(f"  {i}. {axis['name']:<20} ({axis['size']:>3} pairs, "
              f"coherence={axis['coherence_score']:.3f}, "
              f"separation={axis['separation_score']:.3f})")
    print()
    print(f"Results saved to: {args.output_dir}")
    print("=" * 100)


if __name__ == '__main__':
    main()
