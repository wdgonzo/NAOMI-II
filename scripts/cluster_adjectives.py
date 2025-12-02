"""
Cluster adjectives to discover semantic axes (Phase 1.5).

This script extracts adjectives from WordNet, clusters them by semantic similarity,
and exports the discovered axes in a format compatible with the dimension allocator.

Usage:
    python scripts/cluster_adjectives.py --output-dir data/adjective_axes

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path - import directly to avoid __init__.py issues
src_path = Path(__file__).parent.parent / 'src' / 'embeddings'
sys.path.insert(0, str(src_path))

from adjective_clustering import (
    cluster_adjectives_from_wordnet,
    convert_adjective_clusters_to_axes
)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster adjectives to discover semantic axes (Phase 1.5)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/adjective_axes',
        help='Output directory for adjective axes'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=50,
        help='Target number of clusters'
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=3,
        help='Minimum adjectives per cluster'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=5,
        help='Minimum WordNet synset count for adjective'
    )
    parser.add_argument(
        '--max-adjectives',
        type=int,
        default=2000,
        help='Maximum number of adjectives to cluster'
    )
    parser.add_argument(
        '--min-coherence',
        type=float,
        default=0.3,
        help='Minimum coherence score for valid axis'
    )
    parser.add_argument(
        '--linkage-method',
        type=str,
        default='complete',
        choices=['complete', 'average', 'single', 'ward'],
        help='Hierarchical clustering linkage method'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("ADJECTIVE CLUSTERING (PHASE 1.5)")
    print("=" * 100)
    print(f"Target clusters: {args.n_clusters}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"Max adjectives: {args.max_adjectives}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Min coherence: {args.min_coherence}")
    print(f"Linkage method: {args.linkage_method}")
    print()

    # Cluster adjectives
    adjective_clusters, cluster_assignments = cluster_adjectives_from_wordnet(
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_frequency=args.min_frequency,
        max_adjectives=args.max_adjectives,
        linkage_method=args.linkage_method,
        verbose=True
    )

    print()
    print(f"Discovered {len(adjective_clusters)} adjective clusters")
    print()

    # Convert to axes format
    print("Converting clusters to axis format...")
    axes = convert_adjective_clusters_to_axes(
        adjective_clusters,
        min_coherence=args.min_coherence
    )

    print(f"Created {len(axes)} valid axes (min coherence {args.min_coherence})")
    print()

    # Save axes
    axes_output_path = os.path.join(args.output_dir, 'adjective_axes.json')
    print(f"Saving axes to {axes_output_path}...")

    with open(axes_output_path, 'w') as f:
        json.dump(axes, f, indent=2)

    print("[OK] Saved axes")

    # Save raw clusters
    clusters_output_path = os.path.join(args.output_dir, 'adjective_clusters.json')
    print(f"Saving raw clusters to {clusters_output_path}...")

    with open(clusters_output_path, 'w') as f:
        json.dump(adjective_clusters, f, indent=2)

    print("[OK] Saved clusters")

    # Print summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total adjective clusters: {len(adjective_clusters)}")
    print(f"Valid axes (after coherence filter): {len(axes)}")
    print()

    # Show top 10 axes
    print("Top 10 axes by size:")
    for i, axis in enumerate(axes[:10], 1):
        print(f"  {i}. {axis['name']}: {axis['size']} adjectives (coherence: {axis['coherence_score']:.3f})")
        print(f"     Examples: {', '.join(axis['poles'][0][:5])}")

    print()
    print(f"[OK] Adjective clustering complete!")
    print(f"[OK] Output saved to {args.output_dir}")


if __name__ == '__main__':
    main()
