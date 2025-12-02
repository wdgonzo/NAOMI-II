"""
Test word-based clustering approach on fundamental antonym pairs.

This script tests if clustering WORDS (not PAIRS) can successfully discover
fundamental semantic axes like good/bad, hot/cold, big/small.

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src' / 'embeddings'
sys.path.insert(0, str(src_path))

from word_clustering import cluster_words_and_detect_axes


def main():
    # Load antonym pairs
    pairs_path = Path(__file__).parent.parent / 'data' / 'full_wordnet' / 'antonym_pairs.json'
    print(f"Loading antonym pairs from {pairs_path}...")

    with open(pairs_path, 'r') as f:
        antonym_pairs = json.load(f)

    print(f"Loaded {len(antonym_pairs)} antonym pairs")
    print()

    # Test with small subset first
    print("=" * 100)
    print("TEST 1: Small subset (500 pairs)")
    print("=" * 100)

    test_pairs = antonym_pairs[:500]

    word_clusters, axes = cluster_words_and_detect_axes(
        test_pairs,
        n_clusters=30,
        min_cluster_size=3,
        min_antonym_edges=2,
        linkage_method='average',
        verbose=True
    )

    print()
    print("RESULTS:")
    print(f"  Word clusters: {len(word_clusters)}")
    print(f"  Detected axes: {len(axes)}")
    print()

    # Check if fundamental axes found
    fundamental_keywords = ['good', 'bad', 'hot', 'cold', 'big', 'small', 'large', 'fast', 'slow']

    print("Searching for fundamental axes:")
    for axis in axes[:20]:
        pole_words = set()
        for pole in axis['poles']:
            pole_words.update(pole)

        # Check if contains fundamental keywords
        found_keywords = [kw for kw in fundamental_keywords if kw in pole_words]

        if found_keywords:
            print(f"\n[FOUND] Axis with {len(axis['poles'])} poles ({axis['size']} words, {axis['n_pairs']} pairs):")
            print(f"  Keywords: {found_keywords}")
            print(f"  Pole names: {axis['pole_names']}")
            for i, pole in enumerate(axis['poles']):
                print(f"  Pole {i+1}: {pole[:10]}")

    print()
    print("=" * 100)
    print("TEST 2: Full dataset (all pairs)")
    print("=" * 100)

    word_clusters_full, axes_full = cluster_words_and_detect_axes(
        antonym_pairs,
        n_clusters=60,
        min_cluster_size=3,
        min_antonym_edges=3,
        linkage_method='average',
        verbose=True
    )

    print()
    print("RESULTS:")
    print(f"  Word clusters: {len(word_clusters_full)}")
    print(f"  Detected axes: {len(axes_full)}")
    print()

    # Show top 20 axes
    print("Top 20 axes by size:")
    for i, axis in enumerate(axes_full[:20], 1):
        print(f"  {i}. {' / '.join(axis['pole_names'])}: {axis['size']} words, {axis['n_pairs']} pairs")

        # Check if contains fundamental keywords
        pole_words = set()
        for pole in axis['poles']:
            pole_words.update(pole)
        found_keywords = [kw for kw in fundamental_keywords if kw in pole_words]
        if found_keywords:
            print(f"      [FUNDAMENTAL] Keywords: {found_keywords}")

    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
