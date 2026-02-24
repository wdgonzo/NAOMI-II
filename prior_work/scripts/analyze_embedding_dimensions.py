"""
Analyze Embedding Dimensions for Interpretability

This script analyzes the trained embeddings to discover:
1. Which dimensions have clear semantic meaning
2. How well words cluster on their assigned dimensions
3. Dimension activation patterns and sparsity
4. Top words for each dimension (positive and negative poles)

Usage:
    python scripts/analyze_embedding_dimensions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def load_embeddings() -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Load embeddings and vocabulary."""
    embeddings = np.load("checkpoints/embeddings.npy")

    with open("checkpoints/vocabulary.json", 'r') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = {v: k for k, v in word_to_id.items()}

    return embeddings, word_to_id, id_to_word


def analyze_dimension_sparsity(embeddings: np.ndarray, threshold: float = 0.01) -> Dict:
    """Analyze sparsity patterns across dimensions."""
    vocab_size, embedding_dim = embeddings.shape

    # Per-dimension sparsity
    dim_sparsity = []
    for dim in range(embedding_dim):
        near_zero = np.abs(embeddings[:, dim]) < threshold
        sparsity_pct = 100.0 * np.mean(near_zero)
        dim_sparsity.append(sparsity_pct)

    # Overall statistics
    overall_sparsity = 100.0 * np.mean(np.abs(embeddings) < threshold)

    return {
        'overall_sparsity': overall_sparsity,
        'per_dimension_sparsity': dim_sparsity,
        'mean_dim_sparsity': np.mean(dim_sparsity),
        'std_dim_sparsity': np.std(dim_sparsity),
        'min_dim_sparsity': np.min(dim_sparsity),
        'max_dim_sparsity': np.max(dim_sparsity)
    }


def find_top_words_for_dimension(embeddings: np.ndarray,
                                 id_to_word: Dict[int, str],
                                 dim_idx: int,
                                 top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """Find top words with highest positive and negative values on a dimension."""
    values = embeddings[:, dim_idx]

    # Get indices sorted by value
    sorted_indices = np.argsort(values)

    # Top negative (most negative values)
    top_negative = []
    for idx in sorted_indices[:top_k]:
        word = id_to_word[idx]
        value = values[idx]
        if abs(value) > 0.01:  # Filter near-zero
            top_negative.append((word, float(value)))

    # Top positive (most positive values)
    top_positive = []
    for idx in sorted_indices[-top_k:][::-1]:
        word = id_to_word[idx]
        value = values[idx]
        if abs(value) > 0.01:  # Filter near-zero
            top_positive.append((word, float(value)))

    return {
        'positive': top_positive,
        'negative': top_negative
    }


def analyze_semantic_clusters(embeddings: np.ndarray,
                              word_to_id: Dict[str, int],
                              clusters_path: str = "config/semantic_clusters.json") -> Dict:
    """Analyze how well semantic clusters align with their assigned dimensions."""

    with open(clusters_path, 'r') as f:
        clusters_str = json.load(f)

    clusters = {int(k): v for k, v in clusters_str.items()}

    results = {}

    for dim_idx, cluster_def in clusters.items():
        dim_results = {
            'positive': {'count': 0, 'correct_sign': 0, 'mean_value': 0.0, 'words': []},
            'negative': {'count': 0, 'correct_sign': 0, 'mean_value': 0.0, 'words': []},
            'neutral': {'count': 0, 'near_zero': 0, 'mean_abs_value': 0.0, 'words': []}
        }

        # Analyze positive pole
        positive_values = []
        for word in cluster_def.get('positive', []):
            word_id = word_to_id.get(word.lower())
            if word_id is not None:
                value = embeddings[word_id, dim_idx]
                positive_values.append(value)
                dim_results['positive']['count'] += 1
                if value > 0:
                    dim_results['positive']['correct_sign'] += 1
                dim_results['positive']['words'].append((word, float(value)))

        if positive_values:
            dim_results['positive']['mean_value'] = float(np.mean(positive_values))
            dim_results['positive']['accuracy'] = dim_results['positive']['correct_sign'] / dim_results['positive']['count']

        # Analyze negative pole
        negative_values = []
        for word in cluster_def.get('negative', []):
            word_id = word_to_id.get(word.lower())
            if word_id is not None:
                value = embeddings[word_id, dim_idx]
                negative_values.append(value)
                dim_results['negative']['count'] += 1
                if value < 0:
                    dim_results['negative']['correct_sign'] += 1
                dim_results['negative']['words'].append((word, float(value)))

        if negative_values:
            dim_results['negative']['mean_value'] = float(np.mean(negative_values))
            dim_results['negative']['accuracy'] = dim_results['negative']['correct_sign'] / dim_results['negative']['count']

        # Analyze neutral
        neutral_abs_values = []
        for word in cluster_def.get('neutral', []):
            word_id = word_to_id.get(word.lower())
            if word_id is not None:
                value = embeddings[word_id, dim_idx]
                abs_value = abs(value)
                neutral_abs_values.append(abs_value)
                dim_results['neutral']['count'] += 1
                if abs_value < 0.01:
                    dim_results['neutral']['near_zero'] += 1
                dim_results['neutral']['words'].append((word, float(value)))

        if neutral_abs_values:
            dim_results['neutral']['mean_abs_value'] = float(np.mean(neutral_abs_values))
            dim_results['neutral']['sparsity'] = dim_results['neutral']['near_zero'] / dim_results['neutral']['count']

        results[dim_idx] = dim_results

    return results


def analyze_antonym_polarization(embeddings: np.ndarray,
                                 word_to_id: Dict[str, int],
                                 antonym_path: str = "config/antonym_types.json",
                                 dim_assignments_path: str = "config/dimension_assignments.json") -> Dict:
    """Analyze how well antonym pairs are polarized on their assigned dimensions."""

    with open(antonym_path, 'r') as f:
        antonym_types = json.load(f)

    with open(dim_assignments_path, 'r') as f:
        dimension_assignments = json.load(f)
        dimension_assignments = {k: v for k, v in dimension_assignments.items()
                                if not k.startswith('_')}

    results = {}

    for atype, pairs in antonym_types.items():
        if atype not in dimension_assignments:
            continue

        assigned_dims = dimension_assignments[atype]
        type_results = {
            'assigned_dimensions': assigned_dims,
            'pairs': [],
            'polarized_on_assigned': 0,
            'total_pairs': 0
        }

        for word1, word2 in pairs:
            id1 = word_to_id.get(word1.lower())
            id2 = word_to_id.get(word2.lower())

            if id1 is None or id2 is None:
                continue

            vec1 = embeddings[id1]
            vec2 = embeddings[id2]

            # Check polarization on assigned dimensions
            polarized_dims = []
            for dim in assigned_dims:
                val1 = vec1[dim]
                val2 = vec2[dim]

                # Check if opposite signs
                if np.sign(val1) * np.sign(val2) < 0:
                    polarized_dims.append(dim)

            is_polarized = len(polarized_dims) > 0

            type_results['pairs'].append({
                'word1': word1,
                'word2': word2,
                'polarized_dimensions': polarized_dims,
                'is_polarized': is_polarized
            })

            type_results['total_pairs'] += 1
            if is_polarized:
                type_results['polarized_on_assigned'] += 1

        if type_results['total_pairs'] > 0:
            type_results['polarization_rate'] = type_results['polarized_on_assigned'] / type_results['total_pairs']

        results[atype] = type_results

    return results


def print_dimension_report(dim_idx: int,
                          top_words: Dict,
                          cluster_analysis: Dict,
                          sparsity: float):
    """Print a detailed report for a single dimension."""
    print(f"\n{'='*80}")
    print(f"DIMENSION {dim_idx}")
    print(f"{'='*80}")
    print(f"Sparsity: {sparsity:.1f}% of words near zero")
    print()

    # Print cluster analysis if available
    if dim_idx in cluster_analysis:
        ca = cluster_analysis[dim_idx]

        print("SEMANTIC CLUSTER ANALYSIS:")
        print("-" * 80)

        if ca['positive']['count'] > 0:
            print(f"Positive Pole: {ca['positive']['count']} words")
            print(f"  Correct sign: {ca['positive']['correct_sign']}/{ca['positive']['count']} ({ca['positive']['accuracy']*100:.1f}%)")
            print(f"  Mean value: {ca['positive']['mean_value']:.4f}")
            print(f"  Sample words:")
            for word, val in sorted(ca['positive']['words'], key=lambda x: -x[1])[:5]:
                print(f"    {word:30s} = {val:+.4f}")

        print()

        if ca['negative']['count'] > 0:
            print(f"Negative Pole: {ca['negative']['count']} words")
            print(f"  Correct sign: {ca['negative']['correct_sign']}/{ca['negative']['count']} ({ca['negative']['accuracy']*100:.1f}%)")
            print(f"  Mean value: {ca['negative']['mean_value']:.4f}")
            print(f"  Sample words:")
            for word, val in sorted(ca['negative']['words'], key=lambda x: x[1])[:5]:
                print(f"    {word:30s} = {val:+.4f}")

        print()

        if ca['neutral']['count'] > 0:
            print(f"Neutral: {ca['neutral']['count']} words")
            print(f"  Near zero: {ca['neutral']['near_zero']}/{ca['neutral']['count']} ({ca['neutral']['sparsity']*100:.1f}%)")
            print(f"  Mean |value|: {ca['neutral']['mean_abs_value']:.4f}")

    print()
    print("TOP WORDS (by absolute value):")
    print("-" * 80)

    print(f"Most Positive ({len(top_words['positive'])} words):")
    for word, val in top_words['positive'][:10]:
        print(f"  {word:40s} = {val:+.4f}")

    print()
    print(f"Most Negative ({len(top_words['negative'])} words):")
    for word, val in top_words['negative'][:10]:
        print(f"  {word:40s} = {val:+.4f}")


def main():
    """Main analysis."""
    print("=" * 80)
    print("EMBEDDING DIMENSION ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print("[1/5] Loading embeddings and vocabulary...")
    embeddings, word_to_id, id_to_word = load_embeddings()
    vocab_size, embedding_dim = embeddings.shape
    print(f"  Vocabulary: {vocab_size} words")
    print(f"  Dimensions: {embedding_dim}")
    print()

    # Analyze sparsity
    print("[2/5] Analyzing sparsity patterns...")
    sparsity_stats = analyze_dimension_sparsity(embeddings)
    print(f"  Overall sparsity: {sparsity_stats['overall_sparsity']:.1f}%")
    print(f"  Mean per-dimension sparsity: {sparsity_stats['mean_dim_sparsity']:.1f}% ± {sparsity_stats['std_dim_sparsity']:.1f}%")
    print(f"  Range: {sparsity_stats['min_dim_sparsity']:.1f}% - {sparsity_stats['max_dim_sparsity']:.1f}%")
    print()

    # Analyze semantic clusters
    print("[3/5] Analyzing semantic cluster alignment...")
    cluster_analysis = analyze_semantic_clusters(embeddings, word_to_id)
    print(f"  Analyzed {len(cluster_analysis)} semantic dimensions")

    # Print summary statistics
    total_pos_correct = 0
    total_pos_count = 0
    total_neg_correct = 0
    total_neg_count = 0

    for dim_idx, ca in cluster_analysis.items():
        total_pos_correct += ca['positive']['correct_sign']
        total_pos_count += ca['positive']['count']
        total_neg_correct += ca['negative']['correct_sign']
        total_neg_count += ca['negative']['count']

    if total_pos_count > 0:
        print(f"  Positive pole accuracy: {total_pos_correct}/{total_pos_count} ({100*total_pos_correct/total_pos_count:.1f}%)")
    if total_neg_count > 0:
        print(f"  Negative pole accuracy: {total_neg_correct}/{total_neg_count} ({100*total_neg_correct/total_neg_count:.1f}%)")
    print()

    # Analyze antonym polarization
    print("[4/5] Analyzing antonym polarization...")
    antonym_analysis = analyze_antonym_polarization(embeddings, word_to_id)

    total_polarized = 0
    total_pairs = 0
    for atype, results in antonym_analysis.items():
        total_polarized += results['polarized_on_assigned']
        total_pairs += results['total_pairs']

    if total_pairs > 0:
        print(f"  Antonym pairs polarized: {total_polarized}/{total_pairs} ({100*total_polarized/total_pairs:.1f}%)")
    print()

    # Find top words for each dimension
    print("[5/5] Finding top words for each dimension...")
    top_words_by_dim = {}
    for dim_idx in range(embedding_dim):
        top_words_by_dim[dim_idx] = find_top_words_for_dimension(
            embeddings, id_to_word, dim_idx, top_k=20
        )
    print(f"  Analyzed all {embedding_dim} dimensions")
    print()

    # Print detailed reports for assigned semantic dimensions
    print("=" * 80)
    print("DETAILED DIMENSION REPORTS")
    print("=" * 80)

    # Dimension names from config
    dimension_names = {
        0: "Morality",
        1: "Temperature",
        2: "Size",
        3: "Speed",
        4: "Emotion",
        5: "Light/Dark",
        6: "Strength",
        7: "Difficulty",
        8: "Quantity",
        9: "Age",
        10: "Distance",
        11: "Quality",
        12: "Wealth",
        13: "Knowledge",
        14: "Safety",
        15: "Truth",
        16: "Activity",
        17: "Visibility",
        18: "Orientation",
    }

    # Print reports for assigned dimensions
    for dim_idx in sorted(cluster_analysis.keys()):
        dim_name = dimension_names.get(dim_idx, f"Dimension {dim_idx}")
        print_dimension_report(
            dim_idx,
            top_words_by_dim[dim_idx],
            cluster_analysis,
            sparsity_stats['per_dimension_sparsity'][dim_idx]
        )

    # Print antonym polarization details
    print(f"\n{'='*80}")
    print("ANTONYM POLARIZATION ANALYSIS")
    print(f"{'='*80}")

    for atype, results in sorted(antonym_analysis.items()):
        if results['total_pairs'] == 0:
            continue

        print(f"\n{atype.upper()}:")
        print(f"  Assigned dimensions: {results['assigned_dimensions']}")
        print(f"  Polarization rate: {results['polarized_on_assigned']}/{results['total_pairs']} ({results['polarization_rate']*100:.1f}%)")

        # Show sample pairs
        print(f"  Sample pairs:")
        for pair in results['pairs'][:5]:
            status = "[Y]" if pair['is_polarized'] else "[N]"
            dims_str = ", ".join(map(str, pair['polarized_dimensions'])) if pair['polarized_dimensions'] else "none"
            print(f"    {status} {pair['word1']:20s} / {pair['word2']:20s} (polarized on dims: {dims_str})")

    # Save analysis results
    print(f"\n{'='*80}")
    print("SAVING ANALYSIS RESULTS")
    print(f"{'='*80}")

    output = {
        'sparsity': sparsity_stats,
        'cluster_analysis': cluster_analysis,
        'antonym_analysis': antonym_analysis,
        'top_words_by_dimension': top_words_by_dim
    }

    output_path = Path("checkpoints/dimension_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved analysis to {output_path}")
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
