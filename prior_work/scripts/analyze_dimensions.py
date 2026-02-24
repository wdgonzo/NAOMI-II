"""
Dimensional Analysis Script

Analyzes individual embedding dimensions to discover what semantic aspects
they encode (e.g., morality: good=high/bad=low).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from typing import List, Tuple

from src.embeddings.dimension_analysis import DimensionAnalyzer
from src.embeddings.relationship_analysis import RelationshipDimensionAnalysis
from src.embeddings.dimension_types import DimensionAnalysisReport


def load_embeddings(checkpoint_dir: str):
    """Load trained embeddings and vocabulary."""
    checkpoint_path = Path(checkpoint_dir)

    # Load embeddings
    embeddings = np.load(checkpoint_path / "embeddings.npy")

    # Load vocabulary
    with open(checkpoint_path / "vocabulary.json", 'r') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    print(f"Loaded {len(word_to_id)} words with {embeddings.shape[1]}-dim embeddings")

    return embeddings, word_to_id, id_to_word


def get_antonym_pairs() -> List[Tuple[str, str]]:
    """Get common antonym pairs for analysis."""
    return [
        ("good", "bad"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("right", "wrong"),
        ("light", "dark"),
        ("hard", "soft"),
        ("long", "short"),
        ("high", "low"),
        ("new", "old"),
        ("clean", "dirty"),
        ("rich", "poor"),
        ("strong", "weak"),
        ("easy", "difficult")
    ]


def get_synonym_pairs() -> List[Tuple[str, str]]:
    """Get common synonym pairs for analysis."""
    return [
        ("big", "large"),
        ("small", "little"),
        ("happy", "glad"),
        ("fast", "quick"),
        ("beautiful", "pretty"),
        ("smart", "intelligent"),
        ("help", "assist"),
        ("begin", "start"),
        ("end", "finish"),
        ("buy", "purchase")
    ]


def get_hypernym_pairs() -> List[Tuple[str, str]]:
    """Get (specific, general) pairs for hypernym analysis."""
    return [
        ("dog", "animal"),
        ("cat", "animal"),
        ("chair", "furniture"),
        ("table", "furniture"),
        ("car", "vehicle"),
        ("truck", "vehicle"),
        ("apple", "fruit"),
        ("orange", "fruit")
    ]


def analyze_word_pair_on_all_dimensions(analyzer: DimensionAnalyzer,
                                         word1: str, word2: str,
                                         top_k: int = 10):
    """
    Analyze which dimensions differ most between two words.

    Args:
        analyzer: DimensionAnalyzer instance
        word1: First word
        word2: Second word
        top_k: Number of top dimensions to show
    """
    print(f"\n=== Analyzing: {word1} vs {word2} ===")

    word1_ids = analyzer._find_word_ids(word1)
    word2_ids = analyzer._find_word_ids(word2)

    if not word1_ids:
        print(f"  '{word1}' not found in vocabulary")
        return
    if not word2_ids:
        print(f"  '{word2}' not found in vocabulary")
        return

    # Get embeddings
    emb1 = analyzer.embeddings[word1_ids[0]]
    emb2 = analyzer.embeddings[word2_ids[0]]

    # Calculate differences per dimension
    diffs = np.abs(emb1 - emb2)

    # Find dimensions sorted by difference
    sorted_dims = np.argsort(diffs)[::-1]

    # Count similar vs different dimensions
    similarity_threshold = 0.1
    discriminative_threshold = 0.5

    similar_count = np.sum(diffs < similarity_threshold)
    different_count = np.sum(diffs > discriminative_threshold)
    moderate_count = len(diffs) - similar_count - different_count

    total_dims = len(diffs)
    print(f"  Similar across {similar_count}/{total_dims} dimensions ({similar_count/total_dims*100:.1f}%)")
    print(f"  Differ on {different_count}/{total_dims} dimensions ({different_count/total_dims*100:.1f}%)")
    print(f"  Moderate difference: {moderate_count}/{total_dims} dimensions ({moderate_count/total_dims*100:.1f}%)")

    print(f"\n  Top {top_k} dimensions that differ:")
    for rank, dim_idx in enumerate(sorted_dims[:top_k], 1):
        diff = diffs[dim_idx]
        val1 = emb1[dim_idx]
        val2 = emb2[dim_idx]

        # Get dimension name if anchor
        dim_name = analyzer.anchor_names.get(dim_idx)
        name_str = f'"{dim_name}"' if dim_name else "(learned)"

        print(f"    {rank:2d}. Dim {dim_idx:3d} {name_str:20s}: "
              f"{word1}={val1:6.3f}, {word2}={val2:6.3f}, diff={diff:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze embedding dimensions')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing trained embeddings')
    parser.add_argument('--output-dir', type=str, default='dimension_analysis',
                       help='Directory for analysis output')
    parser.add_argument('--num-anchor-dims', type=int, default=51,
                       help='Number of anchor dimensions')

    args = parser.parse_args()

    print("="*70)
    print("EMBEDDING DIMENSIONAL ANALYSIS")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("[1/6] Loading embeddings...")
    embeddings, word_to_id, id_to_word = load_embeddings(args.checkpoint_dir)
    print()

    # Initialize analyzers
    print("[2/6] Initializing analyzers...")
    dim_analyzer = DimensionAnalyzer(
        embeddings, word_to_id, id_to_word,
        num_anchor_dims=args.num_anchor_dims
    )
    rel_analyzer = RelationshipDimensionAnalysis(dim_analyzer)
    print()

    # Variance analysis
    print("[3/6] Analyzing dimension variance...")
    print("="*70)
    variances = dim_analyzer.compute_dimension_variance()
    high_var = dim_analyzer.find_high_variance_dimensions(top_k=20)
    low_var = dim_analyzer.find_low_variance_dimensions(threshold=0.001)

    print(f"  High variance dimensions (top 20):")
    for rank, (dim_idx, var) in enumerate(high_var, 1):
        dim_name = dim_analyzer.anchor_names.get(dim_idx)
        name_str = f'"{dim_name}"' if dim_name else "(learned)"
        dim_type = "ANCHOR" if dim_idx < args.num_anchor_dims else "LEARNED"
        print(f"    {rank:2d}. Dim {dim_idx:3d} {name_str:20s} [{dim_type:7s}] var={var:.4f}")

    print(f"\n  Low variance dimensions: {len(low_var)} (candidates for pruning)")
    print()

    # Antonym analysis
    print("[4/6] Analyzing antonym dimensions...")
    print("="*70)
    antonym_pairs = get_antonym_pairs()
    antonym_profile = rel_analyzer.analyze_antonym_dimensions(antonym_pairs)

    print(f"\n  Key insight: Antonyms should be SIMILAR in most dims, DIFFER in few")
    print(f"  Results:")
    print(f"    - Discriminative dims: {len(antonym_profile.discriminative_dims)}")
    print(f"    - Similar dims: {len(antonym_profile.similarity_dims)}")
    print(f"    - Ratio: {len(antonym_profile.similarity_dims)/max(len(antonym_profile.discriminative_dims), 1):.1f}:1 similar:different")

    print(f"\n  Top dimensions that separate antonyms:")
    top_antonym_dims = antonym_profile.get_top_discriminative(10)
    for rank, (dim_idx, score) in enumerate(top_antonym_dims, 1):
        dim_name = dim_analyzer.anchor_names.get(dim_idx)
        name_str = f'"{dim_name}"' if dim_name else "(learned)"
        mean_diff = antonym_profile.mean_difference_per_dim[dim_idx]
        print(f"    {rank:2d}. Dim {dim_idx:3d} {name_str:20s} mean_diff={mean_diff:.3f}")

    print()

    # Synonym analysis
    print("[5/6] Analyzing synonym dimensions...")
    print("="*70)
    synonym_pairs = get_synonym_pairs()
    synonym_profile = rel_analyzer.analyze_synonym_dimensions(synonym_pairs)

    print(f"\n  Key insight: Synonyms should be SIMILAR across most dims")
    print(f"  Results:")
    print(f"    - Similar dims: {len(synonym_profile.similarity_dims)}")
    print(f"    - Discriminative dims: {len(synonym_profile.discriminative_dims)}")

    print()

    # Example word pair analyses
    print("[6/6] Detailed word pair analyses...")
    print("="*70)

    # Analyze specific antonym pairs
    analyze_word_pair_on_all_dimensions(dim_analyzer, "good", "bad", top_k=10)
    analyze_word_pair_on_all_dimensions(dim_analyzer, "hot", "cold", top_k=10)
    analyze_word_pair_on_all_dimensions(dim_analyzer, "big", "small", top_k=10)

    # Analyze synonym pairs
    analyze_word_pair_on_all_dimensions(dim_analyzer, "big", "large", top_k=10)
    analyze_word_pair_on_all_dimensions(dim_analyzer, "happy", "glad", top_k=10)

    print()

    # Discover semantic axes
    print("Discovering semantic axes...")
    print("="*70)

    word_groups = {
        'size': {
            'positive': ['big', 'large', 'huge', 'giant'],
            'negative': ['small', 'tiny', 'little', 'miniscule']
        },
        'morality': {
            'positive': ['good', 'virtuous', 'moral'],
            'negative': ['bad', 'evil', 'immoral']
        },
        'temperature': {
            'positive': ['hot', 'warm'],
            'negative': ['cold', 'cool']
        },
        'speed': {
            'positive': ['fast', 'quick', 'rapid'],
            'negative': ['slow']
        }
    }

    semantic_axes = rel_analyzer.discover_semantic_axes(word_groups)

    for axis in semantic_axes:
        dim_name = dim_analyzer.anchor_names.get(axis.primary_dimension)
        name_str = f'"{dim_name}"' if dim_name else "(learned)"
        print(f"\n  Axis: {axis.name}")
        print(f"    Primary dimension: {axis.primary_dimension} {name_str}")
        print(f"    Correlation score: {axis.correlation_score:.3f}")
        print(f"    Positive pole: {', '.join(axis.positive_pole_words)}")
        print(f"    Negative pole: {', '.join(axis.negative_pole_words)}")

    print()

    # Create comprehensive report
    print("Creating comprehensive report...")

    # Get dimension statistics for top 50
    dimension_stats = []
    for dim_idx in range(min(50, embeddings.shape[1])):
        stats = dim_analyzer.get_dimension_statistics(dim_idx, top_k=5)
        dimension_stats.append(stats)

    report = DimensionAnalysisReport(
        model_path=args.checkpoint_dir,
        vocabulary_size=len(word_to_id),
        embedding_dim=embeddings.shape[1],
        num_anchor_dims=args.num_anchor_dims,
        dimension_stats=dimension_stats,
        synonym_profile=synonym_profile,
        antonym_profile=antonym_profile,
        semantic_axes=semantic_axes,
        high_variance_dims=[dim for dim, _ in high_var],
        low_variance_dims=low_var
    )

    # Save report
    report_path = output_dir / "dimension_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"  Saved comprehensive report: {report_path}")
    print()

    # Print summary
    print("="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    summary = report.get_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*70)
    print()

    print(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
