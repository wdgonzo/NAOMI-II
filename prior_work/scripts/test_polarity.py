"""
Test Polarity Structure in Trained Embeddings

Tests whether the trained embeddings exhibit selective polarity structure:
- Antonyms should oppose on 1-5 specific dimensions
- Different antonym pairs should use different dimensions
- NOT operation should work (NOT(good) ≈ bad)

This validates that transparent dimension training is working correctly.

Usage:
    python scripts/test_polarity.py \
        --embeddings checkpoints/embeddings.npy \
        --vocab checkpoints/vocabulary.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


def load_embeddings(embeddings_path: Path, vocab_path: Path):
    """
    Load embeddings and vocabulary.

    Args:
        embeddings_path: Path to embeddings.npy
        vocab_path: Path to vocabulary.json

    Returns:
        (embeddings, word_to_id, id_to_word) tuple
    """
    embeddings = np.load(embeddings_path)

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = {int(k): v for k, v in vocab_data.get('id_to_word', {}).items()}

    # If id_to_word not in file, create from word_to_id
    if not id_to_word:
        id_to_word = {v: k for k, v in word_to_id.items()}

    return embeddings, word_to_id, id_to_word


def find_word_sense(word: str, word_to_id: Dict[str, int]) -> List[str]:
    """
    Find all sense-tagged versions of a word.

    Args:
        word: Base word (e.g., "good")
        word_to_id: Word to ID mapping

    Returns:
        List of sense-tagged versions (e.g., ["good_wn.00_a", "good_wn.01_a"])
    """
    matches = []

    for vocab_word in word_to_id.keys():
        # Match word_wn.XX_pos pattern
        if vocab_word.startswith(f"{word}_wn."):
            matches.append(vocab_word)

    # Also check exact match (non sense-tagged)
    if word in word_to_id:
        matches.append(word)

    return matches


def compute_polarity_structure(emb1: np.ndarray, emb2: np.ndarray,
                               threshold: float = 0.5) -> Dict:
    """
    Analyze polarity structure between two antonym embeddings.

    Args:
        emb1: First embedding
        emb2: Second embedding
        threshold: Threshold for considering dimensions "opposing"

    Returns:
        Dictionary with polarity statistics
    """
    # Compute dimension-wise differences
    diffs = np.abs(emb1 - emb2)

    # Compute dimension-wise products (opposite signs = negative product)
    products = emb1 * emb2

    # Find opposing dimensions (large difference, opposite signs)
    opposing_mask = (diffs > threshold) & (products < 0)
    opposing_dims = np.where(opposing_mask)[0]

    # Find similar dimensions (small difference)
    similar_mask = diffs < 0.1
    similar_dims = np.where(similar_mask)[0]

    # Find dimensions with opposite signs
    opposite_sign_mask = products < 0
    opposite_sign_dims = np.where(opposite_sign_mask)[0]

    # Overall distance
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return {
        'opposing_dims': opposing_dims.tolist(),
        'num_opposing': len(opposing_dims),
        'similar_dims': similar_dims.tolist(),
        'num_similar': len(similar_dims),
        'opposite_sign_dims': opposite_sign_dims.tolist(),
        'num_opposite_signs': len(opposite_sign_dims),
        'euclidean_distance': float(euclidean_dist),
        'cosine_similarity': float(cosine_sim),
        'dimension_diffs': diffs.tolist(),
        'dimension_products': products.tolist(),
    }


def test_antonym_pair(word1: str, word2: str, embeddings: np.ndarray,
                      word_to_id: Dict[str, int]) -> Dict:
    """
    Test polarity structure for a single antonym pair.

    Args:
        word1: First antonym
        word2: Second antonym
        embeddings: Embedding matrix
        word_to_id: Word to ID mapping

    Returns:
        Test results dictionary
    """
    # Find sense-tagged versions
    word1_senses = find_word_sense(word1, word_to_id)
    word2_senses = find_word_sense(word2, word_to_id)

    if not word1_senses:
        return {'error': f"'{word1}' not found in vocabulary"}
    if not word2_senses:
        return {'error': f"'{word2}' not found in vocabulary"}

    # Use first sense (most common)
    word1_tagged = word1_senses[0]
    word2_tagged = word2_senses[0]

    # Get embeddings
    emb1 = embeddings[word_to_id[word1_tagged]]
    emb2 = embeddings[word_to_id[word2_tagged]]

    # Analyze polarity structure
    polarity = compute_polarity_structure(emb1, emb2)

    return {
        'word1': word1,
        'word2': word2,
        'word1_tagged': word1_tagged,
        'word2_tagged': word2_tagged,
        'polarity': polarity
    }


def test_not_operation(word: str, antonym: str, embeddings: np.ndarray,
                       word_to_id: Dict[str, int]) -> Dict:
    """
    Test if NOT(word) ≈ antonym.

    The NOT operation should flip the signs on opposing dimensions.

    Args:
        word: Word to negate
        antonym: Expected antonym
        embeddings: Embedding matrix
        word_to_id: Word to ID mapping

    Returns:
        Test results
    """
    # Find senses
    word_senses = find_word_sense(word, word_to_id)
    antonym_senses = find_word_sense(antonym, word_to_id)

    if not word_senses or not antonym_senses:
        return {'error': 'Words not found'}

    # Get embeddings
    word_emb = embeddings[word_to_id[word_senses[0]]]
    antonym_emb = embeddings[word_to_id[antonym_senses[0]]]

    # NOT operation: flip sign on all dimensions
    not_word_emb = -word_emb

    # Compare NOT(word) to antonym
    distance = np.linalg.norm(not_word_emb - antonym_emb)
    cosine_sim = np.dot(not_word_emb, antonym_emb) / (
        np.linalg.norm(not_word_emb) * np.linalg.norm(antonym_emb)
    )

    # Also compare word to antonym for baseline
    baseline_distance = np.linalg.norm(word_emb - antonym_emb)

    return {
        'word': word,
        'antonym': antonym,
        'not_word_to_antonym_distance': float(distance),
        'not_word_to_antonym_cosine': float(cosine_sim),
        'word_to_antonym_baseline': float(baseline_distance),
        'improvement': float(baseline_distance - distance)
    }


def get_test_antonym_pairs() -> List[Tuple[str, str]]:
    """Get antonym pairs for testing."""
    return [
        ("good", "bad"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("light", "dark"),
        ("high", "low"),
        ("new", "old"),
        ("clean", "dirty"),
        ("strong", "weak"),
        ("rich", "poor"),
        ("easy", "hard"),
        ("right", "wrong"),
        ("long", "short"),
        ("soft", "hard"),
    ]


def main():
    parser = argparse.ArgumentParser(description='Test polarity structure in embeddings')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings.npy')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for detailed results (JSON)')

    args = parser.parse_args()

    print("=" * 70)
    print("POLARITY STRUCTURE TESTS")
    print("=" * 70)
    print()

    # Load embeddings
    print("[1/4] Loading embeddings...")
    embeddings, word_to_id, id_to_word = load_embeddings(
        Path(args.embeddings),
        Path(args.vocab)
    )
    print(f"  Loaded {embeddings.shape[0]} words with {embeddings.shape[1]} dimensions")
    print()

    # Test antonym pairs
    print("[2/4] Testing antonym pair polarity...")
    print("=" * 70)

    antonym_pairs = get_test_antonym_pairs()
    results = []

    for word1, word2 in antonym_pairs:
        result = test_antonym_pair(word1, word2, embeddings, word_to_id)

        if 'error' in result:
            print(f"  {word1:12s} / {word2:12s}  SKIP ({result['error']})")
            continue

        polarity = result['polarity']

        print(f"  {word1:12s} / {word2:12s}  "
              f"opposing: {polarity['num_opposing']:2d} dims, "
              f"similar: {polarity['num_similar']:3d} dims, "
              f"dist: {polarity['euclidean_distance']:.3f}")

        results.append(result)

    print()

    # Analyze cross-pair patterns
    print("[3/4] Analyzing cross-pair selective polarity...")
    print("=" * 70)

    # Collect which dimensions are used by which pairs
    dim_usage = defaultdict(list)

    for result in results:
        pair_name = f"{result['word1']}/{result['word2']}"
        for dim in result['polarity']['opposing_dims']:
            dim_usage[dim].append(pair_name)

    # Find dimensions used by multiple pairs (less selective)
    # vs dimensions used by single pairs (more selective)
    multi_use_dims = {dim: pairs for dim, pairs in dim_usage.items() if len(pairs) > 1}
    single_use_dims = {dim: pairs for dim, pairs in dim_usage.items() if len(pairs) == 1}

    print(f"  Total dimensions used for polarity: {len(dim_usage)}")
    print(f"  Dimensions used by single pair (selective): {len(single_use_dims)}")
    print(f"  Dimensions used by multiple pairs (less selective): {len(multi_use_dims)}")

    if multi_use_dims:
        print(f"\n  Top shared dimensions:")
        sorted_multi = sorted(multi_use_dims.items(), key=lambda x: len(x[1]), reverse=True)
        for dim, pairs in sorted_multi[:10]:
            print(f"    Dim {dim:3d}: used by {len(pairs)} pairs - {', '.join(pairs[:3])}")

    print()

    # Calculate statistics
    num_opposing_list = [r['polarity']['num_opposing'] for r in results]
    num_similar_list = [r['polarity']['num_similar'] for r in results]
    distances = [r['polarity']['euclidean_distance'] for r in results]

    avg_opposing = np.mean(num_opposing_list)
    avg_similar = np.mean(num_similar_list)
    avg_distance = np.mean(distances)

    total_dims = embeddings.shape[1]

    print(f"  Average opposing dimensions: {avg_opposing:.1f} / {total_dims} ({avg_opposing/total_dims*100:.1f}%)")
    print(f"  Average similar dimensions: {avg_similar:.1f} / {total_dims} ({avg_similar/total_dims*100:.1f}%)")
    print(f"  Average distance: {avg_distance:.3f}")
    print()

    # Test NOT operation
    print("[4/4] Testing NOT operation...")
    print("=" * 70)

    not_test_pairs = [
        ("good", "bad"),
        ("hot", "cold"),
        ("big", "small"),
        ("happy", "sad"),
        ("fast", "slow"),
    ]

    not_results = []

    for word, antonym in not_test_pairs:
        result = test_not_operation(word, antonym, embeddings, word_to_id)

        if 'error' in result:
            continue

        print(f"  NOT({word:8s}) → {antonym:8s}  "
              f"dist: {result['not_word_to_antonym_distance']:.3f}  "
              f"baseline: {result['word_to_antonym_baseline']:.3f}  "
              f"improvement: {result['improvement']:.3f}")

        not_results.append(result)

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPolarity Structure:")
    print(f"  ✓ Tested {len(results)} antonym pairs")
    print(f"  ✓ Average {avg_opposing:.1f} opposing dimensions per pair")
    print(f"  ✓ Average {avg_similar:.1f} similar dimensions per pair")

    # Check if selective polarity is working
    selectivity_ratio = len(single_use_dims) / max(len(dim_usage), 1)
    print(f"\nSelective Polarity:")
    print(f"  {'✓' if selectivity_ratio > 0.5 else '✗'} Selectivity ratio: {selectivity_ratio:.1%}")

    if selectivity_ratio > 0.7:
        print(f"    EXCELLENT - High selective polarity (different pairs use different dims)")
    elif selectivity_ratio > 0.4:
        print(f"    GOOD - Moderate selective polarity")
    else:
        print(f"    POOR - Low selective polarity (many pairs share same dims)")

    # Check if NOT operation works
    if not_results:
        avg_improvement = np.mean([r['improvement'] for r in not_results])
        not_works = avg_improvement > 0

        print(f"\nNOT Operation:")
        print(f"  {'✓' if not_works else '✗'} Average improvement: {avg_improvement:.3f}")

        if not_works:
            print(f"    NOT operation works (NOT(word) closer to antonym than word)")
        else:
            print(f"    NOT operation may not be working optimally")

    # Check if sparsity in target range
    sparsity_vals = [np.mean(np.abs(embeddings[i]) < 0.01) for i in range(min(100, len(embeddings)))]
    avg_sparsity = np.mean(sparsity_vals)

    print(f"\nSparsity:")
    print(f"  Average: {avg_sparsity:.1%}")
    if 0.4 <= avg_sparsity <= 0.7:
        print(f"  ✓ Within target range (40-70%)")
    else:
        print(f"  ✗ Outside target range (40-70%)")

    print("=" * 70)
    print()

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'antonym_pair_results': results,
            'not_operation_results': not_results,
            'dimension_usage': {str(k): v for k, v in dim_usage.items()},
            'statistics': {
                'avg_opposing_dims': float(avg_opposing),
                'avg_similar_dims': float(avg_similar),
                'avg_distance': float(avg_distance),
                'selectivity_ratio': float(selectivity_ratio),
                'avg_sparsity': float(avg_sparsity),
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Detailed results saved to: {output_path}")
        print()

    print("Polarity testing complete!")


if __name__ == "__main__":
    main()
