"""
Test Polarity Constraint System

Validates that polarity constraints work correctly by:
1. Testing polarity dimension discovery
2. Validating NOT operation (NOT(good) ≈ bad)
3. Analyzing sign structure on polarity dimensions
4. Comparing models with and without polarity constraints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
from typing import List, Tuple

from src.embeddings.model import EmbeddingModel
from src.embeddings.polarity_discovery import PolarityDimensionDiscovery
from src.embeddings.logical_operators import LogicalOperators, test_logical_operators


def load_embeddings(checkpoint_dir: str):
    """Load trained embeddings from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)

    # Load embeddings
    embeddings = np.load(checkpoint_path / "embeddings.npy")

    # Load vocabulary
    with open(checkpoint_path / "vocabulary.json", 'r') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    # Create model
    model = EmbeddingModel(
        vocab_size=len(word_to_id),
        embedding_dim=embeddings.shape[1]
    )
    model.embeddings = embeddings
    model.word_to_id = word_to_id
    model.id_to_word = id_to_word

    print(f"Loaded {len(word_to_id)} words with {embeddings.shape[1]}-dim embeddings")

    return model


def get_antonym_pairs() -> List[Tuple[str, str]]:
    """Get common antonym pairs for testing."""
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


def test_polarity_discovery(model: EmbeddingModel, num_dims: int = 10):
    """Test polarity dimension discovery."""
    print("=" * 70)
    print("TEST 1: POLARITY DIMENSION DISCOVERY")
    print("=" * 70)
    print()

    antonym_pairs = get_antonym_pairs()

    discovery = PolarityDimensionDiscovery(model)

    # Discover polarity dimensions
    polarity_dims = discovery.discover_polarity_dimensions(
        antonym_pairs,
        top_k=num_dims,
        min_consistency=0.6
    )

    print(f"Discovered {len(polarity_dims)} polarity dimensions:")
    print()

    # Show details for each dimension
    for i, dim_idx in enumerate(polarity_dims, 1):
        print(f"Dimension {dim_idx}:")

        # Calculate statistics for this dimension
        dim_values = model.embeddings[:, dim_idx]
        mean = np.mean(dim_values)
        std = np.std(dim_values)

        # Check antonym pair differences
        diffs = []
        for word1, word2 in antonym_pairs:
            emb1 = model.get_embedding(word1)
            emb2 = model.get_embedding(word2)

            if emb1 is not None and emb2 is not None:
                diff = emb1[dim_idx] - emb2[dim_idx]
                diffs.append(diff)

        if diffs:
            mean_diff = np.mean(diffs)
            sign_consistency = abs(np.mean(np.sign(diffs)))

            print(f"  Mean: {mean:.3f}, Std: {std:.3f}")
            print(f"  Mean antonym difference: {mean_diff:.3f}")
            print(f"  Sign consistency: {sign_consistency:.3f}")

            # Show some example pairs
            print(f"  Example pairs:")
            for word1, word2 in antonym_pairs[:3]:
                emb1 = model.get_embedding(word1)
                emb2 = model.get_embedding(word2)
                if emb1 is not None and emb2 is not None:
                    val1 = emb1[dim_idx]
                    val2 = emb2[dim_idx]
                    print(f"    {word1}={val1:6.3f}, {word2}={val2:6.3f}, diff={val1-val2:6.3f}")
        print()

    return polarity_dims


def test_NOT_operation(model: EmbeddingModel, polarity_dims: List[int]):
    """Test that NOT(word1) ≈ word2 for antonym pairs."""
    print("=" * 70)
    print("TEST 2: NOT OPERATION VALIDATION")
    print("=" * 70)
    print()

    antonym_pairs = get_antonym_pairs()
    operators = LogicalOperators(polarity_dims)

    print(f"Testing NOT operation on {len(polarity_dims)} polarity dimensions")
    print()

    results = []

    for word1, word2 in antonym_pairs:
        # Get embeddings
        emb1 = model.get_embedding(word1)
        emb2 = model.get_embedding(word2)

        if emb1 is None or emb2 is None:
            continue

        # Apply NOT
        not_emb1 = operators.apply_NOT(emb1)

        # Compute similarity with expected antonym
        similarity = np.dot(not_emb1, emb2) / (np.linalg.norm(not_emb1) * np.linalg.norm(emb2))

        # Compute distance
        distance = np.linalg.norm(not_emb1 - emb2)

        results.append({
            'word1': word1,
            'word2': word2,
            'similarity': float(similarity),
            'distance': float(distance),
            'success': similarity > 0.5
        })

        print(f"  NOT({word1:10s}) → {word2:10s}: "
              f"similarity={similarity:6.3f}, "
              f"distance={distance:6.3f}, "
              f"{'✓' if similarity > 0.5 else '✗'}")

    print()

    # Statistics
    if results:
        avg_sim = np.mean([r['similarity'] for r in results])
        avg_dist = np.mean([r['distance'] for r in results])
        success_rate = sum(r['success'] for r in results) / len(results)

        print("Summary:")
        print(f"  Average similarity: {avg_sim:.3f}")
        print(f"  Average distance: {avg_dist:.3f}")
        print(f"  Success rate (>0.5 similarity): {success_rate*100:.1f}%")
    print()

    return results


def analyze_sign_structure(model: EmbeddingModel, polarity_dims: List[int]):
    """Analyze sign structure on polarity dimensions."""
    print("=" * 70)
    print("TEST 3: SIGN STRUCTURE ANALYSIS")
    print("=" * 70)
    print()

    antonym_pairs = get_antonym_pairs()

    print(f"Analyzing sign structure across {len(polarity_dims)} polarity dimensions")
    print()

    for dim_idx in polarity_dims:
        print(f"Dimension {dim_idx}:")

        # Check if antonym pairs have opposite signs
        same_sign_count = 0
        opposite_sign_count = 0
        zero_count = 0

        for word1, word2 in antonym_pairs:
            emb1 = model.get_embedding(word1)
            emb2 = model.get_embedding(word2)

            if emb1 is None or emb2 is None:
                continue

            val1 = emb1[dim_idx]
            val2 = emb2[dim_idx]

            sign_product = np.sign(val1) * np.sign(val2)

            if sign_product > 0:
                same_sign_count += 1
            elif sign_product < 0:
                opposite_sign_count += 1
            else:
                zero_count += 1

        total = same_sign_count + opposite_sign_count + zero_count

        print(f"  Antonym pairs with opposite signs: {opposite_sign_count}/{total} ({opposite_sign_count/total*100:.1f}%)")
        print(f"  Antonym pairs with same signs: {same_sign_count}/{total} ({same_sign_count/total*100:.1f}%)")

        # Show overall distribution
        dim_values = model.embeddings[:, dim_idx]
        positive_count = np.sum(dim_values > 0)
        negative_count = np.sum(dim_values < 0)
        zero_count = np.sum(dim_values == 0)

        print(f"  Overall sign distribution: +{positive_count}, -{negative_count}, 0={zero_count}")
        print()


def compare_before_after(before_checkpoint: str, after_checkpoint: str,
                          polarity_dims: List[int]):
    """Compare embeddings before and after polarity constraint training."""
    print("=" * 70)
    print("TEST 4: BEFORE/AFTER COMPARISON")
    print("=" * 70)
    print()

    # Load both models
    print("Loading models...")
    model_before = load_embeddings(before_checkpoint)
    model_after = load_embeddings(after_checkpoint)
    print()

    antonym_pairs = get_antonym_pairs()
    operators = LogicalOperators(polarity_dims)

    print("Comparing NOT operation accuracy:")
    print()

    # Test NOT on both models
    results_before = []
    results_after = []

    for word1, word2 in antonym_pairs:
        # Before
        emb1_before = model_before.get_embedding(word1)
        emb2_before = model_before.get_embedding(word2)

        # After
        emb1_after = model_after.get_embedding(word1)
        emb2_after = model_after.get_embedding(word2)

        if emb1_before is None or emb2_before is None or emb1_after is None or emb2_after is None:
            continue

        # Apply NOT and compute similarity
        not_emb1_before = operators.apply_NOT(emb1_before)
        not_emb1_after = operators.apply_NOT(emb1_after)

        sim_before = np.dot(not_emb1_before, emb2_before) / (np.linalg.norm(not_emb1_before) * np.linalg.norm(emb2_before))
        sim_after = np.dot(not_emb1_after, emb2_after) / (np.linalg.norm(not_emb1_after) * np.linalg.norm(emb2_after))

        results_before.append(sim_before)
        results_after.append(sim_after)

        improvement = sim_after - sim_before
        print(f"  {word1:10s} → {word2:10s}: "
              f"before={sim_before:6.3f}, "
              f"after={sim_after:6.3f}, "
              f"improvement={improvement:+6.3f}")

    print()

    if results_before and results_after:
        avg_before = np.mean(results_before)
        avg_after = np.mean(results_after)
        improvement = avg_after - avg_before

        print("Summary:")
        print(f"  Before polarity training: {avg_before:.3f} average similarity")
        print(f"  After polarity training: {avg_after:.3f} average similarity")
        print(f"  Improvement: {improvement:+.3f} ({improvement/avg_before*100:+.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Test polarity constraint system')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--polarity-dims-file', type=str,
                       help='Path to polarity dimensions JSON file (optional)')
    parser.add_argument('--num-polarity-dims', type=int, default=10,
                       help='Number of polarity dimensions to discover if not provided')
    parser.add_argument('--compare-before', type=str,
                       help='Checkpoint before polarity training (for comparison)')

    args = parser.parse_args()

    print("=" * 70)
    print("POLARITY CONSTRAINT SYSTEM TESTING")
    print("=" * 70)
    print()

    # Load model
    print("Loading embeddings...")
    model = load_embeddings(args.checkpoint)
    print()

    # Load or discover polarity dimensions
    if args.polarity_dims_file:
        print(f"Loading polarity dimensions from {args.polarity_dims_file}...")
        with open(args.polarity_dims_file, 'r') as f:
            data = json.load(f)
            polarity_dims = data['polarity_dimensions']
        print(f"Loaded {len(polarity_dims)} polarity dimensions")
        print()
    else:
        print("Discovering polarity dimensions...")
        polarity_dims = test_polarity_discovery(model, args.num_polarity_dims)

    # Test NOT operation
    not_results = test_NOT_operation(model, polarity_dims)

    # Analyze sign structure
    analyze_sign_structure(model, polarity_dims)

    # Compare before/after if requested
    if args.compare_before:
        compare_before_after(args.compare_before, args.checkpoint, polarity_dims)

    # Run comprehensive logical operator tests
    print("Running comprehensive logical operator tests...")
    print()
    test_logical_operators(model, polarity_dims)

    print("=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
