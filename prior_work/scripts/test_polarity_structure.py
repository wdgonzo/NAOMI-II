"""
Polarity Structure Validation Script

Tests whether transparent semantic dimensions have been successfully discovered.

Tests:
1. **Polarity Discovery** - Are there 5-20 dimensions with consistent opposite-sign patterns?
2. **Compositional Semantics** - Does NOT(good) ≈ bad work?
3. **Selective Polarity** - Do different antonym types use different dimensions?
4. **Sparsity** - Are embeddings 40-70% sparse?
5. **Dimensional Consistency** - Does each dimension encode one semantic axis?

Usage:
    python scripts/test_polarity_structure.py \\
        --embeddings checkpoints/phase1_bootstrap/embeddings_best.npy \\
        --vocabulary checkpoints/phase1_bootstrap/vocabulary.json \\
        --polarity-dims checkpoints/phase1_bootstrap/polarity_dimensions.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


def load_data(embeddings_path: str, vocab_path: str, polarity_path: str = None):
    """Load embeddings, vocabulary, and polarity dimensions."""
    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word_to_id = vocab_data['word_to_id']
        id_to_word = vocab_data['id_to_word']
        # Convert string keys back to ints
        id_to_word = {int(k): v for k, v in id_to_word.items()}

    # Load polarity dimensions if provided
    polarity_dims = None
    if polarity_path and Path(polarity_path).exists():
        with open(polarity_path, 'r') as f:
            polarity_data = json.load(f)
            polarity_dims = polarity_data.get('polarity_dims', [])

    return embeddings, word_to_id, id_to_word, polarity_dims


def test_1_polarity_discovery(embeddings: np.ndarray,
                                polarity_dims: List[int]) -> Dict:
    """Test 1: Are polarity dimensions discovered?"""
    print("\n" + "="*70)
    print("TEST 1: POLARITY DIMENSION DISCOVERY")
    print("="*70)

    if polarity_dims is None or len(polarity_dims) == 0:
        print("❌ FAIL: No polarity dimensions discovered")
        return {'passed': False, 'num_dims': 0}

    num_dims = len(polarity_dims)
    print(f"✓ Found {num_dims} polarity dimensions: {polarity_dims[:10]}...")

    if num_dims < 5:
        print(f"⚠️ WARNING: Only {num_dims} polarity dims (expected 5-20)")
        print("  May need more training or higher polarity weight")
    elif num_dims > 30:
        print(f"⚠️ WARNING: {num_dims} polarity dims (expected 5-20)")
        print("  May have over-polarization (too many dims = polarity)")
    else:
        print(f"✓ GOOD: {num_dims} polarity dims in expected range (5-20)")

    # Check dimensionality distribution
    for dim_idx in polarity_dims[:5]:
        dim_values = embeddings[:, dim_idx]
        variance = np.var(dim_values)
        activation = np.mean(np.abs(dim_values) > 0.01)
        print(f"  Dim {dim_idx}: variance={variance:.4f}, activation={activation:.1%}")

    passed = 5 <= num_dims <= 30
    return {'passed': passed, 'num_dims': num_dims}


def find_word(word: str, word_to_id: Dict[str, int]) -> int:
    """Find word index (handles sense-tagged versions)."""
    # Try exact match
    if word in word_to_id:
        return word_to_id[word]

    # Try sense-tagged versions
    for vocab_word, idx in word_to_id.items():
        if vocab_word.startswith(word + "_wn."):
            return idx

    return None


def test_2_compositional_semantics(embeddings: np.ndarray,
                                     word_to_id: Dict[str, int],
                                     id_to_word: Dict[int, str],
                                     polarity_dims: List[int]) -> Dict:
    """Test 2: Does NOT(good) ≈ bad work?"""
    print("\n" + "="*70)
    print("TEST 2: COMPOSITIONAL SEMANTICS (NOT OPERATION)")
    print("="*70)

    # Test pairs: (word, expected_opposite)
    test_pairs = [
        ('good', 'bad'),
        ('hot', 'cold'),
        ('big', 'small'),
        ('happy', 'sad'),
        ('fast', 'slow'),
    ]

    results = []

    for word, expected_opposite in test_pairs:
        word_idx = find_word(word, word_to_id)
        if word_idx is None:
            print(f"  {word:10s} → (not in vocabulary)")
            continue

        # Get word embedding
        word_emb = embeddings[word_idx].copy()

        # Apply NOT operation (flip polarity dimensions)
        not_emb = word_emb.copy()
        if polarity_dims:
            not_emb[polarity_dims] *= -1
        else:
            # If no polarity dims, flip all
            not_emb *= -1

        # Find nearest neighbors
        distances = np.linalg.norm(embeddings - not_emb, axis=1)
        nearest_indices = np.argsort(distances)[:5]

        # Check if expected opposite is in top 5
        expected_idx = find_word(expected_opposite, word_to_id)
        rank = None
        if expected_idx is not None:
            try:
                rank = list(nearest_indices).index(expected_idx) + 1
            except ValueError:
                rank = None

        # Display results
        print(f"\n  NOT({word}):")
        for rank_i, idx in enumerate(nearest_indices, 1):
            neighbor_word = id_to_word[idx]
            neighbor_display = neighbor_word.split('_wn.')[0] if '_wn.' in neighbor_word else neighbor_word
            dist = distances[idx]
            marker = " ← TARGET!" if neighbor_display == expected_opposite else ""
            print(f"    {rank_i}. {neighbor_display:15s} (dist: {dist:.3f}){marker}")

        if rank is not None and rank <= 5:
            print(f"  ✓ PASS: '{expected_opposite}' found at rank {rank}")
            results.append(True)
        else:
            print(f"  ❌ FAIL: '{expected_opposite}' not in top 5")
            results.append(False)

    # Overall pass rate
    pass_rate = np.mean(results) if results else 0.0
    print(f"\nOverall success rate: {pass_rate:.1%} ({sum(results)}/{len(results)})")

    passed = pass_rate >= 0.6  # 60% threshold
    if passed:
        print("✓ COMPOSITIONAL SEMANTICS WORKING!")
    else:
        print("❌ COMPOSITIONAL SEMANTICS NOT WORKING")
        print("  Polarity structure may not be strong enough")

    return {'passed': passed, 'success_rate': pass_rate}


def test_3_selective_polarity(embeddings: np.ndarray,
                                word_to_id: Dict[str, int],
                                polarity_dims: List[int]) -> Dict:
    """Test 3: Do different antonym types use different dimensions?"""
    print("\n" + "="*70)
    print("TEST 3: SELECTIVE POLARITY")
    print("="*70)

    if not polarity_dims or len(polarity_dims) < 2:
        print("❌ SKIP: Need at least 2 polarity dims for selective test")
        return {'passed': False, 'skip': True}

    # Different antonym types
    antonym_groups = {
        'morality': [('good', 'bad'), ('moral', 'immoral')],
        'temperature': [('hot', 'cold'), ('warm', 'cool')],
        'size': [('big', 'small'), ('large', 'tiny')],
        'speed': [('fast', 'slow'), ('quick', 'leisurely')],
    }

    # For each group, find which polarity dims are most active
    group_dims = {}

    for group_name, pairs in antonym_groups.items():
        dim_scores = defaultdict(float)

        for word1, word2 in pairs:
            idx1 = find_word(word1, word_to_id)
            idx2 = find_word(word2, word_to_id)

            if idx1 is None or idx2 is None:
                continue

            emb1 = embeddings[idx1]
            emb2 = embeddings[idx2]

            # Score each polarity dim by how much they oppose
            for dim in polarity_dims:
                val1 = emb1[dim]
                val2 = emb2[dim]

                # Opposite signs = good polarity
                if val1 * val2 < 0:  # Opposite signs
                    dim_scores[dim] += abs(val1 - val2)

        # Get top 3 dimensions for this group
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        group_dims[group_name] = [dim for dim, score in sorted_dims[:3]]

        print(f"\n  {group_name.capitalize()} antonyms:")
        for dim, score in sorted_dims[:3]:
            print(f"    Dim {dim}: score={score:.4f}")

    # Check for dimension specialization (different groups use different dims)
    all_dims_used = set()
    for dims in group_dims.values():
        all_dims_used.update(dims)

    # Calculate overlap
    total_dims_used = sum(len(dims) for dims in group_dims.values())
    unique_dims_used = len(all_dims_used)
    overlap_ratio = 1.0 - (unique_dims_used / total_dims_used) if total_dims_used > 0 else 1.0

    print(f"\n  Dimension overlap ratio: {overlap_ratio:.1%}")
    print(f"  Total dims assigned: {total_dims_used}, Unique dims: {unique_dims_used}")

    # Good selective polarity = low overlap (different groups use different dims)
    if overlap_ratio < 0.3:  # Less than 30% overlap
        print("  ✓ EXCELLENT: Different antonym types use different dimensions!")
        passed = True
    elif overlap_ratio < 0.5:
        print("  ✓ GOOD: Some dimensional specialization")
        passed = True
    else:
        print("  ⚠️ POOR: High overlap - polarity may not be selective")
        passed = False

    return {'passed': passed, 'overlap_ratio': overlap_ratio}


def test_4_sparsity(embeddings: np.ndarray) -> Dict:
    """Test 4: Are embeddings 40-70% sparse?"""
    print("\n" + "="*70)
    print("TEST 4: SPARSITY (DIMENSIONAL SPECIALIZATION)")
    print("="*70)

    # Compute sparsity
    near_zero = np.abs(embeddings) < 0.01
    sparsity = np.mean(near_zero)

    print(f"  Overall sparsity: {sparsity:.1%}")

    # Target range: 40-70%
    if 0.4 <= sparsity <= 0.7:
        print(f"  ✓ PASS: Sparsity in target range (40-70%)")
        passed = True
    elif 0.3 <= sparsity < 0.4:
        print(f"  ⚠️ WARNING: Sparsity slightly low ({sparsity:.1%})")
        print("  Embeddings may be too dense (all dims active)")
        passed = True  # Still acceptable
    elif 0.7 < sparsity <= 0.8:
        print(f"  ⚠️ WARNING: Sparsity slightly high ({sparsity:.1%})")
        print("  Embeddings may be too sparse (unused capacity)")
        passed = True  # Still acceptable
    else:
        print(f"  ❌ FAIL: Sparsity out of range ({sparsity:.1%})")
        passed = False

    # Per-dimension activation
    activation_per_dim = np.mean(np.abs(embeddings) > 0.01, axis=0)
    mean_activation = np.mean(activation_per_dim)
    median_activation = np.median(activation_per_dim)

    print(f"  Mean dimension activation: {mean_activation:.1%}")
    print(f"  Median dimension activation: {median_activation:.1%}")

    # Categorize dimensions
    unused = np.sum(activation_per_dim < 0.1)
    sparse = np.sum((activation_per_dim >= 0.1) & (activation_per_dim < 0.3))
    moderate = np.sum((activation_per_dim >= 0.3) & (activation_per_dim < 0.6))
    saturated = np.sum(activation_per_dim >= 0.6)

    print(f"\n  Dimension categories:")
    print(f"    Unused (<10%):    {unused:3d} dims")
    print(f"    Sparse (10-30%):  {sparse:3d} dims")
    print(f"    Moderate (30-60%):{moderate:3d} dims")
    print(f"    Saturated (>60%): {saturated:3d} dims")

    if moderate + sparse > saturated:
        print("  ✓ GOOD: Most dimensions show selective usage!")
    else:
        print("  ⚠️ Most dimensions are saturated (poor specialization)")

    return {
        'passed': passed,
        'sparsity': float(sparsity),
        'mean_activation': float(mean_activation),
        'dim_categories': {'unused': int(unused), 'sparse': int(sparse),
                           'moderate': int(moderate), 'saturated': int(saturated)}
    }


def test_5_dimensional_consistency(embeddings: np.ndarray,
                                     id_to_word: Dict[int, str],
                                     polarity_dims: List[int]) -> Dict:
    """Test 5: Does each dimension encode one semantic axis?"""
    print("\n" + "="*70)
    print("TEST 5: DIMENSIONAL CONSISTENCY")
    print("="*70)

    if not polarity_dims or len(polarity_dims) == 0:
        print("❌ SKIP: No polarity dimensions to analyze")
        return {'passed': False, 'skip': True}

    # For each polarity dim, show top positive and negative words
    for dim_idx in polarity_dims[:5]:  # Show top 5 polarity dims
        dim_values = embeddings[:, dim_idx]

        # Find extremes
        sorted_indices = np.argsort(dim_values)

        print(f"\n  Dimension {dim_idx}:")

        # Positive pole
        print("    Positive pole:")
        for idx in sorted_indices[-5:][::-1]:
            word = id_to_word[idx]
            word_display = word.split('_wn.')[0] if '_wn.' in word else word
            value = dim_values[idx]
            if abs(value) > 0.01:
                print(f"      {word_display:20s} {value:+.3f}")

        # Negative pole
        print("    Negative pole:")
        for idx in sorted_indices[:5]:
            word = id_to_word[idx]
            word_display = word.split('_wn.')[0] if '_wn.' in word else word
            value = dim_values[idx]
            if abs(value) > 0.01:
                print(f"      {word_display:20s} {value:+.3f}")

    print("\n  Manual inspection required:")
    print("  - Do positive/negative poles have consistent semantic meaning?")
    print("  - Can you identify what each dimension represents?")
    print("    (e.g., morality, temperature, size, speed, etc.)")

    # Auto-check: Polarity dimensions should have balanced positive/negative
    balanced_count = 0
    for dim_idx in polarity_dims:
        dim_values = embeddings[:, dim_idx]
        non_zero = dim_values[np.abs(dim_values) > 0.01]

        if len(non_zero) < 10:
            continue

        positive_count = np.sum(non_zero > 0)
        negative_count = np.sum(non_zero < 0)
        balance_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)

        if balance_ratio > 0.3:  # At least 30% balance
            balanced_count += 1

    balance_pct = balanced_count / len(polarity_dims) if polarity_dims else 0
    print(f"\n  Balanced polarity dimensions: {balanced_count}/{len(polarity_dims)} ({balance_pct:.1%})")

    passed = balance_pct >= 0.6  # 60% should be balanced
    if passed:
        print("  ✓ PASS: Polarity dimensions show good balance")
    else:
        print("  ⚠️ WARNING: Many polarity dims are skewed (not balanced)")

    return {'passed': passed, 'balance_pct': balance_pct}


def main():
    parser = argparse.ArgumentParser(description='Validate polarity structure')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Path to embeddings (.npy file)')
    parser.add_argument('--vocabulary', type=str, required=True,
                       help='Path to vocabulary (.json file)')
    parser.add_argument('--polarity-dims', type=str, default=None,
                       help='Path to polarity dimensions (.json file)')
    parser.add_argument('--output', type=str, default='results/polarity_validation.json',
                       help='Output path for validation results')

    args = parser.parse_args()

    print("="*70)
    print("POLARITY STRUCTURE VALIDATION")
    print("="*70)

    # Load data
    print("\nLoading data...")
    embeddings, word_to_id, id_to_word, polarity_dims = load_data(
        args.embeddings, args.vocabulary, args.polarity_dims
    )

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Vocabulary: {len(word_to_id):,} words")
    if polarity_dims:
        print(f"  Polarity dimensions: {len(polarity_dims)}")

    # Run tests
    results = {}

    results['test1'] = test_1_polarity_discovery(embeddings, polarity_dims)
    results['test2'] = test_2_compositional_semantics(embeddings, word_to_id, id_to_word, polarity_dims)
    results['test3'] = test_3_selective_polarity(embeddings, word_to_id, polarity_dims)
    results['test4'] = test_4_sparsity(embeddings)
    results['test5'] = test_5_dimensional_consistency(embeddings, id_to_word, polarity_dims)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    passed_tests = sum(1 for r in results.values() if r.get('passed', False))
    total_tests = len(results)

    print(f"\nTests passed: {passed_tests}/{total_tests}")
    print(f"\nDetailed results:")
    print(f"  Test 1 (Polarity Discovery):      {'✓ PASS' if results['test1']['passed'] else '❌ FAIL'}")
    print(f"  Test 2 (Compositional Semantics): {'✓ PASS' if results['test2']['passed'] else '❌ FAIL'}")
    print(f"  Test 3 (Selective Polarity):      {'✓ PASS' if results['test3']['passed'] else '❌ FAIL'}")
    print(f"  Test 4 (Sparsity):                {'✓ PASS' if results['test4']['passed'] else '❌ FAIL'}")
    print(f"  Test 5 (Dimensional Consistency): {'✓ PASS' if results['test5']['passed'] else '❌ FAIL'}")

    # Overall verdict
    print("\n" + "="*70)
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Transparent dimensions working!")
    elif passed_tests >= 3:
        print("⚠️ PARTIAL SUCCESS: Most tests passed, but improvement needed")
    else:
        print("❌ FAILURE: Transparent dimensions not discovered")
        print("   Recommendations:")
        print("   - Increase polarity weight (try 20.0 or 50.0)")
        print("   - Decrease sparsity weight (try 0.00001)")
        print("   - Train for more epochs (300-500)")
        print("   - Try WordNet-only bootstrap first")
    print("="*70)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
