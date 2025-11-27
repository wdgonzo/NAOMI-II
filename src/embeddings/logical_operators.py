"""
Logical Operators for Compositional Semantics

Implements logical operators (NOT, AND, OR) that operate on embeddings
using polarity dimensions to enable compositional semantics.

Key insight:
- NOT(good) ≈ bad by flipping signs on polarity dimensions
- AND/OR combine embeddings while preserving polarity structure
"""

import numpy as np
from typing import List, Optional, Dict


class LogicalOperators:
    """
    Logical operators for embedding composition.

    Enables compositional semantics through sign manipulation on polarity dimensions.
    """

    def __init__(self, polarity_dims: List[int]):
        """
        Initialize logical operators.

        Args:
            polarity_dims: List of dimension indices used for polarity constraints
        """
        self.polarity_dims = polarity_dims

    def apply_NOT(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply NOT operator by flipping signs on polarity dimensions.

        This implements the key insight: NOT(good) ≈ bad
        by multiplying polarity dimensions by -1.

        Args:
            embedding: Input embedding vector

        Returns:
            Negated embedding with flipped polarity dimensions

        Example:
            >>> # Assuming dimension 34 is a polarity dimension encoding morality
            >>> good_emb = np.array([0.1, 0.2, ..., 0.5, ...])  # dim 34 = 0.5
            >>> bad_emb = operators.apply_NOT(good_emb)
            >>> # bad_emb[34] ≈ -0.5 (sign flipped)
        """
        result = embedding.copy()

        # Flip signs on polarity dimensions
        for dim_idx in self.polarity_dims:
            result[dim_idx] *= -1

        return result

    def apply_AND(self, emb1: np.ndarray, emb2: np.ndarray,
                  method: str = 'average') -> np.ndarray:
        """
        Apply AND operator to combine two embeddings.

        Multiple combination strategies:
        - 'average': Simple average (preserves polarity structure)
        - 'weighted': Weighted by magnitude (emphasizes stronger features)
        - 'min': Element-wise minimum (conservative combination)

        Args:
            emb1: First embedding
            emb2: Second embedding
            method: Combination method

        Returns:
            Combined embedding

        Example:
            >>> # "big dog" = AND(big, dog)
            >>> big_emb = model.get_embedding("big")
            >>> dog_emb = model.get_embedding("dog")
            >>> big_dog = operators.apply_AND(big_emb, dog_emb)
        """
        if method == 'average':
            return (emb1 + emb2) / 2.0

        elif method == 'weighted':
            # Weight by magnitude (stronger features dominate)
            mag1 = np.linalg.norm(emb1)
            mag2 = np.linalg.norm(emb2)

            if mag1 + mag2 < 1e-8:
                return (emb1 + emb2) / 2.0

            w1 = mag1 / (mag1 + mag2)
            w2 = mag2 / (mag1 + mag2)

            return w1 * emb1 + w2 * emb2

        elif method == 'min':
            # Conservative: take minimum absolute values, preserve signs
            result = np.zeros_like(emb1)
            for i in range(len(result)):
                if abs(emb1[i]) < abs(emb2[i]):
                    result[i] = emb1[i]
                else:
                    result[i] = emb2[i]
            return result

        else:
            raise ValueError(f"Unknown AND method: {method}")

    def apply_OR(self, emb1: np.ndarray, emb2: np.ndarray,
                 method: str = 'max') -> np.ndarray:
        """
        Apply OR operator to combine two embeddings.

        Multiple combination strategies:
        - 'max': Element-wise maximum magnitude (liberal combination)
        - 'average': Simple average
        - 'weighted': Weighted average by magnitude

        Args:
            emb1: First embedding
            emb2: Second embedding
            method: Combination method

        Returns:
            Combined embedding

        Example:
            >>> # "cat or dog" = OR(cat, dog)
            >>> cat_emb = model.get_embedding("cat")
            >>> dog_emb = model.get_embedding("dog")
            >>> cat_or_dog = operators.apply_OR(cat_emb, dog_emb)
        """
        if method == 'max':
            # Liberal: take maximum absolute values, preserve signs
            result = np.zeros_like(emb1)
            for i in range(len(result)):
                if abs(emb1[i]) > abs(emb2[i]):
                    result[i] = emb1[i]
                else:
                    result[i] = emb2[i]
            return result

        elif method == 'average':
            return (emb1 + emb2) / 2.0

        elif method == 'weighted':
            # Weight by magnitude
            mag1 = np.linalg.norm(emb1)
            mag2 = np.linalg.norm(emb2)

            if mag1 + mag2 < 1e-8:
                return (emb1 + emb2) / 2.0

            w1 = mag1 / (mag1 + mag2)
            w2 = mag2 / (mag1 + mag2)

            return w1 * emb1 + w2 * emb2

        else:
            raise ValueError(f"Unknown OR method: {method}")

    def validate_NOT_operation(self, model, word_pairs: List[tuple]) -> Dict:
        """
        Validate that NOT operation produces expected results.

        Tests if NOT(word1) ≈ word2 for antonym pairs.

        Args:
            model: EmbeddingModel with get_embedding method
            word_pairs: List of (word, antonym) tuples

        Returns:
            Validation statistics

        Example:
            >>> pairs = [("good", "bad"), ("hot", "cold"), ("big", "small")]
            >>> stats = operators.validate_NOT_operation(model, pairs)
            >>> print(f"NOT accuracy: {stats['average_similarity']:.3f}")
        """
        similarities = []
        results = []

        for word1, word2 in word_pairs:
            # Get embeddings
            emb1 = model.get_embedding(word1)
            emb2 = model.get_embedding(word2)

            if emb1 is None or emb2 is None:
                continue

            # Apply NOT
            not_emb1 = self.apply_NOT(emb1)

            # Compute similarity with expected antonym
            similarity = self._cosine_similarity(not_emb1, emb2)
            similarities.append(similarity)

            results.append({
                'word1': word1,
                'word2': word2,
                'similarity': float(similarity),
                'success': similarity > 0.5  # Threshold for "similar enough"
            })

        # Compute statistics
        if similarities:
            return {
                'num_pairs': len(similarities),
                'average_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'success_rate': sum(r['success'] for r in results) / len(results),
                'results': results
            }
        else:
            return {
                'num_pairs': 0,
                'average_similarity': 0.0,
                'std_similarity': 0.0,
                'success_rate': 0.0,
                'results': []
            }

    def analyze_polarity_structure(self, embeddings: np.ndarray) -> Dict:
        """
        Analyze polarity dimension structure across all embeddings.

        Args:
            embeddings: Full embedding matrix (vocab_size × embedding_dim)

        Returns:
            Analysis statistics
        """
        stats = {}

        for dim_idx in self.polarity_dims:
            dim_values = embeddings[:, dim_idx]

            # Count sign distribution
            positive_count = np.sum(dim_values > 0)
            negative_count = np.sum(dim_values < 0)
            zero_count = np.sum(dim_values == 0)

            stats[dim_idx] = {
                'mean': float(np.mean(dim_values)),
                'std': float(np.std(dim_values)),
                'min': float(np.min(dim_values)),
                'max': float(np.max(dim_values)),
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'zero_count': int(zero_count),
                'balance': float(abs(positive_count - negative_count) / len(dim_values))
            }

        return stats

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


def test_logical_operators(model, polarity_dims: List[int]):
    """
    Test logical operators on a trained model.

    Args:
        model: Trained EmbeddingModel
        polarity_dims: List of polarity dimension indices
    """
    print("=" * 70)
    print("TESTING LOGICAL OPERATORS")
    print("=" * 70)
    print()

    operators = LogicalOperators(polarity_dims)

    # Test NOT operation
    print("[1/3] Testing NOT operation...")
    print("-" * 70)

    antonym_pairs = [
        ("good", "bad"),
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("light", "dark"),
        ("high", "low"),
        ("new", "old"),
        ("clean", "dirty"),
        ("strong", "weak")
    ]

    not_results = operators.validate_NOT_operation(model, antonym_pairs)

    print(f"  Tested {not_results['num_pairs']} antonym pairs")
    print(f"  Average similarity (NOT(word1) vs word2): {not_results['average_similarity']:.3f}")
    print(f"  Standard deviation: {not_results['std_similarity']:.3f}")
    print(f"  Success rate (>0.5 similarity): {not_results['success_rate']*100:.1f}%")
    print()

    # Show top results
    print("  Top 5 best NOT operations:")
    sorted_results = sorted(not_results['results'], key=lambda x: x['similarity'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"    {i}. NOT({result['word1']}) → {result['word2']}: "
              f"similarity = {result['similarity']:.3f}")
    print()

    # Test AND operation
    print("[2/3] Testing AND operation...")
    print("-" * 70)

    # Test "big dog" = AND(big, dog)
    big_emb = model.get_embedding("big")
    dog_emb = model.get_embedding("dog")

    if big_emb is not None and dog_emb is not None:
        big_dog = operators.apply_AND(big_emb, dog_emb)

        # Compare with similar concepts
        print("  AND(big, dog) similarity to:")
        test_words = ["dog", "big", "animal", "pet", "large", "small"]
        for word in test_words:
            word_emb = model.get_embedding(word)
            if word_emb is not None:
                sim = operators._cosine_similarity(big_dog, word_emb)
                print(f"    {word}: {sim:.3f}")
    else:
        print("  Skipping (embeddings not found)")
    print()

    # Test polarity structure
    print("[3/3] Analyzing polarity structure...")
    print("-" * 70)

    structure = operators.analyze_polarity_structure(model.embeddings)

    print(f"  Polarity dimensions: {len(polarity_dims)}")
    print()
    print("  Dimension statistics:")
    for dim_idx in polarity_dims[:10]:  # Show first 10
        stats = structure[dim_idx]
        print(f"    Dim {dim_idx:3d}: "
              f"mean={stats['mean']:6.3f}, "
              f"std={stats['std']:5.3f}, "
              f"pos={stats['positive_count']:4d}, "
              f"neg={stats['negative_count']:4d}, "
              f"balance={stats['balance']:.3f}")

    print()
    print("=" * 70)
    print()

    return {
        'not_results': not_results,
        'polarity_structure': structure
    }
