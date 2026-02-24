"""
Polarity Dimension Discovery

Automatically discovers which embedding dimensions should have opposite-sign
constraints for antonyms (e.g., good=+0.5, bad=-0.5 on morality dimension).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path


class PolarityDimensionDiscovery:
    """
    Discover dimensions where antonyms should have opposite signs.

    Key insight: Antonyms like good/bad should be oppositely signed on
    semantic dimensions (morality: good=+, bad=-) to enable compositional
    semantics (NOT(good) ≈ bad).
    """

    def __init__(self, embeddings: np.ndarray, word_to_id: Dict[str, int],
                 id_to_word: Dict[str, str]):
        """
        Initialize polarity discovery.

        Args:
            embeddings: Embedding matrix (vocab_size × embedding_dim)
            word_to_id: Word to embedding index mapping
            id_to_word: Embedding index to word mapping
        """
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size, self.embedding_dim = embeddings.shape

    def discover_polarity_dimensions(
        self,
        antonym_pairs: List[Tuple[str, str]],
        top_k: int = 10,
        min_consistency: float = 0.6
    ) -> List[int]:
        """
        Find dimensions where antonyms consistently differ with opposite signs.

        Looks for dimensions with:
        1. High discriminative power (large absolute difference)
        2. Sign consistency (one word always higher than other)

        Args:
            antonym_pairs: List of (word1, word2) antonym tuples
            top_k: Number of polarity dimensions to return
            min_consistency: Minimum sign consistency (0-1) to be considered polarity dim

        Returns:
            List of dimension indices suitable for polarity constraints
        """
        print(f"  Discovering polarity dimensions from {len(antonym_pairs)} antonym pairs...")

        # Track signed differences for each dimension
        dim_signed_diffs = defaultdict(list)
        valid_pairs = 0

        for word1, word2 in antonym_pairs:
            emb1 = self._get_embedding(word1)
            emb2 = self._get_embedding(word2)

            if emb1 is None or emb2 is None:
                continue

            valid_pairs += 1

            # Calculate SIGNED difference (preserves direction)
            diff = emb1 - emb2

            for dim_idx in range(self.embedding_dim):
                dim_signed_diffs[dim_idx].append(diff[dim_idx])

        if valid_pairs == 0:
            print("    Warning: No valid antonym pairs found")
            return []

        # Score each dimension
        polarity_scores = {}

        for dim_idx in range(self.embedding_dim):
            if dim_idx not in dim_signed_diffs:
                continue

            diffs = np.array(dim_signed_diffs[dim_idx])

            # Metric 1: Discriminative power (mean absolute difference)
            discriminative_power = float(np.mean(np.abs(diffs)))

            # Metric 2: Sign consistency (do all pairs differ in same direction?)
            signs = np.sign(diffs)
            # Consistency = |mean of signs| (1.0 = perfect, 0.0 = random)
            sign_consistency = float(np.abs(np.mean(signs)))

            # Only consider dimensions with minimum consistency
            if sign_consistency < min_consistency:
                continue

            # Combined score: both discriminative AND consistent
            polarity_scores[dim_idx] = discriminative_power * sign_consistency

        # Sort by score and return top K
        sorted_dims = sorted(polarity_scores.items(), key=lambda x: x[1], reverse=True)
        polarity_dims = [dim for dim, _ in sorted_dims[:top_k]]

        print(f"    Found {len(polarity_dims)} polarity dimensions (consistency >= {min_consistency})")

        # Print details
        for rank, (dim_idx, score) in enumerate(sorted_dims[:top_k], 1):
            diffs = dim_signed_diffs[dim_idx]
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            sign_cons = np.abs(np.mean(np.sign(diffs)))

            print(f"      {rank:2d}. Dim {dim_idx:3d}: score={score:.4f}, "
                  f"mean_diff={mean_diff:+.3f}, std={std_diff:.3f}, "
                  f"sign_consistency={sign_cons:.2f}")

        return polarity_dims

    def validate_polarity_dimensions(
        self,
        polarity_dims: List[int],
        antonym_pairs: List[Tuple[str, str]]
    ) -> Dict:
        """
        Validate that polarity dimensions actually have opposite signs for antonyms.

        Args:
            polarity_dims: Dimensions to validate
            antonym_pairs: Antonym pairs to test

        Returns:
            Validation statistics
        """
        print(f"  Validating {len(polarity_dims)} polarity dimensions...")

        stats = {
            'total_pairs': 0,
            'opposite_sign_count': 0,
            'same_sign_count': 0,
            'per_dimension': {}
        }

        for dim_idx in polarity_dims:
            dim_stats = {
                'opposite_count': 0,
                'same_count': 0,
                'mean_diff': 0.0,
                'examples': []
            }

            diffs = []

            for word1, word2 in antonym_pairs:
                emb1 = self._get_embedding(word1)
                emb2 = self._get_embedding(word2)

                if emb1 is None or emb2 is None:
                    continue

                val1 = emb1[dim_idx]
                val2 = emb2[dim_idx]

                # Check sign
                if np.sign(val1) != np.sign(val2) and np.sign(val1) != 0 and np.sign(val2) != 0:
                    dim_stats['opposite_count'] += 1
                else:
                    dim_stats['same_count'] += 1

                diffs.append(val1 - val2)

                # Store examples
                if len(dim_stats['examples']) < 5:
                    dim_stats['examples'].append({
                        'word1': word1,
                        'word2': word2,
                        'val1': float(val1),
                        'val2': float(val2),
                        'opposite_signs': bool(np.sign(val1) != np.sign(val2))
                    })

            if diffs:
                dim_stats['mean_diff'] = float(np.mean(diffs))

            stats['per_dimension'][dim_idx] = dim_stats
            stats['opposite_sign_count'] += dim_stats['opposite_count']
            stats['same_sign_count'] += dim_stats['same_count']

        stats['total_pairs'] = stats['opposite_sign_count'] + stats['same_sign_count']
        if stats['total_pairs'] > 0:
            stats['opposite_sign_ratio'] = stats['opposite_sign_count'] / stats['total_pairs']
        else:
            stats['opposite_sign_ratio'] = 0.0

        print(f"    Opposite signs: {stats['opposite_sign_count']}/{stats['total_pairs']} "
              f"({stats['opposite_sign_ratio']*100:.1f}%)")

        return stats

    def analyze_polarity_dimension(
        self,
        dim_idx: int,
        antonym_pairs: List[Tuple[str, str]],
        top_k: int = 10
    ) -> Dict:
        """
        Detailed analysis of a single polarity dimension.

        Args:
            dim_idx: Dimension to analyze
            antonym_pairs: Antonym pairs
            top_k: Number of examples to include

        Returns:
            Analysis report
        """
        dim_values = self.embeddings[:, dim_idx]

        # Get words with highest positive and negative values
        sorted_indices = np.argsort(dim_values)

        positive_pole = []
        for idx in sorted_indices[-top_k:][::-1]:
            word = self.id_to_word[str(idx)]
            val = float(dim_values[idx])
            if val > 0:
                positive_pole.append((word, val))

        negative_pole = []
        for idx in sorted_indices[:top_k]:
            word = self.id_to_word[str(idx)]
            val = float(dim_values[idx])
            if val < 0:
                negative_pole.append((word, val))

        # Check antonym pairs on this dimension
        antonym_examples = []
        for word1, word2 in antonym_pairs[:top_k]:
            emb1 = self._get_embedding(word1)
            emb2 = self._get_embedding(word2)

            if emb1 is None or emb2 is None:
                continue

            val1 = float(emb1[dim_idx])
            val2 = float(emb2[dim_idx])

            antonym_examples.append({
                'word1': word1,
                'word2': word2,
                'val1': val1,
                'val2': val2,
                'diff': val1 - val2,
                'opposite_signs': bool(np.sign(val1) != np.sign(val2))
            })

        return {
            'dimension': dim_idx,
            'mean': float(np.mean(dim_values)),
            'std': float(np.std(dim_values)),
            'min': float(np.min(dim_values)),
            'max': float(np.max(dim_values)),
            'positive_pole': positive_pole,
            'negative_pole': negative_pole,
            'antonym_examples': antonym_examples
        }

    def save_polarity_config(self, polarity_dims: List[int], output_path: str):
        """
        Save polarity dimension configuration for later use.

        Args:
            polarity_dims: Polarity dimensions to save
            output_path: Path to save configuration
        """
        config = {
            'polarity_dimensions': polarity_dims,
            'num_dimensions': len(polarity_dims),
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"    Saved polarity config to: {output_path}")

    @staticmethod
    def load_polarity_config(config_path: str) -> List[int]:
        """
        Load polarity dimension configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            List of polarity dimension indices
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        return config['polarity_dimensions']

    def _get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding for a word (handles sense-tagged versions).

        Args:
            word: Word to look up

        Returns:
            Embedding vector or None if not found
        """
        word_lower = word.lower()

        # Try exact match first
        if word_lower in self.word_to_id:
            idx = self.word_to_id[word_lower]
            return self.embeddings[idx]

        # Try sense-tagged versions
        for vocab_word, idx in self.word_to_id.items():
            if vocab_word.startswith(word_lower + "_"):
                return self.embeddings[idx]

        return None
