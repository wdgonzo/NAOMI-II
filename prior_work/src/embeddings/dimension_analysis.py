"""
Embedding Dimension Analysis

Analyzes individual dimensions of word embeddings to understand what semantic
aspects they encode (e.g., morality, size, concreteness).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from .dimension_types import (
    DimensionStats,
    DimensionComparison,
    AnchorValidationResult
)
from .anchors import AnchorDimensions


class DimensionAnalyzer:
    """
    Analyze individual embedding dimensions for semantic patterns.

    This class provides tools to understand what each dimension encodes,
    identify which dimensions are most informative, and validate anchor dimensions.
    """

    def __init__(self, embeddings: np.ndarray, word_to_id: Dict[str, int],
                 id_to_word: Dict[str, str], num_anchor_dims: int = 51):
        """
        Initialize dimension analyzer.

        Args:
            embeddings: Embedding matrix (vocab_size × embedding_dim)
            word_to_id: Mapping from word to embedding index
            id_to_word: Mapping from embedding index to word
            num_anchor_dims: Number of anchor dimensions (default: 51)
        """
        self.embeddings = embeddings
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.num_anchor_dims = num_anchor_dims

        # Load anchor definitions if available
        try:
            self.anchors = AnchorDimensions()
            self.anchor_names = self.anchors.get_dimension_names()
        except Exception:
            # If anchors not available, use None
            self.anchor_names = {i: None for i in range(num_anchor_dims)}

    def compute_dimension_variance(self) -> Dict[int, float]:
        """
        Calculate variance for each dimension across all words.

        High variance = informative dimension
        Low variance = potentially redundant dimension

        Returns:
            Dictionary mapping dimension index to variance
        """
        variances = {}
        for dim_idx in range(self.embedding_dim):
            dim_values = self.embeddings[:, dim_idx]
            variances[dim_idx] = float(np.var(dim_values))

        return variances

    def get_dimension_statistics(self, dim_idx: int,
                                  top_k: int = 10) -> DimensionStats:
        """
        Get comprehensive statistics for a specific dimension.

        Args:
            dim_idx: Index of dimension to analyze
            top_k: Number of top/bottom activations to include

        Returns:
            DimensionStats object with all statistics
        """
        dim_values = self.embeddings[:, dim_idx]

        # Basic statistics
        variance = float(np.var(dim_values))
        mean = float(np.mean(dim_values))
        std = float(np.std(dim_values))
        min_val = float(np.min(dim_values))
        max_val = float(np.max(dim_values))

        # Find top and bottom activations
        sorted_indices = np.argsort(dim_values)

        top_indices = sorted_indices[-top_k:][::-1]  # Highest values
        top_activations = [
            (self.id_to_word[str(idx)], float(dim_values[idx]))
            for idx in top_indices
        ]

        bottom_indices = sorted_indices[:top_k]  # Lowest values
        bottom_activations = [
            (self.id_to_word[str(idx)], float(dim_values[idx]))
            for idx in bottom_indices
        ]

        # Get dimension name if it's an anchor
        dim_name = self.anchor_names.get(dim_idx)

        return DimensionStats(
            index=dim_idx,
            name=dim_name,
            variance=variance,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            top_activations=top_activations,
            bottom_activations=bottom_activations
        )

    def find_high_variance_dimensions(self, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Identify dimensions with highest variance (most informative).

        Args:
            top_k: Number of dimensions to return

        Returns:
            List of (dimension_index, variance) tuples, sorted by variance descending
        """
        variances = self.compute_dimension_variance()
        sorted_dims = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:top_k]

    def find_low_variance_dimensions(self, threshold: float = 0.01) -> List[int]:
        """
        Find dimensions that barely vary (candidates for pruning).

        Args:
            threshold: Maximum variance to be considered "low"

        Returns:
            List of dimension indices with variance below threshold
        """
        variances = self.compute_dimension_variance()
        return [dim for dim, var in variances.items() if var < threshold]

    def compute_activation_heatmap(self, words: List[str]) -> np.ndarray:
        """
        Create heatmap showing which dimensions activate for which words.

        Args:
            words: List of words to analyze

        Returns:
            Numpy array of shape (len(words), embedding_dim) with activation values
        """
        heatmap = []

        for word in words:
            # Find word in vocabulary (try sense-tagged versions)
            word_ids = self._find_word_ids(word)

            if word_ids:
                # Use first occurrence if multiple senses
                embedding = self.embeddings[word_ids[0]]
                heatmap.append(embedding)
            else:
                # Word not found, use zeros
                heatmap.append(np.zeros(self.embedding_dim))

        return np.array(heatmap)

    def compare_words_on_dimension(self, word1: str, word2: str,
                                    dim_idx: int) -> Optional[DimensionComparison]:
        """
        Compare two words on a specific dimension.

        Args:
            word1: First word
            word2: Second word
            dim_idx: Dimension to compare

        Returns:
            DimensionComparison object, or None if words not found
        """
        word1_ids = self._find_word_ids(word1)
        word2_ids = self._find_word_ids(word2)

        if not word1_ids or not word2_ids:
            return None

        # Use first sense if multiple
        value1 = float(self.embeddings[word1_ids[0], dim_idx])
        value2 = float(self.embeddings[word2_ids[0], dim_idx])

        dim_name = self.anchor_names.get(dim_idx)

        return DimensionComparison(
            dimension_idx=dim_idx,
            dimension_name=dim_name,
            word1=word1,
            word2=word2,
            value1=value1,
            value2=value2
        )

    def find_discriminative_dimensions(
        self,
        word_pairs: List[Tuple[str, str]],
        top_k: int = 10,
        min_difference: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Find dimensions that best distinguish word pairs.

        For antonyms: Find dims where pairs differ most
        For synonyms: Find dims where pairs are most similar

        Args:
            word_pairs: List of (word1, word2) tuples
            top_k: Number of dimensions to return
            min_difference: Minimum average difference to be considered discriminative

        Returns:
            List of (dimension_index, avg_difference) sorted by difference descending
        """
        # Collect differences for each dimension
        dim_differences = defaultdict(list)

        for word1, word2 in word_pairs:
            word1_ids = self._find_word_ids(word1)
            word2_ids = self._find_word_ids(word2)

            if not word1_ids or not word2_ids:
                continue

            # Compare embeddings
            emb1 = self.embeddings[word1_ids[0]]
            emb2 = self.embeddings[word2_ids[0]]

            # Calculate difference for each dimension
            diff = np.abs(emb1 - emb2)

            for dim_idx in range(self.embedding_dim):
                dim_differences[dim_idx].append(diff[dim_idx])

        # Calculate average difference for each dimension
        avg_differences = {}
        for dim_idx, diffs in dim_differences.items():
            if diffs:
                avg_diff = float(np.mean(diffs))
                if avg_diff >= min_difference:
                    avg_differences[dim_idx] = avg_diff

        # Sort by average difference (descending)
        sorted_dims = sorted(avg_differences.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:top_k]

    def find_similarity_dimensions(
        self,
        word_pairs: List[Tuple[str, str]],
        top_k: int = 10,
        max_difference: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Find dimensions where word pairs are most similar.

        Args:
            word_pairs: List of (word1, word2) tuples
            top_k: Number of dimensions to return
            max_difference: Maximum average difference to be considered similar

        Returns:
            List of (dimension_index, avg_difference) sorted by difference ascending
        """
        # Collect differences for each dimension
        dim_differences = defaultdict(list)

        for word1, word2 in word_pairs:
            word1_ids = self._find_word_ids(word1)
            word2_ids = self._find_word_ids(word2)

            if not word1_ids or not word2_ids:
                continue

            emb1 = self.embeddings[word1_ids[0]]
            emb2 = self.embeddings[word2_ids[0]]

            diff = np.abs(emb1 - emb2)

            for dim_idx in range(self.embedding_dim):
                dim_differences[dim_idx].append(diff[dim_idx])

        # Calculate average difference for each dimension
        avg_differences = {}
        for dim_idx, diffs in dim_differences.items():
            if diffs:
                avg_diff = float(np.mean(diffs))
                if avg_diff <= max_difference:
                    avg_differences[dim_idx] = avg_diff

        # Sort by average difference (ascending - most similar first)
        sorted_dims = sorted(avg_differences.items(), key=lambda x: x[1])
        return sorted_dims[:top_k]

    def validate_anchor_dimension(
        self,
        dim_idx: int,
        expected_high: List[str],
        expected_low: List[str],
        threshold: float = 0.5
    ) -> AnchorValidationResult:
        """
        Validate that an anchor dimension behaves as expected.

        Args:
            dim_idx: Index of anchor dimension
            expected_high: Words that should have high values
            expected_low: Words that should have low values
            threshold: Minimum separation score to pass validation

        Returns:
            AnchorValidationResult with validation outcome
        """
        dim_name = self.anchor_names.get(dim_idx, f"dim_{dim_idx}")

        # Get dimension values for expected high/low words
        high_values = []
        for word in expected_high:
            word_ids = self._find_word_ids(word)
            if word_ids:
                val = float(self.embeddings[word_ids[0], dim_idx])
                high_values.append((word, val))

        low_values = []
        for word in expected_low:
            word_ids = self._find_word_ids(word)
            if word_ids:
                val = float(self.embeddings[word_ids[0], dim_idx])
                low_values.append((word, val))

        if not high_values or not low_values:
            return AnchorValidationResult(
                anchor_name=dim_name,
                dimension_idx=dim_idx,
                expected_behavior="High for some words, low for others",
                passes_validation=False,
                confidence_score=0.0,
                high_activation_words=[],
                low_activation_words=[],
                separation_score=0.0,
                notes="Insufficient vocabulary coverage for validation"
            )

        # Calculate separation score
        high_mean = np.mean([val for _, val in high_values])
        low_mean = np.mean([val for _, val in low_values])
        separation = abs(high_mean - low_mean)

        # Normalize by range
        all_values = [val for _, val in high_values] + [val for _, val in low_values]
        value_range = max(all_values) - min(all_values)
        separation_score = separation / value_range if value_range > 0 else 0.0

        # Check if passes validation
        passes = separation_score >= threshold

        # Sort by values
        high_values_sorted = sorted(high_values, key=lambda x: x[1], reverse=True)
        low_values_sorted = sorted(low_values, key=lambda x: x[1])

        return AnchorValidationResult(
            anchor_name=dim_name,
            dimension_idx=dim_idx,
            expected_behavior=f"High for {expected_high[:3]}, low for {expected_low[:3]}",
            passes_validation=passes,
            confidence_score=float(separation_score),
            high_activation_words=high_values_sorted,
            low_activation_words=low_values_sorted,
            separation_score=float(separation_score),
            notes=f"High mean: {high_mean:.3f}, Low mean: {low_mean:.3f}"
        )

    def _find_word_ids(self, word: str) -> List[int]:
        """
        Find all vocabulary IDs for a word (including sense-tagged versions).

        Args:
            word: Word to search for

        Returns:
            List of embedding indices
        """
        word_lower = word.lower()
        matching_ids = []

        # Look for exact match first
        if word_lower in self.word_to_id:
            matching_ids.append(self.word_to_id[word_lower])

        # Look for sense-tagged versions (word_wn.XX_pos)
        for vocab_word, idx in self.word_to_id.items():
            if vocab_word.startswith(word_lower + "_"):
                matching_ids.append(idx)

        return matching_ids
