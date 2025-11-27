"""
Relationship-Specific Dimensional Analysis

Analyzes which dimensions encode specific semantic relationships like
synonymy, antonymy, hypernymy, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .dimension_types import RelationshipDimensionProfile, SemanticAxis
from .dimension_analysis import DimensionAnalyzer


class RelationshipDimensionAnalysis:
    """
    Analyze which dimensions encode specific semantic relationships.

    Key insight: Antonyms should be similar in MOST dimensions (both moral concepts,
    both abstract, etc.) but differ strongly in just a FEW key dimensions
    (positive vs negative valence).
    """

    def __init__(self, analyzer: DimensionAnalyzer):
        """
        Initialize relationship analyzer.

        Args:
            analyzer: DimensionAnalyzer instance with loaded embeddings
        """
        self.analyzer = analyzer
        self.embeddings = analyzer.embeddings
        self.word_to_id = analyzer.word_to_id
        self.embedding_dim = analyzer.embedding_dim

    def analyze_synonym_dimensions(
        self,
        synonym_pairs: List[Tuple[str, str]],
        similarity_threshold: float = 0.1,
        top_k: int = 20
    ) -> RelationshipDimensionProfile:
        """
        Find dimensions where synonyms have similar values.

        Synonyms should be similar across MOST dimensions.

        Args:
            synonym_pairs: List of (synonym1, synonym2) tuples
            similarity_threshold: Max difference to be considered "similar"
            top_k: Number of top dimensions to track

        Returns:
            RelationshipDimensionProfile for synonyms
        """
        print(f"  Analyzing {len(synonym_pairs)} synonym pairs...")

        # Calculate per-dimension differences
        dim_differences = defaultdict(list)
        valid_pairs = 0

        for word1, word2 in synonym_pairs:
            word1_ids = self.analyzer._find_word_ids(word1)
            word2_ids = self.analyzer._find_word_ids(word2)

            if not word1_ids or not word2_ids:
                continue

            valid_pairs += 1
            emb1 = self.embeddings[word1_ids[0]]
            emb2 = self.embeddings[word2_ids[0]]

            # Calculate difference for each dimension
            diff = np.abs(emb1 - emb2)

            for dim_idx in range(self.embedding_dim):
                dim_differences[dim_idx].append(float(diff[dim_idx]))

        # Calculate statistics
        mean_diff_per_dim = {}
        std_diff_per_dim = {}
        similarity_dims = []  # Dims where synonyms are similar
        discriminative_dims = []  # Dims where synonyms differ (unusual)

        for dim_idx in range(self.embedding_dim):
            if dim_idx not in dim_differences:
                continue

            diffs = dim_differences[dim_idx]
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs))

            mean_diff_per_dim[dim_idx] = mean_diff
            std_diff_per_dim[dim_idx] = std_diff

            # Classify dimension
            if mean_diff <= similarity_threshold:
                similarity_dims.append(dim_idx)
            else:
                discriminative_dims.append(dim_idx)

        # Calculate importance scores (inverse of mean difference for synonyms)
        importance_scores = {}
        for dim_idx, mean_diff in mean_diff_per_dim.items():
            # Lower difference = higher importance for synonymy
            importance_scores[dim_idx] = 1.0 / (mean_diff + 1e-6)

        print(f"    Valid pairs: {valid_pairs}")
        print(f"    Similar dimensions: {len(similarity_dims)}")
        print(f"    Discriminative dimensions: {len(discriminative_dims)}")

        return RelationshipDimensionProfile(
            relationship_type="synonym",
            num_pairs=valid_pairs,
            discriminative_dims=discriminative_dims[:top_k],
            similarity_dims=similarity_dims[:top_k],
            importance_scores=importance_scores,
            mean_difference_per_dim=mean_diff_per_dim,
            std_difference_per_dim=std_diff_per_dim
        )

    def analyze_antonym_dimensions(
        self,
        antonym_pairs: List[Tuple[str, str]],
        discriminative_threshold: float = 0.5,
        top_k: int = 20
    ) -> RelationshipDimensionProfile:
        """
        Find dimensions where antonyms differ maximally.

        Key insight: Antonyms should be SIMILAR in most dimensions (both moral,
        both abstract) but DIFFER in just a few key dimensions (positive/negative).

        Args:
            antonym_pairs: List of (antonym1, antonym2) tuples
            discriminative_threshold: Min difference to be "discriminative"
            top_k: Number of top dimensions to track

        Returns:
            RelationshipDimensionProfile for antonyms
        """
        print(f"  Analyzing {len(antonym_pairs)} antonym pairs...")

        # Calculate per-dimension differences
        dim_differences = defaultdict(list)
        valid_pairs = 0

        for word1, word2 in antonym_pairs:
            word1_ids = self.analyzer._find_word_ids(word1)
            word2_ids = self.analyzer._find_word_ids(word2)

            if not word1_ids or not word2_ids:
                continue

            valid_pairs += 1
            emb1 = self.embeddings[word1_ids[0]]
            emb2 = self.embeddings[word2_ids[0]]

            # Calculate difference for each dimension
            diff = np.abs(emb1 - emb2)

            for dim_idx in range(self.embedding_dim):
                dim_differences[dim_idx].append(float(diff[dim_idx]))

        # Calculate statistics
        mean_diff_per_dim = {}
        std_diff_per_dim = {}
        discriminative_dims = []  # Dims where antonyms differ (expected: few)
        similarity_dims = []  # Dims where antonyms are similar (expected: many)

        for dim_idx in range(self.embedding_dim):
            if dim_idx not in dim_differences:
                continue

            diffs = dim_differences[dim_idx]
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs))

            mean_diff_per_dim[dim_idx] = mean_diff
            std_diff_per_dim[dim_idx] = std_diff

            # Classify dimension
            if mean_diff >= discriminative_threshold:
                discriminative_dims.append(dim_idx)
            else:
                similarity_dims.append(dim_idx)

        # Calculate importance scores (mean difference for antonyms)
        importance_scores = {}
        for dim_idx, mean_diff in mean_diff_per_dim.items():
            # Higher difference = higher importance for antonymy
            importance_scores[dim_idx] = mean_diff

        # Sort discriminative dims by importance
        discriminative_dims.sort(key=lambda d: importance_scores[d], reverse=True)

        print(f"    Valid pairs: {valid_pairs}")
        print(f"    Discriminative dimensions: {len(discriminative_dims)} (differ strongly)")
        print(f"    Similar dimensions: {len(similarity_dims)} (remain similar)")
        print(f"    Ratio: {len(similarity_dims)/max(len(discriminative_dims), 1):.1f}:1 similar:different")

        return RelationshipDimensionProfile(
            relationship_type="antonym",
            num_pairs=valid_pairs,
            discriminative_dims=discriminative_dims[:top_k],
            similarity_dims=similarity_dims[:top_k],
            importance_scores=importance_scores,
            mean_difference_per_dim=mean_diff_per_dim,
            std_difference_per_dim=std_diff_per_dim
        )

    def analyze_hypernym_dimensions(
        self,
        hypernym_pairs: List[Tuple[str, str]],
        top_k: int = 20
    ) -> RelationshipDimensionProfile:
        """
        Find dimensions encoding is-a relationships.

        Args:
            hypernym_pairs: List of (specific, general) tuples (e.g., dog, animal)
            top_k: Number of top dimensions to track

        Returns:
            RelationshipDimensionProfile for hypernyms
        """
        print(f"  Analyzing {len(hypernym_pairs)} hypernym pairs...")

        # Calculate per-dimension differences
        dim_differences = defaultdict(list)
        valid_pairs = 0

        for specific, general in hypernym_pairs:
            specific_ids = self.analyzer._find_word_ids(specific)
            general_ids = self.analyzer._find_word_ids(general)

            if not specific_ids or not general_ids:
                continue

            valid_pairs += 1
            emb_specific = self.embeddings[specific_ids[0]]
            emb_general = self.embeddings[general_ids[0]]

            # Calculate difference
            diff = np.abs(emb_specific - emb_general)

            for dim_idx in range(self.embedding_dim):
                dim_differences[dim_idx].append(float(diff[dim_idx]))

        # Calculate statistics
        mean_diff_per_dim = {}
        std_diff_per_dim = {}

        for dim_idx, diffs in dim_differences.items():
            mean_diff_per_dim[dim_idx] = float(np.mean(diffs))
            std_diff_per_dim[dim_idx] = float(np.std(diffs))

        # Importance is based on consistency (low std) and moderate difference
        importance_scores = {}
        for dim_idx in mean_diff_per_dim:
            mean_diff = mean_diff_per_dim[dim_idx]
            std_diff = std_diff_per_dim[dim_idx]
            # Prefer moderate differences with low variance
            importance_scores[dim_idx] = mean_diff / (std_diff + 1e-6)

        # Find most discriminative
        sorted_dims = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        discriminative_dims = [dim for dim, _ in sorted_dims[:top_k]]

        print(f"    Valid pairs: {valid_pairs}")

        return RelationshipDimensionProfile(
            relationship_type="hypernym",
            num_pairs=valid_pairs,
            discriminative_dims=discriminative_dims,
            similarity_dims=[],
            importance_scores=importance_scores,
            mean_difference_per_dim=mean_diff_per_dim,
            std_difference_per_dim=std_diff_per_dim
        )

    def discover_semantic_axis(
        self,
        axis_name: str,
        positive_words: List[str],
        negative_words: List[str]
    ) -> Optional[SemanticAxis]:
        """
        Discover a semantic axis from example words.

        For example:
        - axis_name="size", positive=["big", "huge"], negative=["small", "tiny"]
        - axis_name="morality", positive=["good", "virtuous"], negative=["bad", "evil"]

        Args:
            axis_name: Name of the semantic concept
            positive_words: Words at the high end of the axis
            negative_words: Words at the low end of the axis

        Returns:
            SemanticAxis object, or None if axis cannot be discovered
        """
        # Get embeddings for positive/negative words
        positive_embeds = []
        for word in positive_words:
            word_ids = self.analyzer._find_word_ids(word)
            if word_ids:
                positive_embeds.append(self.embeddings[word_ids[0]])

        negative_embeds = []
        for word in negative_words:
            word_ids = self.analyzer._find_word_ids(word)
            if word_ids:
                negative_embeds.append(self.embeddings[word_ids[0]])

        if not positive_embeds or not negative_embeds:
            return None

        # Calculate mean embeddings
        positive_mean = np.mean(positive_embeds, axis=0)
        negative_mean = np.mean(negative_embeds, axis=0)

        # Find dimensions with largest difference
        diff = np.abs(positive_mean - negative_mean)

        # Find primary dimension (largest difference)
        primary_dim = int(np.argmax(diff))
        primary_diff = float(diff[primary_dim])

        # Find contributing dimensions (top 5)
        sorted_dims = np.argsort(diff)[::-1]
        contributing_dims = sorted_dims[:5].tolist()

        # Calculate correlation score (how well primary dim separates the words)
        positive_values = [emb[primary_dim] for emb in positive_embeds]
        negative_values = [emb[primary_dim] for emb in negative_embeds]

        # Separation score
        positive_mean_val = np.mean(positive_values)
        negative_mean_val = np.mean(negative_values)
        separation = abs(positive_mean_val - negative_mean_val)

        # Normalize by std
        all_values = positive_values + negative_values
        std = np.std(all_values)
        correlation_score = separation / (std + 1e-6)

        return SemanticAxis(
            name=axis_name,
            primary_dimension=primary_dim,
            correlation_score=float(correlation_score),
            positive_pole_words=positive_words,
            negative_pole_words=negative_words,
            contributing_dimensions=contributing_dims
        )

    def discover_semantic_axes(
        self,
        word_groups: Dict[str, Dict[str, List[str]]]
    ) -> List[SemanticAxis]:
        """
        Discover multiple semantic axes.

        Args:
            word_groups: Dictionary mapping axis_name to {'positive': [...], 'negative': [...]}

        Returns:
            List of discovered SemanticAxis objects
        """
        axes = []

        for axis_name, groups in word_groups.items():
            positive_words = groups.get('positive', [])
            negative_words = groups.get('negative', [])

            axis = self.discover_semantic_axis(axis_name, positive_words, negative_words)
            if axis:
                axes.append(axis)

        return axes
