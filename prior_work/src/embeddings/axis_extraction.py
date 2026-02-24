"""
Semantic axis extraction and validation from antonym clusters.

This module extracts interpretable semantic axes from clusters of antonym pairs,
including automatic axis naming, pole extraction, coherence validation, and
hierarchical sub-axis detection.

Author: NAOMI-II Development Team
Date: 2025-11-30
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from collections import Counter
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

# Import n-ary pole detection
try:
    from .pole_detection import detect_poles_from_pairs
except ImportError:
    try:
        from pole_detection import detect_poles_from_pairs
    except ImportError:
        detect_poles_from_pairs = None

STOPWORDS = set(stopwords.words('english'))


class AxisExtractor:
    """Extracts and validates semantic axes from antonym clusters."""

    def __init__(self, antonym_pairs: List[Dict], cluster_assignments: np.ndarray):
        """
        Initialize axis extractor.

        Args:
            antonym_pairs: List of antonym pair dicts
            cluster_assignments: Cluster ID for each pair (from fcluster)
        """
        self.antonym_pairs = antonym_pairs
        self.clusters = cluster_assignments
        self.unique_clusters = sorted(set(cluster_assignments))

        # Cache for WordNet lookups
        self._synset_cache = {}
        self._path_similarity_cache = {}

    def _get_synset(self, synset_name: str) -> Optional[Any]:
        """Get synset with caching."""
        if synset_name not in self._synset_cache:
            try:
                self._synset_cache[synset_name] = wn.synset(synset_name)
            except Exception:
                self._synset_cache[synset_name] = None
        return self._synset_cache[synset_name]

    def extract_all_axes(
        self,
        min_size: int = 3,
        min_coherence: float = 0.3,
        min_separation: float = 0.5,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Extract all semantic axes from clusters.

        Args:
            min_size: Minimum number of pairs for valid axis
            min_coherence: Minimum pole coherence score
            min_separation: Minimum pole separation score
            verbose: Print progress

        Returns:
            List of axis dicts
        """
        print(f"Extracting semantic axes from {len(self.unique_clusters)} clusters...")

        axes = []
        rejected = []

        for cluster_id in self.unique_clusters:
            # Get cluster members
            member_indices = np.where(self.clusters == cluster_id)[0]
            member_pairs = [self.antonym_pairs[i] for i in member_indices]

            # Extract axis info
            axis = self._extract_axis_info(cluster_id, member_pairs)

            # Validate
            is_valid, reason = self._validate_axis(
                axis,
                min_size=min_size,
                min_coherence=min_coherence,
                min_separation=min_separation
            )

            if is_valid:
                axes.append(axis)
            else:
                rejected.append((axis, reason))

        # Sort by size (largest first)
        axes.sort(key=lambda x: x['size'], reverse=True)

        # Assign final axis IDs
        for i, axis in enumerate(axes):
            axis['axis_id'] = i

        if verbose:
            print(f"\nExtracted {len(axes)} valid axes")
            print(f"Rejected {len(rejected)} clusters:")
            rejection_reasons = Counter(r for _, r in rejected)
            for reason, count in rejection_reasons.most_common():
                print(f"  - {reason}: {count}")

        return axes

    def _extract_axis_info(self, cluster_id: int, member_pairs: List[Dict]) -> Dict:
        """
        Extract axis information from cluster members.

        Args:
            cluster_id: Cluster ID
            member_pairs: List of antonym pairs in cluster

        Returns:
            Axis dict with poles, name, metrics, detection method
        """
        # Extract n-ary poles using hybrid detection (WordNet, triadic, frequency)
        pole_sets, pole_names, detection_method, confidence = self._extract_poles_nary(member_pairs)

        # Name axis from shared definition keywords
        axis_name = self._extract_axis_name(member_pairs)

        # Compute coherence (how related are words within each pole?)
        # Pass detection_method to use appropriate coherence calculation
        coherence = self._compute_pole_coherence_nary(pole_sets, detection_method=detection_method)

        # Compute separation (how distant are the poles from each other?)
        separation = self._compute_pole_separation_nary(pole_sets)

        # Get representative pairs (most central to cluster)
        representative_pairs = self._get_representative_pairs_nary(
            member_pairs,
            pole_sets,
            top_k=5
        )

        return {
            'axis_id': cluster_id,  # Temporary, will be reassigned
            'name': axis_name,
            'poles': [sorted(list(pole)) for pole in pole_sets],
            'pole_names': pole_names,
            'representative_pairs': representative_pairs,
            'member_pairs': member_pairs,
            'coherence_score': coherence,
            'separation_score': separation,
            'size': len(member_pairs),
            'cluster_id': cluster_id,  # Original cluster ID
            'source': 'antonym_clustering',
            'detection_method': detection_method,
            'detection_confidence': confidence
        }

    def _extract_poles_nary(
        self,
        member_pairs: List[Dict],
        resolution: float = 1.0,
        min_pole_size: int = 1
    ) -> Tuple[List[Set[str]], List[str], str, float]:
        """
        Extract n-ary poles from member pairs using hybrid detection strategy.

        Args:
            member_pairs: List of antonym pair dicts
            resolution: Louvain resolution parameter (higher = more communities)
            min_pole_size: Minimum words per pole

        Returns:
            (pole_word_sets, pole_names, detection_method, confidence)
        """
        if detect_poles_from_pairs is None:
            # Fallback to binary detection
            positive_pole, negative_pole = self._extract_poles(member_pairs)
            return ([positive_pole, negative_pole], ['positive', 'negative'], 'frequency_binary', 0.5)

        # Use hybrid pole detection (tries WordNet, triadic, then frequency)
        pole_sets, pole_names, method, confidence = detect_poles_from_pairs(
            member_pairs,
            resolution=resolution,
            min_pole_size=min_pole_size
        )

        # Fallback to binary if detection fails
        if not pole_sets or len(pole_sets) == 0:
            positive_pole, negative_pole = self._extract_poles(member_pairs)
            return ([positive_pole, negative_pole], ['positive', 'negative'], 'frequency_binary_fallback', 0.3)

        return (pole_sets, pole_names, method, confidence)

    def _extract_poles(self, member_pairs: List[Dict]) -> Tuple[Set[str], Set[str]]:
        """
        Extract positive and negative poles from member pairs (binary fallback).

        Uses frequency-based heuristic: words appearing more often on one side
        of antonym pairs are assigned to that pole.

        Args:
            member_pairs: List of antonym pair dicts

        Returns:
            (positive_pole_words, negative_pole_words)
        """
        # Count appearances of each word on each side
        word1_counts = Counter()  # Positive pole
        word2_counts = Counter()  # Negative pole

        for pair in member_pairs:
            word1_counts[pair['word1']] += 1
            word2_counts[pair['word2']] += 1

        # Handle words appearing on both sides (ambiguous)
        all_words_1 = set(word1_counts.keys())
        all_words_2 = set(word2_counts.keys())
        ambiguous = all_words_1 & all_words_2

        # Assign ambiguous words to side where they appear more
        for word in ambiguous:
            if word1_counts[word] > word2_counts[word]:
                del word2_counts[word]
            else:
                del word1_counts[word]

        positive_pole = set(word1_counts.keys())
        negative_pole = set(word2_counts.keys())

        return positive_pole, negative_pole

    def _extract_axis_name(self, member_pairs: List[Dict]) -> str:
        """
        Extract axis name from shared definition keywords using TF-IDF.

        Args:
            member_pairs: List of antonym pair dicts

        Returns:
            Most representative concept name (e.g., "morality", "size")
        """
        # Collect all definitions
        all_definitions = []
        for pair in member_pairs:
            syn1 = self._get_synset(pair['synset1'])
            syn2 = self._get_synset(pair['synset2'])

            if syn1:
                all_definitions.append(syn1.definition())
            if syn2:
                all_definitions.append(syn2.definition())

        if not all_definitions:
            return "unknown"

        # Extract content words
        combined_text = " ".join(all_definitions)
        tokens = word_tokenize(combined_text.lower())
        content_words = [t for t in tokens if t.isalpha() and t not in STOPWORDS]

        # Get most common meaningful words
        word_freq = Counter(content_words)

        # Filter out very common words (appear in >50% of definitions)
        n_defs = len(all_definitions)
        filtered_words = [
            word for word, count in word_freq.items()
            if count <= n_defs * 0.5  # Not too common
        ]

        if not filtered_words:
            # Fall back to most common if filtering removed everything
            filtered_words = [word for word, _ in word_freq.most_common(10)]

        # Get top keywords (prefer nouns and adjectives)
        top_keywords = Counter(filtered_words).most_common(5)

        if top_keywords:
            return top_keywords[0][0]
        else:
            return "unknown"

    def _compute_pole_coherence_nary(
        self,
        pole_sets: List[Set[str]],
        detection_method: str = 'unknown'
    ) -> float:
        """
        Compute coherence within all poles using WordNet path similarity.

        Uses detection-method-aware fallback:
        - WordNet/triadic: Require WordNet paths (high quality)
        - Frequency-based: Use default coherence if no paths (coverage)

        Args:
            pole_sets: List of pole word sets
            detection_method: How poles were detected

        Returns:
            Average path similarity within poles [0, 1]
        """
        all_similarities = []

        # Compute pairwise similarities within each pole
        for pole in pole_sets:
            for w1 in pole:
                for w2 in pole:
                    if w1 < w2:  # Only compute each pair once
                        sim = self._wordnet_path_similarity(w1, w2)
                        if sim is not None:
                            all_similarities.append(sim)

        if all_similarities:
            return np.mean(all_similarities)
        else:
            # Fallback coherence when no WordNet paths available
            if detection_method in ['frequency_binary', 'frequency_binary_fallback']:
                # Frequency-based poles: assume moderate coherence
                # They were grouped by clustering, so have some implicit coherence
                return 0.4
            elif detection_method == 'triadic_closure':
                # Triadic poles: assume reasonable coherence
                return 0.5
            else:
                # WordNet-based or unknown: require actual paths
                return 0.0

    def _compute_pole_coherence(
        self,
        positive_pole: Set[str],
        negative_pole: Set[str]
    ) -> float:
        """
        Compute coherence within each pole using WordNet path similarity (binary).

        Args:
            positive_pole: Set of positive pole words
            negative_pole: Set of negative pole words

        Returns:
            Average path similarity within poles [0, 1]
        """
        return self._compute_pole_coherence_nary([positive_pole, negative_pole])

    def _compute_pole_separation_nary(self, pole_sets: List[Set[str]]) -> float:
        """
        Compute separation between all pole pairs (poles should be far apart).

        Args:
            pole_sets: List of pole word sets

        Returns:
            1 - average_path_similarity [0, 1]
        """
        cross_similarities = []

        # Compute pairwise similarities between all pole pairs
        for i, pole1 in enumerate(pole_sets):
            for j, pole2 in enumerate(pole_sets):
                if i < j:  # Only compute each pair once
                    for w1 in pole1:
                        for w2 in pole2:
                            sim = self._wordnet_path_similarity(w1, w2)
                            if sim is not None:
                                cross_similarities.append(sim)

        if cross_similarities:
            # High similarity → low separation
            # Return 1 - similarity for separation score
            return 1.0 - np.mean(cross_similarities)
        else:
            # No similarities computable → assume separated
            return 0.5

    def _compute_pole_separation(
        self,
        positive_pole: Set[str],
        negative_pole: Set[str]
    ) -> float:
        """
        Compute separation between poles (antonyms should be far apart) (binary).

        Args:
            positive_pole: Set of positive pole words
            negative_pole: Set of negative pole words

        Returns:
            1 - average_path_similarity [0, 1]
        """
        return self._compute_pole_separation_nary([positive_pole, negative_pole])

    def _wordnet_path_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Compute WordNet path similarity between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Path similarity [0, 1] or None if not computable
        """
        # Check cache
        cache_key = tuple(sorted([word1, word2]))
        if cache_key in self._path_similarity_cache:
            return self._path_similarity_cache[cache_key]

        # Get all synsets for both words
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if not synsets1 or not synsets2:
            self._path_similarity_cache[cache_key] = None
            return None

        # Find maximum similarity across all synset pairs
        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim is not None and sim > max_sim:
                    max_sim = sim

        result = max_sim if max_sim > 0.0 else None
        self._path_similarity_cache[cache_key] = result
        return result

    def _get_representative_pairs_nary(
        self,
        member_pairs: List[Dict],
        pole_sets: List[Set[str]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get most representative antonym pairs for this axis (n-ary).

        Args:
            member_pairs: All pairs in cluster
            pole_sets: List of pole word sets
            top_k: Number of representative pairs to return

        Returns:
            List of top_k most representative pairs
        """
        # Score each pair by how "central" it is to the poles
        scored_pairs = []

        for pair in member_pairs:
            # Higher score if both words are in their respective poles
            score = 0.0

            # Check if word1 and word2 are in different poles
            word1_pole = -1
            word2_pole = -1

            for i, pole in enumerate(pole_sets):
                if pair['word1'] in pole:
                    word1_pole = i
                if pair['word2'] in pole:
                    word2_pole = i

            # Give score if both words are in poles and in different poles
            if word1_pole >= 0 and word2_pole >= 0:
                if word1_pole != word2_pole:
                    score = 2.0  # Both words in different poles
                else:
                    score = 0.5  # Both words in same pole (less representative)

            scored_pairs.append((score, pair))

        # Sort by score (descending) and take top_k
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in scored_pairs[:top_k]]

    def _get_representative_pairs(
        self,
        member_pairs: List[Dict],
        positive_pole: Set[str],
        negative_pole: Set[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get most representative antonym pairs for this axis (binary).

        Args:
            member_pairs: All pairs in cluster
            positive_pole: Positive pole words
            negative_pole: Negative pole words
            top_k: Number of representative pairs to return

        Returns:
            List of top_k most representative pairs
        """
        return self._get_representative_pairs_nary(
            member_pairs,
            [positive_pole, negative_pole],
            top_k
        )

    def _validate_axis(
        self,
        axis: Dict,
        min_size: int,
        min_coherence: float,
        min_separation: float
    ) -> Tuple[bool, str]:
        """
        Validate that axis meets quality criteria.

        Args:
            axis: Axis dict
            min_size: Minimum number of pairs
            min_coherence: Minimum coherence score
            min_separation: Minimum separation score

        Returns:
            (is_valid, rejection_reason)
        """
        # Check size
        if axis['size'] < min_size:
            return False, f"too_small (size={axis['size']} < {min_size})"

        # Check coherence
        if axis['coherence_score'] < min_coherence:
            return False, f"low_coherence (coherence={axis['coherence_score']:.3f} < {min_coherence})"

        # Check separation
        if axis['separation_score'] < min_separation:
            return False, f"low_separation (separation={axis['separation_score']:.3f} < {min_separation})"

        return True, "valid"

    def generate_axis_report(self, axes: List[Dict]) -> str:
        """
        Generate human-readable report of discovered axes.

        Args:
            axes: List of axis dicts

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 100)
        lines.append("DISCOVERED SEMANTIC AXES")
        lines.append("=" * 100)
        lines.append(f"Total axes: {len(axes)}")
        lines.append("")

        for i, axis in enumerate(axes, 1):
            lines.append(f"\n{'=' * 100}")
            lines.append(f"AXIS {i}: {axis['name'].upper()}")
            lines.append(f"{'=' * 100}")
            lines.append(f"Cluster ID: {axis['cluster_id']}")
            lines.append(f"Size: {axis['size']} pairs/words")
            lines.append(f"Source: {axis.get('source', 'unknown')}")
            lines.append(f"Coherence: {axis['coherence_score']:.4f}")
            lines.append(f"Separation: {axis['separation_score']:.4f}")
            lines.append("")

            # Handle both old binary format and new n-ary format
            if 'poles' in axis and 'pole_names' in axis:
                # N-ary format
                for pole_name, pole_words in zip(axis['pole_names'], axis['poles']):
                    lines.append(f"{pole_name.upper()} POLE ({len(pole_words)} words):")
                    pole_str = ", ".join(pole_words[:20])
                    if len(pole_words) > 20:
                        pole_str += f", ... (+{len(pole_words) - 20} more)"
                    lines.append(f"  {pole_str}")
                    lines.append("")
            else:
                # Old binary format (backward compatibility)
                lines.append(f"POSITIVE POLE ({len(axis.get('positive_pole', []))} words):")
                pole_str = ", ".join(axis.get('positive_pole', [])[:20])
                if len(axis.get('positive_pole', [])) > 20:
                    pole_str += f", ... (+{len(axis.get('positive_pole', [])) - 20} more)"
                lines.append(f"  {pole_str}")
                lines.append("")

                lines.append(f"NEGATIVE POLE ({len(axis.get('negative_pole', []))} words):")
                pole_str = ", ".join(axis.get('negative_pole', [])[:20])
                if len(axis.get('negative_pole', [])) > 20:
                    pole_str += f", ... (+{len(axis.get('negative_pole', [])) - 20} more)"
                lines.append(f"  {pole_str}")
                lines.append("")

            if axis.get('representative_pairs'):
                lines.append("REPRESENTATIVE PAIRS:")
                for j, pair in enumerate(axis['representative_pairs'], 1):
                    lines.append(f"  {j}. {pair['word1']} ↔ {pair['word2']}")

        return "\n".join(lines)

    def export_axes_to_json(self, axes: List[Dict]) -> Dict:
        """
        Export axes to JSON-serializable format.

        Args:
            axes: List of axis dicts

        Returns:
            JSON-serializable dict
        """
        def convert_to_python(obj):
            """Convert numpy types to Python types recursively."""
            if isinstance(obj, (np.integer, np.intc, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python(item) for item in obj]
            elif isinstance(obj, set):
                return sorted(list(obj))
            else:
                return obj

        return {
            'axes': [
                {
                    'axis_id': convert_to_python(axis['axis_id']),
                    'name': axis['name'],
                    'cluster_id': convert_to_python(axis['cluster_id']),
                    'size': convert_to_python(axis['size']),
                    'coherence_score': float(axis['coherence_score']),
                    'separation_score': float(axis['separation_score']),
                    'source': axis.get('source', 'unknown'),
                    # Handle both old binary format and new n-ary format
                    'poles': axis.get('poles', [axis.get('positive_pole', []), axis.get('negative_pole', [])]),
                    'pole_names': axis.get('pole_names', ['positive', 'negative']),
                    'representative_pairs': [
                        {
                            'word1': p['word1'],
                            'word2': p['word2'],
                            'synset1': p['synset1'],
                            'synset2': p['synset2']
                        }
                        for p in axis.get('representative_pairs', [])
                    ]
                }
                for axis in axes
            ],
            'metadata': {
                'total_axes': len(axes),
                'mean_size': float(np.mean([a['size'] for a in axes])) if axes else 0.0,
                'mean_coherence': float(np.mean([a['coherence_score'] for a in axes])) if axes else 0.0,
                'mean_separation': float(np.mean([a['separation_score'] for a in axes])) if axes else 0.0
            }
        }
