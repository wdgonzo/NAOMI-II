"""
Similarity signal computation for antonym pair clustering.

This module implements 4 weighted similarity signals to measure how likely
two antonym pairs belong to the same semantic axis:

1. Antonym Transitivity (0.4 weight) - Pairs sharing endpoints
2. Similar-to Quadrilaterals (0.3 weight) - WordNet similar_to overlaps
3. Definition TF-IDF Cosine (0.2 weight) - Definition keyword overlap
4. Hypernym Path Overlap (0.1 weight) - Shared abstract ancestors

Author: NAOMI-II Development Team
Date: 2025-11-30
"""

from typing import Dict, Set, List, Tuple, Optional, Any
from collections import Counter
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

# Load stopwords once
STOPWORDS = set(stopwords.words('english'))


class SimilaritySignals:
    """Computes similarity signals between antonym pairs."""

    def __init__(self, all_antonym_pairs: List[Dict]):
        """
        Initialize with all antonym pairs for corpus statistics.

        Args:
            all_antonym_pairs: List of dicts with keys:
                - word1, word2: String words
                - synset1, synset2: WordNet synset names (e.g., 'good.a.01')
        """
        self.all_pairs = all_antonym_pairs

        # Build IDF statistics from all definitions
        self._build_idf_statistics()

        # Cache for synset lookups
        self._synset_cache = {}
        self._similar_to_cache = {}
        self._hypernym_cache = {}

    def _build_idf_statistics(self):
        """Build IDF statistics from all antonym pair definitions."""
        print("Building IDF statistics from definitions...")

        # Collect all documents (each pair's combined definitions = 1 document)
        documents = []
        for pair in self.all_pairs:
            try:
                syn1 = wn.synset(pair['synset1'])
                syn2 = wn.synset(pair['synset2'])

                def1 = syn1.definition()
                def2 = syn2.definition()

                combined = def1 + " " + def2
                documents.append(combined)
            except Exception as e:
                # Skip if synset not found
                continue

        # Count document frequencies
        df = Counter()  # document frequency for each term

        for doc in documents:
            words = self._extract_content_words(doc)
            unique_words = set(words)
            for word in unique_words:
                df[word] += 1

        # Compute IDF
        n_docs = len(documents)
        self.idf = {}
        for word, count in df.items():
            self.idf[word] = math.log(n_docs / (1 + count))

        print(f"Built IDF statistics: {len(self.idf)} unique content words")

    def _extract_content_words(self, text: str) -> List[str]:
        """
        Extract content words (nouns, verbs, adjectives, adverbs) from text.

        Args:
            text: Input text

        Returns:
            List of lowercase content words (stopwords removed)
        """
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        content = [t for t in tokens if t.isalpha() and t not in STOPWORDS]

        return content

    def _get_synset(self, synset_name: str) -> Optional[Any]:
        """Get synset with caching."""
        if synset_name not in self._synset_cache:
            try:
                self._synset_cache[synset_name] = wn.synset(synset_name)
            except Exception:
                self._synset_cache[synset_name] = None
        return self._synset_cache[synset_name]

    def _get_similar_tos(self, synset_name: str) -> Set[str]:
        """Get similar_to synsets with caching."""
        if synset_name not in self._similar_to_cache:
            syn = self._get_synset(synset_name)
            if syn:
                similar = {s.name() for s in syn.similar_tos()}
            else:
                similar = set()
            self._similar_to_cache[synset_name] = similar
        return self._similar_to_cache[synset_name]

    def _get_all_hypernyms(self, synset_name: str) -> Set[str]:
        """Get all hypernyms (recursive) with caching."""
        if synset_name not in self._hypernym_cache:
            syn = self._get_synset(synset_name)
            if syn:
                hypernyms = set()
                for path in syn.hypernym_paths():
                    hypernyms.update(s.name() for s in path)
                self._hypernym_cache[synset_name] = hypernyms
            else:
                self._hypernym_cache[synset_name] = set()
        return self._hypernym_cache[synset_name]

    def compute_transitivity(self, pair1: Dict, pair2: Dict) -> float:
        """
        Signal 1: Antonym Transitivity (0.4 weight).

        Check if pairs share endpoints (shared word or synset).
        Example: (good/bad) and (good/evil) share "good" → 1.0

        Args:
            pair1: First antonym pair
            pair2: Second antonym pair

        Returns:
            1.0 if shared endpoint, 0.0 otherwise
        """
        # Check exact word matches
        words1 = {pair1['word1'], pair1['word2']}
        words2 = {pair2['word1'], pair2['word2']}

        if words1 & words2:
            return 1.0

        # Check synset-level matches (same lemma, different surface form)
        synsets1 = {pair1['synset1'], pair1['synset2']}
        synsets2 = {pair2['synset1'], pair2['synset2']}

        if synsets1 & synsets2:
            return 1.0

        return 0.0

    def compute_quadrilateral(self, pair1: Dict, pair2: Dict) -> float:
        """
        Signal 2: Similar-to Quadrilaterals (0.3 weight).

        Check if similar_tos of pair1 overlap with pair2's words/synsets.
        Pattern: A1~A2 (similar), B1~B2 (similar), A1↔B1 (antonym), A2↔B2 (antonym)

        Args:
            pair1: First antonym pair
            pair2: Second antonym pair

        Returns:
            Jaccard similarity of similar_to overlaps [0, 1]
        """
        # Get similar_tos for each synset
        A1_similar = self._get_similar_tos(pair1['synset1'])
        B1_similar = self._get_similar_tos(pair1['synset2'])
        A2_similar = self._get_similar_tos(pair2['synset1'])
        B2_similar = self._get_similar_tos(pair2['synset2'])

        # Check if A1's similar_tos overlap with A2's similar_tos
        # AND B1's similar_tos overlap with B2's similar_tos
        if not A1_similar and not A2_similar:
            overlap_A = 0.0
        else:
            intersection_A = len(A1_similar & A2_similar)
            union_A = len(A1_similar | A2_similar)
            overlap_A = intersection_A / (union_A + 1e-6)

        if not B1_similar and not B2_similar:
            overlap_B = 0.0
        else:
            intersection_B = len(B1_similar & B2_similar)
            union_B = len(B1_similar | B2_similar)
            overlap_B = intersection_B / (union_B + 1e-6)

        # Average overlap across both poles
        return (overlap_A + overlap_B) / 2.0

    def compute_definition_similarity(self, pair1: Dict, pair2: Dict) -> float:
        """
        Signal 3: Definition TF-IDF Cosine (0.2 weight).

        Compare definitions using TF-IDF weighted cosine similarity.

        Args:
            pair1: First antonym pair
            pair2: Second antonym pair

        Returns:
            Cosine similarity [0, 1]
        """
        # Get definitions
        syn1_A = self._get_synset(pair1['synset1'])
        syn1_B = self._get_synset(pair1['synset2'])
        syn2_A = self._get_synset(pair2['synset1'])
        syn2_B = self._get_synset(pair2['synset2'])

        if not all([syn1_A, syn1_B, syn2_A, syn2_B]):
            return 0.0

        # Combine definitions for each pair
        def1_text = syn1_A.definition() + " " + syn1_B.definition()
        def2_text = syn2_A.definition() + " " + syn2_B.definition()

        # Extract content words
        words1 = self._extract_content_words(def1_text)
        words2 = self._extract_content_words(def2_text)

        # Compute TF-IDF vectors
        vec1 = self._compute_tfidf_vector(words1)
        vec2 = self._compute_tfidf_vector(words2)

        # Cosine similarity
        return self._cosine_similarity(vec1, vec2)

    def _compute_tfidf_vector(self, words: List[str]) -> Dict[str, float]:
        """
        Compute TF-IDF vector for word list.

        Args:
            words: List of words

        Returns:
            Dict mapping word to TF-IDF score
        """
        # Compute term frequency
        tf = Counter(words)
        total = len(words)

        # Compute TF-IDF
        tfidf = {}
        for word, count in tf.items():
            tf_score = count / total
            idf_score = self.idf.get(word, 0.0)
            tfidf[word] = tf_score * idf_score

        return tfidf

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.

        Args:
            vec1: First vector (word -> score)
            vec2: Second vector (word -> score)

        Returns:
            Cosine similarity [0, 1]
        """
        if not vec1 or not vec2:
            return 0.0

        # Get all words
        all_words = set(vec1.keys()) | set(vec2.keys())

        # Compute dot product and norms
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for word in all_words:
            v1 = vec1.get(word, 0.0)
            v2 = vec2.get(word, 0.0)

            dot_product += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2

        # Compute cosine
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def compute_hypernym_overlap(self, pair1: Dict, pair2: Dict) -> float:
        """
        Signal 4: Hypernym Path Overlap (0.1 weight).

        Check if both pairs share common abstract hypernyms.

        Args:
            pair1: First antonym pair
            pair2: Second antonym pair

        Returns:
            Jaccard similarity of hypernym paths [0, 1]
        """
        # Get all hypernyms for both synsets in each pair
        hyp1_A = self._get_all_hypernyms(pair1['synset1'])
        hyp1_B = self._get_all_hypernyms(pair1['synset2'])
        hyp2_A = self._get_all_hypernyms(pair2['synset1'])
        hyp2_B = self._get_all_hypernyms(pair2['synset2'])

        # Combine hypernyms for each pair
        pair1_hypernyms = hyp1_A | hyp1_B
        pair2_hypernyms = hyp2_A | hyp2_B

        # Jaccard similarity
        if not pair1_hypernyms and not pair2_hypernyms:
            return 0.0

        intersection = len(pair1_hypernyms & pair2_hypernyms)
        union = len(pair1_hypernyms | pair2_hypernyms)

        return intersection / (union + 1e-6)

    def compute_combined_similarity(
        self,
        pair1: Dict,
        pair2: Dict,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute combined similarity using weighted sum of all signals.

        Args:
            pair1: First antonym pair
            pair2: Second antonym pair
            weights: Optional custom weights (default: transitivity=0.4,
                     quadrilateral=0.3, definition=0.2, hypernym=0.1)

        Returns:
            Combined similarity score [0, 1]
        """
        # Default weights from OPUS analysis
        if weights is None:
            weights = {
                'transitivity': 0.4,
                'quadrilateral': 0.3,
                'definition': 0.2,
                'hypernym': 0.1
            }

        # Compute all signals
        trans_score = self.compute_transitivity(pair1, pair2)
        quad_score = self.compute_quadrilateral(pair1, pair2)
        def_score = self.compute_definition_similarity(pair1, pair2)
        hyp_score = self.compute_hypernym_overlap(pair1, pair2)

        # Weighted combination
        total = (
            weights['transitivity'] * trans_score +
            weights['quadrilateral'] * quad_score +
            weights['definition'] * def_score +
            weights['hypernym'] * hyp_score
        )

        return total

    def compute_similarity_matrix(
        self,
        progress_callback: Optional[callable] = None
    ) -> np.ndarray:
        """
        Build pairwise similarity matrix for all antonym pairs.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            (N × N) symmetric similarity matrix
        """
        n_pairs = len(self.all_pairs)
        similarity_matrix = np.zeros((n_pairs, n_pairs), dtype=np.float32)

        print(f"Computing similarity matrix for {n_pairs} antonym pairs...")

        # Compute upper triangle (matrix is symmetric)
        total_comparisons = (n_pairs * (n_pairs - 1)) // 2
        comparison_count = 0

        for i in range(n_pairs):
            # Diagonal is 1.0 (pair is identical to itself)
            similarity_matrix[i, i] = 1.0

            for j in range(i + 1, n_pairs):
                sim = self.compute_combined_similarity(
                    self.all_pairs[i],
                    self.all_pairs[j]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

                comparison_count += 1

                # Progress callback
                if progress_callback and comparison_count % 10000 == 0:
                    progress_callback(comparison_count, total_comparisons)

        print(f"Similarity matrix complete: {n_pairs} × {n_pairs}")
        return similarity_matrix
