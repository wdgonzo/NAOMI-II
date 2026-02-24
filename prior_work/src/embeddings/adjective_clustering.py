"""
Adjective clustering for semantic axis discovery (Phase 1.5).

This module clusters semantically related adjectives using hierarchical clustering
on WordNet similarity, discovering additional semantic axes beyond antonym pairs.

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=UserWarning)

STOPWORDS = set(stopwords.words('english'))


class AdjectiveClusterer:
    """Cluster adjectives by semantic similarity to discover axes."""

    def __init__(self, adjectives: List[str], verbose: bool = True):
        """
        Initialize adjective clusterer.

        Args:
            adjectives: List of adjective words
            verbose: Print progress
        """
        self.adjectives = adjectives
        self.verbose = verbose
        self._synset_cache = {}
        self._path_similarity_cache = {}

    def cluster_adjectives(
        self,
        n_clusters: int = 50,
        min_cluster_size: int = 3,
        linkage_method: str = 'complete'
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Cluster adjectives by semantic similarity.

        Args:
            n_clusters: Target number of clusters
            min_cluster_size: Minimum adjectives per cluster
            linkage_method: Hierarchical clustering linkage method

        Returns:
            (adjective_clusters, cluster_assignments)
        """
        if self.verbose:
            print(f"Clustering {len(self.adjectives)} adjectives...")

        # Build similarity matrix
        if self.verbose:
            print("Computing similarity matrix...")

        similarity_matrix = self._build_similarity_matrix()

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1.0 - similarity_matrix

        # Flatten to condensed distance matrix for linkage
        condensed_distances = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        if self.verbose:
            print(f"Performing hierarchical clustering ({linkage_method})...")

        Z = linkage(condensed_distances, method=linkage_method)

        # Cut dendrogram to get clusters
        cluster_assignments = fcluster(Z, t=n_clusters, criterion='maxclust')

        # Group adjectives by cluster
        clusters_dict = defaultdict(list)
        for i, cluster_id in enumerate(cluster_assignments):
            clusters_dict[cluster_id].append(self.adjectives[i])

        # Filter small clusters
        adjective_clusters = [
            cluster for cluster in clusters_dict.values()
            if len(cluster) >= min_cluster_size
        ]

        # Sort by size (descending)
        adjective_clusters.sort(key=len, reverse=True)

        if self.verbose:
            print(f"Found {len(adjective_clusters)} clusters (min size {min_cluster_size})")

        return (adjective_clusters, cluster_assignments)

    def _build_similarity_matrix(self) -> np.ndarray:
        """
        Build pairwise similarity matrix using WordNet path similarity.

        Returns:
            Similarity matrix of shape (n_adjectives, n_adjectives)
        """
        n = len(self.adjectives)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._wordnet_path_similarity(
                        self.adjectives[i],
                        self.adjectives[j]
                    )

                    # Default to 0.0 if similarity not computable
                    if sim is None:
                        sim = 0.0

                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix

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

        # Get all synsets for both words (adjectives only)
        synsets1 = wn.synsets(word1, pos=wn.ADJ)
        synsets2 = wn.synsets(word2, pos=wn.ADJ)

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

    def _get_synset(self, synset_name: str) -> Optional[any]:
        """Get synset with caching."""
        if synset_name not in self._synset_cache:
            try:
                self._synset_cache[synset_name] = wn.synset(synset_name)
            except Exception:
                self._synset_cache[synset_name] = None
        return self._synset_cache[synset_name]


def extract_adjectives_from_wordnet(
    min_frequency: int = 5,
    max_adjectives: int = 2000
) -> List[str]:
    """
    Extract common adjectives from WordNet.

    Args:
        min_frequency: Minimum synset count for adjective
        max_adjectives: Maximum number of adjectives to return

    Returns:
        List of adjective strings
    """
    adjective_counts = Counter()

    # Count adjectives across all synsets
    for synset in wn.all_synsets(pos=wn.ADJ):
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ')
            adjective_counts[word] += 1

    # Filter by minimum frequency
    common_adjectives = [
        word for word, count in adjective_counts.items()
        if count >= min_frequency
    ]

    # Sort by frequency (most common first)
    common_adjectives.sort(key=lambda w: adjective_counts[w], reverse=True)

    # Take top N
    return common_adjectives[:max_adjectives]


def cluster_adjectives_from_wordnet(
    n_clusters: int = 50,
    min_cluster_size: int = 3,
    min_frequency: int = 5,
    max_adjectives: int = 2000,
    linkage_method: str = 'complete',
    verbose: bool = True
) -> Tuple[List[List[str]], np.ndarray]:
    """
    Convenience function to extract and cluster adjectives from WordNet.

    Args:
        n_clusters: Target number of clusters
        min_cluster_size: Minimum adjectives per cluster
        min_frequency: Minimum synset count for adjective
        max_adjectives: Maximum number of adjectives
        linkage_method: Hierarchical clustering linkage method
        verbose: Print progress

    Returns:
        (adjective_clusters, cluster_assignments)
    """
    if verbose:
        print("Extracting adjectives from WordNet...")

    adjectives = extract_adjectives_from_wordnet(
        min_frequency=min_frequency,
        max_adjectives=max_adjectives
    )

    if verbose:
        print(f"Extracted {len(adjectives)} common adjectives")

    clusterer = AdjectiveClusterer(adjectives, verbose=verbose)

    return clusterer.cluster_adjectives(
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        linkage_method=linkage_method
    )


def convert_adjective_clusters_to_axes(
    adjective_clusters: List[List[str]],
    min_coherence: float = 0.3
) -> List[Dict]:
    """
    Convert adjective clusters to axis format.

    Args:
        adjective_clusters: List of adjective word lists
        min_coherence: Minimum coherence score

    Returns:
        List of axis dicts compatible with dimension_allocator
    """
    axes = []

    for i, cluster in enumerate(adjective_clusters):
        # For adjective clusters, create a single pole
        # (multi-pole structure discovered in Phase 2)

        # Compute basic coherence
        coherence = _compute_cluster_coherence(cluster)

        if coherence < min_coherence:
            continue

        # Extract cluster name
        cluster_name = _extract_cluster_name(cluster)

        axis = {
            'axis_id': i,
            'name': cluster_name,
            'poles': [cluster],  # Single pole for now
            'pole_names': [cluster_name],
            'representative_pairs': [],  # No pairs for adjective clusters
            'member_pairs': [],
            'coherence_score': coherence,
            'separation_score': 0.0,  # No separation for single pole
            'size': len(cluster),
            'cluster_id': i,
            'source': 'adjective_clustering'
        }

        axes.append(axis)

    return axes


def _compute_cluster_coherence(cluster: List[str]) -> float:
    """
    Compute coherence within adjective cluster.

    Args:
        cluster: List of adjective words

    Returns:
        Average path similarity [0, 1]
    """
    if len(cluster) < 2:
        return 0.0

    similarities = []
    for i, w1 in enumerate(cluster):
        for j, w2 in enumerate(cluster):
            if i < j:
                synsets1 = wn.synsets(w1, pos=wn.ADJ)
                synsets2 = wn.synsets(w2, pos=wn.ADJ)

                if synsets1 and synsets2:
                    max_sim = 0.0
                    for s1 in synsets1:
                        for s2 in synsets2:
                            sim = s1.path_similarity(s2)
                            if sim is not None and sim > max_sim:
                                max_sim = sim

                    if max_sim > 0.0:
                        similarities.append(max_sim)

    if similarities:
        return np.mean(similarities)
    else:
        return 0.0


def _extract_cluster_name(cluster: List[str]) -> str:
    """
    Extract semantic name for adjective cluster using TF-IDF.

    Args:
        cluster: List of adjective words

    Returns:
        Most representative concept name
    """
    if not cluster:
        return "unknown"

    # Collect all definitions
    all_definitions = []
    for word in cluster:
        synsets = wn.synsets(word, pos=wn.ADJ)
        if synsets:
            all_definitions.append(synsets[0].definition())

    if not all_definitions:
        # Fallback: use first word
        return cluster[0]

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

    # Get top keywords
    top_keywords = Counter(filtered_words).most_common(5)

    if top_keywords:
        return top_keywords[0][0]
    else:
        # Ultimate fallback: first word in cluster
        return cluster[0]
