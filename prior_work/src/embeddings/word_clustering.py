"""
Word-based clustering for semantic axis discovery.

This module clusters individual WORDS (not pairs) from antonym relationships,
then detects which clusters form axes by analyzing antonym connections.

Key improvement over pair-based clustering:
- (good, bad), (good, evil), (excellent, terrible) all contribute to ONE morality axis
- Instead of fragmenting into separate clusters

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from nltk.corpus import wordnet as wn
import networkx as nx
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=UserWarning)


class WordClusterer:
    """Cluster words from antonym pairs, then detect axis structure."""

    def __init__(self, antonym_pairs: List[Dict], verbose: bool = True):
        """
        Initialize word clusterer.

        Args:
            antonym_pairs: List of antonym pair dicts with word1, word2, synset1, synset2
            verbose: Print progress
        """
        self.antonym_pairs = antonym_pairs
        self.verbose = verbose

        # Extract all unique words
        self.words = self._extract_words()
        self.n_words = len(self.words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}

        # Caches
        self._synset_cache = {}
        self._similarity_cache = {}

    def _extract_words(self) -> List[str]:
        """Extract all unique words from antonym pairs."""
        words = set()
        for pair in self.antonym_pairs:
            words.add(pair['word1'])
            words.add(pair['word2'])
        return sorted(list(words))

    def cluster_words(
        self,
        n_clusters: int = 50,
        min_cluster_size: int = 3,
        linkage_method: str = 'average'
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Cluster words by semantic similarity.

        Args:
            n_clusters: Target number of clusters
            min_cluster_size: Minimum words per cluster
            linkage_method: Hierarchical clustering linkage method

        Returns:
            (word_clusters, cluster_assignments)
        """
        if self.verbose:
            print(f"Clustering {self.n_words} words from {len(self.antonym_pairs)} antonym pairs...")

        # Build similarity matrix
        if self.verbose:
            print("Computing word similarity matrix...")

        similarity_matrix = self._build_similarity_matrix()

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Ensure valid distance matrix (symmetric, non-negative, zero diagonal)
        distance_matrix = np.maximum(distance_matrix, 0.0)
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # Flatten to condensed distance matrix for linkage
        condensed_distances = squareform(distance_matrix, checks=False)

        # Perform hierarchical clustering
        if self.verbose:
            print(f"Performing hierarchical clustering ({linkage_method})...")

        Z = linkage(condensed_distances, method=linkage_method)

        # Cut dendrogram to get clusters
        cluster_assignments = fcluster(Z, t=n_clusters, criterion='maxclust')

        # Group words by cluster
        clusters_dict = defaultdict(list)
        for i, cluster_id in enumerate(cluster_assignments):
            clusters_dict[cluster_id].append(self.words[i])

        # Filter small clusters
        word_clusters = [
            cluster for cluster in clusters_dict.values()
            if len(cluster) >= min_cluster_size
        ]

        # Sort by size (descending)
        word_clusters.sort(key=len, reverse=True)

        if self.verbose:
            print(f"Found {len(word_clusters)} clusters (min size {min_cluster_size})")

        return (word_clusters, cluster_assignments)

    def _build_similarity_matrix(self) -> np.ndarray:
        """
        Build pairwise similarity matrix for words.

        Uses 4 signals:
        1. WordNet path similarity
        2. Shared antonyms (if A↔B and A↔C, then B and C are similar)
        3. Shared hypernyms
        4. Definition cosine similarity (TF-IDF)

        Returns:
            Similarity matrix of shape (n_words, n_words)
        """
        n = self.n_words
        similarity_matrix = np.zeros((n, n))

        # Build antonym graph for shared antonym signal
        antonym_graph = self._build_antonym_graph()

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._compute_word_similarity(
                        self.words[i],
                        self.words[j],
                        antonym_graph
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix

    def _compute_word_similarity(
        self,
        word1: str,
        word2: str,
        antonym_graph: nx.Graph
    ) -> float:
        """
        Compute similarity between two words using 4 signals + antonym penalty.

        CRITICAL: If words are direct antonyms, return 0.0 (maximum distance).
        This ensures antonyms cluster into separate poles.

        Args:
            word1: First word
            word2: Second word
            antonym_graph: Graph of antonym relationships

        Returns:
            Combined similarity score [0, 1]
        """
        # Check cache
        cache_key = tuple(sorted([word1, word2]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # CRITICAL: If direct antonyms, force zero similarity (they must separate)
        if antonym_graph.has_edge(word1, word2):
            self._similarity_cache[cache_key] = 0.0
            return 0.0

        similarities = []

        # Signal 1: WordNet path similarity
        path_sim = self._wordnet_path_similarity(word1, word2)
        if path_sim is not None:
            similarities.append(path_sim)

        # Signal 2: Shared antonyms (Jaccard similarity of antonym sets)
        # Words that share antonyms are likely in the same pole
        antonyms1 = set(antonym_graph.neighbors(word1)) if word1 in antonym_graph else set()
        antonyms2 = set(antonym_graph.neighbors(word2)) if word2 in antonym_graph else set()

        if antonyms1 or antonyms2:
            shared_antonyms = len(antonyms1 & antonyms2)
            total_antonyms = len(antonyms1 | antonyms2)
            antonym_sim = shared_antonyms / total_antonyms if total_antonyms > 0 else 0.0
            # Weight this signal heavily (2x) since it's very reliable
            similarities.append(antonym_sim)
            similarities.append(antonym_sim)

        # Signal 3: Shared hypernyms
        hypernym_sim = self._shared_hypernyms_similarity(word1, word2)
        if hypernym_sim > 0.0:
            similarities.append(hypernym_sim)

        # Signal 4: Definition similarity (TF-IDF cosine)
        def_sim = self._definition_similarity(word1, word2)
        if def_sim > 0.0:
            similarities.append(def_sim)

        # Combine signals (mean of available signals)
        if similarities:
            result = np.mean(similarities)
        else:
            result = 0.0

        self._similarity_cache[cache_key] = result
        return result

    def _build_antonym_graph(self) -> nx.Graph:
        """Build graph where edges connect antonyms."""
        G = nx.Graph()
        for pair in self.antonym_pairs:
            G.add_edge(pair['word1'], pair['word2'])
        return G

    def _wordnet_path_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Compute WordNet path similarity between two words.

        Returns:
            Path similarity [0, 1] or None if not computable
        """
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if not synsets1 or not synsets2:
            return None

        # Find maximum similarity across all synset pairs
        max_sim = 0.0
        for s1 in synsets1:
            for s2 in synsets2:
                sim = s1.path_similarity(s2)
                if sim is not None and sim > max_sim:
                    max_sim = sim

        return max_sim if max_sim > 0.0 else None

    def _shared_hypernyms_similarity(self, word1: str, word2: str) -> float:
        """
        Compute similarity based on shared hypernyms.

        Returns:
            Jaccard similarity of hypernym sets [0, 1]
        """
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if not synsets1 or not synsets2:
            return 0.0

        # Get all hypernyms for each word
        hypernyms1 = set()
        for synset in synsets1:
            for hypernym in synset.hypernyms():
                hypernyms1.add(hypernym.name())

        hypernyms2 = set()
        for synset in synsets2:
            for hypernym in synset.hypernyms():
                hypernyms2.add(hypernym.name())

        if not hypernyms1 or not hypernyms2:
            return 0.0

        # Jaccard similarity
        shared = len(hypernyms1 & hypernyms2)
        total = len(hypernyms1 | hypernyms2)

        return shared / total if total > 0 else 0.0

    def _definition_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between word definitions using TF-IDF.

        Returns:
            Cosine similarity [0, 1]
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if not synsets1 or not synsets2:
            return 0.0

        # Get definitions
        def1 = synsets1[0].definition()
        def2 = synsets2[0].definition()

        # Compute TF-IDF cosine similarity
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([def1, def2])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(sim)
        except:
            return 0.0


def detect_axes_from_word_clusters(
    word_clusters: List[List[str]],
    antonym_pairs: List[Dict],
    min_antonym_edges: int = 3
) -> List[Dict]:
    """
    Detect semantic axes from word clusters by analyzing antonym connections.

    Args:
        word_clusters: List of word clusters
        antonym_pairs: Original antonym pairs
        min_antonym_edges: Minimum antonym edges between clusters to form axis

    Returns:
        List of axis dicts with detected pole structure
    """
    # Build antonym graph
    antonym_graph = nx.Graph()
    for pair in antonym_pairs:
        antonym_graph.add_edge(pair['word1'], pair['word2'])

    # Map words to cluster IDs
    word_to_cluster = {}
    for i, cluster in enumerate(word_clusters):
        for word in cluster:
            word_to_cluster[word] = i

    # Count antonym edges between clusters
    cluster_antonym_counts = defaultdict(lambda: defaultdict(int))
    for pair in antonym_pairs:
        c1 = word_to_cluster.get(pair['word1'])
        c2 = word_to_cluster.get(pair['word2'])

        if c1 is not None and c2 is not None and c1 != c2:
            cluster_antonym_counts[c1][c2] += 1
            cluster_antonym_counts[c2][c1] += 1

    # Detect axes: clusters with strong antonym connections form poles of same axis
    axes = []
    used_clusters = set()

    for c1 in range(len(word_clusters)):
        if c1 in used_clusters:
            continue

        # Find clusters with strong antonym connections to c1
        connected_clusters = [c1]

        for c2, count in cluster_antonym_counts[c1].items():
            if count >= min_antonym_edges and c2 not in used_clusters:
                connected_clusters.append(c2)

        if len(connected_clusters) >= 2:
            # We have an axis!
            poles = [word_clusters[c] for c in connected_clusters]
            pole_names = [_extract_pole_name(pole) for pole in poles]

            # Count total antonym pairs in this axis
            member_pairs = []
            for pair in antonym_pairs:
                c1_pair = word_to_cluster.get(pair['word1'])
                c2_pair = word_to_cluster.get(pair['word2'])
                if c1_pair in connected_clusters and c2_pair in connected_clusters and c1_pair != c2_pair:
                    member_pairs.append(pair)

            axis = {
                'poles': poles,
                'pole_names': pole_names,
                'member_pairs': member_pairs,
                'representative_pairs': member_pairs[:5],
                'size': sum(len(pole) for pole in poles),
                'n_pairs': len(member_pairs),
                'cluster_ids': connected_clusters
            }

            axes.append(axis)
            used_clusters.update(connected_clusters)

    # Sort by size
    axes.sort(key=lambda a: a['size'], reverse=True)

    return axes


def _extract_pole_name(pole_words: List[str]) -> str:
    """
    Extract semantic name for pole using TF-IDF on definitions.

    Args:
        pole_words: List of words in pole

    Returns:
        Most representative concept name
    """
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words('english'))

    if not pole_words:
        return "unknown"

    # Collect all definitions
    all_definitions = []
    for word in pole_words:
        synsets = wn.synsets(word)
        if synsets:
            all_definitions.append(synsets[0].definition())

    if not all_definitions:
        return pole_words[0]

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
        if count <= n_defs * 0.5
    ]

    if not filtered_words:
        filtered_words = [word for word, _ in word_freq.most_common(10)]

    # Get top keywords
    top_keywords = Counter(filtered_words).most_common(5)

    if top_keywords:
        return top_keywords[0][0]
    else:
        return pole_words[0]


def cluster_words_and_detect_axes(
    antonym_pairs: List[Dict],
    n_clusters: int = 50,
    min_cluster_size: int = 3,
    min_antonym_edges: int = 3,
    linkage_method: str = 'average',
    verbose: bool = True
) -> Tuple[List[List[str]], List[Dict]]:
    """
    Convenience function to cluster words and detect axes.

    Args:
        antonym_pairs: List of antonym pair dicts
        n_clusters: Target number of word clusters
        min_cluster_size: Minimum words per cluster
        min_antonym_edges: Minimum antonym edges between clusters to form axis
        linkage_method: Hierarchical clustering linkage method
        verbose: Print progress

    Returns:
        (word_clusters, axes)
    """
    # Cluster words
    clusterer = WordClusterer(antonym_pairs, verbose=verbose)
    word_clusters, cluster_assignments = clusterer.cluster_words(
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        linkage_method=linkage_method
    )

    if verbose:
        print(f"\nDetecting axes from {len(word_clusters)} word clusters...")

    # Detect axes
    axes = detect_axes_from_word_clusters(
        word_clusters,
        antonym_pairs,
        min_antonym_edges=min_antonym_edges
    )

    if verbose:
        print(f"Detected {len(axes)} semantic axes")

    return (word_clusters, axes)
