"""
N-ary pole detection via graph community detection.

This module detects n-pole structures (binary, ternary, quaternary, etc.) from
antonym pair clusters using Louvain community detection on word co-occurrence graphs.

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter
import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

STOPWORDS = set(stopwords.words('english'))


class PoleDetector:
    """Detects n-ary pole structure from antonym pair clusters."""

    def __init__(self, member_pairs: List[Dict]):
        """
        Initialize pole detector.

        Args:
            member_pairs: List of antonym pair dicts in cluster
        """
        self.member_pairs = member_pairs
        self._synset_cache = {}

    def detect_poles(
        self,
        resolution: float = 1.0,
        min_pole_size: int = 1
    ) -> Tuple[List[Set[str]], List[str], str, float]:
        """
        Detect n poles from member pairs using hybrid detection strategy.

        Tries multiple methods in order:
        1. WordNet similarity graph (best quality)
        2. Triadic closure detection (for n-ary patterns)
        3. Frequency-based binary split (fallback)

        Args:
            resolution: Louvain resolution parameter (higher = more communities)
            min_pole_size: Minimum words per pole

        Returns:
            (pole_word_sets, pole_names, detection_method, confidence)
        """
        # Strategy 1: Try WordNet similarity graph
        graph = self._build_word_graph()

        if len(graph.nodes()) > 0 and graph.number_of_edges() > 0:
            communities = list(nx.community.louvain_communities(
                graph,
                resolution=resolution,
                seed=42
            ))

            # Filter small poles
            communities = [c for c in communities if len(c) >= min_pole_size]

            if len(communities) >= 2:
                # WordNet similarity worked!
                communities.sort(key=len, reverse=True)
                pole_names = [self._extract_pole_name(pole) for pole in communities]
                confidence = self._compute_graph_confidence(graph, communities)
                return (communities, pole_names, 'wordnet_similarity', confidence)

        # Strategy 2: Try triadic closure detection
        triadic_poles = self._detect_triadic_closure()
        if len(triadic_poles) >= 3:
            pole_names = [self._extract_pole_name(pole) for pole in triadic_poles]
            return (triadic_poles, pole_names, 'triadic_closure', 0.7)

        # Strategy 3: Fallback to frequency-based binary split
        binary_poles = self._frequency_based_binary_split()
        if len(binary_poles) == 2:
            pole_names = [self._extract_pole_name(pole) for pole in binary_poles]
            return (binary_poles, pole_names, 'frequency_binary', 0.5)

        # No valid detection
        return ([], [], 'failed', 0.0)

    def _build_word_graph(self) -> nx.Graph:
        """
        Build undirected graph where edges connect semantically similar words.

        Uses WordNet relations (synonyms, similar-to, also-see) to connect words
        that should be in the same pole. Antonym edges are NOT added since they
        define axis opposition, not pole membership.

        Returns:
            NetworkX graph with words as nodes
        """
        G = nx.Graph()

        # Collect all words from pairs
        all_words = set()
        for pair in self.member_pairs:
            all_words.add(pair['word1'])
            all_words.add(pair['word2'])

        # Add nodes
        for word in all_words:
            G.add_node(word)

        # Build similarity edges using WordNet relations
        for word in all_words:
            synsets = wn.synsets(word)
            if not synsets:
                continue

            # Get related words through various WordNet relations
            related_words = set()

            for synset in synsets:
                # 1. Synonyms (same synset)
                for lemma in synset.lemmas():
                    related_word = lemma.name().replace('_', ' ')
                    if related_word in all_words and related_word != word:
                        related_words.add(related_word)

                # 2. Similar-to relation (for adjectives)
                for similar_synset in synset.similar_tos():
                    for lemma in similar_synset.lemmas():
                        related_word = lemma.name().replace('_', ' ')
                        if related_word in all_words and related_word != word:
                            related_words.add(related_word)

                # 3. Also-see relation
                for also_synset in synset.also_sees():
                    for lemma in also_synset.lemmas():
                        related_word = lemma.name().replace('_', ' ')
                        if related_word in all_words and related_word != word:
                            related_words.add(related_word)

            # Add edges for related words
            for related_word in related_words:
                if G.has_edge(word, related_word):
                    G[word][related_word]['weight'] += 1.0
                else:
                    G.add_edge(word, related_word, weight=1.0)

        return G

    def _extract_pole_name(self, pole_words: Set[str]) -> str:
        """
        Extract semantic name for pole using TF-IDF on definitions.

        Args:
            pole_words: Set of words in pole

        Returns:
            Most representative concept name
        """
        if not pole_words:
            return "unknown"

        # Collect all definitions
        all_definitions = []
        for word in pole_words:
            synsets = wn.synsets(word)
            if synsets:
                # Use first synset definition
                all_definitions.append(synsets[0].definition())

        if not all_definitions:
            # Fallback: use most common word
            return sorted(pole_words)[0]

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
            # Return most common keyword
            return top_keywords[0][0]
        else:
            # Ultimate fallback: first word alphabetically
            return sorted(pole_words)[0]

    def _detect_triadic_closure(self) -> List[Set[str]]:
        """
        Detect n-ary pole structure using triadic closure.

        Looks for patterns like: A↔B, B↔C, A↔C (all three pair with each other)
        indicating a 3-pole structure.

        Returns:
            List of pole sets (empty if no triadic pattern found)
        """
        # Build antonym graph (opposite of similarity graph)
        antonym_graph = nx.Graph()
        for pair in self.member_pairs:
            antonym_graph.add_edge(pair['word1'], pair['word2'])

        # Find triangles (3-cliques)
        triangles = []
        for clique in nx.enumerate_all_cliques(antonym_graph):
            if len(clique) == 3:
                triangles.append(set(clique))
            elif len(clique) > 3:
                break  # enumerate_all_cliques returns in size order

        # If we found triangles, each vertex is a separate pole
        if triangles:
            # Use the largest triangle
            triangle = max(triangles, key=len)
            # Each word in the triangle is its own pole
            return [set([word]) for word in triangle]

        return []

    def _frequency_based_binary_split(self) -> List[Set[str]]:
        """
        Split words into two poles using frequency heuristic.

        Words appearing more on the left side of pairs go to pole 1,
        words appearing more on the right go to pole 2.

        Returns:
            List of 2 pole sets
        """
        from collections import Counter

        word1_counts = Counter()
        word2_counts = Counter()

        for pair in self.member_pairs:
            word1_counts[pair['word1']] += 1
            word2_counts[pair['word2']] += 1

        # Handle ambiguous words (appear on both sides)
        all_words_1 = set(word1_counts.keys())
        all_words_2 = set(word2_counts.keys())
        ambiguous = all_words_1 & all_words_2

        for word in ambiguous:
            if word1_counts[word] > word2_counts[word]:
                del word2_counts[word]
            else:
                del word1_counts[word]

        pole1 = set(word1_counts.keys())
        pole2 = set(word2_counts.keys())

        if pole1 and pole2:
            return [pole1, pole2]
        return []

    def _compute_graph_confidence(
        self,
        graph: nx.Graph,
        communities: List[Set[str]]
    ) -> float:
        """
        Compute confidence score for graph-based detection.

        Higher score when:
        - More edges within communities
        - Fewer edges between communities
        - Higher graph density

        Returns:
            Confidence score [0, 1]
        """
        if not communities or len(graph.nodes()) == 0:
            return 0.0

        # Compute modularity-like score
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            return 0.0

        internal_edges = 0
        for community in communities:
            subgraph = graph.subgraph(community)
            internal_edges += subgraph.number_of_edges()

        modularity = internal_edges / total_edges if total_edges > 0 else 0.0

        # Scale to [0.5, 1.0] range for high-quality detections
        confidence = 0.5 + (modularity * 0.5)

        return confidence

    def _get_synset(self, synset_name: str) -> Optional[Any]:
        """Get synset with caching."""
        if synset_name not in self._synset_cache:
            try:
                self._synset_cache[synset_name] = wn.synset(synset_name)
            except Exception:
                self._synset_cache[synset_name] = None
        return self._synset_cache[synset_name]


def detect_poles_from_pairs(
    member_pairs: List[Dict],
    resolution: float = 1.0,
    min_pole_size: int = 1
) -> Tuple[List[Set[str]], List[str], str, float]:
    """
    Convenience function to detect poles from member pairs using hybrid strategy.

    Args:
        member_pairs: List of antonym pair dicts
        resolution: Louvain resolution parameter
        min_pole_size: Minimum words per pole

    Returns:
        (pole_word_sets, pole_names, detection_method, confidence)
    """
    detector = PoleDetector(member_pairs)
    return detector.detect_poles(
        resolution=resolution,
        min_pole_size=min_pole_size
    )
