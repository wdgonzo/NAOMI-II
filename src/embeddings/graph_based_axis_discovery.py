"""
Graph-based semantic axis discovery using transitive closure.

This module discovers axes by analyzing the structure of the antonym graph:
- Words that share many antonyms belong in the same pole
- Connected components of the "shared antonym" graph form poles
- Poles with strong antonym connections form axes

Author: NAOMI-II Development Team
Date: 2025-12-01
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import networkx as nx


def discover_axes_from_antonym_graph(
    antonym_pairs: List[Dict],
    min_shared_antonyms: int = 1,
    min_axis_pairs: int = 3,
    verbose: bool = True
) -> List[Dict]:
    """
    Discover semantic axes using graph-based transitive closure.

    Algorithm:
    1. Build antonym graph (edges = antonym relationships)
    2. Build "same pole" graph (edges = shared antonyms)
    3. Find connected components in same-pole graph (these are poles)
    4. Find poles with strong antonym connections (these are axes)

    Args:
        antonym_pairs: List of antonym pair dicts
        min_shared_antonyms: Minimum shared antonyms to connect words in same pole
        min_axis_pairs: Minimum antonym pairs between poles to form axis
        verbose: Print progress

    Returns:
        List of discovered axes with pole structure
    """
    if verbose:
        print(f"Discovering axes from {len(antonym_pairs)} antonym pairs...")

    # Step 1: Build antonym graph
    antonym_graph = nx.Graph()
    for pair in antonym_pairs:
        antonym_graph.add_edge(pair['word1'], pair['word2'], pair=pair)

    if verbose:
        print(f"  Antonym graph: {antonym_graph.number_of_nodes()} words, {antonym_graph.number_of_edges()} edges")

    # Step 2: Build "same pole" graph based on shared antonyms
    same_pole_graph = nx.Graph()
    all_words = list(antonym_graph.nodes())

    for word in all_words:
        same_pole_graph.add_node(word)

    for i, word1 in enumerate(all_words):
        for word2 in all_words[i+1:]:
            # Skip if they're direct antonyms (opposite poles, not same pole)
            if antonym_graph.has_edge(word1, word2):
                continue

            # Count shared antonyms
            antonyms1 = set(antonym_graph.neighbors(word1))
            antonyms2 = set(antonym_graph.neighbors(word2))
            shared = antonyms1 & antonyms2

            if len(shared) >= min_shared_antonyms:
                # Words that share antonyms belong in same pole
                same_pole_graph.add_edge(word1, word2, shared_antonyms=shared)

    if verbose:
        print(f"  Same-pole graph: {same_pole_graph.number_of_nodes()} words, {same_pole_graph.number_of_edges()} edges")

    # Step 3: Find poles (connected components in same-pole graph)
    pole_components = list(nx.connected_components(same_pole_graph))

    if verbose:
        print(f"  Found {len(pole_components)} pole candidates")

    # Step 4: Find axes (poles with strong antonym connections)
    axes = []
    used_poles = set()

    for i, pole1 in enumerate(pole_components):
        if i in used_poles:
            continue

        # Find other poles that have antonym connections to this pole
        connected_poles = [(i, pole1)]
        pole_antonym_counts = defaultdict(int)

        for j, pole2 in enumerate(pole_components):
            if j <= i or j in used_poles:
                continue

            # Count antonym edges between pole1 and pole2
            antonym_count = 0
            for w1 in pole1:
                for w2 in pole2:
                    if antonym_graph.has_edge(w1, w2):
                        antonym_count += 1

            if antonym_count >= min_axis_pairs:
                connected_poles.append((j, pole2))
                pole_antonym_counts[j] = antonym_count

        # If we have 2+ poles connected by antonyms, it's an axis
        if len(connected_poles) >= 2:
            pole_ids = [idx for idx, _ in connected_poles]
            poles = [pole for _, pole in connected_poles]

            # Get all antonym pairs for this axis
            member_pairs = []
            for pair in antonym_pairs:
                w1, w2 = pair['word1'], pair['word2']
                # Check if this pair connects poles in this axis
                pole1_idx = None
                pole2_idx = None

                for idx, pole in enumerate(poles):
                    if w1 in pole:
                        pole1_idx = idx
                    if w2 in pole:
                        pole2_idx = idx

                if pole1_idx is not None and pole2_idx is not None and pole1_idx != pole2_idx:
                    member_pairs.append(pair)

            # Create axis
            axis = {
                'poles': [sorted(list(pole)) for pole in poles],
                'pole_names': [_extract_pole_name(pole) for pole in poles],
                'member_pairs': member_pairs,
                'representative_pairs': member_pairs[:5],
                'size': sum(len(pole) for pole in poles),
                'n_pairs': len(member_pairs),
                'n_poles': len(poles),
                'pole_ids': pole_ids
            }

            axes.append(axis)
            used_poles.update(pole_ids)

    # Sort axes by size (number of pairs)
    axes.sort(key=lambda a: a['n_pairs'], reverse=True)

    if verbose:
        print(f"  Discovered {len(axes)} axes")

    return axes


def _extract_pole_name(pole_words: Set[str]) -> str:
    """
    Extract semantic name for pole.

    For now, just use the first word alphabetically.
    Could be enhanced with TF-IDF on definitions later.

    Args:
        pole_words: Set/list of words in pole

    Returns:
        Pole name
    """
    if not pole_words:
        return "unknown"

    # Sort and take first
    sorted_words = sorted(list(pole_words))
    return sorted_words[0]


def discover_axes_with_expansion(
    antonym_pairs: List[Dict],
    min_shared_antonyms: int = 1,
    min_axis_pairs: int = 3,
    expand_threshold: float = 0.7,
    verbose: bool = True
) -> List[Dict]:
    """
    Discover axes with iterative pole expansion.

    After initial discovery, expand poles by adding words that:
    - Share many antonyms with the existing pole
    - Don't have antonyms within the pole (must be consistent)

    Args:
        antonym_pairs: List of antonym pair dicts
        min_shared_antonyms: Minimum shared antonyms to connect words
        min_axis_pairs: Minimum antonym pairs between poles
        expand_threshold: Fraction of pole words that must share antonyms with candidate
        verbose: Print progress

    Returns:
        List of discovered axes with expanded poles
    """
    # First, discover base axes
    axes = discover_axes_from_antonym_graph(
        antonym_pairs,
        min_shared_antonyms=min_shared_antonyms,
        min_axis_pairs=min_axis_pairs,
        verbose=verbose
    )

    if verbose:
        print(f"\nExpanding poles (threshold={expand_threshold})...")

    # Build antonym graph
    antonym_graph = nx.Graph()
    for pair in antonym_pairs:
        antonym_graph.add_edge(pair['word1'], pair['word2'])

    all_words = set(antonym_graph.nodes())

    # Expand each axis's poles
    for axis in axes:
        original_size = axis['size']

        for pole_idx, pole in enumerate(axis['poles']):
            pole_set = set(pole)

            # Find candidate words (not already in any pole of this axis)
            all_axis_words = set()
            for p in axis['poles']:
                all_axis_words.update(p)

            candidates = all_words - all_axis_words

            # Try to add candidates
            added = []
            for candidate in candidates:
                # Check: candidate must share antonyms with threshold fraction of pole
                antonyms_candidate = set(antonym_graph.neighbors(candidate))
                shared_count = 0

                for pole_word in pole_set:
                    antonyms_pole_word = set(antonym_graph.neighbors(pole_word))
                    if antonyms_candidate & antonyms_pole_word:
                        shared_count += 1

                # Check: candidate must NOT have antonyms within the pole
                has_internal_antonym = any(
                    antonym_graph.has_edge(candidate, pw) for pw in pole_set
                )

                if not has_internal_antonym and shared_count >= len(pole_set) * expand_threshold:
                    added.append(candidate)
                    pole_set.add(candidate)

            # Update pole
            axis['poles'][pole_idx] = sorted(list(pole_set))

            if verbose and added:
                print(f"  Pole '{axis['pole_names'][pole_idx]}': added {len(added)} words")

        # Recount pairs after expansion
        new_member_pairs = []
        for pair in antonym_pairs:
            w1, w2 = pair['word1'], pair['word2']
            pole1_idx = None
            pole2_idx = None

            for idx, pole in enumerate(axis['poles']):
                if w1 in pole:
                    pole1_idx = idx
                if w2 in pole:
                    pole2_idx = idx

            if pole1_idx is not None and pole2_idx is not None and pole1_idx != pole2_idx:
                new_member_pairs.append(pair)

        axis['member_pairs'] = new_member_pairs
        axis['representative_pairs'] = new_member_pairs[:5]
        axis['n_pairs'] = len(new_member_pairs)
        axis['size'] = sum(len(pole) for pole in axis['poles'])

        if verbose:
            print(f"  Axis '{' / '.join(axis['pole_names'])}': {original_size} -> {axis['size']} words")

    # Re-sort by size
    axes.sort(key=lambda a: a['n_pairs'], reverse=True)

    return axes
