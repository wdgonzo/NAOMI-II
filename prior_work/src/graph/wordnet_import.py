"""
WordNet Importer

Imports semantic relationships from Princeton WordNet into the knowledge graph.

WordNet provides expert-labeled relationships:
- Synonyms (same meaning)
- Antonyms (opposite meaning)
- Hypernyms (is-a relationships: dog → animal)
- Hyponyms (inverse of hypernyms: animal → dog)
- Meronyms (part-of relationships: wheel → car)
- Holonyms (inverse of meronyms: car → wheel)

These serve as ground truth constraints for training embeddings.
"""

from typing import List, Set, Dict
import nltk
from nltk.corpus import wordnet as wn

from .knowledge_graph import KnowledgeGraph
from .triple_extractor import RelationType


# Extended relation types for WordNet
class WordNetRelationType:
    """Mapping from WordNet relation types to our RelationType."""

    SYNONYM = RelationType.GENERIC_RELATION  # Same meaning
    # We'll use a special marker for synonyms in confidence score

    # For other WordNet relations, we could extend RelationType enum
    # For now, map to existing types


def ensure_wordnet_downloaded():
    """Ensure WordNet data is downloaded."""
    try:
        wn.synsets('dog')
    except LookupError:
        print("Downloading WordNet data...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Open Multilingual WordNet
        print("WordNet downloaded successfully!")


def get_wordnet_synonyms(word: str, pos: str = None) -> Set[str]:
    """
    Get synonyms for a word from WordNet.

    Args:
        word: The word to find synonyms for
        pos: Part of speech filter ('n', 'v', 'a', 'r')

    Returns:
        Set of synonym words
    """
    synonyms = set()

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.add(synonym.lower())

    return synonyms


def get_wordnet_antonyms(word: str, pos: str = None) -> Set[str]:
    """Get antonyms for a word from WordNet."""
    antonyms = set()

    for synset in wn.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                antonym_word = antonym.name().replace('_', ' ')
                antonyms.add(antonym_word.lower())

    return antonyms


def get_wordnet_hypernyms(word: str, pos: str = None) -> Set[str]:
    """
    Get hypernyms (broader categories) for a word.

    Example: dog → canine → animal → organism
    """
    hypernyms = set()

    for synset in wn.synsets(word, pos=pos):
        for hypernym_synset in synset.hypernyms():
            for lemma in hypernym_synset.lemmas():
                hypernym_word = lemma.name().replace('_', ' ')
                hypernyms.add(hypernym_word.lower())

    return hypernyms


def get_wordnet_hyponyms(word: str, pos: str = None) -> Set[str]:
    """
    Get hyponyms (more specific instances) for a word.

    Example: animal → mammal → dog → beagle
    """
    hyponyms = set()

    for synset in wn.synsets(word, pos=pos):
        for hyponym_synset in synset.hyponyms():
            for lemma in hyponym_synset.lemmas():
                hyponym_word = lemma.name().replace('_', ' ')
                hyponyms.add(hyponym_word.lower())

    return hyponyms


def import_wordnet_for_word(word: str, graph: KnowledgeGraph,
                            language: str = "en",
                            max_relations: int = 10) -> int:
    """
    Import WordNet relationships for a single word into the graph.

    Args:
        word: The word to import relationships for
        graph: KnowledgeGraph to add relationships to
        language: Language code (default "en")
        max_relations: Maximum number of relationships per type to import

    Returns:
        Number of edges added
    """
    edges_added = 0

    # Add synonyms (with special confidence marker)
    synonyms = get_wordnet_synonyms(word)
    for synonym in list(synonyms)[:max_relations]:
        graph.add_edge(
            source_word=word,
            target_word=synonym,
            relation=RelationType.GENERIC_RELATION,
            source_lang=language,
            target_lang=language,
            confidence=0.95,  # High confidence for synonyms
            source_type="wordnet_synonym"
        )
        edges_added += 1

    # Add antonyms (we can use confidence to indicate opposite)
    antonyms = get_wordnet_antonyms(word)
    for antonym in list(antonyms)[:max_relations]:
        graph.add_edge(
            source_word=word,
            target_word=antonym,
            relation=RelationType.GENERIC_RELATION,
            source_lang=language,
            target_lang=language,
            confidence=0.05,  # Low confidence indicates antonym (opposite)
            source_type="wordnet_antonym"
        )
        edges_added += 1

    # Add hypernyms (broader categories)
    hypernyms = get_wordnet_hypernyms(word)
    for hypernym in list(hypernyms)[:max_relations]:
        graph.add_edge(
            source_word=word,
            target_word=hypernym,
            relation=RelationType.GENERIC_RELATION,
            source_lang=language,
            target_lang=language,
            confidence=0.8,  # Moderate confidence for hypernyms
            source_type="wordnet_hypernym"
        )
        edges_added += 1

    # Add hyponyms (more specific)
    hyponyms = get_wordnet_hyponyms(word)
    for hyponym in list(hyponyms)[:max_relations]:
        graph.add_edge(
            source_word=word,
            target_word=hyponym,
            relation=RelationType.GENERIC_RELATION,
            source_lang=language,
            target_lang=language,
            confidence=0.8,
            source_type="wordnet_hyponym"
        )
        edges_added += 1

    return edges_added


def import_wordnet_for_vocabulary(vocabulary: List[str], graph: KnowledgeGraph,
                                  language: str = "en",
                                  max_relations: int = 10,
                                  verbose: bool = True) -> int:
    """
    Import WordNet relationships for a list of words.

    Args:
        vocabulary: List of words to import
        graph: KnowledgeGraph to add to
        language: Language code
        max_relations: Max relations per word per type
        verbose: Print progress

    Returns:
        Total number of edges added
    """
    ensure_wordnet_downloaded()

    total_edges = 0

    for i, word in enumerate(vocabulary):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(vocabulary)} words...")

        edges_added = import_wordnet_for_word(word, graph, language, max_relations)
        total_edges += edges_added

    if verbose:
        print(f"\nWordNet import complete!")
        print(f"Total edges added: {total_edges}")

    return total_edges


def get_wordnet_word_senses(word: str, pos: str = None) -> List[Dict]:
    """
    Get all senses (meanings) of a word from WordNet.

    This is useful for sense discovery validation.

    Args:
        word: The word to look up
        pos: Part of speech filter

    Returns:
        List of dicts with sense information
    """
    senses = []

    for i, synset in enumerate(wn.synsets(word, pos=pos)):
        sense_info = {
            'sense_id': i,
            'synset_id': synset.name(),
            'definition': synset.definition(),
            'examples': synset.examples(),
            'lemmas': [l.name() for l in synset.lemmas()],
        }
        senses.append(sense_info)

    return senses


def build_wordnet_graph(vocabulary: List[str],
                       max_relations: int = 10,
                       verbose: bool = True) -> KnowledgeGraph:
    """
    Build a complete knowledge graph from WordNet for a vocabulary.

    Args:
        vocabulary: List of words to include
        max_relations: Max relations per word per type
        verbose: Print progress

    Returns:
        KnowledgeGraph with WordNet relationships
    """
    graph = KnowledgeGraph()

    if verbose:
        print(f"Building WordNet graph for {len(vocabulary)} words...")

    import_wordnet_for_vocabulary(vocabulary, graph, max_relations=max_relations, verbose=verbose)

    if verbose:
        stats = graph.get_statistics()
        print(f"\nGraph statistics:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")

    return graph


# Convenience function for common use case
def add_wordnet_constraints(graph: KnowledgeGraph,
                           max_relations: int = 10,
                           verbose: bool = True) -> int:
    """
    Add WordNet relationships for all words already in the graph.

    Args:
        graph: Existing KnowledgeGraph (e.g., built from parse trees)
        max_relations: Max relations per word
        verbose: Print progress

    Returns:
        Number of edges added
    """
    # Get all English words from graph
    vocabulary = [node.word for node in graph.nodes.values() if node.language == "en"]

    if verbose:
        print(f"Adding WordNet constraints for {len(vocabulary)} existing words...")

    return import_wordnet_for_vocabulary(vocabulary, graph, max_relations=max_relations, verbose=verbose)
