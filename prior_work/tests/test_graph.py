"""
Tests for the graph module (triple extraction and knowledge graph).
"""

import sys
sys.path.insert(0, '.')

from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence
from src.graph import extract_triples, KnowledgeGraph, add_wordnet_constraints


def test_triple_extraction():
    """Test extracting semantic triples from parse trees."""
    print("\n" + "="*60)
    print("TEST: Triple Extraction from Parse Trees")
    print("="*60)

    # Parse a simple sentence
    parser = QuantumParser('grammars/english.json')
    words = tag_sentence("The big dog runs quickly")
    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"\nSentence: 'The big dog runs quickly'")
    print(f"Parse score: {best.score:.3f}")
    print(f"Edges in parse tree: {len(best.edges)}")

    # Extract triples
    triples = extract_triples(best)

    print(f"\nExtracted {len(triples)} semantic triples:")
    for triple in triples:
        print(f"  {triple}")

    assert len(triples) > 0, "Should extract at least one triple"
    print("\n[OK] Triple extraction works!")


def test_knowledge_graph():
    """Test building a knowledge graph from parse trees."""
    print("\n" + "="*60)
    print("TEST: Knowledge Graph Construction")
    print("="*60)

    graph = KnowledgeGraph()

    # Parse multiple sentences
    parser = QuantumParser('grammars/english.json')

    sentences = [
        "The big dog runs quickly",
        "Cats chase mice",
        "The red ball rolls",
    ]

    print(f"\nParsing {len(sentences)} sentences...")

    for sentence in sentences:
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        best = chart.best_hypothesis()

        triples = extract_triples(best)
        graph.add_triples(triples, language="en", source_type="parse")

        print(f"  '{sentence}' -> {len(triples)} triples")

    # Check graph statistics
    stats = graph.get_statistics()
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")

    assert stats['num_nodes'] > 0, "Should have nodes"
    assert stats['num_edges'] > 0, "Should have edges"

    # Test querying
    print(f"\nTesting graph queries...")

    if graph.get_node("dog", "en"):
        dog_neighbors = graph.get_neighbors("dog", "en")
        print(f"  'dog' has {len(dog_neighbors)} neighbors: {dog_neighbors}")

    print("\n[OK] Knowledge graph works!")


def test_wordnet_import():
    """Test importing WordNet relationships."""
    print("\n" + "="*60)
    print("TEST: WordNet Import")
    print("="*60)

    graph = KnowledgeGraph()

    # Import WordNet for a few words
    test_words = ["dog", "cat", "run", "chase", "big", "red"]

    print(f"\nImporting WordNet relationships for {len(test_words)} words...")

    from src.graph.wordnet_import import import_wordnet_for_vocabulary
    edges_added = import_wordnet_for_vocabulary(
        test_words,
        graph,
        max_relations=5,
        verbose=False
    )

    print(f"Added {edges_added} WordNet edges")

    # Check statistics
    stats = graph.get_statistics()
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Edges by source: {dict(stats['edges_by_source'])}")

    # Test synonym lookup
    from src.graph.wordnet_import import get_wordnet_synonyms
    dog_synonyms = get_wordnet_synonyms("dog")
    print(f"\nWordNet synonyms for 'dog': {list(dog_synonyms)[:5]}")

    assert edges_added > 0, "Should add WordNet edges"
    print("\n[OK] WordNet import works!")


def test_combined_graph():
    """Test combining parse-derived triples with WordNet."""
    print("\n" + "="*60)
    print("TEST: Combined Parse + WordNet Graph")
    print("="*60)

    graph = KnowledgeGraph()

    # Parse sentences
    parser = QuantumParser('grammars/english.json')
    sentence = "The big dog runs quickly"

    print(f"\nParsing: '{sentence}'")
    words = tag_sentence(sentence)
    chart = parser.parse(words)
    best = chart.best_hypothesis()

    triples = extract_triples(best)
    graph.add_triples(triples, language="en", source_type="parse")

    print(f"Added {len(triples)} parse-derived triples")

    # Add WordNet constraints
    print("\nAdding WordNet constraints for words in graph...")
    wordnet_edges = add_wordnet_constraints(graph, max_relations=5, verbose=False)

    print(f"Added {wordnet_edges} WordNet edges")

    # Check final statistics
    stats = graph.get_statistics()
    print(f"\nFinal graph statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Edges by source:")
    for source_type, count in stats['edges_by_source'].items():
        print(f"    {source_type}: {count}")

    # Test export for training
    print("\nExporting graph for training...")
    edge_list, vocab = graph.export_for_training()
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Training edges: {len(edge_list)}")
    print(f"  Sample edge: {edge_list[0] if edge_list else 'None'}")

    assert len(vocab) > 0, "Should have vocabulary"
    assert len(edge_list) > 0, "Should have training edges"

    print("\n[OK] Combined graph works!")


def main():
    """Run all graph tests."""
    print("\n" + "="*60)
    print("GRAPH MODULE TESTS")
    print("="*60)

    try:
        test_triple_extraction()
        test_knowledge_graph()
        test_wordnet_import()
        test_combined_graph()

        print("\n" + "="*60)
        print("ALL GRAPH TESTS PASSED! ✅")
        print("="*60)

    except Exception as e:
        print(f"\n[FAILED] Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
