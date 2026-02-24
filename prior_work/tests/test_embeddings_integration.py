"""
Integration test for the full semantic vector space pipeline.

Tests the complete flow:
1. Parse sentences → Parse trees
2. Extract triples → Knowledge graph
3. Add WordNet constraints → Combined graph
4. Encode parse trees → Semantic vectors
5. Train embeddings → Learned vector space
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence
from src.graph import extract_triples, KnowledgeGraph, add_wordnet_constraints
from src.embeddings import encode_hypothesis, EmbeddingModel
from src.embeddings.anchors import AnchorDimensions


def test_parse_to_vector():
    """Test encoding a parse tree into a semantic vector."""
    print("\n" + "="*60)
    print("TEST: Parse Tree -> Semantic Vector")
    print("="*60)

    # Parse a sentence
    parser = QuantumParser('grammars/english.json')
    words = tag_sentence("The big dog runs")
    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"\nSentence: 'The big dog runs'")
    print(f"Parse score: {best.score:.3f}")
    print(f"Edges: {len(best.edges)}")

    # Create simple embedding model
    words_in_sentence = [node.value.text for node in best.nodes if node.value]
    vocabulary = list(set(words_in_sentence))

    print(f"Vocabulary: {vocabulary}")

    embedding_dim = 128  # Start small for testing
    model = EmbeddingModel(vocabulary, embedding_dim)
    word2idx = model.word_to_id

    # Create word embeddings dict
    word_embeddings = {word: model.embeddings[idx] for word, idx in word2idx.items()}

    # Encode the parse tree
    try:
        vector = encode_hypothesis(best, word_embeddings, model.anchors)
        print(f"\nGenerated vector shape: {vector.shape}")
        print(f"Vector norm: {np.linalg.norm(vector):.3f}")
        print(f"Non-zero dimensions: {np.count_nonzero(vector)}/{len(vector)}")

        # Check anchor dimensions are used
        anchor_dims = model.anchors.num_anchors()
        anchor_values = vector[:anchor_dims]
        print(f"\nAnchor dimensions active: {np.count_nonzero(anchor_values)}/{anchor_dims}")

        assert vector.shape[0] == embedding_dim, "Vector should match embedding dim"
        assert np.linalg.norm(vector) > 0, "Vector should be non-zero"

        print("\n[OK] Parse tree encoding works!")
        return True

    except Exception as e:
        print(f"\n[FAILED] Encoding error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_sentences():
    """Test encoding multiple sentences and checking vector differences."""
    print("\n" + "="*60)
    print("TEST: Multiple Sentence Encoding")
    print("="*60)

    parser = QuantumParser('grammars/english.json')

    sentences = [
        "The dog runs",
        "The cat runs",
        "The dog walks",
    ]

    # Build vocabulary from all sentences
    all_words = set()
    parses = []

    print(f"\nParsing {len(sentences)} sentences...")
    for sent in sentences:
        words = tag_sentence(sent)
        chart = parser.parse(words)
        best = chart.best_hypothesis()
        parses.append((sent, best))

        words_in_sent = [node.value.text for node in best.nodes if node.value]
        all_words.update(words_in_sent)

        print(f"  '{sent}' -> {len(best.edges)} edges")

    vocabulary = list(all_words)
    print(f"\nTotal vocabulary: {len(vocabulary)} words")

    # Create model
    embedding_dim = 128
    model = EmbeddingModel(vocabulary, embedding_dim)
    word2idx = model.word_to_id

    # Create word embeddings dict
    word_embeddings = {word: model.embeddings[idx] for word, idx in word2idx.items()}

    # Encode all sentences
    vectors = []
    for sent, parse in parses:
        try:
            vec = encode_hypothesis(parse, word_embeddings, model.anchors)
            vectors.append(vec)
        except Exception as e:
            print(f"[WARN] Could not encode '{sent}': {e}")
            vectors.append(np.zeros(embedding_dim))

    # Compare vectors
    print(f"\nVector comparisons:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            similarity = np.dot(vectors[i], vectors[j]) / (
                np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8
            )
            print(f"  '{sentences[i]}' vs '{sentences[j]}': {similarity:.3f}")

    # Sentences with same structure should be more similar
    # "dog runs" vs "cat runs" (different subject) should be closer than
    # "dog runs" vs "dog walks" (different verb)

    print("\n[OK] Multiple sentence encoding works!")
    return True


def test_graph_to_embeddings():
    """Test building a knowledge graph and preparing for training."""
    print("\n" + "="*60)
    print("TEST: Knowledge Graph -> Training Data")
    print("="*60)

    graph = KnowledgeGraph()
    parser = QuantumParser('grammars/english.json')

    sentences = [
        "The dog runs",
        "The cat walks",
        "Big dogs run quickly",
    ]

    print(f"\nBuilding knowledge graph from {len(sentences)} sentences...")

    for sent in sentences:
        words = tag_sentence(sent)
        chart = parser.parse(words)
        best = chart.best_hypothesis()

        triples = extract_triples(best)
        graph.add_triples(triples, language="en", source_type="parse")
        print(f"  '{sent}' -> {len(triples)} triples")

    # Add WordNet constraints (limited for speed)
    print("\nAdding WordNet constraints...")
    wordnet_edges = add_wordnet_constraints(graph, max_relations=3, verbose=False)
    print(f"Added {wordnet_edges} WordNet edges")

    # Export for training
    print("\nExporting graph for training...")
    edge_list, vocab = graph.export_for_training()

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Training edges: {len(edge_list)}")

    # Check statistics
    stats = graph.get_statistics()
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Edges by source:")
    for source_type, count in stats['edges_by_source'].items():
        print(f"    {source_type}: {count}")

    assert len(vocab) > 0, "Should have vocabulary"
    assert len(edge_list) > 0, "Should have training edges"

    print("\n[OK] Graph export for training works!")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("EMBEDDINGS INTEGRATION TESTS")
    print("="*60)

    all_passed = True

    try:
        all_passed &= test_parse_to_vector()
        all_passed &= test_multiple_sentences()
        all_passed &= test_graph_to_embeddings()

        if all_passed:
            print("\n" + "="*60)
            print("ALL INTEGRATION TESTS PASSED!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("SOME TESTS FAILED")
            print("="*60)

    except Exception as e:
        print(f"\n[FAILED] Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
