"""
Embedding Evaluation Script

Tests the quality of trained embeddings:
1. Word similarities (synonyms should be close)
2. Word analogies (king - man + woman ≈ queen)
3. Nearest neighbors
4. Composition tests

Usage:
    python scripts/evaluate_embeddings.py models/trained_embeddings.pkl
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
from pathlib import Path

from src.embeddings import EmbeddingModel


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


def test_similarities(model: EmbeddingModel):
    """Test word-to-word similarities."""
    print_section("Word Similarity Tests")

    # Test pairs with expected relationships
    test_pairs = [
        # Animals (should be similar)
        ('dog', 'cat', 'similar animals'),
        ('dog', 'bird', 'both animals'),

        # Sizes (synonyms)
        ('big', 'large', 'size synonyms'),
        ('small', 'little', 'size synonyms'),

        # Antonyms (should be dissimilar)
        ('big', 'small', 'size antonyms'),
        ('happy', 'sad', 'emotion antonyms'),
        ('fast', 'slow', 'speed antonyms'),

        # Unrelated
        ('dog', 'table', 'unrelated'),
        ('run', 'red', 'unrelated'),
    ]

    print("\nPairwise Similarities:")
    print(f"{'Word 1':<15} {'Word 2':<15} {'Similarity':<12} {'Expected'}")
    print("-" * 70)

    for w1, w2, expected in test_pairs:
        if model.get_embedding(w1) is not None and model.get_embedding(w2) is not None:
            sim = model.compute_similarity(w1, w2)
            print(f"{w1:<15} {w2:<15} {sim:>10.3f}   {expected}")
        else:
            missing = w1 if model.get_embedding(w1) is None else w2
            print(f"{w1:<15} {w2:<15} {'N/A':>10}   ('{missing}' not in vocab)")


def test_analogies(model: EmbeddingModel):
    """Test word analogies: A is to B as C is to D."""
    print_section("Word Analogy Tests")

    # Classic analogies to test
    analogies = [
        # ('king', 'man', 'woman', 'queen'),  # Classic example
        ('big', 'dog', 'cat', '?'),  # big dog : cat = ?
        ('fast', 'car', 'animal', '?'),  # fast car : animal = ?
    ]

    print("\nAnalogy: A - B + C ~= D")
    print(f"{'A':<10} {'B':<10} {'C':<10} {'Expected D':<12} {'Predicted (Top 3)'}")
    print("-" * 70)

    for a, b, c, expected_d in analogies:
        # Check if all words exist
        if not all([model.get_embedding(w) is not None for w in [a, b, c]]):
            missing = [w for w in [a, b, c] if model.get_embedding(w) is None]
            print(f"{a:<10} {b:<10} {c:<10} {expected_d:<12} Missing: {missing}")
            continue

        # Compute: A - B + C
        vec_a = model.get_embedding(a)
        vec_b = model.get_embedding(b)
        vec_c = model.get_embedding(c)

        result_vec = vec_a - vec_b + vec_c

        # Find nearest neighbors to result
        # (Compute similarities to all words)
        similarities = []
        for word in model.vocabulary:
            if word in [a, b, c]:  # Skip input words
                continue
            vec_w = model.get_embedding(word)
            if vec_w is not None:
                sim = np.dot(result_vec, vec_w) / (np.linalg.norm(result_vec) * np.linalg.norm(vec_w) + 1e-8)
                similarities.append((word, sim))

        # Get top 3
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_3 = [f"{w} ({s:.2f})" for w, s in similarities[:3]]

        print(f"{a:<10} {b:<10} {c:<10} {expected_d:<12} {', '.join(top_3)}")


def test_nearest_neighbors(model: EmbeddingModel):
    """Show nearest neighbors for test words."""
    print_section("Nearest Neighbors")

    test_words = ['dog', 'cat', 'big', 'small', 'run', 'walk', 'red', 'blue', 'happy', 'sad']
    available_words = [w for w in test_words if model.get_embedding(w) is not None]

    print(f"\nShowing top 5 nearest neighbors:\n")

    for word in available_words[:5]:  # Show first 5
        similar = model.get_similar_words(word, top_k=5)
        similar_str = ', '.join([f"{w} ({s:.2f})" for w, s in similar])
        print(f"  {word:<10} -> {similar_str}")


def test_composition(model: EmbeddingModel, parser=None):
    """Test that composition produces meaningful vectors."""
    print_section("Composition Tests")

    if parser is None:
        print("\nSkipping composition tests (no parser provided)")
        return

    from src.parser.pos_tagger import tag_sentence
    from src.embeddings import encode_hypothesis

    # Test sentences
    test_sentences = [
        "the big dog",
        "the small dog",
        "the big cat",
        "the red ball",
    ]

    print("\nEncoded Sentences:")
    print(f"{'Sentence':<20} {'Magnitude':<12} {'Sample dims'}")
    print("-" * 70)

    vectors = {}
    for sent in test_sentences:
        try:
            words = tag_sentence(sent)
            chart = parser.parse(words)
            if chart.hypotheses:
                vec = encode_hypothesis(chart.best_hypothesis(),
                                       model.get_embeddings_dict(),
                                       model.anchors)
                mag = np.linalg.norm(vec)
                sample = vec[:5]  # First 5 dims
                sample_str = ', '.join([f"{v:.2f}" for v in sample])
                print(f"{sent:<20} {mag:>10.3f}   [{sample_str}, ...]")
                vectors[sent] = vec
        except Exception as e:
            print(f"{sent:<20} {'ERROR':>10}   {str(e)[:40]}")

    # Compare compositions
    if len(vectors) >= 2:
        print("\nComposition Similarities:")
        print(f"{'Sentence 1':<20} {'Sentence 2':<20} {'Similarity'}")
        print("-" * 70)

        sents = list(vectors.keys())
        for i, s1 in enumerate(sents):
            for s2 in sents[i+1:]:
                sim = np.dot(vectors[s1], vectors[s2]) / (
                    np.linalg.norm(vectors[s1]) * np.linalg.norm(vectors[s2]) + 1e-8)
                print(f"{s1:<20} {s2:<20} {sim:>10.3f}")


def analyze_vocabulary(model: EmbeddingModel):
    """Analyze the vocabulary and embedding statistics."""
    print_section("Vocabulary Analysis")

    stats = model.get_statistics()

    print(f"\nModel Statistics:")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Embedding dimensions: {stats['embedding_dim']}")
    print(f"    - Anchor dimensions: {stats['num_anchors']}")
    print(f"    - Learned dimensions: {stats['num_learned_dims']}")
    print(f"  Average embedding norm: {stats['embedding_norm_mean']:.3f} ± {stats['embedding_norm_std']:.3f}")

    # Show sample vocabulary
    print(f"\nSample Vocabulary (first 20 words):")
    sample_words = sorted(model.vocabulary)[:20]
    for i, word in enumerate(sample_words):
        if i % 4 == 0:
            print()
            print("  ", end="")
        print(f"{word:<15}", end="")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained embeddings')
    parser.add_argument('model_path', type=str, help='Path to trained model (.pkl)')
    parser.add_argument('--grammar', type=str, default='grammars/english.json',
                       help='Grammar file for composition tests')
    parser.add_argument('--skip-composition', action='store_true',
                       help='Skip composition tests')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = EmbeddingModel.load(args.model_path)
    print(f"Model loaded successfully!")

    # Load parser for composition tests
    sentence_parser = None
    if not args.skip_composition:
        try:
            from src.parser.quantum_parser import QuantumParser
            sentence_parser = QuantumParser(args.grammar)
        except Exception as e:
            print(f"Warning: Could not load parser: {e}")

    # Run evaluations
    analyze_vocabulary(model)
    test_similarities(model)
    test_analogies(model)
    test_nearest_neighbors(model)
    test_composition(model, sentence_parser)

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
