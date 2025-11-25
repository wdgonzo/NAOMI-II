"""
Training Script for NAOMI-II Semantic Embeddings

This script trains word embeddings using:
1. Parse-derived semantic triples
2. WordNet relationship constraints
3. Structure-aware composition loss

Usage:
    python scripts/train_embeddings.py [--epochs 100] [--lr 0.01]
"""

import sys
sys.path.insert(0, '.')

import argparse
from pathlib import Path

from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence
from src.graph import KnowledgeGraph, extract_triples, add_wordnet_constraints
from src.embeddings import (
    EmbeddingModel,
    train_embeddings,
    generate_training_corpus
)


def main():
    parser = argparse.ArgumentParser(description='Train NAOMI-II semantic embeddings')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--sentences', type=int, default=200, help='Number of training sentences')
    parser.add_argument('--output', type=str, default='models/trained_embeddings.pkl',
                       help='Output model path')
    parser.add_argument('--grammar', type=str, default='grammars/english.json',
                       help='Grammar file')

    args = parser.parse_args()

    print("="*70)
    print("NAOMI-II Semantic Embedding Training")
    print("="*70)

    # 1. Generate training corpus
    print("\n[1/7] Generating training corpus...")
    sentences = generate_training_corpus(args.sentences)
    print(f"      Generated {len(sentences)} sentences")

    # 2. Parse sentences
    print("\n[2/7] Parsing sentences...")
    sentence_parser = QuantumParser(args.grammar)
    parsed = []
    graph = KnowledgeGraph()

    for i, sent in enumerate(sentences):
        if (i + 1) % 50 == 0:
            print(f"      Parsed {i + 1}/{len(sentences)} sentences...")

        try:
            words = tag_sentence(sent)
            chart = sentence_parser.parse(words)

            if chart.hypotheses:
                hyp = chart.best_hypothesis()
                parsed.append(hyp)

                # Extract triples
                triples = extract_triples(hyp)
                graph.add_triples(triples, language="en", source_type="parse")
        except Exception as e:
            print(f"      Warning: Failed to parse '{sent}': {e}")
            continue

    print(f"      Successfully parsed {len(parsed)}/{len(sentences)} sentences")

    # 3. Add WordNet constraints
    print("\n[3/7] Adding WordNet constraints...")
    print("      (This may take a minute...)")
    num_wordnet_edges = add_wordnet_constraints(graph, max_relations=5, verbose=False)
    print(f"      Added {num_wordnet_edges} WordNet relationship edges")

    # 4. Build vocabulary
    print("\n[4/7] Building vocabulary...")
    stats = graph.get_statistics()
    vocabulary = list(set(node.word for node in graph.nodes.values()))
    vocabulary.sort()  # Sort for consistency
    print(f"      Vocabulary: {len(vocabulary)} unique words")
    print(f"      Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    # 5. Initialize model
    print("\n[5/7] Initializing embedding model...")
    model = EmbeddingModel(vocabulary, embedding_dim=None)  # Auto-determine dimensionality
    print(f"      Model dimensions: {model.embedding_dim}")
    print(f"      - Anchor dimensions: {model.num_anchors}")
    print(f"      - Learned dimensions: {model.embedding_dim - model.num_anchors}")

    # 6. Train!
    print("\n[6/7] Training embeddings...")
    validation_sentences = [
        "the big dog",
        "cats run",
        "the red ball"
    ]

    history = train_embeddings(
        model=model,
        knowledge_graph=graph,
        parsed_sentences=parsed,
        parser=sentence_parser,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        validation_sentences=validation_sentences
    )

    # 7. Save model
    print("\n[7/7] Saving trained model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"      Model saved to: {output_path}")

    # Display final results
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"  Total Loss: {history['total_loss'][-1]:.4f}")
    print(f"  WordNet Loss: {history['wordnet_loss'][-1]:.4f}")
    print(f"  Composition Loss: {history['composition_loss'][-1]:.4f}")
    print(f"  Constraint Satisfaction: {history['satisfaction_rate'][-1]*100:.1f}%")

    # Show some learned similarities
    print(f"\nLearned Similarities (examples):")
    test_words = ['dog', 'cat', 'big', 'run']
    for word in test_words:
        if model.get_embedding(word) is not None:
            similar = model.get_similar_words(word, top_k=5)
            similar_str = ', '.join([f"{w} ({s:.2f})" for w, s in similar])
            print(f"  {word:10s} -> {similar_str}")

    print(f"\nModel ready for use!")
    print(f"Load with: EmbeddingModel.load('{output_path}')")


if __name__ == "__main__":
    main()
