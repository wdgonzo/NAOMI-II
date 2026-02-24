"""
NAOMI-II Training Data Generation Script

Generates a large training dataset by:
1. Collecting sentences from multiple sources
2. Parsing them with the quantum parser
3. Building a knowledge graph with parse triples + WordNet constraints
4. Exporting for embedding training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle

from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence
from src.graph.triple_extractor import extract_triples
from src.graph.knowledge_graph import KnowledgeGraph
from src.graph.wordnet_import import import_wordnet_for_word


def generate_simple_sentences() -> List[str]:
    """Generate a set of simple training sentences covering basic grammar."""
    subjects = ["The dog", "The cat", "The bird", "A man", "A woman", "The child",
                "The student", "The teacher", "My friend", "The doctor"]
    verbs = ["runs", "walks", "jumps", "sits", "stands", "sleeps", "eats", "drinks",
             "reads", "writes", "thinks", "speaks"]
    objects = ["quickly", "slowly", "happily", "sadly", "loudly", "quietly",
               "a book", "the ball", "some food", "the water"]
    adjectives = ["big", "small", "red", "blue", "happy", "sad", "fast", "slow"]

    sentences = []

    # Subject + Verb
    for subj in subjects:
        for verb in verbs[:6]:
            sentences.append(f"{subj} {verb}")

    # Adjective + Subject + Verb
    for adj in adjectives[:4]:
        for subj in subjects[:5]:
            for verb in verbs[:3]:
                noun = subj.split()[-1]
                sentences.append(f"The {adj} {noun} {verb}")

    # Subject + Verb + Adverb
    for subj in subjects[:5]:
        for verb in verbs[:5]:
            for adv in objects[:6]:
                sentences.append(f"{subj} {verb} {adv}")

    # Subject + Verb + Object
    for subj in subjects[:5]:
        for verb in verbs[6:]:
            for obj in objects[6:]:
                sentences.append(f"{subj} {verb} {obj}")

    return list(set(sentences))  # Remove duplicates


def load_custom_sentences(file_path: str) -> List[str]:
    """Load sentences from a text file (one per line)."""
    if not Path(file_path).exists():
        print(f"[SKIP] File not found: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def build_knowledge_graph_from_sentences(
    sentences: List[str],
    parser: QuantumParser,
    max_sentences: int = None,
    show_progress: bool = True
) -> Tuple[KnowledgeGraph, Dict[str, int], List[Tuple[str, any]]]:
    """
    Parse all sentences and build a unified knowledge graph.

    Args:
        sentences: List of sentences to parse
        parser: Initialized QuantumParser
        max_sentences: Maximum number of sentences to process (None = all)
        show_progress: Show progress bar

    Returns:
        (knowledge_graph, parse_stats, parsed_sentences)
        where parsed_sentences is List[Tuple[sentence, hypothesis]]
    """
    graph = KnowledgeGraph()
    parsed_sentences = []
    stats = {
        'total_sentences': 0,
        'parsed_successfully': 0,
        'parse_failures': 0,
        'total_triples': 0
    }

    if max_sentences:
        sentences = sentences[:max_sentences]

    iterator = tqdm(sentences, desc="Parsing sentences") if show_progress else sentences

    for sentence in iterator:
        stats['total_sentences'] += 1

        try:
            # Tokenize and POS tag
            words = tag_sentence(sentence)

            # Parse
            chart = parser.parse(words)
            hypothesis = chart.best_hypothesis()

            if hypothesis and hypothesis.score > 0:
                stats['parsed_successfully'] += 1

                # Save parse hypothesis for structure-aware training
                parsed_sentences.append((sentence, hypothesis))

                # Extract semantic triples
                triples = extract_triples(hypothesis)
                stats['total_triples'] += len(triples)

                # Add to knowledge graph
                for triple in triples:
                    graph.add_edge(
                        triple.subject,
                        triple.object,
                        triple.relation,
                        confidence=1.0,
                        source_type="parse"
                    )
            else:
                stats['parse_failures'] += 1

        except Exception as e:
            stats['parse_failures'] += 1
            if show_progress:
                print(f"\n[ERROR] Failed to parse: '{sentence}' - {str(e)}")

    return graph, stats, parsed_sentences


def add_wordnet_constraints(
    graph: KnowledgeGraph,
    max_words: int = None,
    show_progress: bool = True
) -> Dict[str, int]:
    """
    Add WordNet constraints for words in the graph.

    Args:
        graph: Knowledge graph to augment
        max_words: Maximum words to process (None = all)
        show_progress: Show progress bar

    Returns:
        stats dict
    """
    # Extract unique words from graph nodes (nodes.keys() are tuples (word, lang))
    unique_words = list(set(node.word for node in graph.nodes.values() if node.language == "en"))
    if max_words:
        unique_words = unique_words[:max_words]

    stats = {
        'words_processed': 0,
        'wordnet_edges_added': 0
    }

    iterator = tqdm(unique_words, desc="Adding WordNet constraints") if show_progress else unique_words

    for word in iterator:
        stats['words_processed'] += 1
        edges_before = len(graph.edges)

        try:
            import_wordnet_for_word(word, graph, language="en")
        except Exception as e:
            if show_progress:
                print(f"\n[WARNING] WordNet import failed for '{word}': {e}")

        edges_after = len(graph.edges)
        stats['wordnet_edges_added'] += (edges_after - edges_before)

    return stats


def main():
    print("=" * 60)
    print("NAOMI-II TRAINING DATA GENERATION")
    print("=" * 60)
    print()

    # Configuration
    OUTPUT_DIR = Path("data/training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    MAX_SENTENCES = 5000  # Adjust based on available memory/time
    MAX_WORDNET_WORDS = 1000  # Limit WordNet expansion

    # Step 1: Collect sentences
    print("[1/5] Collecting training sentences...")
    sentences = []

    # Generate simple sentences
    simple = generate_simple_sentences()
    print(f"  Generated {len(simple)} simple sentences")
    sentences.extend(simple)

    # Load custom sentences if available
    custom_file = OUTPUT_DIR.parent / "custom_sentences.txt"
    custom = load_custom_sentences(str(custom_file))
    if custom:
        print(f"  Loaded {len(custom)} custom sentences from {custom_file}")
        sentences.extend(custom)

    print(f"  Total sentences collected: {len(sentences)}")
    print()

    # Step 2: Initialize parser
    print("[2/5] Initializing quantum parser...")
    grammar_path = Path(__file__).parent.parent / "grammars" / "english.json"
    parser = QuantumParser(str(grammar_path))
    print(f"  Parser initialized")
    print()

    # Step 3: Parse sentences and build graph
    print(f"[3/5] Parsing sentences (max: {MAX_SENTENCES})...")
    graph, parse_stats, parsed_sentences = build_knowledge_graph_from_sentences(
        sentences,
        parser,
        max_sentences=MAX_SENTENCES,
        show_progress=True
    )
    print()
    print("  Parse Statistics:")
    print(f"    Total sentences: {parse_stats['total_sentences']}")
    print(f"    Parsed successfully: {parse_stats['parsed_successfully']}")
    print(f"    Parse failures: {parse_stats['parse_failures']}")
    print(f"    Success rate: {parse_stats['parsed_successfully']/parse_stats['total_sentences']*100:.1f}%")
    print(f"    Total triples extracted: {parse_stats['total_triples']}")
    print(f"    Parsed hypotheses saved: {len(parsed_sentences)}")
    print()

    # Step 4: Add WordNet constraints
    print(f"[4/5] Adding WordNet constraints (max words: {MAX_WORDNET_WORDS})...")
    wordnet_stats = add_wordnet_constraints(
        graph,
        max_words=MAX_WORDNET_WORDS,
        show_progress=True
    )
    print()
    print("  WordNet Statistics:")
    print(f"    Words processed: {wordnet_stats['words_processed']}")
    print(f"    WordNet edges added: {wordnet_stats['wordnet_edges_added']}")
    print()

    # Step 5: Export training data
    print("[5/5] Exporting training data...")

    # Export graph
    train_edges, word_to_id = graph.export_for_training()
    id_to_word = {v: k for k, v in word_to_id.items()}

    # Save graph object
    graph_file = OUTPUT_DIR / "knowledge_graph.pkl"
    with open(graph_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"  Saved graph to: {graph_file}")

    # Save training edges
    edges_file = OUTPUT_DIR / "training_edges.pkl"
    with open(edges_file, 'wb') as f:
        pickle.dump(train_edges, f)
    print(f"  Saved {len(train_edges)} training edges to: {edges_file}")

    # Save vocabulary
    vocab_file = OUTPUT_DIR / "vocabulary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'vocab_size': len(word_to_id)
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary ({len(word_to_id)} words) to: {vocab_file}")

    # Save parsed sentences for structure-aware training
    parsed_file = OUTPUT_DIR / "parsed_sentences.pkl"
    with open(parsed_file, 'wb') as f:
        pickle.dump(parsed_sentences, f)
    print(f"  Saved {len(parsed_sentences)} parsed hypotheses to: {parsed_file}")

    # Save metadata
    metadata_file = OUTPUT_DIR / "metadata.json"
    metadata = {
        'total_sentences': parse_stats['total_sentences'],
        'parsed_successfully': parse_stats['parsed_successfully'],
        'parse_failures': parse_stats['parse_failures'],
        'total_triples': parse_stats['total_triples'],
        'wordnet_words': wordnet_stats['words_processed'],
        'wordnet_edges': wordnet_stats['wordnet_edges_added'],
        'vocab_size': len(word_to_id),
        'training_edges': len(train_edges),
        'graph_nodes': len(graph.nodes),
        'graph_edges': len(graph.edges)
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to: {metadata_file}")
    print()

    # Final summary
    print("=" * 60)
    print("TRAINING DATA GENERATION COMPLETE!")
    print("=" * 60)
    print()
    print(f"Vocabulary size: {len(word_to_id)} words")
    print(f"Training edges: {len(train_edges)} constraints")
    print(f"Graph nodes: {len(graph.nodes)}")
    print(f"Graph edges: {len(graph.edges)}")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/train_embeddings.py")
    print("  2. Evaluate: python scripts/evaluate_embeddings.py")
    print()


if __name__ == "__main__":
    main()
