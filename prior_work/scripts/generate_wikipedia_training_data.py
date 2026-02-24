"""Generate training edges from parsed Wikipedia corpus.

This script converts parsed Wikipedia sentences into training edges (semantic relations)
suitable for embedding training.

Usage:
    python scripts/generate_wikipedia_training_data.py

Input:
    data/wikipedia_parsed/parsed_corpus.pkl - Parse trees from parse_wikipedia_corpus.py

Output:
    data/wikipedia_training/training_edges.pkl - Training edges (~2GB, 484M edges)
    data/wikipedia_training/vocabulary.json - Vocabulary mapping
"""

import pickle
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
from tqdm import tqdm


def extract_edges_from_parse(parse_result: Dict, min_confidence: float = 0.3) -> List[Tuple[str, str, str, float]]:
    """Extract semantic edges from a parse result.

    Args:
        parse_result: Parse result dictionary
        min_confidence: Minimum confidence threshold

    Returns:
        List of (source_word, relation_type, target_word, confidence) tuples
    """
    if not parse_result['success']:
        return []

    edges = []
    words = parse_result.get('words', [])
    relations = parse_result.get('relations', [])

    # Extract co-occurrence edges from relations
    for rel in relations:
        source = rel.get('source', '').lower()
        target = rel.get('target', '').lower()
        rel_type = rel.get('type', 'co-occur')
        distance = rel.get('distance', 1)

        # Skip empty words
        if not source or not target:
            continue

        # Skip very short words (likely stopwords)
        if len(source) < 2 or len(target) < 2:
            continue

        # Confidence based on distance (closer = higher confidence)
        confidence = max(0.3, 1.0 - (distance * 0.1))

        if confidence >= min_confidence:
            edges.append((source, rel_type, target, confidence))

    return edges


def build_vocabulary(edges: List[Tuple[str, str, str, float]]) -> Dict:
    """Build vocabulary from edges.

    Args:
        edges: List of (source, relation, target, confidence) tuples

    Returns:
        Dictionary with word_to_id, id_to_word, vocab_size
    """
    # Collect all unique words
    words = set()
    for source, _, target, _ in edges:
        words.add(source)
        words.add(target)

    # Sort for consistent ordering
    sorted_words = sorted(words)

    # Create mappings
    word_to_id = {word: idx for idx, word in enumerate(sorted_words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'vocab_size': len(words)
    }


def filter_edges_by_frequency(
    edges: List[Tuple[str, str, str, float]],
    min_frequency: int = 2
) -> List[Tuple[str, str, str, float]]:
    """Filter edges, keeping only those where both words appear frequently enough.

    Args:
        edges: List of edges
        min_frequency: Minimum word frequency

    Returns:
        Filtered list of edges
    """
    # Count word frequencies
    word_freq = Counter()
    for source, _, target, _ in edges:
        word_freq[source] += 1
        word_freq[target] += 1

    # Filter edges
    filtered = []
    for source, relation, target, confidence in edges:
        if word_freq[source] >= min_frequency and word_freq[target] >= min_frequency:
            filtered.append((source, relation, target, confidence))

    return filtered


def aggregate_edge_weights(
    edges: List[Tuple[str, str, str, float]]
) -> List[Tuple[int, str, int, float]]:
    """Aggregate edges, combining duplicate edges by averaging confidence.

    Args:
        edges: List of (source_id, relation, target_id, confidence) tuples

    Returns:
        Aggregated list with averaged confidences
    """
    # Group by (source, relation, target)
    edge_groups = defaultdict(list)

    for source, relation, target, confidence in edges:
        key = (source, relation, target)
        edge_groups[key].append(confidence)

    # Average confidences
    aggregated = []
    for (source, relation, target), confidences in edge_groups.items():
        avg_confidence = sum(confidences) / len(confidences)
        aggregated.append((source, relation, target, avg_confidence))

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Generate training edges from parsed Wikipedia corpus"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/wikipedia_parsed/parsed_corpus.pkl"),
        help="Input parsed corpus file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wikipedia_training"),
        help="Output directory for training data"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum edge confidence (default: 0.3)"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum word frequency to include (default: 2)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Wikipedia Training Data Generation")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Min word frequency: {args.min_frequency}")
    print("=" * 70)

    # Load parsed corpus
    print("\n[1/6] Loading parsed corpus...")
    with open(args.input_file, 'rb') as f:
        parsed_corpus = pickle.load(f)

    successful_parses = [p for p in parsed_corpus if p.get('success', False)]
    print(f"  Loaded {len(parsed_corpus):,} parse results")
    print(f"  Successful: {len(successful_parses):,}")

    # Extract edges
    print("\n[2/6] Extracting edges from parse trees...")
    all_edges = []

    for parse_result in tqdm(successful_parses, desc="Extracting"):
        edges = extract_edges_from_parse(parse_result, args.min_confidence)
        all_edges.extend(edges)

    print(f"  Extracted {len(all_edges):,} raw edges")

    # Filter by frequency
    print("\n[3/6] Filtering by word frequency...")
    filtered_edges = filter_edges_by_frequency(all_edges, args.min_frequency)
    print(f"  Retained {len(filtered_edges):,} edges")
    print(f"  Filtered out: {len(all_edges) - len(filtered_edges):,} ({(1-len(filtered_edges)/len(all_edges))*100:.1f}%)")

    # Build vocabulary
    print("\n[4/6] Building vocabulary...")
    vocab = build_vocabulary(filtered_edges)
    print(f"  Vocabulary size: {vocab['vocab_size']:,} words")

    # Convert to ID-based edges
    print("\n[5/6] Converting to ID-based edges...")
    id_edges = []

    for source, relation, target, confidence in tqdm(filtered_edges, desc="Converting"):
        source_id = vocab['word_to_id'][source]
        target_id = vocab['word_to_id'][target]
        id_edges.append((source_id, relation, target_id, confidence))

    # Aggregate duplicate edges
    print("  Aggregating duplicate edges...")
    aggregated_edges = aggregate_edge_weights(id_edges)
    print(f"  After aggregation: {len(aggregated_edges):,} unique edges")

    # Save
    print("\n[6/6] Saving training data...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges_file = args.output_dir / "training_edges.pkl"
    vocab_file = args.output_dir / "vocabulary.json"

    with open(edges_file, 'wb') as f:
        pickle.dump(aggregated_edges, f)

    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

    # Statistics
    edges_size_mb = edges_file.stat().st_size / (1024**2)
    vocab_size_mb = vocab_file.stat().st_size / (1024**2)

    # Relation type distribution
    relation_counts = Counter(rel for _, rel, _, _ in aggregated_edges)

    print("\n" + "=" * 70)
    print("Training Data Generation Complete!")
    print("=" * 70)
    print(f"  Edges file: {edges_file} ({edges_size_mb:.1f} MB)")
    print(f"  Vocabulary file: {vocab_file} ({vocab_size_mb:.1f} MB)")
    print(f"  Vocabulary size: {vocab['vocab_size']:,} words")
    print(f"  Training edges: {len(aggregated_edges):,}")
    print(f"\n  Relation type distribution:")
    for rel_type, count in relation_counts.most_common(10):
        print(f"    {rel_type}: {count:,} ({count/len(aggregated_edges)*100:.1f}%)")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Run: python scripts/merge_datasets.py")
    print("  2. This will merge WordNet + Wikipedia datasets")


if __name__ == "__main__":
    main()
