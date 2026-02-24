"""Merge WordNet and Wikipedia training datasets.

This script combines WordNet and Wikipedia datasets into a unified training dataset
with a shared vocabulary space. Outputs both merged and separate edge lists for
cycled training.

Usage:
    python scripts/merge_datasets.py

Input:
    data/wordnet_training/training_edges.pkl
    data/wordnet_training/vocabulary.json
    data/wikipedia_training/training_edges.pkl
    data/wikipedia_training/vocabulary.json

Output:
    data/combined_training/training_edges.pkl - All edges combined
    data/combined_training/vocabulary.json - Unified vocabulary
    data/combined_training/wordnet_edges.pkl - WordNet edges (remapped IDs)
    data/combined_training/wikipedia_edges.pkl - Wikipedia edges (remapped IDs)
"""

import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from tqdm import tqdm


def load_dataset(data_dir: Path) -> Tuple[List, Dict]:
    """Load training edges and vocabulary from a dataset directory.

    Args:
        data_dir: Directory containing training_edges.pkl and vocabulary.json

    Returns:
        Tuple of (edges, vocabulary)
    """
    edges_file = data_dir / "training_edges.pkl"
    vocab_file = data_dir / "vocabulary.json"

    with open(edges_file, 'rb') as f:
        edges = pickle.load(f)

    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    return edges, vocab


def merge_vocabularies(
    wordnet_vocab: Dict,
    wikipedia_vocab: Dict
) -> Tuple[Dict, Dict, Dict]:
    """Merge two vocabularies into a unified vocabulary space.

    Args:
        wordnet_vocab: WordNet vocabulary dictionary
        wikipedia_vocab: Wikipedia vocabulary dictionary

    Returns:
        Tuple of (unified_vocab, wordnet_id_map, wikipedia_id_map)
        - unified_vocab: New merged vocabulary
        - wordnet_id_map: Mapping from old WordNet IDs to new IDs
        - wikipedia_id_map: Mapping from old Wikipedia IDs to new IDs
    """
    # Get all unique words from both vocabularies
    wordnet_words = set(wordnet_vocab['word_to_id'].keys())
    wikipedia_words = set(wikipedia_vocab['word_to_id'].keys())
    all_words = wordnet_words | wikipedia_words

    # Statistics
    shared_words = wordnet_words & wikipedia_words
    wordnet_only = wordnet_words - wikipedia_words
    wikipedia_only = wikipedia_words - wordnet_words

    print(f"\n  Vocabulary statistics:")
    print(f"    WordNet only: {len(wordnet_only):,}")
    print(f"    Wikipedia only: {len(wikipedia_only):,}")
    print(f"    Shared: {len(shared_words):,}")
    print(f"    Total unique: {len(all_words):,}")

    # Create unified vocabulary (sorted for consistency)
    sorted_words = sorted(all_words)
    unified_word_to_id = {word: idx for idx, word in enumerate(sorted_words)}
    unified_id_to_word = {idx: word for word, idx in unified_word_to_id.items()}

    # Create ID mapping dictionaries
    wordnet_id_map = {}
    for old_id_str, word in wordnet_vocab['id_to_word'].items():
        old_id = int(old_id_str)
        new_id = unified_word_to_id[word]
        wordnet_id_map[old_id] = new_id

    wikipedia_id_map = {}
    for old_id_str, word in wikipedia_vocab['id_to_word'].items():
        old_id = int(old_id_str)
        new_id = unified_word_to_id[word]
        wikipedia_id_map[old_id] = new_id

    unified_vocab = {
        'word_to_id': unified_word_to_id,
        'id_to_word': unified_id_to_word,
        'vocab_size': len(all_words),
        'sources': {
            'wordnet_words': len(wordnet_words),
            'wikipedia_words': len(wikipedia_words),
            'shared_words': len(shared_words)
        }
    }

    return unified_vocab, wordnet_id_map, wikipedia_id_map


def remap_edges(
    edges: List[Tuple],
    id_map: Dict[int, int],
    source_tag: str
) -> List[Tuple[int, str, int, float]]:
    """Remap edge IDs from old vocabulary to new unified vocabulary.

    Args:
        edges: List of (source_id, relation, target_id, confidence) tuples
        id_map: Mapping from old IDs to new IDs
        source_tag: Tag to prepend to relation types (e.g., 'wordnet' or 'wikipedia')

    Returns:
        List of remapped edges with tagged relation types
    """
    remapped = []

    for source_id, relation, target_id, confidence in tqdm(edges, desc=f"Remapping {source_tag}"):
        # Map to new IDs
        new_source_id = id_map[source_id]
        new_target_id = id_map[target_id]

        # Tag relation type with source
        tagged_relation = f"{source_tag}:{relation}"

        remapped.append((new_source_id, tagged_relation, new_target_id, confidence))

    return remapped


def main():
    parser = argparse.ArgumentParser(
        description="Merge WordNet and Wikipedia training datasets"
    )
    parser.add_argument(
        "--wordnet-dir",
        type=Path,
        default=Path("data/wordnet_training"),
        help="WordNet training data directory"
    )
    parser.add_argument(
        "--wikipedia-dir",
        type=Path,
        default=Path("data/wikipedia_training"),
        help="Wikipedia training data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/combined_training"),
        help="Output directory for merged data"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Dataset Merging: WordNet + Wikipedia")
    print("=" * 70)
    print(f"WordNet directory: {args.wordnet_dir}")
    print(f"Wikipedia directory: {args.wikipedia_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Load WordNet dataset
    print("\n[1/5] Loading WordNet dataset...")
    wordnet_edges, wordnet_vocab = load_dataset(args.wordnet_dir)
    print(f"  WordNet edges: {len(wordnet_edges):,}")
    print(f"  WordNet vocabulary: {wordnet_vocab['vocab_size']:,} words")

    # Load Wikipedia dataset
    print("\n[2/5] Loading Wikipedia dataset...")
    wikipedia_edges, wikipedia_vocab = load_dataset(args.wikipedia_dir)
    print(f"  Wikipedia edges: {len(wikipedia_edges):,}")
    print(f"  Wikipedia vocabulary: {wikipedia_vocab['vocab_size']:,} words")

    # Merge vocabularies
    print("\n[3/5] Merging vocabularies...")
    unified_vocab, wordnet_id_map, wikipedia_id_map = merge_vocabularies(
        wordnet_vocab,
        wikipedia_vocab
    )
    print(f"  Unified vocabulary: {unified_vocab['vocab_size']:,} words")

    # Remap edges
    print("\n[4/5] Remapping edge IDs to unified vocabulary...")
    wordnet_remapped = remap_edges(wordnet_edges, wordnet_id_map, 'wordnet')
    wikipedia_remapped = remap_edges(wikipedia_edges, wikipedia_id_map, 'wikipedia')

    # Combine edges
    combined_edges = wordnet_remapped + wikipedia_remapped
    print(f"\n  Combined edges: {len(combined_edges):,}")
    print(f"    WordNet: {len(wordnet_remapped):,} ({len(wordnet_remapped)/len(combined_edges)*100:.1f}%)")
    print(f"    Wikipedia: {len(wikipedia_remapped):,} ({len(wikipedia_remapped)/len(combined_edges)*100:.1f}%)")

    # Save merged dataset
    print("\n[5/5] Saving merged dataset...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save combined edges
    combined_edges_file = args.output_dir / "training_edges.pkl"
    with open(combined_edges_file, 'wb') as f:
        pickle.dump(combined_edges, f)

    # Save unified vocabulary
    vocab_file = args.output_dir / "vocabulary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(unified_vocab, f, indent=2)

    # Save separate edge lists for cycled training
    wordnet_edges_file = args.output_dir / "wordnet_edges.pkl"
    with open(wordnet_edges_file, 'wb') as f:
        pickle.dump(wordnet_remapped, f)

    wikipedia_edges_file = args.output_dir / "wikipedia_edges.pkl"
    with open(wikipedia_edges_file, 'wb') as f:
        pickle.dump(wikipedia_remapped, f)

    # Statistics
    combined_size_mb = combined_edges_file.stat().st_size / (1024**2)
    vocab_size_mb = vocab_file.stat().st_size / (1024**2)
    wordnet_size_mb = wordnet_edges_file.stat().st_size / (1024**2)
    wikipedia_size_mb = wikipedia_edges_file.stat().st_size / (1024**2)

    total_size_mb = combined_size_mb + vocab_size_mb + wordnet_size_mb + wikipedia_size_mb

    print("\n" + "=" * 70)
    print("Dataset Merging Complete!")
    print("=" * 70)
    print(f"  Output directory: {args.output_dir}")
    print(f"\n  Files created:")
    print(f"    training_edges.pkl: {combined_size_mb:.1f} MB ({len(combined_edges):,} edges)")
    print(f"    vocabulary.json: {vocab_size_mb:.1f} MB ({unified_vocab['vocab_size']:,} words)")
    print(f"    wordnet_edges.pkl: {wordnet_size_mb:.1f} MB ({len(wordnet_remapped):,} edges)")
    print(f"    wikipedia_edges.pkl: {wikipedia_size_mb:.1f} MB ({len(wikipedia_remapped):,} edges)")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print("=" * 70)

    print("\n  Vocabulary breakdown:")
    print(f"    WordNet words: {unified_vocab['sources']['wordnet_words']:,}")
    print(f"    Wikipedia words: {unified_vocab['sources']['wikipedia_words']:,}")
    print(f"    Shared words: {unified_vocab['sources']['shared_words']:,}")
    print(f"    Unique words: {unified_vocab['vocab_size']:,}")

    print("\n  Edge breakdown:")
    print(f"    WordNet edges: {len(wordnet_remapped):,}")
    print(f"    Wikipedia edges: {len(wikipedia_remapped):,}")
    print(f"    Total edges: {len(combined_edges):,}")
    print(f"    Ratio: 1 WordNet : {len(wikipedia_remapped)/len(wordnet_remapped):.1f} Wikipedia")

    print("\nNext steps:")
    print("  1. Upload data to Google Drive (for Colab access)")
    print("  2. Run training on A100: python scripts/train_embeddings.py --cycled-training")


if __name__ == "__main__":
    main()
