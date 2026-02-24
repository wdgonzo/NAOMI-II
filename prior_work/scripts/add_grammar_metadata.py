"""
Add Grammar Metadata to Vocabulary

This script takes an existing vocabulary (word_to_id mapping) and creates
a corresponding grammar_metadata.json file with categorical grammatical features.

Usage:
    python scripts/add_grammar_metadata.py --vocab checkpoints/vocabulary.json
    python scripts/add_grammar_metadata.py --vocab data/full_wordnet/vocabulary.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from src.utils.grammar_extraction import create_grammar_metadata_for_vocab


def main():
    parser = argparse.ArgumentParser(description='Add grammar metadata to vocabulary')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grammar_metadata.json (default: same dir as vocab)')

    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"ERROR: Vocabulary file not found: {vocab_path}")
        return

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    # Handle both formats: {'word_to_id': {...}} or just {...}
    if 'word_to_id' in vocab_data:
        word_to_id = vocab_data['word_to_id']
    else:
        word_to_id = vocab_data

    print(f"  Loaded {len(word_to_id)} words")

    # Extract grammar metadata
    print("\nExtracting grammar metadata...")
    grammar_metadata = create_grammar_metadata_for_vocab(word_to_id)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = vocab_path.parent / 'grammar_metadata.json'

    # Save grammar metadata
    print(f"\nSaving grammar metadata to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grammar_metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved metadata for {len(grammar_metadata)} words")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Vocabulary: {len(word_to_id)} words")
    print(f"Grammar metadata: {len(grammar_metadata)} entries")
    print(f"Output: {output_path}")
    print()
    print("Grammar metadata is now stored separately from semantic embeddings.")
    print("During parsing, both will be used:")
    print("  - Semantic vectors: similarity/distance computations")
    print("  - Grammar metadata: morphological/syntactic constraints")


if __name__ == "__main__":
    main()
