"""
Generate Training Data from Full WordNet

Converts extracted WordNet semantic relations into training edges
with distance constraints for embedding training.

Usage:
    python scripts/generate_wordnet_training_data.py --input data/full_wordnet --output data/wordnet_training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict


def load_wordnet_data(input_dir: Path):
    """Load extracted WordNet data."""
    print("Loading extracted WordNet data...")

    with open(input_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    with open(input_dir / "semantic_relations.json", 'r', encoding='utf-8') as f:
        relations = json.load(f)

    with open(input_dir / "antonym_pairs.json", 'r', encoding='utf-8') as f:
        antonym_pairs = json.load(f)

    print(f"  Loaded {len(vocab_data['sense_tagged'])} words")
    print(f"  Loaded {sum(len(pairs) for pairs in relations.values())} semantic relations")
    print(f"  Loaded {len(antonym_pairs)} antonym pairs")

    return vocab_data, relations, antonym_pairs


def build_vocabulary_mapping(vocab_data: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word_to_id and id_to_word mappings."""
    print("\nBuilding vocabulary mapping...")

    # Use sense-tagged vocabulary
    words = vocab_data['sense_tagged']

    word_to_id = {word: idx for idx, word in enumerate(words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    print(f"  Vocabulary size: {len(word_to_id)}")

    return word_to_id, id_to_word


def create_relation_to_distance_map() -> Dict[str, float]:
    """
    Map semantic relation types to target distances.

    Closer distances mean more semantically similar.
    """
    return {
        'hypernym': 0.3,      # is-a (dog -> animal): fairly close
        'hyponym': 0.3,       # reverse is-a: fairly close
        'similar': 0.2,       # similar adjectives: very close
        'meronym': 0.5,       # part-of (wheel -> car): moderate distance
        'holonym': 0.5,       # has-part: moderate distance
        'antonym': 1.0,       # opposites: far apart
    }


def generate_training_edges(relations: Dict,
                            antonym_pairs: List[Dict],
                            word_to_id: Dict[str, int]) -> List[Tuple[int, str, int, float]]:
    """
    Generate training edges from semantic relations.

    Returns:
        List of tuples: (source_id, relation_type, target_id, confidence)
    """
    print("\nGenerating training edges...")

    edges = []
    relation_counts = defaultdict(int)
    skipped_counts = defaultdict(int)

    # Process semantic relations
    for relation_type, word_pairs in relations.items():
        for word1, word2 in word_pairs:
            # Find matching sense-tagged words
            # Try to match words in vocabulary (which are sense-tagged)
            word1_matches = [w for w in word_to_id.keys() if w.startswith(f"{word1}_")]
            word2_matches = [w for w in word_to_id.keys() if w.startswith(f"{word2}_")]

            # For each combination, create an edge
            for w1 in word1_matches:
                for w2 in word2_matches:
                    id1 = word_to_id[w1]
                    id2 = word_to_id[w2]
                    edges.append((id1, relation_type, id2, 1.0))
                    relation_counts[relation_type] += 1

            if not word1_matches or not word2_matches:
                skipped_counts[relation_type] += 1

    # Process antonym pairs
    for pair in antonym_pairs:
        word1 = pair['word1']
        word2 = pair['word2']
        synset1 = pair['synset1']
        synset2 = pair['synset2']

        # Build sense-tagged IDs
        # Format synset: word.pos.sense_num -> word_wn.sense_num_pos
        try:
            parts1 = synset1.split('.')
            parts2 = synset2.split('.')

            word1_tagged = f"{word1}_wn.{parts1[2]}_{parts1[1]}"
            word2_tagged = f"{word2}_wn.{parts2[2]}_{parts2[1]}"

            if word1_tagged in word_to_id and word2_tagged in word_to_id:
                id1 = word_to_id[word1_tagged]
                id2 = word_to_id[word2_tagged]
                edges.append((id1, 'antonym', id2, 1.0))
                relation_counts['antonym'] += 1
            else:
                skipped_counts['antonym'] += 1
        except:
            skipped_counts['antonym'] += 1

    print(f"\n  Generated edges by relation type:")
    for relation_type, count in sorted(relation_counts.items()):
        print(f"    {relation_type}: {count}")

    print(f"\n  Skipped (word not in vocabulary):")
    for relation_type, count in sorted(skipped_counts.items()):
        print(f"    {relation_type}: {count}")

    print(f"\n  Total edges: {len(edges)}")

    return edges


def save_training_data(output_dir: Path,
                       edges: List,
                       word_to_id: Dict[str, int],
                       id_to_word: Dict[int, str]):
    """Save training data to files."""
    print(f"\nSaving training data to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save edges
    with open(output_dir / "training_edges.pkl", 'wb') as f:
        pickle.dump(edges, f)
    print(f"  Saved {len(edges)} edges to training_edges.pkl")

    # Save vocabulary
    vocab_data = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'vocab_size': len(word_to_id)
    }
    with open(output_dir / "vocabulary.json", 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary ({len(word_to_id)} words) to vocabulary.json")


def main():
    parser = argparse.ArgumentParser(description='Generate training data from WordNet')
    parser.add_argument('--input', type=str, default='data/full_wordnet',
                       help='Input directory with extracted WordNet data')
    parser.add_argument('--output', type=str, default='data/wordnet_training',
                       help='Output directory for training data')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print("=" * 80)
    print("WORDNET TRAINING DATA GENERATION")
    print("=" * 80)
    print()

    # Load extracted WordNet data
    vocab_data, relations, antonym_pairs = load_wordnet_data(input_dir)

    # Build vocabulary mapping
    word_to_id, id_to_word = build_vocabulary_mapping(vocab_data)

    # Generate training edges
    edges = generate_training_edges(relations, antonym_pairs, word_to_id)

    # Save training data
    save_training_data(output_dir, edges, word_to_id, id_to_word)

    print()
    print("=" * 80)
    print("TRAINING DATA GENERATION COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Vocabulary: {len(word_to_id)} words")
    print(f"  Training edges: {len(edges)}")
    print()
    print("Next step:")
    print("  python scripts/train_embeddings.py \\")
    print("    --unsupervised \\")
    print("    --dynamic-dims \\")
    print("    --embedding-dim 128 \\")
    print("    --max-dims 512 \\")
    print("    --epochs 200")


if __name__ == "__main__":
    main()
