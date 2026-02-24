"""
Extract Full WordNet Data for Unsupervised Training

This script extracts ALL of WordNet's semantic knowledge:
- All 117,659 synsets
- All 148,730 unique words (lemmas)
- All 3,989 antonym pairs
- All semantic relationships (hypernyms, meronyms, etc.)

The output is used for large-scale unsupervised embedding training.

Usage:
    python scripts/extract_full_wordnet.py --output data/full_wordnet/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from nltk.corpus import wordnet as wn


def extract_all_synsets() -> Dict:
    """Extract all WordNet synsets with metadata."""
    print("Extracting all WordNet synsets...")

    synsets_data = {}
    pos_counts = defaultdict(int)

    for synset in wn.all_synsets():
        synset_id = synset.name()
        pos_counts[synset.pos()] += 1

        # Extract lemmas (word forms)
        lemmas = [lemma.name() for lemma in synset.lemmas()]

        # Extract definition and examples
        definition = synset.definition()
        examples = synset.examples()

        # Extract hypernyms (is-a relationships)
        hypernyms = [h.name() for h in synset.hypernyms()]

        # Extract hyponyms (reverse is-a)
        hyponyms = [h.name() for h in synset.hyponyms()]

        # Extract meronyms (part-of relationships)
        meronyms = [m.name() for m in synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms()]

        # Extract holonyms (has-part relationships)
        holonyms = [h.name() for h in synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms()]

        # Extract similar_tos (for adjectives)
        similar = [s.name() for s in synset.similar_tos()]

        synsets_data[synset_id] = {
            'lemmas': lemmas,
            'pos': synset.pos(),
            'definition': definition,
            'examples': examples,
            'hypernyms': hypernyms,
            'hyponyms': hyponyms,
            'meronyms': meronyms,
            'holonyms': holonyms,
            'similar': similar
        }

    print(f"  Extracted {len(synsets_data)} synsets")
    print(f"  POS distribution:")
    for pos, count in sorted(pos_counts.items()):
        pos_name = {'n': 'nouns', 'v': 'verbs', 'a': 'adjectives', 'r': 'adverbs', 's': 'adjective satellites'}
        print(f"    {pos_name.get(pos, pos)}: {count}")

    return synsets_data


def extract_all_antonym_pairs() -> List[Tuple[str, str, str, str]]:
    """Extract ALL antonym pairs from WordNet.

    Returns:
        List of tuples: (word1, word2, synset1_id, synset2_id)
    """
    print("\nExtracting all antonym pairs...")

    antonym_pairs = set()

    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            for antonym_lemma in lemma.antonyms():
                word1 = lemma.name().lower()
                word2 = antonym_lemma.name().lower()
                synset1 = synset.name()
                synset2 = antonym_lemma.synset().name()

                # Store as sorted tuple to avoid duplicates
                pair = tuple(sorted([
                    (word1, synset1),
                    (word2, synset2)
                ]))
                antonym_pairs.add(pair)

    # Convert to list format
    antonym_list = [
        (p[0][0], p[1][0], p[0][1], p[1][1])
        for p in antonym_pairs
    ]

    print(f"  Found {len(antonym_list)} unique antonym pairs")

    return antonym_list


def build_vocabulary(synsets_data: Dict) -> Dict[str, Set[str]]:
    """Build vocabulary with all unique words, grouped by POS."""
    print("\nBuilding vocabulary...")

    vocab_by_pos = defaultdict(set)

    for synset_id, synset_info in synsets_data.items():
        pos = synset_info['pos']
        for lemma in synset_info['lemmas']:
            word = lemma.lower().replace('_', ' ')  # Handle multi-word expressions
            vocab_by_pos[pos].add(word)

    # Also create sense-tagged vocabulary
    sense_tagged_vocab = set()
    for synset_id, synset_info in synsets_data.items():
        for lemma in synset_info['lemmas']:
            # Format: word_wn.XX_pos
            sense_tagged = f"{lemma.lower()}_wn.{synset_id.split('.')[1]}_{synset_info['pos']}"
            sense_tagged_vocab.add(sense_tagged)

    total_words = sum(len(words) for words in vocab_by_pos.values())
    print(f"  Total unique base words: {total_words}")
    print(f"  Total sense-tagged words: {len(sense_tagged_vocab)}")

    for pos, words in sorted(vocab_by_pos.items()):
        pos_name = {'n': 'nouns', 'v': 'verbs', 'a': 'adjectives', 'r': 'adverbs', 's': 'adj_satellites'}
        print(f"    {pos_name.get(pos, pos)}: {len(words)}")

    return {
        'by_pos': {pos: list(words) for pos, words in vocab_by_pos.items()},
        'sense_tagged': list(sense_tagged_vocab)
    }


def extract_semantic_relations(synsets_data: Dict) -> Dict:
    """Extract all semantic relationships for training constraints."""
    print("\nExtracting semantic relationships...")

    relations = {
        'hypernym': [],      # is-a (dog is-a animal)
        'hyponym': [],       # reverse is-a
        'meronym': [],       # part-of (wheel part-of car)
        'holonym': [],       # has-part (car has-part wheel)
        'similar': [],       # similar-to (for adjectives)
    }

    for synset_id, synset_info in synsets_data.items():
        # Get first lemma as representative
        if not synset_info['lemmas']:
            continue
        word = synset_info['lemmas'][0].lower()

        # Hypernyms
        for hypernym_id in synset_info['hypernyms']:
            if hypernym_id in synsets_data:
                hypernym_word = synsets_data[hypernym_id]['lemmas'][0].lower()
                relations['hypernym'].append((word, hypernym_word))

        # Hyponyms
        for hyponym_id in synset_info['hyponyms']:
            if hyponym_id in synsets_data:
                hyponym_word = synsets_data[hyponym_id]['lemmas'][0].lower()
                relations['hyponym'].append((word, hyponym_word))

        # Meronyms
        for meronym_id in synset_info['meronyms']:
            if meronym_id in synsets_data:
                meronym_word = synsets_data[meronym_id]['lemmas'][0].lower()
                relations['meronym'].append((word, meronym_word))

        # Holonyms
        for holonym_id in synset_info['holonyms']:
            if holonym_id in synsets_data:
                holonym_word = synsets_data[holonym_id]['lemmas'][0].lower()
                relations['holonym'].append((word, holonym_word))

        # Similar
        for similar_id in synset_info['similar']:
            if similar_id in synsets_data:
                similar_word = synsets_data[similar_id]['lemmas'][0].lower()
                relations['similar'].append((word, similar_word))

    print(f"  Relationship counts:")
    for rel_type, pairs in relations.items():
        print(f"    {rel_type}: {len(pairs)}")

    return relations


def save_extraction_results(output_dir: Path,
                            synsets_data: Dict,
                            antonym_pairs: List,
                            vocabulary: Dict,
                            relations: Dict):
    """Save all extracted data to JSON files."""
    print(f"\nSaving extraction results to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save synsets
    synsets_path = output_dir / "synsets.json"
    with open(synsets_path, 'w', encoding='utf-8') as f:
        json.dump(synsets_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(synsets_data)} synsets to {synsets_path}")

    # Save antonym pairs
    antonyms_path = output_dir / "antonym_pairs.json"
    antonym_dict = [
        {
            'word1': pair[0],
            'word2': pair[1],
            'synset1': pair[2],
            'synset2': pair[3]
        }
        for pair in antonym_pairs
    ]
    with open(antonyms_path, 'w', encoding='utf-8') as f:
        json.dump(antonym_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(antonym_pairs)} antonym pairs to {antonyms_path}")

    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary to {vocab_path}")

    # Save semantic relations
    relations_path = output_dir / "semantic_relations.json"
    # Convert tuples to lists for JSON serialization
    relations_json = {
        rel_type: [[w1, w2] for w1, w2 in pairs]
        for rel_type, pairs in relations.items()
    }
    with open(relations_path, 'w', encoding='utf-8') as f:
        json.dump(relations_json, f, indent=2, ensure_ascii=False)
    print(f"  Saved semantic relations to {relations_path}")

    # Save summary statistics
    stats = {
        'total_synsets': len(synsets_data),
        'total_antonym_pairs': len(antonym_pairs),
        'total_base_words': sum(len(words) for words in vocabulary['by_pos'].values()),
        'total_sense_tagged_words': len(vocabulary['sense_tagged']),
        'pos_distribution': {pos: len(words) for pos, words in vocabulary['by_pos'].items()},
        'relation_counts': {rel: len(pairs) for rel, pairs in relations.items()}
    }

    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract full WordNet data')
    parser.add_argument('--output', type=str, default='data/full_wordnet',
                       help='Output directory for extracted data')

    args = parser.parse_args()
    output_dir = Path(args.output)

    print("=" * 80)
    print("FULL WORDNET EXTRACTION")
    print("=" * 80)
    print()

    # Extract all synsets
    synsets_data = extract_all_synsets()

    # Extract all antonym pairs
    antonym_pairs = extract_all_antonym_pairs()

    # Build vocabulary
    vocabulary = build_vocabulary(synsets_data)

    # Extract semantic relations
    relations = extract_semantic_relations(synsets_data)

    # Save everything
    save_extraction_results(output_dir, synsets_data, antonym_pairs, vocabulary, relations)

    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Synsets: {len(synsets_data)}")
    print(f"  Antonym pairs: {len(antonym_pairs)}")
    print(f"  Base words: {sum(len(words) for words in vocabulary['by_pos'].values())}")
    print(f"  Sense-tagged words: {len(vocabulary['sense_tagged'])}")
    print()
    print("Next steps:")
    print("  1. Generate training data from semantic relations")
    print("  2. Train embeddings with --unsupervised flag")
    print("  3. Run dimension discovery analysis")


if __name__ == "__main__":
    main()
