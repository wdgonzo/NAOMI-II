"""
Test Sense Mapping on Parsed Corpus

Loads parsed corpus and applies sense mapping to see how well
WordNet disambiguation works on real sentences.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import json
from collections import Counter
from typing import List, Dict

from src.embeddings.sense_mapper import SenseMapper


def analyze_sense_mapping(parsed_corpus_path: str, num_sentences: int = 10):
    """
    Analyze sense mapping quality on parsed corpus.

    Args:
        parsed_corpus_path: Path to parsed_corpus.pkl
        num_sentences: Number of sentences to analyze in detail
    """
    print("="*70)
    print("SENSE MAPPING ANALYSIS")
    print("="*70)
    print()

    # Load parsed corpus
    print(f"[1/4] Loading parsed corpus from {parsed_corpus_path}...")
    with open(parsed_corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    print(f"  Loaded {len(corpus)} parsed sentences")
    print()

    # Initialize sense mapper
    print("[2/4] Initializing sense mapper...")
    mapper = SenseMapper()
    print()

    # Process sentences
    print(f"[3/4] Mapping senses for all sentences...")

    stats = {
        'total_words': 0,
        'words_in_wordnet': 0,
        'words_not_in_wordnet': 0,
        'polysemous_words': 0,  # Words with multiple senses
        'monosemous_words': 0,  # Words with single sense
        'sense_tags': Counter(),
        'ambiguous_words': []  # Words that needed disambiguation
    }

    all_sense_assignments = {}

    for result in corpus:
        if not result['success']:
            continue

        word_contexts = result['word_contexts']

        for word_pos, (word, context) in enumerate(word_contexts):
            stats['total_words'] += 1

            # Get senses for word
            senses = mapper.get_senses(word, context.pos_tag)

            if not senses:
                stats['words_not_in_wordnet'] += 1
                sense_tag = mapper.create_sense_tag(word, -1, context.pos_tag)
            else:
                stats['words_in_wordnet'] += 1

                if len(senses) == 1:
                    stats['monosemous_words'] += 1
                    sense_tag = mapper.create_sense_tag(word, 0, context.pos_tag)
                else:
                    stats['polysemous_words'] += 1

                    # Disambiguate
                    sense_idx, confidence = mapper.match_context_to_sense(word, context)
                    sense_tag = mapper.create_sense_tag(word, sense_idx, context.pos_tag)

                    stats['ambiguous_words'].append({
                        'word': word,
                        'sentence_id': context.sentence_id,
                        'num_senses': len(senses),
                        'chosen_sense': sense_idx,
                        'confidence': confidence,
                        'neighbors': context.neighbors[:5]
                    })

            stats['sense_tags'][sense_tag] += 1

            key = (result['sentence_id'], word_pos)
            all_sense_assignments[key] = sense_tag

    print(f"  Processed {stats['total_words']} word occurrences")
    print()

    # Print statistics
    print("[4/4] Sense Mapping Statistics")
    print("="*70)
    print(f"Total word occurrences: {stats['total_words']}")
    print(f"Words in WordNet: {stats['words_in_wordnet']} ({stats['words_in_wordnet']/stats['total_words']*100:.1f}%)")
    print(f"Words NOT in WordNet: {stats['words_not_in_wordnet']} ({stats['words_not_in_wordnet']/stats['total_words']*100:.1f}%)")
    print()
    print(f"Monosemous (single sense): {stats['monosemous_words']}")
    print(f"Polysemous (multiple senses): {stats['polysemous_words']}")
    print()
    print(f"Unique sense-tagged vocabulary size: {len(stats['sense_tags'])}")
    print(f"Words requiring disambiguation: {len(stats['ambiguous_words'])}")
    print()

    # Show most common sense tags
    print("Most common sense tags:")
    for tag, count in stats['sense_tags'].most_common(20):
        print(f"  {tag:30s}: {count:4d} occurrences")
    print()

    # Show detailed disambiguation examples
    print(f"\nDetailed Disambiguation Examples (first {num_sentences}):")
    print("="*70)

    shown = 0
    for amb in stats['ambiguous_words']:
        if shown >= num_sentences:
            break

        word = amb['word']
        sense_idx = amb['chosen_sense']

        # Get sense info
        sense_info = mapper.get_sense_info(word, sense_idx)

        if sense_info['found']:
            print(f"\nWord: '{word}' (sentence {amb['sentence_id']})")
            print(f"  Senses available: {amb['num_senses']}")
            print(f"  Chosen sense: {sense_idx} (confidence: {amb['confidence']:.3f})")
            print(f"  Definition: {sense_info['definition']}")
            print(f"  Context neighbors: {', '.join(amb['neighbors'])}")
            shown += 1

    print()
    print("="*70)
    print("Analysis complete!")

    return stats, all_sense_assignments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test sense mapping on parsed corpus')
    parser.add_argument('--corpus', type=str,
                       default='data/test_parse/parsed_corpus.pkl',
                       help='Path to parsed corpus')
    parser.add_argument('--num-examples', type=int, default=10,
                       help='Number of detailed examples to show')

    args = parser.parse_args()

    stats, assignments = analyze_sense_mapping(args.corpus, args.num_examples)
