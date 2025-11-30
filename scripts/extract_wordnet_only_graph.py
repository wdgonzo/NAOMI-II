"""
WordNet-Only Knowledge Graph Builder

Builds a knowledge graph from PURE WordNet relations (no Wikipedia parsing).
This is Phase 1 of the 2-step bootstrap strategy to discover transparent dimensions.

Strategy:
1. Extract ALL words from WordNet vocabulary
2. Create sense-tagged vocabulary (word_wn.XX_pos)
3. Build graph from ONLY WordNet semantic relations:
   - Antonyms (524 pairs) ← CRITICAL for polarity discovery
   - Synonyms (98K+ relations) ← Clustering
   - Hypernyms (44K+ relations) ← Hierarchies
   - All other WordNet relations

Why this works:
- Clean semantic structure (no parse noise)
- Strong polarity signal from antonyms
- No chicken-and-egg dependency on embeddings
- Discovers 10-20 interpretable semantic axes

Output:
- Sense-tagged vocabulary
- Pure WordNet knowledge graph
- Training data for Phase 1 bootstrap
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from nltk.corpus import wordnet as wn


def extract_wordnet_vocabulary(min_senses: int = 1,
                                max_words: int = None) -> Set[str]:
    """
    Extract ALL sense-tagged words from WordNet.

    Args:
        min_senses: Minimum number of senses per word (default: 1)
        max_words: Maximum vocabulary size (None = unlimited)

    Returns:
        Set of sense-tagged words (e.g., 'bank_wn.01_n', 'run_wn.02_v')
    """
    print("  Extracting WordNet vocabulary...")

    sense_tags = set()
    pos_map = {'n': 'n', 'v': 'v', 'a': 'a', 'r': 'r', 's': 'a'}  # Map satellite to adjective

    # Iterate over all synsets
    for synset in tqdm(list(wn.all_synsets()), desc="Processing synsets"):
        # Get word from synset name (e.g., 'bank.n.01' -> 'bank')
        word = synset.name().split('.')[0]
        pos = synset.pos()
        pos_tag = pos_map.get(pos, 'n')

        # Get all lemmas (synonyms) in this synset
        for lemma in synset.lemmas():
            lemma_word = lemma.name().replace('_', ' ').lower()

            # Find sense index for this lemma
            lemma_synsets = wn.synsets(lemma_word, pos=pos)
            try:
                sense_idx = lemma_synsets.index(synset)
            except ValueError:
                sense_idx = 0

            # Create sense tag
            sense_tag = f"{lemma_word}_wn.{sense_idx:02d}_{pos_tag}"
            sense_tags.add(sense_tag)

            if max_words and len(sense_tags) >= max_words:
                break

        if max_words and len(sense_tags) >= max_words:
            break

    print(f"    Extracted {len(sense_tags)} sense-tagged words from WordNet")

    return sense_tags


def build_wordnet_only_graph(vocabulary: Set[str]) -> List[Tuple[str, str, str]]:
    """
    Build knowledge graph from PURE WordNet relations.

    Args:
        vocabulary: Set of sense-tagged words

    Returns:
        List of (subject, relation, object) triples
    """
    print("  Building WordNet-only knowledge graph...")

    triples = []
    added = {
        'synonyms': 0, 'hypernyms': 0, 'hyponyms': 0,
        'antonyms': 0,  # Critical for polarity!
        'meronyms': 0, 'holonyms': 0,
        'entailments': 0, 'causes': 0,
        'similar_tos': 0, 'also_sees': 0,
        'attributes': 0, 'verb_groups': 0,
        'pertainyms': 0, 'derivationally_related': 0
    }

    pos_map = {'n': 'n', 'v': 'v', 'a': 'a', 'r': 'r', 's': 'a'}

    for sense_tag in tqdm(list(vocabulary), desc="Adding relations"):
        # Parse sense tag: word_wn.XX_pos
        if '_wn.' not in sense_tag:
            continue

        parts = sense_tag.rsplit('_', 1)  # ['word_wn.XX', 'pos']
        if len(parts) != 2:
            continue

        word_sense = parts[0]  # 'word_wn.XX'
        pos_tag = parts[1]  # 'n', 'v', 'a', 'r'
        word_parts = word_sense.split('_wn.')  # ['word', 'XX']
        if len(word_parts) != 2:
            continue

        word = word_parts[0]
        try:
            sense_idx = int(word_parts[1])
        except ValueError:
            continue

        # Get synset
        synsets = wn.synsets(word, pos=pos_tag)
        if sense_idx >= len(synsets):
            continue
        synset = synsets[sense_idx]

        # 1. SYNONYMS (other lemmas in same synset)
        for lemma in synset.lemmas():
            lemma_word = lemma.name().replace('_', ' ').lower()
            lemma_synsets = wn.synsets(lemma_word, pos=pos_tag)
            try:
                lemma_sense_idx = lemma_synsets.index(synset)
            except ValueError:
                lemma_sense_idx = 0
            lemma_tag = f"{lemma_word}_wn.{lemma_sense_idx:02d}_{pos_tag}"

            if lemma_tag != sense_tag and lemma_tag in vocabulary:
                triples.append((sense_tag, 'SYNONYM', lemma_tag))
                added['synonyms'] += 1

        # 2. HYPERNYMS/HYPONYMS (is-a hierarchy)
        for hypernym in synset.hypernyms():
            hypernym_word = hypernym.name().split('.')[0]
            hypernym_pos = hypernym.pos()
            hypernym_synsets = wn.synsets(hypernym_word, pos=hypernym_pos)
            try:
                hypernym_sense_idx = hypernym_synsets.index(hypernym)
            except ValueError:
                hypernym_sense_idx = 0
            hypernym_tag = f"{hypernym_word}_wn.{hypernym_sense_idx:02d}_{pos_map.get(hypernym_pos, 'n')}"

            if hypernym_tag in vocabulary:
                triples.append((sense_tag, 'HYPERNYM', hypernym_tag))
                triples.append((hypernym_tag, 'HYPONYM', sense_tag))
                added['hypernyms'] += 1
                added['hyponyms'] += 1

        # 3. ANTONYMS (opposite meaning) - CRITICAL FOR POLARITY!
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower():
                for antonym_lemma in lemma.antonyms():
                    antonym_word = antonym_lemma.name().replace('_', ' ').lower()
                    antonym_synset = antonym_lemma.synset()
                    antonym_synsets = wn.synsets(antonym_word)
                    try:
                        antonym_sense_idx = antonym_synsets.index(antonym_synset)
                    except ValueError:
                        antonym_sense_idx = 0
                    antonym_pos = antonym_synset.pos()
                    antonym_tag = f"{antonym_word}_wn.{antonym_sense_idx:02d}_{pos_map.get(antonym_pos, 'n')}"

                    if antonym_tag in vocabulary:
                        triples.append((sense_tag, 'ANTONYM', antonym_tag))
                        added['antonyms'] += 1

        # 4. MERONYMS (part-whole)
        for meronym in synset.part_meronyms() + synset.member_meronyms() + synset.substance_meronyms():
            meronym_word = meronym.name().split('.')[0]
            meronym_pos = meronym.pos()
            meronym_synsets = wn.synsets(meronym_word, pos=meronym_pos)
            try:
                meronym_sense_idx = meronym_synsets.index(meronym)
            except ValueError:
                meronym_sense_idx = 0
            meronym_tag = f"{meronym_word}_wn.{meronym_sense_idx:02d}_{pos_map.get(meronym_pos, 'n')}"

            if meronym_tag in vocabulary:
                triples.append((sense_tag, 'MERONYM', meronym_tag))
                added['meronyms'] += 1

        # 5. HOLONYMS (whole-part)
        for holonym in synset.part_holonyms() + synset.member_holonyms() + synset.substance_holonyms():
            holonym_word = holonym.name().split('.')[0]
            holonym_pos = holonym.pos()
            holonym_synsets = wn.synsets(holonym_word, pos=holonym_pos)
            try:
                holonym_sense_idx = holonym_synsets.index(holonym)
            except ValueError:
                holonym_sense_idx = 0
            holonym_tag = f"{holonym_word}_wn.{holonym_sense_idx:02d}_{pos_map.get(holonym_pos, 'n')}"

            if holonym_tag in vocabulary:
                triples.append((sense_tag, 'HOLONYM', holonym_tag))
                added['holonyms'] += 1

        # 6. ENTAILMENTS (verb implies another verb)
        for entailment in synset.entailments():
            entailment_word = entailment.name().split('.')[0]
            entailment_synsets = wn.synsets(entailment_word, pos='v')
            try:
                entailment_sense_idx = entailment_synsets.index(entailment)
            except ValueError:
                entailment_sense_idx = 0
            entailment_tag = f"{entailment_word}_wn.{entailment_sense_idx:02d}_v"

            if entailment_tag in vocabulary:
                triples.append((sense_tag, 'ENTAILMENT', entailment_tag))
                added['entailments'] += 1

        # 7. CAUSES (verb causes another)
        for cause in synset.causes():
            cause_word = cause.name().split('.')[0]
            cause_synsets = wn.synsets(cause_word, pos='v')
            try:
                cause_sense_idx = cause_synsets.index(cause)
            except ValueError:
                cause_sense_idx = 0
            cause_tag = f"{cause_word}_wn.{cause_sense_idx:02d}_v"

            if cause_tag in vocabulary:
                triples.append((sense_tag, 'CAUSE', cause_tag))
                added['causes'] += 1

        # 8. SIMILAR_TO (similar adjectives)
        for similar in synset.similar_tos():
            similar_word = similar.name().split('.')[0]
            similar_pos = similar.pos()
            similar_synsets = wn.synsets(similar_word, pos=similar_pos)
            try:
                similar_sense_idx = similar_synsets.index(similar)
            except ValueError:
                similar_sense_idx = 0
            similar_tag = f"{similar_word}_wn.{similar_sense_idx:02d}_{pos_map.get(similar_pos, 'a')}"

            if similar_tag in vocabulary:
                triples.append((sense_tag, 'SIMILAR_TO', similar_tag))
                added['similar_tos'] += 1

        # 9. ALSO_SEE (related concepts)
        for also_see in synset.also_sees():
            also_see_word = also_see.name().split('.')[0]
            also_see_pos = also_see.pos()
            also_see_synsets = wn.synsets(also_see_word, pos=also_see_pos)
            try:
                also_see_sense_idx = also_see_synsets.index(also_see)
            except ValueError:
                also_see_sense_idx = 0
            also_see_tag = f"{also_see_word}_wn.{also_see_sense_idx:02d}_{pos_map.get(also_see_pos, 'n')}"

            if also_see_tag in vocabulary:
                triples.append((sense_tag, 'ALSO_SEE', also_see_tag))
                added['also_sees'] += 1

        # 10. ATTRIBUTES (adjective → noun)
        for attribute in synset.attributes():
            attribute_word = attribute.name().split('.')[0]
            attribute_synsets = wn.synsets(attribute_word, pos='n')
            try:
                attribute_sense_idx = attribute_synsets.index(attribute)
            except ValueError:
                attribute_sense_idx = 0
            attribute_tag = f"{attribute_word}_wn.{attribute_sense_idx:02d}_n"

            if attribute_tag in vocabulary:
                triples.append((sense_tag, 'ATTRIBUTE', attribute_tag))
                added['attributes'] += 1

        # 11. VERB_GROUPS
        for verb_group in synset.verb_groups():
            vg_word = verb_group.name().split('.')[0]
            vg_synsets = wn.synsets(vg_word, pos='v')
            try:
                vg_sense_idx = vg_synsets.index(verb_group)
            except ValueError:
                vg_sense_idx = 0
            vg_tag = f"{vg_word}_wn.{vg_sense_idx:02d}_v"

            if vg_tag in vocabulary:
                triples.append((sense_tag, 'VERB_GROUP', vg_tag))
                added['verb_groups'] += 1

        # 12. PERTAINYMS & DERIVATIONALLY_RELATED (lemma-level)
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower():
                # Pertainyms
                for pertainym in lemma.pertainyms():
                    pertainym_word = pertainym.name().replace('_', ' ').lower()
                    pertainym_synset = pertainym.synset()
                    pertainym_pos = pertainym_synset.pos()
                    pertainym_synsets = wn.synsets(pertainym_word, pos=pertainym_pos)
                    try:
                        pertainym_sense_idx = pertainym_synsets.index(pertainym_synset)
                    except ValueError:
                        pertainym_sense_idx = 0
                    pertainym_tag = f"{pertainym_word}_wn.{pertainym_sense_idx:02d}_{pos_map.get(pertainym_pos, 'n')}"

                    if pertainym_tag in vocabulary:
                        triples.append((sense_tag, 'PERTAINYM', pertainym_tag))
                        added['pertainyms'] += 1

                # Derivationally related
                for derived in lemma.derivationally_related_forms():
                    derived_word = derived.name().replace('_', ' ').lower()
                    derived_synset = derived.synset()
                    derived_pos = derived_synset.pos()
                    derived_synsets = wn.synsets(derived_word, pos=derived_pos)
                    try:
                        derived_sense_idx = derived_synsets.index(derived_synset)
                    except ValueError:
                        derived_sense_idx = 0
                    derived_tag = f"{derived_word}_wn.{derived_sense_idx:02d}_{pos_map.get(derived_pos, 'n')}"

                    if derived_tag in vocabulary:
                        triples.append((sense_tag, 'DERIVATIONALLY_RELATED', derived_tag))
                        added['derivationally_related'] += 1

    print(f"\n    WordNet Relations Added:")
    print(f"      Synonyms: {added['synonyms']}")
    print(f"      Hypernyms: {added['hypernyms']}")
    print(f"      Hyponyms: {added['hyponyms']}")
    print(f"      Antonyms: {added['antonyms']} ← KEY FOR POLARITY!")
    print(f"      Meronyms: {added['meronyms']}")
    print(f"      Holonyms: {added['holonyms']}")
    print(f"      Entailments: {added['entailments']}")
    print(f"      Causes: {added['causes']}")
    print(f"      Similar-tos: {added['similar_tos']}")
    print(f"      Also-sees: {added['also_sees']}")
    print(f"      Attributes: {added['attributes']}")
    print(f"      Verb-groups: {added['verb_groups']}")
    print(f"      Pertainyms: {added['pertainyms']}")
    print(f"      Derivationally-related: {added['derivationally_related']}")

    total_relations = sum(added.values())
    print(f"      TOTAL: {total_relations} relations")
    print(f"    Total triples: {len(triples)}")

    return triples


def create_vocabulary_mappings(vocabulary: Set[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Create word_to_id and id_to_word mappings."""
    sorted_vocab = sorted(vocabulary)
    word_to_id = {word: idx for idx, word in enumerate(sorted_vocab)}
    id_to_word = {str(idx): word for idx, word in enumerate(sorted_vocab)}
    return word_to_id, id_to_word


def create_training_examples(triples: List[Tuple[str, str, str]],
                              word_to_id: Dict[str, int]) -> List[Tuple[int, str, int]]:
    """Convert triples to training examples with word IDs."""
    print("  Creating training examples...")

    examples = []
    for subj, rel, obj in triples:
        if subj in word_to_id and obj in word_to_id:
            examples.append((word_to_id[subj], rel, word_to_id[obj]))

    print(f"    Created {len(examples)} training examples")

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Build WordNet-only knowledge graph for Phase 1 bootstrap'
    )
    parser.add_argument('--output-dir', type=str,
                       default='data/wordnet_only_graph',
                       help='Output directory for knowledge graph')
    parser.add_argument('--max-words', type=int, default=None,
                       help='Maximum vocabulary size (None = unlimited)')
    parser.add_argument('--min-senses', type=int, default=1,
                       help='Minimum senses per word (default: 1)')

    args = parser.parse_args()

    print("="*70)
    print("WORDNET-ONLY KNOWLEDGE GRAPH BUILDER")
    print("Phase 1: Bootstrap Transparent Dimensions")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract vocabulary
    print("[1/4] Extracting WordNet vocabulary...")
    vocabulary = extract_wordnet_vocabulary(
        min_senses=args.min_senses,
        max_words=args.max_words
    )
    print()

    # Step 2: Build graph
    print("[2/4] Building WordNet-only knowledge graph...")
    triples = build_wordnet_only_graph(vocabulary)
    print()

    # Step 3: Create vocabulary mappings
    print("[3/4] Creating vocabulary mappings...")
    word_to_id, id_to_word = create_vocabulary_mappings(vocabulary)
    print(f"  Vocabulary size: {len(word_to_id)} sense-tagged words")
    print()

    # Step 4: Create training examples
    print("[4/4] Creating training data...")
    training_examples = create_training_examples(triples, word_to_id)
    print()

    # Save outputs
    print("Saving knowledge graph...")

    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump({
            'word_to_id': word_to_id,
            'id_to_word': id_to_word
        }, f, indent=2)
    print(f"  Saved vocabulary: {vocab_path}")

    # Save triples
    triples_path = output_dir / "triples.pkl"
    with open(triples_path, 'wb') as f:
        pickle.dump(triples, f)
    print(f"  Saved triples: {triples_path}")

    # Save training examples
    training_path = output_dir / "training_examples.pkl"
    with open(training_path, 'wb') as f:
        pickle.dump(training_examples, f)
    print(f"  Saved training examples: {training_path}")

    # Save statistics
    stats = {
        'vocabulary_size': len(word_to_id),
        'num_triples': len(triples),
        'num_training_examples': len(training_examples),
        'source': 'WordNet only (no Wikipedia parsing)'
    }

    stats_path = output_dir / "graph_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics: {stats_path}")

    print()
    print("="*70)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*70)
    print(f"Vocabulary size: {stats['vocabulary_size']} sense-tagged words")
    print(f"Total triples: {stats['num_triples']}")
    print(f"Training examples: {stats['num_training_examples']}")
    print(f"Source: {stats['source']}")
    print("="*70)
    print()
    print(f"Knowledge graph saved to: {output_dir}")
    print()
    print("Next step: Train Phase 1 bootstrap embeddings")
    print("  python scripts/train_wordnet_bootstrap.py \\")
    print(f"    --graph-dir {output_dir}")


if __name__ == "__main__":
    main()
