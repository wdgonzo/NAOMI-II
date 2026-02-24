"""
Sense-Tagged Knowledge Graph Builder

Builds a knowledge graph from parsed corpus with:
1. Sense-tagged vocabulary (word occurrences → WordNet senses)
2. Parse-based triples (subject-relation-object)
3. WordNet semantic relations (synonyms, hypernyms, etc.)

Output:
- Sense-tagged vocabulary (word_to_id, id_to_word)
- Knowledge graph triples with sense tags
- Training data for embedding model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

from src.embeddings.sense_mapper import SenseMapper
from src.graph.triple_extractor import extract_triples
from src.parser.enums import Tag


def build_sense_tagged_vocabulary(parsed_corpus: List[Dict],
                                   mapper: SenseMapper) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Build sense-tagged vocabulary from parsed corpus.

    Args:
        parsed_corpus: List of parsed sentence results
        mapper: SenseMapper instance

    Returns:
        (word_to_id, id_to_word) dictionaries
    """
    print("  Building sense-tagged vocabulary...")

    # Collect all sense-tagged words
    sense_tags = set()

    for result in parsed_corpus:
        if not result['success']:
            continue

        word_contexts = result['word_contexts']

        # word_contexts is a list of serialized WordContext dicts
        for ctx_dict in word_contexts:
            if isinstance(ctx_dict, dict):
                word = ctx_dict.get('word')
                pos_str = ctx_dict.get('pos')  # e.g., "Tag.DET"
                neighbors = ctx_dict.get('neighbors', [])

                if word and pos_str:
                    # Parse pos_tag from string (e.g., "Tag.DET" -> Tag.DET)
                    try:
                        # Extract the tag name (e.g., "DET" from "Tag.DET")
                        tag_name = pos_str.split('.')[-1] if '.' in pos_str else pos_str
                        pos_tag = Tag[tag_name] if hasattr(Tag, tag_name) else None
                    except (KeyError, AttributeError):
                        pos_tag = None

                    if pos_tag:
                        # Reconstruct WordContext object
                        from src.embeddings.sense_mapper import WordContext
                        context = WordContext(
                            sentence_id=result['sentence_id'],
                            word=word,
                            pos_tag=pos_tag,
                            syntactic_role=None,  # Not needed for WSD
                            neighbors=neighbors,
                            parse_tree=None  # Don't need full hypothesis
                        )

                        # Get sense for this word occurrence
                        sense_idx, _ = mapper.match_context_to_sense(word, context)
                        sense_tag = mapper.create_sense_tag(word, sense_idx, pos_str)
                        sense_tags.add(sense_tag)

    # Create vocabulary (sorted for consistency)
    sorted_tags = sorted(sense_tags)
    word_to_id = {tag: idx for idx, tag in enumerate(sorted_tags)}
    id_to_word = {str(idx): tag for idx, tag in enumerate(sorted_tags)}

    print(f"    Vocabulary size: {len(word_to_id)} sense-tagged words")

    return word_to_id, id_to_word


def extract_sense_tagged_triples(parsed_corpus: List[Dict],
                                  mapper: SenseMapper) -> List[Tuple[str, str, str]]:
    """
    Extract triples with sense-tagged words.

    Args:
        parsed_corpus: List of parsed sentence results
        mapper: SenseMapper instance

    Returns:
        List of (subject, relation, object) tuples with sense tags
    """
    print("  Extracting sense-tagged triples...")

    all_triples = []

    # Build word → sense mapping for each sentence
    for result in parsed_corpus:
        if not result['success']:
            continue

        # Build mapping: word_text → sense_tag for this sentence
        word_to_sense = {}
        for ctx_dict in result['word_contexts']:
            if isinstance(ctx_dict, dict):
                word = ctx_dict.get('word')
                pos_str = ctx_dict.get('pos')  # e.g., "Tag.DET"
                neighbors = ctx_dict.get('neighbors', [])

                if word and pos_str:
                    # Parse pos_tag from string
                    try:
                        tag_name = pos_str.split('.')[-1] if '.' in pos_str else pos_str
                        pos_tag = Tag[tag_name] if hasattr(Tag, tag_name) else None
                    except (KeyError, AttributeError):
                        pos_tag = None

                    if pos_tag:
                        # Reconstruct WordContext object
                        from src.embeddings.sense_mapper import WordContext
                        context = WordContext(
                            sentence_id=result['sentence_id'],
                            word=word,
                            pos_tag=pos_tag,
                            syntactic_role=None,
                            neighbors=neighbors,
                            parse_tree=None
                        )

                        sense_idx, _ = mapper.match_context_to_sense(word, context)
                        sense_tag = mapper.create_sense_tag(word, sense_idx, pos_str)
                        # Use word text as key (first occurrence wins)
                        if word.lower() not in word_to_sense:
                            word_to_sense[word.lower()] = sense_tag

        # Get raw triples from parse
        raw_triples = result['triples']

        # Convert to sense-tagged triples
        for triple in raw_triples:
            # Triple is serialized as dict
            if isinstance(triple, dict):
                subj = triple.get('subject', '')
                rel = triple.get('relation', '')
                obj = triple.get('object', '')

                if subj and rel and obj:
                    # Map to sense tags
                    subj_sense = word_to_sense.get(subj.lower(), subj)
                    obj_sense = word_to_sense.get(obj.lower(), obj)

                    # Append as tuple with relation string
                    all_triples.append((subj_sense, rel, obj_sense))

    print(f"    Extracted {len(all_triples)} sense-tagged triples")

    return all_triples


def add_wordnet_relations(triples: List[Tuple[str, str, str]],
                          vocabulary: Set[str],
                          mapper: SenseMapper) -> List[Tuple[str, str, str]]:
    """
    Add ALL WordNet semantic relations to knowledge graph.

    For each sense-tagged word in vocabulary, add:
    - Synonym relations (from synset lemmas)
    - Hypernym/Hyponym relations (is-a hierarchies)
    - Antonym relations (opposite meaning) ← CRITICAL FOR POLARITY!
    - Meronym/Holonym relations (part-whole relationships)
    - Attribute relations (adjective-noun attributes)
    - Entailment/Cause relations (verb implications)
    - Similarity relations (similar adjectives)
    - Derivational relations (same root words)
    - And more...

    Args:
        triples: Existing triples
        vocabulary: Set of sense-tagged words
        mapper: SenseMapper instance

    Returns:
        Extended triples list with ALL WordNet relations
    """
    print("  Adding WordNet semantic relations...")
    from nltk.corpus import wordnet as wn

    extended_triples = list(triples)
    added = {
        'synonyms': 0, 'hypernyms': 0, 'hyponyms': 0,
        'antonyms': 0,  # Critical for polarity!
        'meronyms': 0, 'holonyms': 0,
        'entailments': 0, 'causes': 0,
        'similar_tos': 0, 'also_sees': 0,
        'attributes': 0, 'verb_groups': 0,
        'pertainyms': 0, 'derivationally_related': 0
    }

    for sense_tag in vocabulary:
        # Parse sense tag to extract word and sense index
        # Format: word_wn.XX_pos or word_pos
        if '_wn.' not in sense_tag:
            # Not in WordNet, skip
            continue

        parts = sense_tag.rsplit('_', 1)  # Split from right: ['word_wn.XX', 'pos']
        if len(parts) != 2:
            continue

        word_sense = parts[0]  # 'word_wn.XX'
        word_parts = word_sense.split('_wn.')  # ['word', 'XX']
        if len(word_parts) != 2:
            continue

        word = word_parts[0]
        try:
            sense_idx = int(word_parts[1])
        except ValueError:
            continue

        # Get sense info
        sense_info = mapper.get_sense_info(word, sense_idx)
        if not sense_info['found']:
            continue

        # Get the actual synset for more detailed relations
        synsets = wn.synsets(word)
        if sense_idx >= len(synsets):
            continue
        synset = synsets[sense_idx]

        # 1. SYNONYMS (other lemmas in same synset)
        for lemma in sense_info['lemmas']:
            lemma_tag = f"{lemma}_wn.{sense_idx:02d}_{parts[1]}"
            if lemma_tag != sense_tag:  # Don't create self-loops
                extended_triples.append((sense_tag, 'SYNONYM', lemma_tag))
                added['synonyms'] += 1

        # 2. HYPERNYMS/HYPONYMS (is-a hierarchy)
        for hypernym_synset_id in sense_info['hypernyms']:
            hypernym_word = hypernym_synset_id.split('.')[0]
            hypernym_tag = f"{hypernym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'HYPERNYM', hypernym_tag))
            extended_triples.append((hypernym_tag, 'HYPONYM', sense_tag))
            added['hypernyms'] += 1
            added['hyponyms'] += 1

        # 3. ANTONYMS (opposite meaning) - CRITICAL FOR POLARITY!
        # Antonyms are lemma-level, not synset-level
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower():
                for antonym_lemma in lemma.antonyms():
                    antonym_word = antonym_lemma.name().replace('_', ' ').lower()
                    antonym_synset = antonym_lemma.synset()
                    # Try to find the sense index
                    antonym_synsets = wn.synsets(antonym_word)
                    try:
                        antonym_sense_idx = antonym_synsets.index(antonym_synset)
                    except ValueError:
                        antonym_sense_idx = 0
                    antonym_tag = f"{antonym_word}_wn.{antonym_sense_idx:02d}_{parts[1]}"
                    extended_triples.append((sense_tag, 'ANTONYM', antonym_tag))
                    added['antonyms'] += 1

        # 4. MERONYMS (has-part, has-member, has-substance)
        for meronym in synset.part_meronyms():
            meronym_word = meronym.name().split('.')[0]
            meronym_tag = f"{meronym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'PART_MERONYM', meronym_tag))
            added['meronyms'] += 1

        for meronym in synset.member_meronyms():
            meronym_word = meronym.name().split('.')[0]
            meronym_tag = f"{meronym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'MEMBER_MERONYM', meronym_tag))
            added['meronyms'] += 1

        for meronym in synset.substance_meronyms():
            meronym_word = meronym.name().split('.')[0]
            meronym_tag = f"{meronym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'SUBSTANCE_MERONYM', meronym_tag))
            added['meronyms'] += 1

        # 5. HOLONYMS (part-of, member-of, substance-of)
        for holonym in synset.part_holonyms():
            holonym_word = holonym.name().split('.')[0]
            holonym_tag = f"{holonym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'PART_HOLONYM', holonym_tag))
            added['holonyms'] += 1

        for holonym in synset.member_holonyms():
            holonym_word = holonym.name().split('.')[0]
            holonym_tag = f"{holonym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'MEMBER_HOLONYM', holonym_tag))
            added['holonyms'] += 1

        for holonym in synset.substance_holonyms():
            holonym_word = holonym.name().split('.')[0]
            holonym_tag = f"{holonym_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'SUBSTANCE_HOLONYM', holonym_tag))
            added['holonyms'] += 1

        # 6. ENTAILMENTS (verb implies another verb: snore → sleep)
        for entailment in synset.entailments():
            entailment_word = entailment.name().split('.')[0]
            entailment_tag = f"{entailment_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'ENTAILMENT', entailment_tag))
            added['entailments'] += 1

        # 7. CAUSES (verb causes another: kill → die)
        for cause in synset.causes():
            cause_word = cause.name().split('.')[0]
            cause_tag = f"{cause_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'CAUSE', cause_tag))
            added['causes'] += 1

        # 8. SIMILAR_TO (similar adjectives)
        for similar in synset.similar_tos():
            similar_word = similar.name().split('.')[0]
            similar_tag = f"{similar_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'SIMILAR_TO', similar_tag))
            added['similar_tos'] += 1

        # 9. ALSO_SEE (related concepts)
        for also_see in synset.also_sees():
            also_see_word = also_see.name().split('.')[0]
            also_see_tag = f"{also_see_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'ALSO_SEE', also_see_tag))
            added['also_sees'] += 1

        # 10. ATTRIBUTES (adjective relates to noun: heavy → weight)
        for attribute in synset.attributes():
            attribute_word = attribute.name().split('.')[0]
            # Attributes might have different POS
            attribute_pos = attribute.pos()
            pos_map = {'n': 'n', 'v': 'v', 'a': 'a', 'r': 'r', 's': 'a'}
            attribute_tag = f"{attribute_word}_wn.00_{pos_map.get(attribute_pos, 'n')}"
            extended_triples.append((sense_tag, 'ATTRIBUTE', attribute_tag))
            added['attributes'] += 1

        # 11. VERB_GROUPS (verbs in same group)
        for verb_group in synset.verb_groups():
            vg_word = verb_group.name().split('.')[0]
            vg_tag = f"{vg_word}_wn.00_{parts[1]}"
            extended_triples.append((sense_tag, 'VERB_GROUP', vg_tag))
            added['verb_groups'] += 1

        # 12. PERTAINYMS & DERIVATIONALLY_RELATED (lemma-level)
        for lemma in synset.lemmas():
            if lemma.name().lower() == word.lower():
                # Pertainyms (adjective → noun: facial → face)
                for pertainym in lemma.pertainyms():
                    pertainym_word = pertainym.name().replace('_', ' ').lower()
                    pertainym_synset = pertainym.synset()
                    pertainym_synsets = wn.synsets(pertainym_word)
                    try:
                        pertainym_sense_idx = pertainym_synsets.index(pertainym_synset)
                    except ValueError:
                        pertainym_sense_idx = 0
                    pertainym_pos = pertainym_synset.pos()
                    pos_map = {'n': 'n', 'v': 'v', 'a': 'a', 'r': 'r', 's': 'a'}
                    pertainym_tag = f"{pertainym_word}_wn.{pertainym_sense_idx:02d}_{pos_map.get(pertainym_pos, 'n')}"
                    extended_triples.append((sense_tag, 'PERTAINYM', pertainym_tag))
                    added['pertainyms'] += 1

                # Derivationally related (same root: destruction → destroy)
                for derived in lemma.derivationally_related_forms():
                    derived_word = derived.name().replace('_', ' ').lower()
                    derived_synset = derived.synset()
                    derived_synsets = wn.synsets(derived_word)
                    try:
                        derived_sense_idx = derived_synsets.index(derived_synset)
                    except ValueError:
                        derived_sense_idx = 0
                    derived_pos = derived_synset.pos()
                    pos_map = {'n': 'n', 'v': 'v', 'a': 'a', 'r': 'r', 's': 'a'}
                    derived_tag = f"{derived_word}_wn.{derived_sense_idx:02d}_{pos_map.get(derived_pos, 'n')}"
                    extended_triples.append((sense_tag, 'DERIVATIONALLY_RELATED', derived_tag))
                    added['derivationally_related'] += 1

    print(f"    Added {added['synonyms']} synonym relations")
    print(f"    Added {added['hypernyms']} hypernym relations")
    print(f"    Added {added['hyponyms']} hyponym relations")
    print(f"    Added {added['antonyms']} antonym relations <-- KEY FOR POLARITY!")
    print(f"    Added {added['meronyms']} meronym relations (part-whole)")
    print(f"    Added {added['holonyms']} holonym relations (whole-part)")
    print(f"    Added {added['entailments']} entailment relations")
    print(f"    Added {added['causes']} cause relations")
    print(f"    Added {added['similar_tos']} similar_to relations")
    print(f"    Added {added['also_sees']} also_see relations")
    print(f"    Added {added['attributes']} attribute relations")
    print(f"    Added {added['verb_groups']} verb_group relations")
    print(f"    Added {added['pertainyms']} pertainym relations")
    print(f"    Added {added['derivationally_related']} derivationally_related relations")

    total_wordnet = sum(added.values())
    print(f"    TOTAL WordNet relations added: {total_wordnet}")

    return extended_triples


def create_training_examples(triples: List[Tuple[str, str, str]],
                             word_to_id: Dict[str, int]) -> List[Tuple[int, str, int]]:
    """
    Convert triples to training examples with word IDs.

    Args:
        triples: Sense-tagged triples
        word_to_id: Vocabulary mapping

    Returns:
        List of (subject_id, relation, object_id) tuples
    """
    print("  Creating training examples...")

    examples = []

    for subj, rel, obj in triples:
        # Map to IDs (skip if not in vocabulary)
        if subj in word_to_id and obj in word_to_id:
            examples.append((word_to_id[subj], rel, word_to_id[obj]))

    print(f"    Created {len(examples)} training examples")

    return examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build sense-tagged knowledge graph')
    parser.add_argument('--corpus', type=str,
                       default='data/test_parse/parsed_corpus.pkl',
                       help='Path to parsed corpus')
    parser.add_argument('--output-dir', type=str,
                       default='data/sense_graph',
                       help='Output directory for knowledge graph')
    parser.add_argument('--add-wordnet', action='store_true',
                       help='Add WordNet semantic relations')

    args = parser.parse_args()

    print("="*70)
    print("SENSE-TAGGED KNOWLEDGE GRAPH BUILDER")
    print("="*70)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load parsed corpus
    print(f"[1/5] Loading parsed corpus...")
    with open(args.corpus, 'rb') as f:
        corpus = pickle.load(f)
    print(f"  Loaded {len(corpus)} parsed sentences")
    print()

    # Initialize sense mapper
    print("[2/5] Initializing sense mapper...")
    mapper = SenseMapper()
    print()

    # Build vocabulary
    print("[3/5] Building sense-tagged vocabulary...")
    word_to_id, id_to_word = build_sense_tagged_vocabulary(corpus, mapper)
    print()

    # Extract triples
    print("[4/5] Extracting sense-tagged triples...")
    triples = extract_sense_tagged_triples(corpus, mapper)

    # Add WordNet relations if requested
    if args.add_wordnet:
        triples = add_wordnet_relations(triples, set(word_to_id.keys()), mapper)

    print()

    # Create training examples
    print("[5/5] Creating training data...")
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
        'num_sentences': len(corpus)
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
    print(f"Source sentences: {stats['num_sentences']}")
    print("="*70)
    print()
    print(f"Knowledge graph saved to: {output_dir}")


if __name__ == "__main__":
    main()
