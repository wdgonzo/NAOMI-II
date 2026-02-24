"""
Grammar Metadata Extraction

Extracts categorical grammatical features from WordNet and other sources.
These features are stored as metadata alongside semantic embeddings, not learned.

Grammatical features are discrete categories (enums), not continuous values:
- tense: past/present/future
- number: singular/plural
- person: 1st/2nd/3rd
- etc.

Usage:
    from src.utils.grammar_extraction import extract_grammar_metadata

    metadata = extract_grammar_metadata("run_wn.01_v")
    # Returns: {'pos': 'verb', 'tense': None, 'number': None, ...}
"""

from typing import Dict, Optional
from nltk.corpus import wordnet as wn


# The 51 grammatical dimensions from anchors.py
GRAMMAR_DIMENSIONS = {
    # Core grammatical features (15)
    'tense': ['past', 'present', 'future', 'none'],
    'aspect': ['simple', 'progressive', 'perfect', 'perfect_progressive', 'none'],
    'mood': ['indicative', 'subjunctive', 'imperative', 'conditional', 'none'],
    'voice': ['active', 'passive', 'middle', 'none'],
    'person': ['1st', '2nd', '3rd', 'none'],
    'number': ['singular', 'plural', 'dual', 'none'],
    'gender': ['masculine', 'feminine', 'neuter', 'none'],
    'case': ['nominative', 'accusative', 'genitive', 'dative', 'none'],
    'definiteness': ['definite', 'indefinite', 'none'],
    'polarity': ['positive', 'negative'],
    'animacy': ['animate', 'inanimate', 'none'],
    'countability': ['countable', 'uncountable', 'none'],
    'degree': ['positive', 'comparative', 'superlative', 'none'],
    'transitivity': ['transitive', 'intransitive', 'ditransitive', 'none'],
    'evidentiality': ['direct', 'indirect', 'none'],

    # Semantic roles (10) - these are contextual, set to 'none' for base words
    'role_subject': ['none'],
    'role_object': ['none'],
    'role_instrument': ['none'],
    'role_location': ['none'],
    'role_time': ['none'],
    'role_manner': ['none'],
    'role_purpose': ['none'],
    'role_source': ['none'],
    'role_goal': ['none'],
    'role_experiencer': ['none'],

    # Nominal features (6)
    'determinatory': ['determined', 'undetermined', 'none'],
    'personal': ['personal', 'impersonal', 'none'],
    'living': ['living', 'nonliving', 'none'],
    'permanence': ['permanent', 'temporary', 'none'],
    'embodiment': ['physical', 'abstract', 'none'],
    'magnitude': ['small', 'medium', 'large', 'none'],

    # Logical operators (9) - not applicable to base words
    'logical_and': ['none'],
    'logical_or': ['none'],
    'logical_xor': ['none'],
    'logical_nand': ['none'],
    'logical_if': ['none'],
    'logical_xif': ['none'],
    'logical_not': ['none'],
    'logical_nor': ['none'],
    'logical_xnor': ['none'],

    # Scopes (11) - contextual, set to 'none' for base words
    'scope_temporal': ['none'],
    'scope_frequency': ['none'],
    'scope_location': ['none'],
    'scope_manner': ['none'],
    'scope_extent': ['none'],
    'scope_reason': ['none'],
    'scope_attitude': ['none'],
    'scope_relative': ['none'],
    'scope_direction': ['none'],
    'scope_spatial_extent': ['none'],
    'scope_beneficiary': ['none'],
}


def extract_grammar_metadata(word_id: str) -> Dict[str, str]:
    """
    Extract grammatical metadata for a sense-tagged word.

    Args:
        word_id: Sense-tagged word like "run_wn.01_v" or "dog_wn.01_n"

    Returns:
        Dictionary of grammatical features with categorical values
    """
    # Initialize all features to 'none'
    metadata = {feature: 'none' for feature in GRAMMAR_DIMENSIONS.keys()}

    # Parse word_id format: word_wn.XX_pos
    try:
        parts = word_id.split('_')
        if len(parts) < 3 or parts[1][:3] != 'wn.':
            # Not a sense-tagged word, return defaults
            return metadata

        base_word = parts[0]
        sense_num = parts[1].split('.')[1]
        pos = parts[2]

        # Map POS to full name
        pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb', 's': 'adjective_satellite'}
        metadata['pos'] = pos_map.get(pos, pos)

        # Try to get WordNet synset (format: word.pos.sense_num)
        synset_name = f"{base_word}.{pos}.{sense_num}"
        try:
            synset = wn.synset(synset_name)
        except:
            # Synset not found, return defaults with just POS
            return metadata

        # Extract POS-specific features
        if pos == 'n':  # Noun
            metadata.update(_extract_noun_features(synset, base_word))
        elif pos == 'v':  # Verb
            metadata.update(_extract_verb_features(synset, base_word))
        elif pos in ['a', 's']:  # Adjective
            metadata.update(_extract_adjective_features(synset, base_word))
        elif pos == 'r':  # Adverb
            metadata.update(_extract_adverb_features(synset, base_word))

    except Exception as e:
        # If parsing fails, return defaults
        pass

    return metadata


def _extract_noun_features(synset, word: str) -> Dict[str, str]:
    """Extract noun-specific grammatical features."""
    features = {}

    # Number: Check if word ends in 's' (heuristic, not perfect)
    if word.endswith('s') and not word.endswith('ss'):
        features['number'] = 'plural'
    else:
        features['number'] = 'singular'

    # Animacy: Check if it's a person/animal
    hypernyms = set()
    for path in synset.hypernym_paths():
        hypernyms.update([s.name() for s in path])

    if any('person' in h or 'animal' in h or 'organism' in h for h in hypernyms):
        features['animacy'] = 'animate'
    else:
        features['animacy'] = 'inanimate'

    # Living: Similar to animacy
    if any('organism' in h or 'living_thing' in h for h in hypernyms):
        features['living'] = 'living'
    else:
        features['living'] = 'nonliving'

    # Embodiment: Physical vs abstract
    if any('abstraction' in h or 'psychological_feature' in h for h in hypernyms):
        features['embodiment'] = 'abstract'
    else:
        features['embodiment'] = 'physical'

    # Countability: Heuristic based on definition
    definition = synset.definition().lower()
    if any(word in definition for word in ['amount', 'mass', 'substance']):
        features['countability'] = 'uncountable'
    else:
        features['countability'] = 'countable'

    return features


def _extract_verb_features(synset, word: str) -> Dict[str, str]:
    """Extract verb-specific grammatical features."""
    features = {}

    # Tense: Base form is present (morphology would give past/future)
    features['tense'] = 'present'

    # Aspect: Default to simple
    features['aspect'] = 'simple'

    # Voice: Default to active
    features['voice'] = 'active'

    # Mood: Default to indicative
    features['mood'] = 'indicative'

    # Transitivity: Check verb frames using lemmas
    try:
        frames = []
        for lemma in synset.lemmas():
            frames.extend(lemma.frame_strings())

        if frames:
            has_object = any('something' in f or 'somebody' in f for f in frames)
            has_two_objects = any(f.count('somebody') + f.count('something') >= 2 for f in frames)

            if has_two_objects:
                features['transitivity'] = 'ditransitive'
            elif has_object:
                features['transitivity'] = 'transitive'
            else:
                features['transitivity'] = 'intransitive'
        else:
            # No frame info, use definition heuristic
            definition = synset.definition().lower()
            if 'something' in definition or 'someone' in definition:
                features['transitivity'] = 'transitive'
    except:
        # If frame extraction fails, leave as 'none'
        pass

    return features


def _extract_adjective_features(synset, word: str) -> Dict[str, str]:
    """Extract adjective-specific grammatical features."""
    features = {}

    # Degree: Check if it's a comparative/superlative
    if word.endswith('er'):
        features['degree'] = 'comparative'
    elif word.endswith('est'):
        features['degree'] = 'superlative'
    else:
        features['degree'] = 'positive'

    # Polarity: Check if it's a negative adjective
    negative_markers = ['un', 'in', 'im', 'il', 'ir', 'non', 'dis']
    if any(word.startswith(prefix) for prefix in negative_markers):
        features['polarity'] = 'negative'
    else:
        features['polarity'] = 'positive'

    return features


def _extract_adverb_features(synset, word: str) -> Dict[str, str]:
    """Extract adverb-specific grammatical features."""
    features = {}

    # Degree: Check if it's comparative/superlative
    if word.endswith('er'):
        features['degree'] = 'comparative'
    elif word.endswith('est'):
        features['degree'] = 'superlative'
    else:
        features['degree'] = 'positive'

    return features


def create_grammar_metadata_for_vocab(word_to_id: Dict[str, int]) -> Dict[str, Dict[str, str]]:
    """
    Create grammar metadata for entire vocabulary.

    Args:
        word_to_id: Vocabulary mapping word IDs to indices

    Returns:
        Dictionary mapping word IDs to grammar metadata
    """
    print("Extracting grammar metadata for vocabulary...")

    grammar_metadata = {}

    for word_id in word_to_id.keys():
        grammar_metadata[word_id] = extract_grammar_metadata(word_id)

    print(f"  Extracted metadata for {len(grammar_metadata)} words")

    # Print statistics
    feature_counts = {}
    for feature in GRAMMAR_DIMENSIONS.keys():
        value_counts = {}
        for word_metadata in grammar_metadata.values():
            value = word_metadata.get(feature, 'none')
            value_counts[value] = value_counts.get(value, 0) + 1
        feature_counts[feature] = value_counts

    # Print interesting features (those with variation)
    print("\n  Grammar feature distribution (non-trivial features):")
    for feature, value_counts in sorted(feature_counts.items()):
        # Skip features where everything is 'none'
        if len(value_counts) > 1 or 'none' not in value_counts:
            print(f"    {feature}:")
            for value, count in sorted(value_counts.items(), key=lambda x: -x[1])[:5]:
                pct = 100 * count / len(grammar_metadata)
                print(f"      {value}: {count} ({pct:.1f}%)")

    return grammar_metadata
