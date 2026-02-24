"""
Anchor Dimension Initializer

Populates the first 51 anchor dimensions with semantic features from WordNet.
This solves the "wasted anchor capacity" problem - instead of frozen zeros,
anchors are initialized with meaningful linguistic features.

Anchor Dimensions (51 total):
- 27 semantic: morality, temperature, size, animacy, etc.
- 15 grammatical: tense, aspect, mood, gender, number, etc.
- 9 logical: AND, OR, NOT, XOR, IF, etc.

Strategy:
1. For each word in vocabulary, query WordNet for semantic features
2. Populate anchor dimensions based on synset properties:
   - Morality: good/bad, moral/immoral, right/wrong
   - Gender: masculine/feminine (from lemma properties)
   - Animacy: person/animal vs object
   - Size: large/small (from hypernyms)
   - Temperature: hot/cold (from definitions)
   - etc.
3. Save initialized embeddings to be used as starting point for training

This provides baseline interpretability and helps guide polarity discovery.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from tqdm import tqdm

from nltk.corpus import wordnet as wn
from src.embeddings.anchors import AnchorDimensions


# Semantic feature indicators (expanded from known antonym pairs)
SEMANTIC_FEATURES = {
    'morality': {
        'positive': {'good', 'moral', 'right', 'ethical', 'virtuous', 'righteous'},
        'negative': {'bad', 'immoral', 'wrong', 'unethical', 'evil', 'wicked'}
    },
    'temperature': {
        'hot': {'hot', 'warm', 'heated', 'burning', 'scorching'},
        'cold': {'cold', 'cool', 'chilly', 'freezing', 'icy'}
    },
    'size': {
        'large': {'large', 'big', 'huge', 'enormous', 'giant', 'massive'},
        'small': {'small', 'tiny', 'little', 'miniature', 'minute'}
    },
    'quality': {
        'high': {'excellent', 'superior', 'quality', 'fine', 'premium'},
        'low': {'poor', 'inferior', 'bad', 'cheap', 'shoddy'}
    },
    'intensity': {
        'strong': {'strong', 'powerful', 'intense', 'vigorous', 'robust'},
        'weak': {'weak', 'feeble', 'mild', 'faint', 'delicate'}
    },
    'speed': {
        'fast': {'fast', 'quick', 'rapid', 'swift', 'speedy'},
        'slow': {'slow', 'sluggish', 'leisurely', 'gradual'}
    },
    'difficulty': {
        'hard': {'hard', 'difficult', 'challenging', 'tough'},
        'easy': {'easy', 'simple', 'effortless', 'straightforward'}
    },
    'brightness': {
        'bright': {'bright', 'light', 'luminous', 'brilliant', 'radiant'},
        'dark': {'dark', 'dim', 'shadowy', 'gloomy', 'murky'}
    },
    'wetness': {
        'wet': {'wet', 'moist', 'damp', 'humid', 'soggy'},
        'dry': {'dry', 'arid', 'parched', 'dehydrated'}
    },
    'age': {
        'young': {'young', 'youthful', 'new', 'fresh', 'modern'},
        'old': {'old', 'aged', 'ancient', 'elderly', 'antique'}
    },
    'safety': {
        'safe': {'safe', 'secure', 'protected', 'guarded'},
        'dangerous': {'dangerous', 'unsafe', 'hazardous', 'risky', 'perilous'}
    },
    'happiness': {
        'happy': {'happy', 'joyful', 'cheerful', 'glad', 'delighted'},
        'sad': {'sad', 'unhappy', 'sorrowful', 'miserable', 'depressed'}
    },
    'wealth': {
        'rich': {'rich', 'wealthy', 'affluent', 'prosperous'},
        'poor': {'poor', 'impoverished', 'destitute', 'penniless'}
    },
    'health': {
        'healthy': {'healthy', 'well', 'fit', 'robust', 'vigorous'},
        'sick': {'sick', 'ill', 'unwell', 'diseased', 'ailing'}
    },
    'beauty': {
        'beautiful': {'beautiful', 'pretty', 'attractive', 'lovely', 'gorgeous'},
        'ugly': {'ugly', 'unattractive', 'hideous', 'unsightly'}
    },
}


def extract_wordnet_features(word: str, sense_idx: int, pos: str) -> Dict[str, float]:
    """
    Extract semantic features from WordNet for a single sense.

    Args:
        word: Word lemma (e.g., 'good')
        sense_idx: Sense index (e.g., 0 for first sense)
        pos: Part of speech ('n', 'v', 'a', 'r')

    Returns:
        Dictionary of feature_name -> value (range: -1.0 to 1.0)
    """
    features = {}

    # Get synset
    synsets = wn.synsets(word, pos=pos)
    if sense_idx >= len(synsets):
        return features

    synset = synsets[sense_idx]

    # Check semantic features based on word/synset properties
    word_lower = word.lower()

    for feature_name, polarity_dict in SEMANTIC_FEATURES.items():
        # Check if word matches any polarity indicator
        for polarity, indicator_words in polarity_dict.items():
            if word_lower in indicator_words:
                # Assign value based on polarity
                if polarity in ['positive', 'hot', 'large', 'high', 'strong', 'fast',
                                'hard', 'bright', 'wet', 'young', 'safe', 'happy',
                                'rich', 'healthy', 'beautiful']:
                    features[feature_name] = 1.0
                else:
                    features[feature_name] = -1.0
                break

    # Animacy from hypernyms
    if pos == 'n':
        hypernyms = set()
        for hypernym in synset.closure(lambda s: s.hypernyms(), depth=5):
            hypernyms.add(hypernym.name().split('.')[0])

        if 'person' in hypernyms or 'animal' in hypernyms or 'organism' in hypernyms:
            features['animacy'] = 1.0
        elif 'object' in hypernyms or 'artifact' in hypernyms:
            features['animacy'] = -1.0

    # Gender from lemma properties (if available)
    for lemma in synset.lemmas():
        lemma_name = lemma.name().lower()
        # Simple heuristics
        if any(m in lemma_name for m in ['man', 'boy', 'male', 'father', 'brother', 'son']):
            features['gender'] = -1.0  # Masculine
        elif any(f in lemma_name for f in ['woman', 'girl', 'female', 'mother', 'sister', 'daughter']):
            features['gender'] = 1.0  # Feminine

    # Polarity (grammatical negation) from definition
    definition = synset.definition().lower()
    if any(neg in definition for neg in ['not', 'lack', 'without', 'absence']):
        features['polarity'] = -1.0  # Negative
    else:
        features['polarity'] = 0.5  # Default to affirmative

    return features


def initialize_anchor_embeddings(vocabulary: Set[str],
                                  word_to_id: Dict[str, int],
                                  embedding_dim: int) -> np.ndarray:
    """
    Initialize embeddings with anchor dimensions populated from WordNet.

    Args:
        vocabulary: Set of sense-tagged words
        word_to_id: Vocabulary mapping
        embedding_dim: Total embedding dimensions

    Returns:
        Initialized embeddings (vocab_size, embedding_dim)
    """
    print("  Initializing anchor dimensions from WordNet...")

    # Get anchor definitions
    anchors = AnchorDimensions()
    num_anchors = anchors.num_anchors()

    print(f"    Anchor dimensions: {num_anchors}")
    print(f"    Total dimensions: {embedding_dim}")

    # Create embeddings (start with small random for learned dims)
    vocab_size = len(word_to_id)
    embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01

    # Map feature names to anchor indices
    feature_to_anchor_idx = {
        'morality': 0,  # Custom semantic (not in default anchors, use first slot)
        'temperature': 1,
        'size': anchors.get_dimension_index('magnitude'),
        'animacy': anchors.get_dimension_index('animacy'),
        'gender': anchors.get_dimension_index('gender'),
        'polarity': anchors.get_dimension_index('polarity'),
    }

    # Add custom semantic dimensions for discovered features
    custom_semantic_idx = 6  # Start after built-in semantic anchors
    for feature_name in SEMANTIC_FEATURES.keys():
        if feature_name not in feature_to_anchor_idx:
            if custom_semantic_idx < num_anchors:
                feature_to_anchor_idx[feature_name] = custom_semantic_idx
                custom_semantic_idx += 1

    # Populate anchor dimensions for each word
    populated_count = defaultdict(int)

    for sense_tag in tqdm(list(vocabulary), desc="Populating anchors"):
        # Parse sense tag
        if '_wn.' not in sense_tag:
            continue

        parts = sense_tag.rsplit('_', 1)
        if len(parts) != 2:
            continue

        word_sense = parts[0]
        pos_tag = parts[1]
        word_parts = word_sense.split('_wn.')
        if len(word_parts) != 2:
            continue

        word = word_parts[0]
        try:
            sense_idx = int(word_parts[1])
        except ValueError:
            continue

        # Get word index
        word_idx = word_to_id.get(sense_tag)
        if word_idx is None:
            continue

        # Extract features
        features = extract_wordnet_features(word, sense_idx, pos_tag)

        # Populate anchor dimensions
        for feature_name, value in features.items():
            anchor_idx = feature_to_anchor_idx.get(feature_name)
            if anchor_idx is not None and anchor_idx < num_anchors:
                embeddings[word_idx, anchor_idx] = value
                populated_count[feature_name] += 1

    print(f"\n    Anchor population statistics:")
    for feature_name, count in sorted(populated_count.items()):
        print(f"      {feature_name}: {count} words")

    total_populated = sum(populated_count.values())
    print(f"    Total anchor values populated: {total_populated}")

    return embeddings


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize anchor dimensions from WordNet features'
    )
    parser.add_argument('--graph-dir', type=str,
                       default='data/wordnet_only_graph',
                       help='Knowledge graph directory (contains vocabulary.json)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Total embedding dimensions (default: 128)')
    parser.add_argument('--output-file', type=str,
                       default='checkpoints/initialized_embeddings.npy',
                       help='Output file for initialized embeddings')

    args = parser.parse_args()

    print("="*70)
    print("ANCHOR DIMENSION INITIALIZER")
    print("="*70)
    print()

    # Load vocabulary
    print("[1/2] Loading vocabulary...")
    graph_dir = Path(args.graph_dir)
    vocab_path = graph_dir / "vocabulary.json"

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
        word_to_id = vocab_data['word_to_id']

    vocabulary = set(word_to_id.keys())
    print(f"  Loaded {len(vocabulary)} sense-tagged words")
    print()

    # Initialize anchors
    print("[2/2] Initializing anchor dimensions...")
    embeddings = initialize_anchor_embeddings(
        vocabulary=vocabulary,
        word_to_id=word_to_id,
        embedding_dim=args.embedding_dim
    )
    print()

    # Save initialized embeddings
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, embeddings)
    print(f"Saved initialized embeddings: {output_file}")
    print(f"  Shape: {embeddings.shape}")

    print()
    print("="*70)
    print("INITIALIZATION COMPLETE")
    print("="*70)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Anchor dimensions: {AnchorDimensions().num_anchors()}")
    print(f"Learned dimensions: {args.embedding_dim - AnchorDimensions().num_anchors()}")
    print()
    print("Next step: Train Phase 1 bootstrap with initialized anchors")
    print("  python scripts/train_wordnet_bootstrap.py \\")
    print(f"    --graph-dir {args.graph_dir} \\")
    print(f"    --init-embeddings {output_file}")


if __name__ == "__main__":
    main()
