"""
Discover Polarity Dimensions (Unsupervised)

This script analyzes trained embeddings to discover which dimensions
have interpretable polarity patterns, WITHOUT any manual pre-assignment.

The goal is to prove that semantic structure emerges naturally from
distance + sparsity constraints alone.

Usage:
    python scripts/discover_polarity_dimensions.py \
        --embeddings checkpoints/embeddings.npy \
        --vocabulary checkpoints/vocabulary.json \
        --output checkpoints/discovered_dimensions.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from nltk.corpus import wordnet as wn


def load_embeddings_and_vocab(embeddings_path: str, vocab_path: str):
    """Load embeddings and vocabulary."""
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)

    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = {v: k for k, v in word_to_id.items()}

    print(f"  Vocabulary: {len(word_to_id)} words")
    print(f"  Dimensions: {embeddings.shape[1]}")

    return embeddings, word_to_id, id_to_word


def extract_all_antonym_pairs(word_to_id: Dict[str, int]) -> List[Tuple[str, str]]:
    """Extract all antonym pairs that exist in vocabulary."""
    print("\nExtracting antonym pairs from vocabulary...")

    antonym_pairs = set()
    vocab_base_words = set()

    # Extract base words from sense-tagged vocabulary
    for word in word_to_id.keys():
        if '_wn.' in word:
            base = word.split('_wn.')[0]
            vocab_base_words.add(base)
        elif '_' in word:
            parts = word.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['n', 'v', 'a', 'r', 'x']:
                vocab_base_words.add(parts[0])
            else:
                vocab_base_words.add(word)
        else:
            vocab_base_words.add(word)

    # Find antonyms using WordNet
    for base_word in vocab_base_words:
        synsets = wn.synsets(base_word)

        for synset in synsets[:3]:  # Check first 3 senses
            for lemma in synset.lemmas():
                for antonym_lemma in lemma.antonyms():
                    antonym_base = antonym_lemma.name().lower()

                    if antonym_base in vocab_base_words and antonym_base != base_word:
                        # Get actual vocabulary words (with POS tags)
                        word1_candidates = [w for w in word_to_id.keys()
                                          if w.lower().startswith(base_word)]
                        word2_candidates = [w for w in word_to_id.keys()
                                          if w.lower().startswith(antonym_base)]

                        if word1_candidates and word2_candidates:
                            # Use first matching candidate for each
                            pair = tuple(sorted([word1_candidates[0], word2_candidates[0]]))
                            antonym_pairs.add(pair)

    antonym_list = list(antonym_pairs)
    print(f"  Found {len(antonym_list)} antonym pairs in vocabulary")

    return antonym_list


def analyze_dimension_polarity(embeddings: np.ndarray,
                               antonym_pairs: List[Tuple[str, str]],
                               word_to_id: Dict[str, int],
                               dim_idx: int) -> Dict:
    """Analyze polarity patterns for a single dimension."""
    opposite_sign_count = 0
    same_sign_count = 0
    zero_count = 0
    polarized_pairs = []

    for word1, word2 in antonym_pairs:
        id1 = word_to_id.get(word1.lower())
        id2 = word_to_id.get(word2.lower())

        if id1 is None or id2 is None:
            continue

        val1 = embeddings[id1, dim_idx]
        val2 = embeddings[id2, dim_idx]

        sign1 = np.sign(val1)
        sign2 = np.sign(val2)

        if sign1 * sign2 < 0:  # Opposite signs
            opposite_sign_count += 1
            polarized_pairs.append((word1, word2, val1, val2))
        elif sign1 * sign2 > 0:  # Same sign
            same_sign_count += 1
        else:  # One or both are zero
            zero_count += 1

    total = opposite_sign_count + same_sign_count + zero_count

    if total == 0:
        consistency = 0.0
    else:
        consistency = opposite_sign_count / total

    return {
        'opposite_sign': opposite_sign_count,
        'same_sign': same_sign_count,
        'zero': zero_count,
        'total': total,
        'consistency': consistency,
        'polarized_pairs': polarized_pairs
    }


def infer_semantic_type(polarized_pairs: List[Tuple[str, str, float, float]]) -> str:
    """Infer semantic type from polarized antonym pairs."""
    # Extract base words
    base_pairs = []
    for word1, word2, _, _ in polarized_pairs:
        base1 = word1.split('_')[0] if '_' in word1 else word1
        base2 = word2.split('_')[0] if '_' in word2 else word2
        base_pairs.append((base1.lower(), base2.lower()))

    # Semantic categories with keywords
    categories = {
        'size': ['big', 'small', 'large', 'tiny', 'huge', 'little', 'enlarge', 'reduce'],
        'morality': ['good', 'bad', 'right', 'wrong', 'virtue', 'evil'],
        'emotion': ['happy', 'sad', 'joy', 'sorrow', 'glad', 'gloomy'],
        'temperature': ['hot', 'cold', 'warm', 'cool'],
        'speed': ['fast', 'slow', 'quick', 'sluggish'],
        'light': ['light', 'dark', 'bright', 'dim', 'heavy'],
        'strength': ['strong', 'weak', 'powerful', 'feeble'],
        'difficulty': ['easy', 'hard', 'simple', 'difficult'],
        'quantity': ['many', 'few', 'more', 'less'],
        'age': ['young', 'old', 'new', 'ancient'],
        'distance': ['near', 'far', 'close', 'distant'],
        'wealth': ['rich', 'poor', 'wealthy', 'impoverished'],
        'knowledge': ['wise', 'foolish', 'intelligent', 'stupid'],
        'safety': ['safe', 'dangerous', 'secure', 'risky'],
        'truth': ['true', 'false', 'real', 'fake'],
    }

    # Score each category
    category_scores = defaultdict(int)
    for base1, base2 in base_pairs:
        for category, keywords in categories.items():
            if base1 in keywords or base2 in keywords:
                category_scores[category] += 1

    if not category_scores:
        return 'unknown'

    # Return category with highest score
    best_category = max(category_scores.items(), key=lambda x: x[1])
    return best_category[0]


def discover_polarity_dimensions(embeddings: np.ndarray,
                                 antonym_pairs: List[Tuple[str, str]],
                                 word_to_id: Dict[str, int],
                                 threshold: float = 0.7) -> Dict[int, Dict]:
    """Discover dimensions with clear polarity patterns."""
    print(f"\nDiscovering polarity dimensions (threshold={threshold*100:.0f}%)...")

    discovered_dims = {}
    embedding_dim = embeddings.shape[1]

    for dim_idx in range(embedding_dim):
        analysis = analyze_dimension_polarity(embeddings, antonym_pairs, word_to_id, dim_idx)

        if analysis['consistency'] >= threshold:
            # Infer semantic type
            semantic_type = infer_semantic_type(analysis['polarized_pairs'])

            discovered_dims[dim_idx] = {
                'consistency': float(analysis['consistency']),
                'opposite_sign_count': analysis['opposite_sign'],
                'same_sign_count': analysis['same_sign'],
                'total_pairs': analysis['total'],
                'semantic_type': semantic_type,
                'sample_pairs': analysis['polarized_pairs'][:10]  # Top 10 pairs
            }

    print(f"  Discovered {len(discovered_dims)} polarity dimensions")

    return discovered_dims


def compute_dimension_statistics(embeddings: np.ndarray) -> Dict[int, Dict]:
    """Compute statistics for each dimension."""
    print("\nComputing dimension statistics...")

    stats = {}
    embedding_dim = embeddings.shape[1]

    for dim_idx in range(embedding_dim):
        values = embeddings[:, dim_idx]

        # Sparsity (percentage near zero)
        near_zero = np.abs(values) < 0.01
        sparsity = 100.0 * np.mean(near_zero)

        stats[dim_idx] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'sparsity': float(sparsity),
            'variance': float(np.var(values))
        }

    return stats


def print_discovery_report(discovered_dims: Dict[int, Dict],
                           dim_stats: Dict[int, Dict],
                           id_to_word: Dict[int, str]):
    """Print detailed discovery report."""
    print("\n" + "=" * 80)
    print("POLARITY DIMENSION DISCOVERY REPORT")
    print("=" * 80)

    if not discovered_dims:
        print("\nNo polarity dimensions discovered above threshold.")
        print("This suggests the model may need:")
        print("  - More training epochs")
        print("  - Different hyperparameters")
        print("  - Larger vocabulary with more antonym pairs")
        return

    # Sort by consistency
    sorted_dims = sorted(discovered_dims.items(), key=lambda x: x[1]['consistency'], reverse=True)

    for dim_idx, dim_info in sorted_dims:
        stats = dim_stats[dim_idx]

        print(f"\n{'='*80}")
        print(f"DIMENSION {dim_idx}: {dim_info['semantic_type'].upper()}")
        print(f"{'='*80}")

        print(f"Polarity consistency: {dim_info['consistency']*100:.1f}%")
        print(f"Polarized pairs: {dim_info['opposite_sign_count']}/{dim_info['total_pairs']}")
        print(f"Sparsity: {stats['sparsity']:.1f}%")
        print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"Std dev: {stats['std']:.4f}")

        print(f"\nSample polarized pairs:")
        for word1, word2, val1, val2 in dim_info['sample_pairs'][:10]:
            # Clean up word display
            w1_display = word1.split('_')[0] if '_' in word1 else word1
            w2_display = word2.split('_')[0] if '_' in word2 else word2
            print(f"  {w1_display:20s} ({val1:+.4f}) <-> {w2_display:20s} ({val2:+.4f})")


def save_discovery_results(discovered_dims: Dict[int, Dict],
                           dim_stats: Dict[int, Dict],
                           output_path: str):
    """Save discovery results to JSON."""
    print(f"\nSaving discovery results to {output_path}...")

    # Convert sample_pairs to JSON-serializable format
    output_data = {}
    for dim_idx, dim_info in discovered_dims.items():
        output_data[str(dim_idx)] = {
            'consistency': dim_info['consistency'],
            'opposite_sign_count': dim_info['opposite_sign_count'],
            'same_sign_count': dim_info['same_sign_count'],
            'total_pairs': dim_info['total_pairs'],
            'semantic_type': dim_info['semantic_type'],
            'statistics': dim_stats[dim_idx],
            'sample_pairs': [
                {
                    'word1': pair[0],
                    'word2': pair[1],
                    'value1': float(pair[2]),
                    'value2': float(pair[3])
                }
                for pair in dim_info['sample_pairs']
            ]
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(discovered_dims)} discovered dimensions")


def main():
    parser = argparse.ArgumentParser(description='Discover polarity dimensions from trained embeddings')
    parser.add_argument('--embeddings', type=str, default='checkpoints/embeddings.npy',
                       help='Path to embeddings file')
    parser.add_argument('--vocabulary', type=str, default='checkpoints/vocabulary.json',
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default='checkpoints/discovered_dimensions.json',
                       help='Output path for discovery results')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Minimum opposite-sign consistency threshold (0-1)')

    args = parser.parse_args()

    print("=" * 80)
    print("UNSUPERVISED POLARITY DIMENSION DISCOVERY")
    print("=" * 80)
    print()

    # Load data
    embeddings, word_to_id, id_to_word = load_embeddings_and_vocab(
        args.embeddings, args.vocabulary
    )

    # Extract antonym pairs from vocabulary
    antonym_pairs = extract_all_antonym_pairs(word_to_id)

    if len(antonym_pairs) == 0:
        print("\nERROR: No antonym pairs found in vocabulary!")
        print("This vocabulary may be too small or lack antonym pairs.")
        return

    # Discover polarity dimensions
    discovered_dims = discover_polarity_dimensions(
        embeddings, antonym_pairs, word_to_id, threshold=args.threshold
    )

    # Compute dimension statistics
    dim_stats = compute_dimension_statistics(embeddings)

    # Print report
    print_discovery_report(discovered_dims, dim_stats, id_to_word)

    # Save results
    save_discovery_results(discovered_dims, dim_stats, args.output)

    # Summary
    print("\n" + "=" * 80)
    print("DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"Total dimensions analyzed: {embeddings.shape[1]}")
    print(f"Polarity dimensions discovered: {len(discovered_dims)}")
    print(f"Discovery rate: {100*len(discovered_dims)/embeddings.shape[1]:.1f}%")
    print(f"Total antonym pairs: {len(antonym_pairs)}")

    if discovered_dims:
        avg_consistency = np.mean([d['consistency'] for d in discovered_dims.values()])
        print(f"Average consistency: {avg_consistency*100:.1f}%")

        semantic_types = [d['semantic_type'] for d in discovered_dims.values()]
        unique_types = set(semantic_types)
        print(f"Unique semantic types: {len(unique_types)}")
        print(f"Semantic types: {', '.join(sorted(unique_types))}")

    print()
    print("=" * 80)
    print("DISCOVERY COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
