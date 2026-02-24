"""
Test Trained Embeddings

Analyzes embedding quality by checking:
1. Similar word distances (should be close)
2. Dissimilar word distances (should be far)
3. Sense separation (different senses should be separated)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import List, Tuple
import argparse


def load_embeddings(checkpoint_dir: str = "checkpoints"):
    """Load embeddings and vocabulary."""
    checkpoint_path = Path(checkpoint_dir)

    embeddings = np.load(checkpoint_path / "embeddings.npy")
    with open(checkpoint_path / "vocabulary.json", 'r') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']

    return embeddings, word_to_id, id_to_word


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate normalized Euclidean distance."""
    return np.linalg.norm(vec1 - vec2) / np.sqrt(len(vec1))


def find_word(word: str, word_to_id: dict) -> List[str]:
    """Find all sense-tagged versions of a word in vocabulary."""
    matches = []
    word_lower = word.lower()

    for vocab_word in word_to_id.keys():
        # Check if this is the word we're looking for
        if vocab_word.startswith(word_lower + "_"):
            matches.append(vocab_word)
        elif vocab_word == word_lower:
            matches.append(vocab_word)

    return matches


def test_word_similarity(embeddings: np.ndarray, word_to_id: dict,
                         word1: str, word2: str) -> None:
    """Test similarity between two words."""
    # Find all senses
    matches1 = find_word(word1, word_to_id)
    matches2 = find_word(word2, word_to_id)

    if not matches1:
        print(f"  '{word1}' not found in vocabulary")
        return
    if not matches2:
        print(f"  '{word2}' not found in vocabulary")
        return

    print(f"\nComparing '{word1}' and '{word2}':")
    print(f"  Found {len(matches1)} sense(s) of '{word1}': {matches1[:3]}")
    print(f"  Found {len(matches2)} sense(s) of '{word2}': {matches2[:3]}")

    # Test all combinations
    for w1 in matches1[:3]:  # Limit to first 3 senses
        for w2 in matches2[:3]:
            vec1 = embeddings[word_to_id[w1]]
            vec2 = embeddings[word_to_id[w2]]

            cos_sim = cosine_similarity(vec1, vec2)
            euc_dist = euclidean_distance(vec1, vec2)

            print(f"  {w1:40s} <-> {w2:40s}")
            print(f"    Cosine similarity: {cos_sim:6.3f}")
            print(f"    Euclidean distance: {euc_dist:6.3f}")


def test_sense_separation(embeddings: np.ndarray, word_to_id: dict, word: str) -> None:
    """Test that different senses of a word are separated."""
    matches = find_word(word, word_to_id)

    if len(matches) < 2:
        print(f"\n'{word}' has only {len(matches)} sense(s) in vocabulary")
        return

    print(f"\nSense separation for '{word}' ({len(matches)} senses):")

    # Compare all pairs of senses
    for i, w1 in enumerate(matches[:5]):
        for j, w2 in enumerate(matches[:5]):
            if i >= j:
                continue

            vec1 = embeddings[word_to_id[w1]]
            vec2 = embeddings[word_to_id[w2]]

            cos_sim = cosine_similarity(vec1, vec2)
            euc_dist = euclidean_distance(vec1, vec2)

            print(f"  {w1:40s} <-> {w2:40s}")
            print(f"    Cosine similarity: {cos_sim:6.3f}, Distance: {euc_dist:6.3f}")


def find_nearest_neighbors(embeddings: np.ndarray, word_to_id: dict,
                           id_to_word: dict, word: str, top_k: int = 10) -> None:
    """Find nearest neighbors of a word."""
    matches = find_word(word, word_to_id)

    if not matches:
        print(f"\n'{word}' not found in vocabulary")
        return

    target_word = matches[0]
    print(f"\nNearest neighbors of '{target_word}':")

    target_vec = embeddings[word_to_id[target_word]]

    # Calculate distances to all words
    distances = []
    for vocab_word, idx in word_to_id.items():
        if vocab_word == target_word:
            continue

        vec = embeddings[idx]
        dist = euclidean_distance(target_vec, vec)
        cos_sim = cosine_similarity(target_vec, vec)
        distances.append((vocab_word, dist, cos_sim))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Print top K
    for i, (vocab_word, dist, cos_sim) in enumerate(distances[:top_k], 1):
        print(f"  {i:2d}. {vocab_word:40s}  dist={dist:.3f}  sim={cos_sim:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Test trained embeddings')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing embeddings')

    args = parser.parse_args()

    print("="*70)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*70)

    # Load embeddings
    print("\n[1/3] Loading embeddings...")
    embeddings, word_to_id, id_to_word = load_embeddings(args.checkpoint_dir)
    print(f"  Loaded {len(word_to_id)} words, embedding dim: {embeddings.shape[1]}")

    # Test similar words
    print("\n[2/3] Testing similar word pairs...")
    print("="*70)

    # Common similar pairs
    similar_pairs = [
        ("dog", "cat"),      # Both animals
        ("big", "large"),    # Synonyms
        ("run", "walk"),     # Both movement verbs
        ("good", "bad"),     # Antonyms (should be far)
        ("the", "a"),        # Determiners
    ]

    for word1, word2 in similar_pairs:
        test_word_similarity(embeddings, word_to_id, word1, word2)

    # Test sense separation
    print("\n[3/3] Testing sense separation...")
    print("="*70)

    # Words that should have multiple senses
    polysemous_words = ["bank", "run", "light", "play", "right"]

    for word in polysemous_words:
        test_sense_separation(embeddings, word_to_id, word)

    # Find nearest neighbors
    print("\n[Bonus] Nearest neighbor analysis...")
    print("="*70)

    interesting_words = ["dog", "run", "good"]
    for word in interesting_words:
        find_nearest_neighbors(embeddings, word_to_id, id_to_word, word, top_k=10)

    print("\n" + "="*70)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
