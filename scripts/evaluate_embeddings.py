"""
NAOMI-II Embedding Evaluation Script

Evaluates the quality of trained embeddings by testing:
1. Nearest neighbors (similar words)
2. Distance relationships (synonyms vs antonyms)
3. Vector space structure
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from typing import List, Tuple


def load_embeddings(checkpoint_dir: Path):
    """Load trained embeddings and vocabulary."""
    embeddings = np.load(checkpoint_dir / "embeddings.npy")
    with open(checkpoint_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return embeddings, vocab_data


def find_nearest_neighbors(
    word: str,
    embeddings: np.ndarray,
    word_to_id: dict,
    id_to_word: dict,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find k nearest neighbors of a word."""
    if word not in word_to_id:
        return []

    word_id = word_to_id[word]
    word_vec = embeddings[word_id]

    # Compute cosine similarities
    similarities = np.dot(embeddings, word_vec)

    # Get top-k (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1]
    neighbors = []

    for idx in top_indices:
        if idx == word_id:
            continue
        neighbor_word = id_to_word[str(idx)]
        similarity = similarities[idx]
        neighbors.append((neighbor_word, float(similarity)))
        if len(neighbors) >= top_k:
            break

    return neighbors


def compute_distance(
    word1: str,
    word2: str,
    embeddings: np.ndarray,
    word_to_id: dict
) -> float:
    """Compute Euclidean distance between two words."""
    if word1 not in word_to_id or word2 not in word_to_id:
        return -1.0

    vec1 = embeddings[word_to_id[word1]]
    vec2 = embeddings[word_to_id[word2]]

    return float(np.linalg.norm(vec1 - vec2))



def main():
    print("=" * 70)
    print("NAOMI-II EMBEDDING EVALUATION")
    print("=" * 70)
    print()

    # Load embeddings
    checkpoint_dir = Path("checkpoints")
    print("[1/3] Loading embeddings...")
    embeddings, vocab_data = load_embeddings(checkpoint_dir)
    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']
    vocab_size = vocab_data['vocab_size']
    embedding_dim = vocab_data['embedding_dim']

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print()

    # Show vocabulary
    print("[2/3] Vocabulary:")
    words = sorted(word_to_id.keys())
    print(f"  {', '.join(words[:20])}")
    if len(words) > 20:
        print(f"  ... and {len(words) - 20} more")
    print()

    # Test nearest neighbors
    print("[3/3] Nearest Neighbors:")
    print()

    test_words = ['dog_en', 'cat_en', 'runs_en', 'walks_en', 'big_en', 'small_en', 'The_en', 'the_en']
    for word in test_words:
        if word not in word_to_id:
            continue

        neighbors = find_nearest_neighbors(word, embeddings, word_to_id, id_to_word, top_k=5)
        # Strip language tags for display
        neighbor_str = ', '.join([f"{w.replace('_en', '')} ({s:.3f})" for w, s in neighbors])
        print(f"  '{word.replace('_en', '')}' -> {neighbor_str}")

    print()

    # Test specific distances
    print("Distance Analysis:")
    print()

    test_pairs = [
        ('dog_en', 'cat_en', 'Similar animals'),
        ('dog_en', 'runs_en', 'Agent-action'),
        ('runs_en', 'walks_en', 'Similar verbs'),
        ('big_en', 'small_en', 'Antonyms'),
        ('The_en', 'the_en', 'Case variants'),
        ('dog_en', 'big_en', 'Unrelated'),
    ]

    for word1, word2, description in test_pairs:
        distance = compute_distance(word1, word2, embeddings, word_to_id)
        if distance >= 0:
            w1_display = word1.replace('_en', '')
            w2_display = word2.replace('_en', '')
            print(f"  {w1_display:8s} <-> {w2_display:8s}  ({description:20s}): {distance:.4f}")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print()
    print("Notes:")
    print("  - Lower similarity scores mean words are more similar (normalized vectors)")
    print("  - Distances should reflect semantic relationships")
    print("  - With more training data, relationships will be clearer")
    print()


if __name__ == "__main__":
    main()
