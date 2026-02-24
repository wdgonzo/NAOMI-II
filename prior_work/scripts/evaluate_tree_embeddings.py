"""
NAOMI-II Tree-LSTM Embedding Evaluation

Evaluates structure-aware embeddings by testing:
1. Word-level embeddings (nearest neighbors, distances)
2. Tree composition (sentence encoding via Tree-LSTM)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import numpy as np
import torch
from typing import List, Tuple

from src.embeddings.tree_lstm import TreeLSTMEncoder
from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence


def load_tree_model(checkpoint_dir: Path):
    """Load trained Tree-LSTM model and vocabulary."""
    # Load vocabulary
    with open(checkpoint_dir / "tree_vocabulary.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data['word_to_id']
    id_to_word = vocab_data['id_to_word']
    vocab_size = vocab_data['vocab_size']
    embedding_dim = vocab_data['embedding_dim']
    hidden_dim = vocab_data['hidden_dim']

    # Load model
    model = TreeLSTMEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_edge_types=20,
        dropout=0.0  # Disable dropout for evaluation
    )

    # Load weights
    checkpoint = torch.load(checkpoint_dir / "tree_best_model.pt",
                           map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Store vocab in model
    model.word_to_id = word_to_id

    # Extract word embeddings
    word_embeddings = model.word_embeddings.weight.detach().cpu().numpy()

    return model, word_embeddings, word_to_id, id_to_word


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

    # Normalize
    word_vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-10)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    # Compute cosine similarities
    similarities = np.dot(embeddings_norm, word_vec_norm)

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


def encode_sentence(sentence: str, model: TreeLSTMEncoder, parser: QuantumParser,
                    device: torch.device) -> torch.Tensor:
    """Parse and encode a sentence using Tree-LSTM."""
    try:
        # Parse
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        hypothesis = chart.best_hypothesis()

        if hypothesis and hypothesis.score > 0:
            # Encode with Tree-LSTM
            with torch.no_grad():
                encoding = model(hypothesis, model.word_to_id, device)
            return encoding
        else:
            return None
    except Exception as e:
        print(f"Error encoding '{sentence}': {e}")
        return None


def compute_sentence_similarity(enc1: torch.Tensor, enc2: torch.Tensor) -> float:
    """Compute cosine similarity between two sentence encodings."""
    if enc1 is None or enc2 is None:
        return -1.0

    enc1_norm = enc1 / (torch.norm(enc1) + 1e-10)
    enc2_norm = enc2 / (torch.norm(enc2) + 1e-10)

    similarity = torch.dot(enc1_norm.squeeze(), enc2_norm.squeeze())
    return float(similarity)


def main():
    print("=" * 70)
    print("NAOMI-II TREE-LSTM EMBEDDING EVALUATION")
    print("=" * 70)
    print()

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    device = torch.device('cpu')

    # Load model
    print("[1/4] Loading Tree-LSTM model...")
    model, word_embeddings, word_to_id, id_to_word = load_tree_model(checkpoint_dir)
    vocab_size = len(word_to_id)
    embedding_dim = word_embeddings.shape[1]

    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dim: {embedding_dim}")
    print()

    # Word-level evaluation
    print("[2/4] Word-Level Embeddings:")
    print()

    test_words = ['dog_en', 'cat_en', 'runs_en', 'walks_en', 'big_en', 'small_en']
    for word in test_words:
        if word not in word_to_id:
            continue

        neighbors = find_nearest_neighbors(word, word_embeddings, word_to_id, id_to_word, top_k=5)
        neighbor_str = ', '.join([f"{w.replace('_en', '')} ({s:.3f})" for w, s in neighbors])
        print(f"  '{word.replace('_en', '')}' -> {neighbor_str}")

    print()

    # Distance analysis
    print("[3/4] Distance Analysis:")
    print()

    test_pairs = [
        ('dog_en', 'cat_en', 'Similar animals'),
        ('dog_en', 'runs_en', 'Agent-action'),
        ('runs_en', 'walks_en', 'Similar verbs'),
        ('big_en', 'small_en', 'Antonyms'),
        ('dog_en', 'big_en', 'Unrelated'),
    ]

    for word1, word2, description in test_pairs:
        distance = compute_distance(word1, word2, word_embeddings, word_to_id)
        if distance >= 0:
            w1_display = word1.replace('_en', '')
            w2_display = word2.replace('_en', '')
            print(f"  {w1_display:8s} <-> {w2_display:8s}  ({description:20s}): {distance:.4f}")

    print()

    # Sentence composition evaluation
    print("[4/4] Tree Composition (Sentence Encoding):")
    print()

    # Initialize parser
    grammar_path = Path(__file__).parent.parent / "grammars" / "english.json"
    parser = QuantumParser(str(grammar_path))

    # Test sentences
    test_sentences = [
        ("The dog runs", "Similar"),
        ("The cat runs", "Similar"),
        ("The dog walks", "Similar"),
        ("The big dog runs", "Related"),
        ("The small cat walks", "Related"),
        ("The bird jumps", "Different"),
    ]

    # Encode all sentences
    encodings = []
    for sentence, _ in test_sentences:
        enc = encode_sentence(sentence, model, parser, device)
        encodings.append((sentence, enc))

    # Compare first sentence to all others
    base_sentence, base_enc = encodings[0]
    print(f"  Base sentence: '{base_sentence}'")
    print()

    for sentence, enc in encodings[1:]:
        similarity = compute_sentence_similarity(base_enc, enc)
        print(f"    vs '{sentence:25s}': {similarity:.4f}")

    print()

    # All pairwise comparisons
    print("  Pairwise sentence similarities:")
    print()
    for i, (sent1, enc1) in enumerate(encodings):
        for j, (sent2, enc2) in enumerate(encodings):
            if i < j:
                sim = compute_sentence_similarity(enc1, enc2)
                print(f"    '{sent1}' <-> '{sent2}': {sim:.4f}")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print()
    print("Notes:")
    print("  - Word embeddings learned from both tree structure AND distance constraints")
    print("  - Sentence encodings use Tree-LSTM composition")
    print("  - Higher similarity = more semantically related")
    print()


if __name__ == "__main__":
    main()
