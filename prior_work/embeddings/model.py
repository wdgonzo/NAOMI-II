"""
Embedding Model

The embedding model that learns word vectors such that:
1. Parse tree composition produces meaningful sentence vectors
2. Semantic relationships (WordNet constraints) are preserved
3. Cross-lingual alignments hold (for bilingual training)

Key design:
- Anchor dimensions (first N dims) stay relatively fixed
- Learned dimensions (remaining dims) are trained
- Dimensionality is flexible - determined by what's needed
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

from .anchors import AnchorDimensions, get_anchor_vector
from ..parser.enums import SubType


class EmbeddingModel:
    """
    Embedding model for learning semantic word vectors.

    Attributes:
        embedding_dim: Total embedding dimensions
        num_anchors: Number of fixed anchor dimensions
        vocabulary: List of words in vocabulary
        word_to_id: Mapping from word to integer ID
        embeddings: The actual embedding matrix (vocab_size x embedding_dim)
        anchors: AnchorDimensions object
    """

    def __init__(self, vocabulary: List[str], embedding_dim: int = None,
                 anchors: AnchorDimensions = None):
        """
        Initialize embedding model.

        Args:
            vocabulary: List of words to embed
            embedding_dim: Total dimensions (if None, determined automatically)
            anchors: AnchorDimensions object (if None, created new)
        """
        self.vocabulary = vocabulary
        self.word_to_id = {word: i for i, word in enumerate(vocabulary)}
        self.id_to_word = {i: word for i, word in enumerate(vocabulary)}

        # Initialize anchors
        if anchors is None:
            self.anchors = AnchorDimensions()
        else:
            self.anchors = anchors

        self.num_anchors = self.anchors.num_anchors()

        # Determine embedding dimensionality
        if embedding_dim is None:
            # Auto-determine: anchors + learned dimensions
            # Start with 200 learned dimensions, can expand later
            self.embedding_dim = self.num_anchors + 200
        else:
            self.embedding_dim = embedding_dim

        # Initialize embeddings with small random values
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> np.ndarray:
        """
        Initialize embedding matrix.

        Anchor dimensions start at zero, learned dimensions are random.
        """
        vocab_size = len(self.vocabulary)
        embeddings = np.random.randn(vocab_size, self.embedding_dim).astype(np.float32) * 0.01

        # Zero out anchor dimensions initially
        embeddings[:, :self.num_anchors] = 0.0

        return embeddings

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        word_id = self.word_to_id.get(word.lower())
        if word_id is None:
            return None
        return self.embeddings[word_id]

    def get_embeddings_dict(self) -> Dict[str, np.ndarray]:
        """Get embeddings as a dictionary."""
        return {word: self.embeddings[word_id]
                for word, word_id in self.word_to_id.items()}

    def set_embedding(self, word: str, vector: np.ndarray):
        """Set embedding for a word."""
        word_id = self.word_to_id.get(word.lower())
        if word_id is not None:
            self.embeddings[word_id] = vector

    def update_embeddings(self, gradients: np.ndarray, learning_rate: float = 0.01):
        """
        Update embeddings with gradients.

        Args:
            gradients: Gradient matrix (same shape as embeddings)
            learning_rate: Learning rate
        """
        # Update learned dimensions only (preserve anchors)
        self.embeddings[:, self.num_anchors:] -= learning_rate * gradients[:, self.num_anchors:]

    def freeze_anchor_dimensions(self):
        """Ensure anchor dimensions don't change during training."""
        # This is already handled in update_embeddings, but can be called explicitly
        pass

    def expand_dimensions(self, num_new_dims: int):
        """
        Expand embedding dimensionality by adding new learned dimensions.

        This allows the model to discover it needs more capacity.

        Args:
            num_new_dims: Number of new dimensions to add
        """
        vocab_size = len(self.vocabulary)
        new_dims = np.random.randn(vocab_size, num_new_dims).astype(np.float32) * 0.01

        self.embeddings = np.concatenate([self.embeddings, new_dims], axis=1)
        self.embedding_dim += num_new_dims

        print(f"Expanded embeddings to {self.embedding_dim} dimensions")

    def normalize_embeddings(self):
        """Normalize embeddings to unit length (useful for cosine similarity)."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        self.embeddings = self.embeddings / norms

    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to a given word using cosine similarity.

        Args:
            word: Query word
            top_k: Number of results to return

        Returns:
            List of (word, similarity) tuples
        """
        word_vec = self.get_embedding(word)
        if word_vec is None:
            return []

        # Compute cosine similarities
        word_vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-8)
        all_vecs_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        similarities = np.dot(all_vecs_norm, word_vec_norm)

        # Get top-k (excluding the word itself)
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            candidate_word = self.id_to_word[idx]
            if candidate_word.lower() != word.lower():
                results.append((candidate_word, similarities[idx]))

        return results

    def compute_distance(self, word1: str, word2: str) -> float:
        """
        Compute Euclidean distance between two word embeddings.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Distance (0 = identical, larger = more different)
        """
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)

        if vec1 is None or vec2 is None:
            return float('inf')

        return np.linalg.norm(vec1 - vec2)

    def compute_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two word embeddings.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Similarity (-1 to 1, higher = more similar)
        """
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)

        if vec1 is None or vec2 is None:
            return 0.0

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def save(self, filepath: str):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'vocabulary': self.vocabulary,
            'embeddings': self.embeddings,
            'embedding_dim': self.embedding_dim,
            'num_anchors': self.num_anchors,
            'word_to_id': self.word_to_id,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'EmbeddingModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(vocabulary=data['vocabulary'], embedding_dim=data['embedding_dim'])
        model.embeddings = data['embeddings']
        model.num_anchors = data['num_anchors']
        model.word_to_id = data['word_to_id']
        model.id_to_word = {i: w for w, i in model.word_to_id.items()}

        print(f"Model loaded from {filepath}")
        return model

    def get_statistics(self) -> Dict:
        """Get model statistics."""
        return {
            'vocabulary_size': len(self.vocabulary),
            'embedding_dim': self.embedding_dim,
            'num_anchors': self.num_anchors,
            'num_learned_dims': self.embedding_dim - self.num_anchors,
            'embedding_norm_mean': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
            'embedding_norm_std': float(np.std(np.linalg.norm(self.embeddings, axis=1))),
        }

    def __repr__(self):
        return f"EmbeddingModel(vocab={len(self.vocabulary)}, " \
               f"dim={self.embedding_dim}, anchors={self.num_anchors})"
