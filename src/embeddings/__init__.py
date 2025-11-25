"""
Embeddings module for semantic vector space training.

This module provides:
- Anchor dimension definitions (semantic/grammatical/logical)
- Parse tree encoder (tree → vector composition)
- Embedding model architecture
- Fuzzy constraint loss functions
- Training loop with dual-source loss
"""

from .anchors import AnchorDimensions, get_anchor_vector
from .encoder import encode_hypothesis, compose_tree
from .model import EmbeddingModel

__all__ = [
    'AnchorDimensions',
    'get_anchor_vector',
    'encode_hypothesis',
    'compose_tree',
    'EmbeddingModel',
]
