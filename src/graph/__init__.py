"""
Graph module for semantic triple extraction and knowledge graph operations.

This module provides:
- Triple extraction from parse trees
- Knowledge graph data structures
- WordNet import utilities
"""

from .triple_extractor import extract_triples, SemanticTriple, RelationType
from .knowledge_graph import KnowledgeGraph, GraphNode, GraphEdge
from .wordnet_import import (
    import_wordnet_for_word,
    import_wordnet_for_vocabulary,
    add_wordnet_constraints,
    build_wordnet_graph,
    get_wordnet_synonyms,
    get_wordnet_antonyms,
)

__all__ = [
    # Triple extraction
    'extract_triples',
    'SemanticTriple',
    'RelationType',

    # Knowledge graph
    'KnowledgeGraph',
    'GraphNode',
    'GraphEdge',

    # WordNet import
    'import_wordnet_for_word',
    'import_wordnet_for_vocabulary',
    'add_wordnet_constraints',
    'build_wordnet_graph',
    'get_wordnet_synonyms',
    'get_wordnet_antonyms',
]
