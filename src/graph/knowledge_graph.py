"""
Knowledge Graph

A graph data structure for storing semantic triples and word relationships.
Supports fuzzy relationships with confidence scores and relation type tracking.

This combines:
- Parse-derived triples (from parse trees)
- WordNet triples (expert-labeled relationships)
- Cross-lingual alignments (for bilingual training)
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import json

from .triple_extractor import SemanticTriple, RelationType


@dataclass
class GraphNode:
    """
    A node in the knowledge graph representing a word or concept.

    Attributes:
        word: The word text (e.g., "dog", "run")
        language: Language code ("en", "es", etc.)
        frequency: Number of times seen in corpus
        metadata: Additional information (POS tags, senses, etc.)
    """
    word: str
    language: str = "en"
    frequency: int = 0
    metadata: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash((self.word, self.language))

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.word == other.word and self.language == other.language

    def __repr__(self):
        return f"Node({self.word}[{self.language}])"


@dataclass
class GraphEdge:
    """
    An edge in the knowledge graph representing a relationship.

    Attributes:
        source: Source node
        target: Target node
        relation: Type of relationship
        confidence: Confidence score (0.0-1.0)
        source_type: Where this edge came from ("wordnet", "parse", "alignment")
    """
    source: GraphNode
    target: GraphNode
    relation: RelationType
    confidence: float = 1.0
    source_type: str = "parse"

    def __repr__(self):
        return f"{self.source} --[{self.relation.value}:{self.confidence:.2f}]--> {self.target}"


class KnowledgeGraph:
    """
    Knowledge graph for storing word relationships from multiple sources.

    This graph combines:
    1. Parse-derived relationships (from triple extraction)
    2. WordNet relationships (synonyms, hypernyms, etc.)
    3. Cross-lingual alignments (for translation)

    The graph supports:
    - Adding nodes and edges
    - Querying relationships
    - Computing statistics
    - Exporting for training
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.nodes: Dict[Tuple[str, str], GraphNode] = {}  # (word, lang) -> Node
        self.edges: List[GraphEdge] = []

        # Adjacency lists for fast lookup
        self.outgoing: Dict[GraphNode, List[GraphEdge]] = defaultdict(list)
        self.incoming: Dict[GraphNode, List[GraphEdge]] = defaultdict(list)

        # Relation type index
        self.edges_by_relation: Dict[RelationType, List[GraphEdge]] = defaultdict(list)

    def add_node(self, word: str, language: str = "en", **metadata) -> GraphNode:
        """
        Add a node to the graph or retrieve existing node.

        Args:
            word: The word text
            language: Language code
            **metadata: Additional metadata

        Returns:
            GraphNode object
        """
        key = (word, language)

        if key in self.nodes:
            # Update frequency
            self.nodes[key].frequency += 1
            # Merge metadata
            self.nodes[key].metadata.update(metadata)
            return self.nodes[key]

        node = GraphNode(word=word, language=language, frequency=1, metadata=metadata)
        self.nodes[key] = node
        return node

    def add_edge(self, source_word: str, target_word: str, relation: RelationType,
                 source_lang: str = "en", target_lang: str = "en",
                 confidence: float = 1.0, source_type: str = "parse") -> GraphEdge:
        """
        Add an edge to the graph.

        Args:
            source_word: Source word text
            target_word: Target word text
            relation: Relation type
            source_lang: Source language code
            target_lang: Target language code
            confidence: Confidence score (0.0-1.0)
            source_type: Source of this edge ("wordnet", "parse", "alignment")

        Returns:
            GraphEdge object
        """
        # Create or get nodes
        source_node = self.add_node(source_word, source_lang)
        target_node = self.add_node(target_word, target_lang)

        # Create edge
        edge = GraphEdge(
            source=source_node,
            target=target_node,
            relation=relation,
            confidence=confidence,
            source_type=source_type
        )

        self.edges.append(edge)
        self.outgoing[source_node].append(edge)
        self.incoming[target_node].append(edge)
        self.edges_by_relation[relation].append(edge)

        return edge

    def add_triple(self, triple: SemanticTriple, language: str = "en",
                   source_type: str = "parse") -> GraphEdge:
        """
        Add a semantic triple to the graph.

        Args:
            triple: SemanticTriple object
            language: Language code
            source_type: Source of this triple

        Returns:
            GraphEdge object
        """
        return self.add_edge(
            source_word=triple.subject,
            target_word=triple.object,
            relation=triple.relation,
            source_lang=language,
            target_lang=language,
            confidence=triple.confidence,
            source_type=source_type
        )

    def add_triples(self, triples: List[SemanticTriple], language: str = "en",
                    source_type: str = "parse") -> List[GraphEdge]:
        """Add multiple triples at once."""
        return [self.add_triple(t, language, source_type) for t in triples]

    def get_node(self, word: str, language: str = "en") -> Optional[GraphNode]:
        """Get a node by word and language."""
        return self.nodes.get((word, language))

    def get_outgoing_edges(self, word: str, language: str = "en",
                          relation: Optional[RelationType] = None) -> List[GraphEdge]:
        """
        Get all outgoing edges from a word.

        Args:
            word: Source word
            language: Language code
            relation: Optional filter by relation type

        Returns:
            List of outgoing edges
        """
        node = self.get_node(word, language)
        if not node:
            return []

        edges = self.outgoing[node]

        if relation:
            edges = [e for e in edges if e.relation == relation]

        return edges

    def get_incoming_edges(self, word: str, language: str = "en",
                          relation: Optional[RelationType] = None) -> List[GraphEdge]:
        """Get all incoming edges to a word."""
        node = self.get_node(word, language)
        if not node:
            return []

        edges = self.incoming[node]

        if relation:
            edges = [e for e in edges if e.relation == relation]

        return edges

    def get_neighbors(self, word: str, language: str = "en",
                     relation: Optional[RelationType] = None) -> Set[str]:
        """
        Get all neighboring words (both incoming and outgoing).

        Returns:
            Set of neighbor word strings
        """
        outgoing = self.get_outgoing_edges(word, language, relation)
        incoming = self.get_incoming_edges(word, language, relation)

        neighbors = set()
        neighbors.update(e.target.word for e in outgoing)
        neighbors.update(e.source.word for e in incoming)

        return neighbors

    def get_related_by_type(self, word: str, language: str = "en",
                           relation: RelationType = None) -> List[str]:
        """Get words related by a specific relation type (outgoing only)."""
        edges = self.get_outgoing_edges(word, language, relation)
        return [e.target.word for e in edges]

    def get_statistics(self) -> Dict:
        """
        Compute graph statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_languages': len(set(n.language for n in self.nodes.values())),
            'edges_by_relation': {
                rel.value: len(edges)
                for rel, edges in self.edges_by_relation.items()
            },
            'edges_by_source': defaultdict(int),
            'avg_degree': 0,
        }

        # Count by source type
        for edge in self.edges:
            stats['edges_by_source'][edge.source_type] += 1

        # Average degree
        if self.nodes:
            total_degree = sum(len(self.outgoing[n]) + len(self.incoming[n])
                             for n in self.nodes.values())
            stats['avg_degree'] = total_degree / len(self.nodes)

        return stats

    def export_for_training(self) -> Tuple[List[Tuple], Dict]:
        """
        Export graph in format suitable for embedding training.

        Returns:
            - List of (source_id, relation_id, target_id, confidence) tuples
            - Dict mapping words to integer IDs
        """
        # Create vocabulary
        vocab = {}
        vocab_idx = 0

        for node in self.nodes.values():
            key = f"{node.word}_{node.language}"
            if key not in vocab:
                vocab[key] = vocab_idx
                vocab_idx += 1

        # Create relation vocabulary
        relation_vocab = {rel: idx for idx, rel in enumerate(RelationType)}

        # Export edges
        edge_list = []
        for edge in self.edges:
            source_key = f"{edge.source.word}_{edge.source.language}"
            target_key = f"{edge.target.word}_{edge.target.language}"

            edge_tuple = (
                vocab[source_key],
                relation_vocab[edge.relation],
                vocab[target_key],
                edge.confidence
            )
            edge_list.append(edge_tuple)

        return edge_list, vocab

    def save(self, filepath: str):
        """Save graph to JSON file."""
        data = {
            'nodes': [
                {
                    'word': node.word,
                    'language': node.language,
                    'frequency': node.frequency,
                    'metadata': node.metadata
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source.word,
                    'target': edge.target.word,
                    'source_lang': edge.source.language,
                    'target_lang': edge.target.language,
                    'relation': edge.relation.value,
                    'confidence': edge.confidence,
                    'source_type': edge.source_type
                }
                for edge in self.edges
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = cls()

        # Add nodes
        for node_data in data['nodes']:
            graph.add_node(**node_data)

        # Add edges
        for edge_data in data['edges']:
            # Convert relation string back to enum
            relation = RelationType(edge_data['relation'])

            graph.add_edge(
                source_word=edge_data['source'],
                target_word=edge_data['target'],
                relation=relation,
                source_lang=edge_data['source_lang'],
                target_lang=edge_data['target_lang'],
                confidence=edge_data['confidence'],
                source_type=edge_data['source_type']
            )

        return graph

    def __repr__(self):
        return f"KnowledgeGraph({len(self.nodes)} nodes, {len(self.edges)} edges)"
