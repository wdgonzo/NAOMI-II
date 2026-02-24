"""
Semantic Triple Extractor

Converts parse trees (Hypothesis objects) into semantic triples for knowledge graph construction.

A semantic triple is: (subject, relation, object)
Example: (dog, is-agent-of, runs)

This flattens the hierarchical parse tree into word-to-word relationships
that can be used for training embeddings and building knowledge graphs.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple
from enum import Enum

from ..parser.data_structures import Hypothesis, Node, Edge
from ..parser.enums import ConnectionType


class RelationType(Enum):
    """Semantic relation types derived from parse tree edge types."""

    # Core argument relations
    IS_AGENT_OF = "is-agent-of"           # Subject of action
    IS_PATIENT_OF = "is-patient-of"       # Object of action
    IS_THEME_OF = "is-theme-of"           # Theme/topic
    IS_EXPERIENCER_OF = "is-experiencer-of"  # Experiences state

    # Modification relations
    MODIFIES = "modifies"                 # General modification
    MODIFIES_MANNER = "modifies-manner"   # How something happens
    MODIFIES_DEGREE = "modifies-degree"   # Intensity/degree

    # Nominal relations
    DETERMINES = "determines"             # Determiner relation
    DESCRIBES = "describes"               # Adjective/descriptor
    POSSESSES = "possesses"              # Possessive relation

    # Prepositional relations
    HAS_LOCATION = "has-location"
    HAS_TIME = "has-time"
    HAS_INSTRUMENT = "has-instrument"
    HAS_SOURCE = "has-source"
    HAS_GOAL = "has-goal"
    HAS_BENEFICIARY = "has-beneficiary"

    # Structural relations
    COORDINATES_WITH = "coordinates-with"  # Coordination (and/or)
    SUBORDINATES = "subordinates"         # Subordinate clause

    # Comparative relations
    COMPARED_TO = "compared-to"

    # Other
    GENERIC_RELATION = "related-to"       # Catch-all


@dataclass
class SemanticTriple:
    """
    A semantic triple representing a relationship between two words.

    Attributes:
        subject: The source word (as string)
        relation: The type of relationship
        object: The target word (as string)
        confidence: Optional confidence score (0.0-1.0)
    """
    subject: str
    relation: RelationType
    object: str
    confidence: float = 1.0

    def __repr__(self):
        return f"({self.subject} --[{self.relation.value}]--> {self.object})"

    def to_tuple(self) -> Tuple[str, str, str]:
        """Return as simple (subject, relation, object) tuple."""
        return (self.subject, self.relation.value, self.object)


def _map_edge_to_relation(edge_type: ConnectionType, preposition: str = None) -> RelationType:
    """
    Map parse tree edge types to semantic relation types.

    Args:
        edge_type: The ConnectionType from the parse tree
        preposition: Optional preposition word for PP relations

    Returns:
        Corresponding RelationType
    """
    mapping = {
        ConnectionType.SUBJECT: RelationType.IS_AGENT_OF,
        ConnectionType.OBJECT: RelationType.IS_PATIENT_OF,
        ConnectionType.SUBJECT_COMPLEMENT: RelationType.IS_THEME_OF,
        ConnectionType.DESCRIPTION: RelationType.DESCRIBES,
        ConnectionType.MODIFICATION: RelationType.MODIFIES_MANNER,
        ConnectionType.SPECIFICATION: RelationType.MODIFIES_DEGREE,
        ConnectionType.COORDINATION: RelationType.COORDINATES_WITH,
        ConnectionType.SUBORDINATION: RelationType.SUBORDINATES,
    }

    # Special handling for prepositional phrases
    if edge_type == ConnectionType.PREPOSITION and preposition:
        # Map common prepositions to semantic relations
        prep_mapping = {
            'in': RelationType.HAS_LOCATION,
            'on': RelationType.HAS_LOCATION,
            'at': RelationType.HAS_LOCATION,
            'to': RelationType.HAS_GOAL,
            'from': RelationType.HAS_SOURCE,
            'with': RelationType.HAS_INSTRUMENT,
            'for': RelationType.HAS_BENEFICIARY,
            'by': RelationType.HAS_INSTRUMENT,
            'during': RelationType.HAS_TIME,
            'before': RelationType.HAS_TIME,
            'after': RelationType.HAS_TIME,
        }
        return prep_mapping.get(preposition.lower(), RelationType.HAS_LOCATION)

    return mapping.get(edge_type, RelationType.GENERIC_RELATION)


def _get_head_word(node: Node, hypothesis: Hypothesis) -> str:
    """
    Extract the head word from a node.

    For leaf nodes with word values, returns the word text.
    For constituent nodes, finds the head child recursively.
    For EQUIVALENCE predicates (single nouns), returns special marker.

    Args:
        node: The node to extract head word from
        hypothesis: The hypothesis containing the node

    Returns:
        Head word as string
    """
    # Check if this is an EQUIVALENCE predicate (from single noun normalization)
    from ..parser.enums import SubType
    if hasattr(node, 'flags') and SubType.EQUIVALENCE in node.flags:
        return "[EQUIVALENCE]"  # Logical operator, not a word

    # If node has a word value, return it
    if node.value and hasattr(node.value, 'text'):
        return node.value.text

    # Find the head child (first unconsumed child or first child)
    node_id = hypothesis.nodes.index(node)
    children = [e for e in hypothesis.edges if e.parent == node_id]

    if not children:
        # No children, return node type as placeholder
        return f"[{node.type.value}]"

    # For most relations, the child is the head
    # But for SUBJECT/OBJECT, the parent is the head
    first_child_id = children[0].child
    first_child = hypothesis.nodes[first_child_id]

    return _get_head_word(first_child, hypothesis)


def _extract_preposition(node: Node, hypothesis: Hypothesis) -> str:
    """Extract preposition word from a prepositional phrase node."""
    # Look for a child with original_type PREP
    node_id = hypothesis.nodes.index(node)

    for edge in hypothesis.edges:
        if edge.parent == node_id:
            child = hypothesis.nodes[edge.child]
            if child.value and hasattr(child.value, 'tag'):
                from ..parser.enums import Tag
                if child.value.tag == Tag.PREP:
                    return child.value.text

    return None


def extract_triples(hypothesis: Hypothesis) -> List[SemanticTriple]:
    """
    Extract semantic triples from a parse tree hypothesis.

    This converts the hierarchical tree structure into flat word-to-word
    relationships suitable for knowledge graph construction.

    Args:
        hypothesis: A parsed hypothesis containing nodes and edges

    Returns:
        List of SemanticTriple objects

    Example:
        >>> # Parse: "The big dog runs quickly"
        >>> # Tree: runs(CLAUSE) --SUBJECT--> dog(NOMINAL) --DESCRIPTION--> big
        >>> #                    --MODIFICATION--> quickly
        >>> triples = extract_triples(hypothesis)
        >>> print(triples)
        [(dog, is-agent-of, runs),
         (big, describes, dog),
         (the, determines, dog),
         (quickly, modifies-manner, runs)]
    """
    triples = []

    for edge in hypothesis.edges:
        parent_node = hypothesis.nodes[edge.parent]
        child_node = hypothesis.nodes[edge.child]

        # Get head words
        parent_word = _get_head_word(parent_node, hypothesis)
        child_word = _get_head_word(child_node, hypothesis)

        # Map edge type to relation
        preposition = None
        if edge.type == ConnectionType.PREPOSITION:
            preposition = _extract_preposition(child_node, hypothesis)

        relation = _map_edge_to_relation(edge.type, preposition)

        # Create triple based on relation semantics
        # For SUBJECT/OBJECT, child is agent/patient OF parent
        if relation in [RelationType.IS_AGENT_OF, RelationType.IS_PATIENT_OF,
                       RelationType.IS_THEME_OF, RelationType.IS_EXPERIENCER_OF]:
            triple = SemanticTriple(
                subject=child_word,
                relation=relation,
                object=parent_word
            )
        else:
            # For modification/description, child MODIFIES/DESCRIBES parent
            triple = SemanticTriple(
                subject=child_word,
                relation=relation,
                object=parent_word
            )

        triples.append(triple)

    return triples


def extract_triples_recursive(hypothesis: Hypothesis, include_transitive: bool = False) -> List[SemanticTriple]:
    """
    Extract triples including transitive relationships.

    For example, if "big" describes "dog" and "dog" is-agent-of "runs",
    we might also extract (big, modifies-agent-of, runs).

    Args:
        hypothesis: A parsed hypothesis
        include_transitive: Whether to include transitive relationships

    Returns:
        List of SemanticTriple objects
    """
    # Start with direct triples
    triples = extract_triples(hypothesis)

    if not include_transitive:
        return triples

    # TODO: Add transitive relationship extraction if needed
    # This could be useful for richer semantic graphs

    return triples


def triples_to_dict(triples: List[SemanticTriple]) -> dict:
    """
    Convert list of triples to dictionary format for easier manipulation.

    Returns:
        Dict mapping (subject, object) -> list of relations
    """
    result = {}
    for triple in triples:
        key = (triple.subject, triple.object)
        if key not in result:
            result[key] = []
        result[key].append(triple.relation)

    return result


def filter_triples_by_relation(triples: List[SemanticTriple],
                               relation_types: Set[RelationType]) -> List[SemanticTriple]:
    """Filter triples to only include specific relation types."""
    return [t for t in triples if t.relation in relation_types]
