"""
Parse Tree Normalizer

Post-processes parse hypotheses to add implied elements and markers:
1. Implied subjects for imperatives ("Run!" → "you run")
2. Passive voice markers for passive constructions
3. Equivalence predicates for single nouns (using XNOR logical anchor)

These normalizations create consistent parse structures without adding
learned words - using logical anchors and grammatical features instead.
"""

from typing import Optional
from .data_structures import Hypothesis, Node, Edge
from .enums import NodeType, ConnectionType, SubType, Tag
from .pos_tagger import Word


def normalize_hypothesis(hypothesis: Optional[Hypothesis]) -> Optional[Hypothesis]:
    """
    Normalize a parse hypothesis by adding implied elements.

    Args:
        hypothesis: Parse hypothesis to normalize (may be None)

    Returns:
        Normalized hypothesis (or None if input was None)
    """
    if hypothesis is None or hypothesis.score <= 0:
        return hypothesis

    # Apply normalizations in sequence
    hypothesis = add_implied_subject_to_imperatives(hypothesis)
    hypothesis = mark_passive_voice(hypothesis)
    hypothesis = add_equivalence_to_single_nouns(hypothesis)

    return hypothesis


def add_implied_subject_to_imperatives(hypothesis: Hypothesis) -> Hypothesis:
    """
    Add implied "you" subject to imperative sentences.

    Detects predicates without subjects and adds an implied "you" nominal.

    Args:
        hypothesis: Parse hypothesis

    Returns:
        Hypothesis with implied subjects added
    """
    # Find PREDICATE nodes without subjects
    for node_idx, node in enumerate(hypothesis.nodes):
        if node.type != NodeType.PREDICATE:
            continue

        # Check if this predicate has a subject edge
        has_subject = any(
            e.parent == node_idx and e.type == ConnectionType.SUBJECT
            for e in hypothesis.edges
        )

        if not has_subject:
            # This is an imperative - add implied "you" subject
            _add_implied_you_subject(hypothesis, node_idx)

    return hypothesis


def _add_implied_you_subject(hypothesis: Hypothesis, predicate_idx: int):
    """
    Add an implied "you" nominal as subject of predicate.

    Args:
        hypothesis: Parse hypothesis to modify
        predicate_idx: Index of predicate node needing subject
    """
    # Create "you" word
    you_word = Word(
        text="you",
        pos=Tag.PRON,
        subtypes=[SubType.SECOND_PERSON, SubType.SINGULAR]
    )

    # Create NOMINAL node for "you"
    you_node = Node(
        type=NodeType.NOMINAL,
        original_type=NodeType.NOMINAL,
        value=you_word,
        pos=Tag.PRON,
        flags=[SubType.SECOND_PERSON, SubType.SINGULAR],
        index=-1  # Constructed node
    )

    # Add node to hypothesis
    you_idx = len(hypothesis.nodes)
    hypothesis.nodes.append(you_node)

    # Create SUBJECT edge from predicate to "you"
    subject_edge = Edge(
        parent=predicate_idx,
        child=you_idx,
        type=ConnectionType.SUBJECT
    )

    hypothesis.edges.append(subject_edge)


def mark_passive_voice(hypothesis: Hypothesis) -> Hypothesis:
    """
    Mark passive voice constructions on verbal nodes.

    Detects passive constructions (auxiliary + past participle) and
    adds PASSIVE subtype to the verbal node.

    Args:
        hypothesis: Parse hypothesis

    Returns:
        Hypothesis with passive markers added
    """
    for node_idx, node in enumerate(hypothesis.nodes):
        if node.type != NodeType.VERBAL:
            continue

        # Check if this verbal is part of passive construction
        # Look for: be/was/were + past participle pattern
        if _is_passive_construction(hypothesis, node_idx, node):
            # Add PASSIVE subtype if not already present
            if SubType.PASSIVE not in node.flags:
                node.flags.append(SubType.PASSIVE)

    return hypothesis


def _is_passive_construction(hypothesis: Hypothesis, node_idx: int, node: Node) -> bool:
    """
    Detect if a verbal node is part of a passive construction.

    Looks for auxiliary "be" + past participle pattern.

    Args:
        hypothesis: Parse hypothesis
        node_idx: Index of verbal node
        node: The verbal node

    Returns:
        True if passive construction detected
    """
    # Check if node has past participle form
    if not node.value or not hasattr(node.value, 'text'):
        return False

    # Check for children that are auxiliaries
    child_edges = [e for e in hypothesis.edges if e.parent == node_idx]

    for edge in child_edges:
        if edge.type != ConnectionType.MODIFICATION:
            continue

        child_node = hypothesis.nodes[edge.child]

        # Check if child is auxiliary "be" form
        if child_node.value and hasattr(child_node.value, 'text'):
            word_text = child_node.value.text.lower()
            if word_text in ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being']:
                # Check if this node has past participle marker
                if SubType.PAST_PARTICIPLE in node.flags:
                    return True

    return False


def add_equivalence_to_single_nouns(hypothesis: Hypothesis) -> Hypothesis:
    """
    Add EQUIVALENCE predicate to single-noun parse trees.

    When parse tree is just a bare noun (no predicate/clause), adds
    an EQUIVALENCE predicate that activates the XNOR logical anchor.
    This represents "dog" as "dog [EQUIVALENCE]" without adding the word "is".

    Args:
        hypothesis: Parse hypothesis

    Returns:
        Hypothesis with equivalence predicates added
    """
    # Check if this is a single-noun tree (no PREDICATE or CLAUSE nodes)
    has_predicate = any(
        n.type in [NodeType.PREDICATE, NodeType.CLAUSE]
        for n in hypothesis.nodes
    )

    if has_predicate:
        # Already has predicate structure, no need to add equivalence
        return hypothesis

    # Check if we have exactly one NOMINAL node as root
    nominal_nodes = [
        (idx, n) for idx, n in enumerate(hypothesis.nodes)
        if n.type == NodeType.NOMINAL
    ]

    if len(nominal_nodes) != 1:
        # Not a simple single-noun case
        return hypothesis

    nominal_idx, nominal_node = nominal_nodes[0]

    # Add EQUIVALENCE predicate
    _add_equivalence_predicate(hypothesis, nominal_idx)

    return hypothesis


def _add_equivalence_predicate(hypothesis: Hypothesis, nominal_idx: int):
    """
    Add an EQUIVALENCE predicate for a nominal node.

    Creates a PREDICATE node that represents the XNOR (equivalence)
    logical operator, without adding the word "is" to vocabulary.

    Args:
        hypothesis: Parse hypothesis to modify
        nominal_idx: Index of nominal node
    """
    # Create PREDICATE node representing EQUIVALENCE
    # This activates the XNOR anchor dimension, not a word
    equivalence_node = Node(
        type=NodeType.PREDICATE,
        original_type=NodeType.PREDICATE,
        value=None,  # No word - this is a logical operator
        pos=Tag.LOG,  # Logical operator tag
        flags=[SubType.EQUIVALENCE],  # Mark as equivalence predicate
        index=-1  # Constructed node
    )

    # Add node to hypothesis
    pred_idx = len(hypothesis.nodes)
    hypothesis.nodes.append(equivalence_node)

    # Create SUBJECT edge from predicate to nominal
    subject_edge = Edge(
        parent=pred_idx,
        child=nominal_idx,
        type=ConnectionType.SUBJECT
    )

    hypothesis.edges.append(subject_edge)

    # Create CLAUSE node to complete structure
    clause_node = Node(
        type=NodeType.CLAUSE,
        original_type=NodeType.CLAUSE,
        value=None,
        pos=Tag.X,  # Unknown/constructed
        flags=[],
        index=-1  # Constructed node
    )

    clause_idx = len(hypothesis.nodes)
    hypothesis.nodes.append(clause_node)

    # Link clause to predicate
    clause_edge = Edge(
        parent=clause_idx,
        child=pred_idx,
        type=ConnectionType.COMPLEMENT
    )

    hypothesis.edges.append(clause_edge)
