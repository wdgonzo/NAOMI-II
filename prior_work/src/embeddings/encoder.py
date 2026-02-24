"""
Parse Tree Encoder

Composes parse trees into semantic vectors through recursive composition.

The key insight: PARSE TREE STRUCTURE determines HOW vectors compose.
- Edge types define composition operations
- Tree hierarchy is preserved in the vector
- Meaning emerges from grammatical relationships

This is NOT token-based averaging - it's structure-based composition!
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from ..parser.data_structures import Hypothesis, Node, Edge
from ..parser.enums import ConnectionType, NodeType
from .anchors import AnchorDimensions


def compose_vectors(parent_vec: np.ndarray, child_vec: np.ndarray,
                   edge_type: ConnectionType,
                   anchors: AnchorDimensions) -> np.ndarray:
    """
    Compose two vectors based on their grammatical relationship.

    Different edge types use different composition operations to preserve
    the semantic nature of the relationship.

    Args:
        parent_vec: Parent node vector
        child_vec: Child node vector
        edge_type: Type of edge connecting them
        anchors: Anchor dimensions object

    Returns:
        Composed vector
    """

    # SUBJECT edge: child fills AGENT role in parent
    if edge_type == ConnectionType.SUBJECT:
        result = parent_vec.copy()
        # Activate subject/agent dimension
        subj_idx = anchors.get_dimension_index("subject")
        if subj_idx >= 0:
            result[subj_idx] = 1.0
        # Add child vector weighted to role dimensions
        # This "fills" the agent role with the child's semantics
        result = result + 0.5 * child_vec
        return result

    # OBJECT edge: child fills PATIENT role in parent
    elif edge_type == ConnectionType.OBJECT:
        result = parent_vec.copy()
        # Activate object/patient dimension
        obj_idx = anchors.get_dimension_index("objects")
        if obj_idx >= 0:
            result[obj_idx] = 1.0
        # Add child vector weighted
        result = result + 0.5 * child_vec
        return result

    # DESCRIPTION edge: child CONJUNCTS with parent (AND operation)
    # "big red dog" = big AND red AND dog
    elif edge_type == ConnectionType.DESCRIPTION:
        # Element-wise multiplication approximates logical AND
        # Both properties must be present
        and_idx = anchors.get_dimension_index("AND")
        result = parent_vec * child_vec
        if and_idx >= 0:
            result[and_idx] = 1.0
        return result

    # MODIFICATION edge: child modifies scope of parent
    elif edge_type == ConnectionType.MODIFICATION:
        result = parent_vec.copy()
        # Activate manner dimension
        manner_idx = anchors.get_dimension_index("manner")
        if manner_idx >= 0:
            result[manner_idx] = 1.0
        # Add modifier semantics to scope dimensions
        result = result + 0.3 * child_vec
        return result

    # SPECIFICATION edge: child specifies degree/extent
    elif edge_type == ConnectionType.SPECIFICATION:
        result = parent_vec.copy()
        extent_idx = anchors.get_dimension_index("extent")
        if extent_idx >= 0:
            result[extent_idx] = np.dot(parent_vec, child_vec)  # Use similarity as degree
        # Scale parent by specifier
        result = result * (1.0 + 0.2 * np.mean(child_vec))
        return result

    # COORDINATION edge: child coordinates with parent (OR operation)
    # "cats or dogs" = cats OR dogs
    elif edge_type == ConnectionType.COORDINATION:
        # Average approximates logical OR (either could be true)
        or_idx = anchors.get_dimension_index("OR")
        result = (parent_vec + child_vec) / 2.0
        if or_idx >= 0:
            result[or_idx] = 1.0
        return result

    # PREPOSITION edge: prepositional relationship
    elif edge_type == ConnectionType.PREPOSITION:
        result = parent_vec.copy()
        # Activate location dimension (default for PPs)
        loc_idx = anchors.get_dimension_index("location")
        if loc_idx >= 0:
            result[loc_idx] = 1.0
        # Add prepositional object semantics
        result = result + 0.4 * child_vec
        return result

    # SUBORDINATION edge: subordinate clause modifies parent
    elif edge_type == ConnectionType.SUBORDINATION:
        result = parent_vec.copy()
        # Add subordinate clause semantics
        result = result + 0.6 * child_vec
        return result

    # Default: simple addition
    else:
        return parent_vec + 0.5 * child_vec


def compose_tree(node_id: int, hypothesis: Hypothesis,
                word_embeddings: Dict[str, np.ndarray],
                anchors: AnchorDimensions,
                visited: Optional[set] = None) -> np.ndarray:
    """
    Recursively compose a parse tree into a single semantic vector.

    This is the core of structure-based encoding:
    1. Start at a node
    2. Get its word embedding
    3. For each child, recursively compose its subtree
    4. Combine child vectors using edge-type-specific operations
    5. Return composed vector

    Args:
        node_id: ID of node to compose from
        hypothesis: Parse hypothesis containing tree
        word_embeddings: Dict mapping words to embedding vectors
        anchors: Anchor dimensions object
        visited: Set of visited node IDs (to avoid cycles)

    Returns:
        Composed semantic vector for this subtree
    """
    if visited is None:
        visited = set()

    if node_id in visited:
        # Avoid cycles (shouldn't happen in tree, but safety check)
        embedding_dim = len(next(iter(word_embeddings.values())))
        return np.zeros(embedding_dim)

    visited.add(node_id)

    node = hypothesis.nodes[node_id]

    # Get base vector for this node
    if node.value and hasattr(node.value, 'text'):
        word_text = node.value.text.lower()
        if word_text in word_embeddings:
            base_vector = word_embeddings[word_text].copy()
        else:
            # Unknown word - use zero vector
            embedding_dim = len(next(iter(word_embeddings.values())))
            base_vector = np.zeros(embedding_dim)
    else:
        # Constituent node without word - start with zero
        embedding_dim = len(next(iter(word_embeddings.values())))
        base_vector = np.zeros(embedding_dim)

    # Find all edges where this node is the parent
    child_edges = [e for e in hypothesis.edges if e.parent == node_id]

    # If no children, return base vector
    if not child_edges:
        return base_vector

    # Compose with each child
    result = base_vector
    for edge in child_edges:
        # Recursively compose child subtree
        child_vector = compose_tree(edge.child, hypothesis, word_embeddings,
                                    anchors, visited)

        # Compose with parent using edge-specific operation
        result = compose_vectors(result, child_vector, edge.type, anchors)

    return result


def encode_hypothesis(hypothesis: Hypothesis,
                     word_embeddings: Dict[str, np.ndarray],
                     anchors: AnchorDimensions) -> np.ndarray:
    """
    Encode an entire parse hypothesis into a single semantic vector.

    This finds the root node (typically a CLAUSE) and composes the entire
    tree from there.

    Args:
        hypothesis: Parse hypothesis to encode
        word_embeddings: Dict mapping words to embeddings
        anchors: Anchor dimensions object

    Returns:
        Semantic vector representing the entire sentence/clause
    """

    # Find root node (node with no parent edges)
    all_children = set(e.child for e in hypothesis.edges)
    root_candidates = [i for i, node in enumerate(hypothesis.nodes)
                      if i not in all_children]

    if not root_candidates:
        # No clear root - use first unconsumed node
        unconsumed = hypothesis.get_unconsumed()
        if unconsumed:
            root_id = hypothesis.nodes.index(unconsumed[0])
        else:
            root_id = 0
    else:
        # Use first root candidate
        root_id = root_candidates[0]

    # Compose from root
    return compose_tree(root_id, hypothesis, word_embeddings, anchors)


def encode_multiple_hypotheses(hypotheses: List[Hypothesis],
                               word_embeddings: Dict[str, np.ndarray],
                               anchors: AnchorDimensions) -> List[Tuple[np.ndarray, float]]:
    """
    Encode multiple hypotheses and return with their scores.

    Args:
        hypotheses: List of parse hypotheses
        word_embeddings: Word embeddings
        anchors: Anchor dimensions

    Returns:
        List of (vector, score) tuples
    """
    results = []
    for hyp in hypotheses:
        vector = encode_hypothesis(hyp, word_embeddings, anchors)
        results.append((vector, hyp.score))

    return results


def batch_encode_sentences(sentences: List[str],
                           parser,
                           word_embeddings: Dict[str, np.ndarray],
                           anchors: AnchorDimensions) -> np.ndarray:
    """
    Batch encode a list of sentences.

    Args:
        sentences: List of sentence strings
        parser: Parser object with parse() method
        word_embeddings: Word embeddings
        anchors: Anchor dimensions

    Returns:
        Matrix of shape (num_sentences, embedding_dim)
    """
    vectors = []

    for sentence in sentences:
        # Parse sentence
        from ..parser.pos_tagger import tag_sentence
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        best = chart.best_hypothesis()

        # Encode
        vector = encode_hypothesis(best, word_embeddings, anchors)
        vectors.append(vector)

    return np.array(vectors)


def get_word_context_vector(word: str, sentence: str, parser,
                            word_embeddings: Dict[str, np.ndarray],
                            anchors: AnchorDimensions) -> np.ndarray:
    """
    Get the contextualized vector for a word in a sentence.

    This parses the sentence and extracts the composed vector for the
    specific word node, showing how context affects meaning.

    Args:
        word: Target word
        sentence: Sentence containing the word
        parser: Parser object
        word_embeddings: Word embeddings
        anchors: Anchor dimensions

    Returns:
        Contextualized vector for the word
    """
    from ..parser.pos_tagger import tag_sentence

    # Parse sentence
    words = tag_sentence(sentence)
    chart = parser.parse(words)
    best = chart.best_hypothesis()

    # Find node containing target word
    target_node_id = None
    for i, node in enumerate(best.nodes):
        if node.value and hasattr(node.value, 'text'):
            if node.value.text.lower() == word.lower():
                target_node_id = i
                break

    if target_node_id is None:
        # Word not found - return base embedding
        return word_embeddings.get(word.lower(), np.zeros(len(next(iter(word_embeddings.values())))))

    # Compose subtree for this node
    return compose_tree(target_node_id, best, word_embeddings, anchors)
