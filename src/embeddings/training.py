"""
Embedding Training

Trains word embeddings using dual-source loss:
1. WordNet constraints (expert knowledge)
2. Parse composition coherence (structural patterns)

Guiding philosophy:
- Adverbs → amplification (learned, not rigid 2x)
- Adjectives → dimensional modification (emergent)
- Verbs → relationship establishment (from usage patterns)
- Structure determines composition operations
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .model import EmbeddingModel
from .constraints import ConstraintLoss
from .encoder import encode_hypothesis, compose_tree
from ..graph.knowledge_graph import KnowledgeGraph
from ..parser.data_structures import Hypothesis
from ..parser.enums import ConnectionType, NodeType


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)


def get_constituent_nodes(hypothesis: Hypothesis) -> List[int]:
    """
    Get IDs of major constituent nodes in a parse tree.

    These are nodes that represent meaningful sub-parts of the sentence.
    """
    constituents = []

    for i, node in enumerate(hypothesis.nodes):
        # Skip leaf nodes with just words
        if node.type in [NodeType.NOMINAL, NodeType.PREDICATE, NodeType.CLAUSE]:
            constituents.append(i)

    return constituents


def extract_structural_patterns(parsed_sentences: List[Hypothesis]) -> Dict[str, List[Tuple]]:
    """
    Extract structural patterns from parsed sentences.

    Returns dict mapping pattern types to instances.
    e.g., {"ADJ_NOUN": [(big, dog), (red, ball)], ...}
    """
    patterns = {
        'adj_noun': [],
        'adv_adj': [],
        'noun_verb': [],
        'description': [],
        'modification': [],
    }

    for hyp in parsed_sentences:
        for edge in hyp.edges:
            parent = hyp.nodes[edge.parent]
            child = hyp.nodes[edge.child]

            if edge.type == ConnectionType.DESCRIPTION:
                patterns['description'].append((hyp, edge.parent, edge.child))
            elif edge.type == ConnectionType.MODIFICATION:
                patterns['modification'].append((hyp, edge.parent, edge.child))
            elif edge.type == ConnectionType.SUBJECT:
                patterns['noun_verb'].append((hyp, edge.child, edge.parent))

    return patterns


def find_modification_pairs(parsed_sentences: List[Hypothesis]) -> List[Tuple[Hypothesis, Hypothesis]]:
    """
    Find pairs of (modified, base) phrases for amplification testing.

    e.g., ("very big", "big"), ("quickly runs", "runs")
    """
    # For now, return empty - would need more sophisticated pattern matching
    # This is a placeholder for future enhancement
    return []


def compute_composition_loss(model: EmbeddingModel,
                            parsed_sentences: List[Hypothesis],
                            anchors) -> float:
    """
    Compute composition loss - tests that tree composition produces coherent vectors.

    Three tests:
    1. Constituency preservation - parts relate to wholes
    2. Structural consistency - same patterns compose similarly
    3. Amplification intuition - modifications affect magnitude

    Args:
        model: Embedding model
        parsed_sentences: List of parsed hypotheses
        anchors: AnchorDimensions object

    Returns:
        Composition loss value
    """
    loss = 0.0
    num_tests = 0

    embeddings_dict = model.get_embeddings_dict()

    # Test 1: Constituency preservation
    for sentence in parsed_sentences:
        try:
            whole_vec = encode_hypothesis(sentence, embeddings_dict, anchors)

            # Get major constituents
            for constituent_id in get_constituent_nodes(sentence):
                constituent_vec = compose_tree(constituent_id, sentence, embeddings_dict, anchors)

                # Constituent should relate to whole (positive correlation)
                similarity = cosine_similarity(whole_vec, constituent_vec)

                if similarity < 0.3:  # Too dissimilar
                    loss += (0.3 - similarity) ** 2

                num_tests += 1
        except Exception:
            # Skip if encoding fails (e.g., unknown words)
            continue

    # Test 2: Structural consistency
    patterns = extract_structural_patterns(parsed_sentences)

    for pattern_type, instances in patterns.items():
        if len(instances) < 2:
            continue

        vectors = []
        for hyp, parent_id, child_id in instances[:10]:  # Sample max 10 per pattern
            try:
                parent_vec = compose_tree(parent_id, hyp, embeddings_dict, anchors)
                child_vec = compose_tree(child_id, hyp, embeddings_dict, anchors)

                # Compute relative vector (how child relates to parent)
                if np.linalg.norm(parent_vec) > 1e-8 and np.linalg.norm(child_vec) > 1e-8:
                    # Normalized difference captures structural relationship
                    rel_vec = (parent_vec / np.linalg.norm(parent_vec)) - (child_vec / np.linalg.norm(child_vec))
                    vectors.append(rel_vec)
            except Exception:
                continue

        # Same pattern should produce consistent relative vectors (low variance)
        if len(vectors) > 1:
            vectors_array = np.array(vectors)
            variance = np.var(vectors_array, axis=0).mean()
            loss += variance * 0.1  # Weight down - this is a soft constraint
            num_tests += 1

    # Test 3: Amplification test (placeholder for now)
    # Would test that modifications change vectors in consistent ways
    modification_pairs = find_modification_pairs(parsed_sentences)
    for modified, base in modification_pairs:
        try:
            vec_modified = encode_hypothesis(modified, embeddings_dict, anchors)
            vec_base = encode_hypothesis(base, embeddings_dict, anchors)

            mag_modified = np.linalg.norm(vec_modified)
            mag_base = np.linalg.norm(vec_base)

            if mag_modified <= mag_base:
                loss += (mag_base - mag_modified) ** 2 * 0.5

            num_tests += 1
        except Exception:
            continue

    # Normalize by number of tests
    if num_tests > 0:
        loss = loss / num_tests

    return loss


def compute_regularization_loss(model: EmbeddingModel) -> float:
    """
    Compute L2 regularization loss on learned dimensions.

    Keeps embeddings from growing too large.
    """
    learned_embeddings = model.embeddings[:, model.num_anchors:]
    return np.sum(learned_embeddings ** 2) / learned_embeddings.size


def compute_gradients_finite_diff(model: EmbeddingModel,
                                  loss_fn,
                                  step_size: float = 0.001,
                                  sample_size: int = 50) -> np.ndarray:
    """
    Compute gradients via finite differences.

    Only perturbs learned dimensions (preserves anchors).
    Samples a subset of words per iteration for efficiency.

    Args:
        model: Embedding model
        loss_fn: Function that takes model and returns scalar loss
        step_size: Perturbation step size
        sample_size: Number of words to sample per iteration

    Returns:
        Gradient matrix (same shape as embeddings)
    """
    gradients = np.zeros_like(model.embeddings)

    # Sample words
    num_words = len(model.vocabulary)
    sample_size = min(sample_size, num_words)
    word_ids = np.random.choice(num_words, sample_size, replace=False)

    # Also sample dimensions for very large embedding spaces
    num_learned_dims = model.embedding_dim - model.num_anchors
    dim_sample_size = min(50, num_learned_dims)  # Max 50 dims per iteration
    learned_dims = np.random.choice(num_learned_dims, dim_sample_size, replace=False) + model.num_anchors

    for word_id in word_ids:
        for dim in learned_dims:
            # Perturb up
            model.embeddings[word_id, dim] += step_size
            loss_plus = loss_fn(model)

            # Perturb down
            model.embeddings[word_id, dim] -= 2 * step_size
            loss_minus = loss_fn(model)

            # Restore
            model.embeddings[word_id, dim] += step_size

            # Gradient approximation
            gradients[word_id, dim] = (loss_plus - loss_minus) / (2 * step_size)

    return gradients


def train_embeddings(model: EmbeddingModel,
                    knowledge_graph: KnowledgeGraph,
                    parsed_sentences: List[Hypothesis],
                    parser,
                    num_epochs: int = 100,
                    learning_rate: float = 0.01,
                    batch_size: int = 32,
                    validation_sentences: Optional[List[str]] = None,
                    checkpoint_dir: str = 'checkpoints') -> Dict[str, List[float]]:
    """
    Train embeddings with dual-source loss.

    Args:
        model: Embedding model to train
        knowledge_graph: Graph with WordNet + parse triples
        parsed_sentences: Pre-parsed training sentences
        parser: Parser object for validation
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size (not used yet, for future mini-batch impl)
        validation_sentences: Optional validation sentences
        checkpoint_dir: Directory to save checkpoints

    Returns:
        Training history dict
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup constraint loss
    constraint_loss_fn = ConstraintLoss()
    constraint_loss_fn.add_constraints_from_graph(knowledge_graph)

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Vocabulary: {len(model.vocabulary)} words")
    print(f"Embedding dimensions: {model.embedding_dim} ({model.num_anchors} anchors)")
    print(f"Constraints: {len(constraint_loss_fn.constraints)}")
    print(f"Training sentences: {len(parsed_sentences)}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")

    # Training history
    history = {
        'total_loss': [],
        'wordnet_loss': [],
        'composition_loss': [],
        'reg_loss': [],
        'satisfaction_rate': []
    }

    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Shuffle sentences
        random.shuffle(parsed_sentences)

        # Compute WordNet constraint loss
        wordnet_loss, stats = constraint_loss_fn.compute_total_loss(model)

        # Compute composition loss
        composition_loss = compute_composition_loss(model, parsed_sentences, model.anchors)

        # Compute regularization
        reg_loss = compute_regularization_loss(model)

        # Combined loss
        total_loss = (0.6 * wordnet_loss +
                     0.4 * composition_loss +
                     0.1 * reg_loss)

        # Store history
        history['total_loss'].append(total_loss)
        history['wordnet_loss'].append(wordnet_loss)
        history['composition_loss'].append(composition_loss)
        history['reg_loss'].append(reg_loss)
        history['satisfaction_rate'].append(stats['satisfaction_rate'])

        # Compute gradients
        def combined_loss_fn(m):
            wn_loss, _ = constraint_loss_fn.compute_total_loss(m)
            comp_loss = compute_composition_loss(m, parsed_sentences, m.anchors)
            r_loss = compute_regularization_loss(m)
            return 0.6 * wn_loss + 0.4 * comp_loss + 0.1 * r_loss

        gradients = compute_gradients_finite_diff(model, combined_loss_fn, step_size=0.001)

        # Update embeddings
        model.update_embeddings(gradients, learning_rate)

        # Periodic normalization
        if epoch % 10 == 0 and epoch > 0:
            model.normalize_embeddings()

        # Logging
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  - WordNet: {wordnet_loss:.4f} ({stats['satisfaction_rate']*100:.1f}% satisfied)")
            print(f"  - Composition: {composition_loss:.4f}")
            print(f"  - Regularization: {reg_loss:.4f}")

            # Validation
            if validation_sentences:
                validate(model, parser, validation_sentences, model.anchors)

            # Save best model
            if total_loss < best_loss:
                best_loss = total_loss
                model.save(f'{checkpoint_dir}/best_model.pkl')
                print(f"  [Saved best model: loss={best_loss:.4f}]")

        # Learning rate decay
        if epoch % 50 == 0 and epoch > 0:
            learning_rate *= 0.9
            print(f"  [Learning rate adjusted: {learning_rate:.5f}]")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final satisfaction rate: {history['satisfaction_rate'][-1]*100:.1f}%")

    return history


def validate(model: EmbeddingModel, parser, test_sentences: List[str], anchors):
    """
    Quick validation during training.

    Tests:
    - Synonym search for common words
    - Basic composition encoding
    """
    from ..parser.pos_tagger import tag_sentence

    print("\n  Validation:")

    # Test synonym search
    test_words = ['dog', 'cat', 'run', 'big', 'red']
    found_words = [w for w in test_words if model.get_embedding(w) is not None]

    if found_words:
        for word in found_words[:3]:  # Show first 3
            similar = model.get_similar_words(word, top_k=3)
            similar_words = [w for w, s in similar]
            print(f"    '{word}' -> {similar_words}")

    # Test composition
    if test_sentences:
        test_sent = test_sentences[0]
        try:
            words = tag_sentence(test_sent)
            chart = parser.parse(words)
            if chart.hypotheses:
                vec = encode_hypothesis(chart.best_hypothesis(),
                                       model.get_embeddings_dict(), anchors)
                print(f"    Encoded '{test_sent}': |v|={np.linalg.norm(vec):.3f}")
        except Exception as e:
            print(f"    Encoding failed: {e}")


def generate_training_corpus(num_sentences: int = 200) -> List[str]:
    """
    Generate controlled training sentences covering various constructions.

    Args:
        num_sentences: Target number of sentences

    Returns:
        List of sentence strings
    """
    sentences = []

    # Basic subject-verb
    subjects = ['the dog', 'the cat', 'the bird', 'a man', 'a woman', 'the child']
    verbs = ['runs', 'walks', 'sleeps', 'eats', 'jumps', 'sits']

    for subj in subjects:
        for verb in verbs:
            sentences.append(f"{subj} {verb}")

    # Adjective-noun
    adjectives = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'happy', 'sad']
    nouns = ['dog', 'cat', 'ball', 'car', 'house', 'tree', 'book']

    for adj in adjectives:
        for noun in nouns:
            sentences.append(f"the {adj} {noun}")
            if len(sentences) < num_sentences // 2:
                sentences.append(f"the {adj} {noun} runs")

    # With adverbs
    adverbs = ['quickly', 'slowly', 'very', 'really', 'extremely']
    for adv in adverbs:
        for adj in adjectives[:4]:
            sentences.append(f"the {adv} {adj} ball")

    # Complex sentences
    sentences.extend([
        "the big red dog runs quickly",
        "a small blue bird flies",
        "the cat and the dog",
        "dogs and cats run",
        "the fast car drives",
        "the happy child plays",
        "the sad man walks slowly",
    ])

    return sentences[:num_sentences]
