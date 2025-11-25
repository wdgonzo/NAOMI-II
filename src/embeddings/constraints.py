"""
Fuzzy Semantic Constraints

Implements fuzzy constraint loss functions for training embeddings.

Key insight from user feedback:
- NOT hard equality constraints (dog = perro)
- USE range-based fuzzy constraints (distance(dog, perro) ∈ [0.1, 0.3])
- Preserves cross-linguistic nuance while ensuring semantic similarity

Constraint types:
- Synonyms: close but not identical (0.0-0.2)
- Antonyms: distant (0.7-1.0)
- Hypernyms: positive correlation (0.4-0.7)
- Cross-lingual: close with nuance (0.1-0.3)
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from ..graph.knowledge_graph import KnowledgeGraph, GraphEdge


@dataclass
class FuzzyConstraint:
    """
    A fuzzy constraint on the relationship between two words.

    Attributes:
        word1: First word
        word2: Second word
        constraint_type: Type of constraint (synonym, antonym, etc.)
        target_distance_min: Minimum acceptable distance
        target_distance_max: Maximum acceptable distance
        weight: Importance weight for this constraint
    """
    word1: str
    word2: str
    constraint_type: str
    target_distance_min: float
    target_distance_max: float
    weight: float = 1.0

    def compute_loss(self, distance: float) -> float:
        """
        Compute loss for this constraint given actual distance.

        Loss is 0 if distance is within range, increases quadratically outside.

        Args:
            distance: Actual distance between word embeddings

        Returns:
            Loss value (0 = satisfied, positive = violated)
        """
        if self.target_distance_min <= distance <= self.target_distance_max:
            # Constraint satisfied
            return 0.0
        elif distance < self.target_distance_min:
            # Too close
            violation = self.target_distance_min - distance
            return self.weight * (violation ** 2)
        else:
            # Too far
            violation = distance - self.target_distance_max
            return self.weight * (violation ** 2)


class ConstraintLoss:
    """
    Manages and computes fuzzy constraint losses for training.
    """

    def __init__(self):
        """Initialize constraint loss manager."""
        self.constraints: List[FuzzyConstraint] = []

        # Define constraint type ranges
        self.constraint_ranges = {
            'wordnet_synonym': (0.0, 0.2, 2.0),      # (min, max, weight)
            'wordnet_antonym': (0.7, 1.0, 1.5),
            'wordnet_hypernym': (0.4, 0.7, 1.0),
            'wordnet_hyponym': (0.4, 0.7, 1.0),
            'cross_lingual': (0.1, 0.3, 1.5),        # Close but with nuance
            'parse_derived': (0.3, 0.6, 0.8),        # Moderate similarity
        }

    def add_constraint(self, word1: str, word2: str, constraint_type: str,
                      custom_range: Tuple[float, float] = None,
                      weight: float = None):
        """
        Add a fuzzy constraint.

        Args:
            word1: First word
            word2: Second word
            constraint_type: Type of constraint
            custom_range: Optional custom (min, max) range
            weight: Optional custom weight
        """
        if custom_range:
            min_dist, max_dist = custom_range
            w = weight if weight is not None else 1.0
        elif constraint_type in self.constraint_ranges:
            min_dist, max_dist, w = self.constraint_ranges[constraint_type]
            if weight is not None:
                w = weight
        else:
            # Default to moderate similarity
            min_dist, max_dist, w = (0.3, 0.6, 1.0)

        constraint = FuzzyConstraint(
            word1=word1,
            word2=word2,
            constraint_type=constraint_type,
            target_distance_min=min_dist,
            target_distance_max=max_dist,
            weight=w
        )

        self.constraints.append(constraint)

    def add_constraints_from_graph(self, graph: KnowledgeGraph):
        """
        Extract constraints from a knowledge graph.

        Args:
            graph: KnowledgeGraph with edges representing relationships
        """
        for edge in graph.edges:
            source_word = edge.source.word
            target_word = edge.target.word

            # Determine constraint type from edge source type
            constraint_type = edge.source_type

            self.add_constraint(source_word, target_word, constraint_type)

    def compute_total_loss(self, model) -> Tuple[float, Dict]:
        """
        Compute total constraint loss across all constraints.

        Args:
            model: EmbeddingModel with get_embedding method

        Returns:
            Total loss and statistics dict
        """
        total_loss = 0.0
        num_satisfied = 0
        num_violated = 0
        losses_by_type = {}

        for constraint in self.constraints:
            # Get embeddings
            vec1 = model.get_embedding(constraint.word1)
            vec2 = model.get_embedding(constraint.word2)

            if vec1 is None or vec2 is None:
                continue

            # Compute distance
            distance = np.linalg.norm(vec1 - vec2)

            # Compute loss
            loss = constraint.compute_loss(distance)
            total_loss += loss

            # Track statistics
            if loss == 0.0:
                num_satisfied += 1
            else:
                num_violated += 1

            # Track by type
            if constraint.constraint_type not in losses_by_type:
                losses_by_type[constraint.constraint_type] = []
            losses_by_type[constraint.constraint_type].append(loss)

        # Compute statistics
        stats = {
            'total_loss': total_loss,
            'num_constraints': len(self.constraints),
            'num_satisfied': num_satisfied,
            'num_violated': num_violated,
            'satisfaction_rate': num_satisfied / len(self.constraints) if self.constraints else 0.0,
            'losses_by_type': {k: sum(v) / len(v) for k, v in losses_by_type.items()},
        }

        return total_loss, stats

    def compute_gradients(self, model) -> np.ndarray:
        """
        Compute gradients of constraint loss with respect to embeddings.

        This is a simplified gradient computation. For production, use autograd.

        Args:
            model: EmbeddingModel

        Returns:
            Gradient matrix (same shape as embeddings)
        """
        gradients = np.zeros_like(model.embeddings)

        for constraint in self.constraints:
            # Get word IDs
            id1 = model.word_to_id.get(constraint.word1.lower())
            id2 = model.word_to_id.get(constraint.word2.lower())

            if id1 is None or id2 is None:
                continue

            vec1 = model.embeddings[id1]
            vec2 = model.embeddings[id2]

            # Compute distance
            diff = vec1 - vec2
            distance = np.linalg.norm(diff)

            if distance < 1e-8:
                continue

            # Compute gradient direction
            if distance < constraint.target_distance_min:
                # Too close - push apart
                grad_direction = -diff / distance
                magnitude = 2 * constraint.weight * (constraint.target_distance_min - distance)
            elif distance > constraint.target_distance_max:
                # Too far - pull together
                grad_direction = diff / distance
                magnitude = 2 * constraint.weight * (distance - constraint.target_distance_max)
            else:
                # Satisfied - no gradient
                continue

            # Apply gradients
            gradients[id1] += magnitude * grad_direction
            gradients[id2] -= magnitude * grad_direction

        return gradients

    def sample_constraints(self, num_samples: int) -> List[FuzzyConstraint]:
        """Sample a subset of constraints for mini-batch training."""
        if len(self.constraints) <= num_samples:
            return self.constraints

        indices = np.random.choice(len(self.constraints), num_samples, replace=False)
        return [self.constraints[i] for i in indices]

    def get_statistics(self) -> Dict:
        """Get constraint statistics."""
        type_counts = {}
        for constraint in self.constraints:
            type_counts[constraint.constraint_type] = type_counts.get(constraint.constraint_type, 0) + 1

        return {
            'total_constraints': len(self.constraints),
            'constraints_by_type': type_counts,
        }

    def __repr__(self):
        return f"ConstraintLoss({len(self.constraints)} constraints)"


def evaluate_constraint_satisfaction(model, constraints: List[FuzzyConstraint]) -> Dict:
    """
    Evaluate how well a model satisfies constraints.

    Args:
        model: EmbeddingModel
        constraints: List of constraints to evaluate

    Returns:
        Evaluation metrics
    """
    results = {
        'total': len(constraints),
        'satisfied': 0,
        'violated': 0,
        'by_type': {},
    }

    for constraint in constraints:
        vec1 = model.get_embedding(constraint.word1)
        vec2 = model.get_embedding(constraint.word2)

        if vec1 is None or vec2 is None:
            continue

        distance = np.linalg.norm(vec1 - vec2)
        loss = constraint.compute_loss(distance)

        # Track overall
        if loss == 0.0:
            results['satisfied'] += 1
        else:
            results['violated'] += 1

        # Track by type
        if constraint.constraint_type not in results['by_type']:
            results['by_type'][constraint.constraint_type] = {
                'total': 0,
                'satisfied': 0,
                'violated': 0,
                'distances': [],
            }

        results['by_type'][constraint.constraint_type]['total'] += 1
        if loss == 0.0:
            results['by_type'][constraint.constraint_type]['satisfied'] += 1
        else:
            results['by_type'][constraint.constraint_type]['violated'] += 1

        results['by_type'][constraint.constraint_type]['distances'].append(distance)

    # Compute satisfaction rates
    if results['total'] > 0:
        results['satisfaction_rate'] = results['satisfied'] / results['total']

    for type_name, type_stats in results['by_type'].items():
        if type_stats['total'] > 0:
            type_stats['satisfaction_rate'] = type_stats['satisfied'] / type_stats['total']
            type_stats['mean_distance'] = np.mean(type_stats['distances'])
            type_stats['std_distance'] = np.std(type_stats['distances'])

    return results
