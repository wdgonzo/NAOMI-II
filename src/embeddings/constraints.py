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
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from ..graph.knowledge_graph import KnowledgeGraph, GraphEdge


class SparsityLoss:
    """
    L1 sparsity regularization to encourage sparse embeddings.

    Penalizes non-zero values in embeddings, encouraging words to only
    activate dimensions that are relevant to their meaning.

    Goal: Achieve 40-70% sparsity (40-70% of values near zero).

    Attributes:
        l1_weight: Regularization strength (higher = more sparse)
    """

    def __init__(self, l1_weight: float = 0.01):
        """
        Initialize sparsity loss.

        Args:
            l1_weight: L1 regularization strength (0.001-0.1 typical range)
        """
        self.l1_weight = l1_weight

    def __call__(self, embeddings: np.ndarray) -> float:
        """
        Compute L1 sparsity loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)

        Returns:
            Sparsity loss value (higher = more non-zero values)
        """
        return self.l1_weight * np.mean(np.abs(embeddings))

    def compute_sparsity(self, embeddings: np.ndarray, threshold: float = 0.01) -> float:
        """
        Compute current sparsity percentage.

        Args:
            embeddings: Word embedding matrix
            threshold: Values below this are considered "zero"

        Returns:
            Sparsity percentage (0-100)
        """
        near_zero = np.abs(embeddings) < threshold
        return 100.0 * np.mean(near_zero)


class SelectivePolarityLoss:
    """
    Selective polarity constraint for antonyms.

    Different antonym types oppose on different specific dimensions:
    - good/bad oppose on dimension 0 (morality)
    - hot/cold oppose on dimension 1 (temperature)
    - big/small oppose on dimension 2 (size)
    - etc.

    This creates interpretable, structured semantic space where
    dimensions have consistent meaning across all words.

    Goal: Each antonym pair opposes on 1-10 dimensions (not all 128!).
    """

    def __init__(self, antonym_dimension_map: Dict[Tuple[str, str], List[int]],
                 polarity_weight: float = 1.0,
                 similarity_weight: float = 0.5):
        """
        Initialize selective polarity loss.

        Args:
            antonym_dimension_map: Maps (word1, word2) -> list of dimensions
                Example: {('good', 'bad'): [0], ('hot', 'cold'): [1]}
            polarity_weight: Weight for opposition on assigned dimensions
            similarity_weight: Weight for similarity on other dimensions
        """
        self.antonym_dimension_map = antonym_dimension_map
        self.polarity_weight = polarity_weight
        self.similarity_weight = similarity_weight

    def __call__(self, embeddings: np.ndarray, word_to_id: Dict[str, int]) -> float:
        """
        Compute selective polarity loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)
            word_to_id: Maps words to embedding indices

        Returns:
            Total selective polarity loss
        """
        total_loss = 0.0
        num_pairs = 0

        for (word1, word2), assigned_dims in self.antonym_dimension_map.items():
            # Get word IDs
            id1 = word_to_id.get(word1.lower())
            id2 = word_to_id.get(word2.lower())

            if id1 is None or id2 is None:
                continue

            vec1 = embeddings[id1]
            vec2 = embeddings[id2]

            # 1. Polarity constraint on ASSIGNED dimensions only
            polarity_loss = 0.0
            for dim in assigned_dims:
                val1 = vec1[dim]
                val2 = vec2[dim]

                sign_product = np.sign(val1) * np.sign(val2)

                if sign_product > 0:
                    # Same sign - penalty
                    polarity_loss += (abs(val1) + abs(val2))
                elif sign_product < 0:
                    # Opposite signs - reward
                    polarity_loss -= (abs(val1) + abs(val2)) * 0.5
                else:
                    # One is zero - small penalty
                    polarity_loss += 0.1

            # 2. Similarity constraint on NON-ASSIGNED dimensions
            # Antonyms should be similar on irrelevant dimensions
            other_dims = [i for i in range(len(vec1)) if i not in assigned_dims]
            if other_dims:
                diff = vec1[other_dims] - vec2[other_dims]
                similarity_loss = np.mean(diff ** 2)
            else:
                similarity_loss = 0.0

            total_loss += self.polarity_weight * polarity_loss + self.similarity_weight * similarity_loss
            num_pairs += 1

        return total_loss / max(num_pairs, 1)

    @staticmethod
    def from_config_files(antonym_types_path: str,
                         dimension_assignments_path: str) -> 'SelectivePolarityLoss':
        """
        Create SelectivePolarityLoss from config files.

        Args:
            antonym_types_path: Path to antonym_types.json
            dimension_assignments_path: Path to dimension_assignments.json

        Returns:
            Configured SelectivePolarityLoss instance
        """
        import json

        # Load antonym types
        with open(antonym_types_path, 'r') as f:
            antonym_types = json.load(f)

        # Load dimension assignments
        with open(dimension_assignments_path, 'r') as f:
            dimension_assignments = json.load(f)
            # Remove comments
            dimension_assignments = {k: v for k, v in dimension_assignments.items()
                                    if not k.startswith('_')}

        # Build antonym-dimension map
        antonym_dimension_map = {}
        for atype, pairs in antonym_types.items():
            if atype in dimension_assignments:
                dims = dimension_assignments[atype]
                for word1, word2 in pairs:
                    antonym_dimension_map[(word1, word2)] = dims

        return SelectivePolarityLoss(antonym_dimension_map)


class DimensionalConsistencyLoss:
    """
    Enforces dimensional consistency across all words.

    Ensures that each dimension has consistent meaning for ALL words:
    - Dimension 0 = morality for EVERY word (not just good/bad)
    - Dimension 1 = gender for EVERY word
    - Dimension 2 = size for EVERY word
    - etc.

    This creates interpretable semantic space where dimensions are transparent.

    Example:
        - All moral words (good, bad, virtue, evil) cluster on dimension 0
        - Moral words have high |value| on dim 0, neutral words near zero
        - Can predict: if |word[0]| > 0.5, word has moral content
    """

    def __init__(self, semantic_clusters: Dict[int, Dict[str, List[str]]],
                 consistency_weight: float = 1.0,
                 sparsity_weight: float = 0.5):
        """
        Initialize dimensional consistency loss.

        Args:
            semantic_clusters: Maps dimension index to cluster definition
                Example: {0: {'positive': ['good'], 'negative': ['bad'], 'neutral': ['chair']}}
            consistency_weight: Weight for clustering on target dimension
            sparsity_weight: Weight for being zero on other dimensions
        """
        self.semantic_clusters = semantic_clusters
        self.consistency_weight = consistency_weight
        self.sparsity_weight = sparsity_weight

    def __call__(self, embeddings: np.ndarray, word_to_id: Dict[str, int]) -> float:
        """
        Compute dimensional consistency loss.

        Args:
            embeddings: Word embedding matrix (vocab_size x embedding_dim)
            word_to_id: Maps words to embedding indices

        Returns:
            Total consistency loss
        """
        total_loss = 0.0
        num_constraints = 0

        for dim_idx, cluster_def in self.semantic_clusters.items():
            # 1. Positive pole: should have positive values on this dim
            for word in cluster_def.get('positive', []):
                word_id = word_to_id.get(word.lower())
                if word_id is None:
                    continue

                vec = embeddings[word_id]
                target_val = vec[dim_idx]

                # Penalty if value is not positive
                if target_val <= 0:
                    total_loss += abs(target_val) + 0.1  # Penalty for wrong sign
                else:
                    total_loss -= target_val * 0.1  # Small reward for correct sign

                # Encourage sparsity on other dimensions
                other_dims = [i for i in range(len(vec)) if i != dim_idx]
                if other_dims:
                    total_loss += self.sparsity_weight * np.mean(np.abs(vec[other_dims]))

                num_constraints += 1

            # 2. Negative pole: should have negative values on this dim
            for word in cluster_def.get('negative', []):
                word_id = word_to_id.get(word.lower())
                if word_id is None:
                    continue

                vec = embeddings[word_id]
                target_val = vec[dim_idx]

                # Penalty if value is not negative
                if target_val >= 0:
                    total_loss += abs(target_val) + 0.1  # Penalty for wrong sign
                else:
                    total_loss -= abs(target_val) * 0.1  # Small reward for correct sign

                # Encourage sparsity on other dimensions
                other_dims = [i for i in range(len(vec)) if i != dim_idx]
                if other_dims:
                    total_loss += self.sparsity_weight * np.mean(np.abs(vec[other_dims]))

                num_constraints += 1

            # 3. Neutral: should be near zero on this dim
            for word in cluster_def.get('neutral', []):
                word_id = word_to_id.get(word.lower())
                if word_id is None:
                    continue

                vec = embeddings[word_id]
                target_val = vec[dim_idx]

                # Strong penalty for being non-zero
                total_loss += abs(target_val) * 2.0

                num_constraints += 1

        return self.consistency_weight * (total_loss / max(num_constraints, 1))

    @staticmethod
    def from_config_file(semantic_clusters_path: str) -> 'DimensionalConsistencyLoss':
        """
        Create DimensionalConsistencyLoss from config file.

        Args:
            semantic_clusters_path: Path to semantic_clusters.json

        Returns:
            Configured DimensionalConsistencyLoss instance
        """
        import json

        with open(semantic_clusters_path, 'r', encoding='utf-8') as f:
            clusters_str = json.load(f)

        # Convert string keys back to integers
        clusters = {int(k): v for k, v in clusters_str.items()}

        return DimensionalConsistencyLoss(clusters)


@dataclass
class PolarityConstraint:
    """
    Opposite-sign polarity constraint for antonyms.

    Forces antonyms to have opposite signs on specific dimensions,
    enabling compositional semantics (NOT(good) ≈ bad).

    Attributes:
        word1: First word (e.g., "good")
        word2: Second word (e.g., "bad")
        polarity_dims: Dimensions where opposite signs should be enforced
        weight: Importance weight
    """
    word1: str
    word2: str
    polarity_dims: List[int]
    weight: float = 1.0

    def compute_loss(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute polarity constraint loss.

        Penalizes same-sign values on polarity dimensions.
        Rewards opposite-sign values with large magnitude.

        Args:
            emb1: First word embedding
            emb2: Second word embedding

        Returns:
            Loss value (negative = good, positive = bad)
        """
        loss = 0.0

        for dim_idx in self.polarity_dims:
            val1 = emb1[dim_idx]
            val2 = emb2[dim_idx]

            # Check sign agreement
            sign_product = np.sign(val1) * np.sign(val2)

            if sign_product > 0:
                # Same sign - penalty proportional to magnitude
                # Larger values with same sign = worse
                loss += (abs(val1) + abs(val2))
            elif sign_product < 0:
                # Opposite signs - reward proportional to magnitude
                # Larger opposite values = better
                loss -= (abs(val1) + abs(val2))
            else:
                # One value is zero - small penalty
                loss += 0.1

        return self.weight * loss


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

    Supports both distance-based and polarity-based constraints.
    """

    def __init__(self, polarity_dims: Optional[List[int]] = None):
        """
        Initialize constraint loss manager.

        Args:
            polarity_dims: Optional list of dimensions for polarity constraints
        """
        self.constraints: List[FuzzyConstraint] = []
        self.polarity_constraints: List[PolarityConstraint] = []
        self.polarity_dims = polarity_dims or []

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

    def add_polarity_constraint(self, word1: str, word2: str, weight: float = 2.0):
        """
        Add a polarity constraint for antonym pair.

        Args:
            word1: First word (e.g., "good")
            word2: Second word (e.g., "bad")
            weight: Importance weight
        """
        if not self.polarity_dims:
            # No polarity dimensions configured yet
            return

        constraint = PolarityConstraint(
            word1=word1,
            word2=word2,
            polarity_dims=self.polarity_dims,
            weight=weight
        )

        self.polarity_constraints.append(constraint)

    def set_polarity_dimensions(self, polarity_dims: List[int]):
        """
        Set or update polarity dimensions.

        Args:
            polarity_dims: List of dimension indices for polarity constraints
        """
        self.polarity_dims = polarity_dims
        print(f"Set {len(polarity_dims)} polarity dimensions: {polarity_dims}")

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

        # Distance-based fuzzy constraints
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

        # Polarity-based constraints for antonyms
        polarity_loss = 0.0
        num_polarity_satisfied = 0
        num_polarity_violated = 0

        for constraint in self.polarity_constraints:
            # Get embeddings
            vec1 = model.get_embedding(constraint.word1)
            vec2 = model.get_embedding(constraint.word2)

            if vec1 is None or vec2 is None:
                continue

            # Compute polarity loss
            loss = constraint.compute_loss(vec1, vec2)
            polarity_loss += loss
            total_loss += loss

            # Track statistics (negative loss = satisfied)
            if loss <= 0.0:
                num_polarity_satisfied += 1
            else:
                num_polarity_violated += 1

        # Track polarity losses separately
        if self.polarity_constraints:
            losses_by_type['polarity_constraints'] = [polarity_loss / len(self.polarity_constraints)]

        # Compute statistics
        total_constraints = len(self.constraints) + len(self.polarity_constraints)
        total_satisfied = num_satisfied + num_polarity_satisfied

        stats = {
            'total_loss': total_loss,
            'distance_loss': total_loss - polarity_loss,
            'polarity_loss': polarity_loss,
            'num_constraints': len(self.constraints),
            'num_polarity_constraints': len(self.polarity_constraints),
            'num_satisfied': num_satisfied,
            'num_violated': num_violated,
            'num_polarity_satisfied': num_polarity_satisfied,
            'num_polarity_violated': num_polarity_violated,
            'satisfaction_rate': num_satisfied / len(self.constraints) if self.constraints else 0.0,
            'polarity_satisfaction_rate': num_polarity_satisfied / len(self.polarity_constraints) if self.polarity_constraints else 0.0,
            'overall_satisfaction_rate': total_satisfied / total_constraints if total_constraints > 0 else 0.0,
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
            'num_polarity_constraints': len(self.polarity_constraints),
            'polarity_dimensions': len(self.polarity_dims),
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
