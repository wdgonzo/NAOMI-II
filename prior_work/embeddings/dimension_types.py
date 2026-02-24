"""
Data structures for embedding dimension analysis.

Defines types used for analyzing individual dimensions of word embeddings
to understand what semantic aspects they encode.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class DimensionStats:
    """Statistics for a single embedding dimension."""

    index: int
    name: Optional[str]  # If it's an anchor dimension
    variance: float
    mean: float
    std: float
    min_val: float
    max_val: float
    top_activations: List[Tuple[str, float]] = field(default_factory=list)
    bottom_activations: List[Tuple[str, float]] = field(default_factory=list)

    def is_anchor(self) -> bool:
        """Check if this is an anchor dimension."""
        return self.name is not None

    def activation_range(self) -> float:
        """Get the range of activation values."""
        return self.max_val - self.min_val

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'name': self.name,
            'variance': float(self.variance),
            'mean': float(self.mean),
            'std': float(self.std),
            'min': float(self.min_val),
            'max': float(self.max_val),
            'range': float(self.activation_range()),
            'top_activations': [(word, float(val)) for word, val in self.top_activations],
            'bottom_activations': [(word, float(val)) for word, val in self.bottom_activations]
        }


@dataclass
class SemanticAxis:
    """A discovered semantic axis in embedding space."""

    name: str
    primary_dimension: int
    correlation_score: float  # How well this dimension captures the concept
    positive_pole_words: List[str]  # Words at high end (e.g., "big", "huge")
    negative_pole_words: List[str]  # Words at low end (e.g., "small", "tiny")
    contributing_dimensions: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'primary_dimension': self.primary_dimension,
            'correlation_score': float(self.correlation_score),
            'positive_pole': self.positive_pole_words,
            'negative_pole': self.negative_pole_words,
            'contributing_dimensions': self.contributing_dimensions
        }


@dataclass
class DimensionComparison:
    """Comparison of two words across a specific dimension."""

    dimension_idx: int
    dimension_name: Optional[str]
    word1: str
    word2: str
    value1: float
    value2: float

    @property
    def difference(self) -> float:
        """Absolute difference between values."""
        return abs(self.value1 - self.value2)

    @property
    def mean(self) -> float:
        """Mean value."""
        return (self.value1 + self.value2) / 2.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'dimension': self.dimension_idx,
            'name': self.dimension_name,
            'word1': self.word1,
            'word2': self.word2,
            'value1': float(self.value1),
            'value2': float(self.value2),
            'difference': float(self.difference),
            'mean': float(self.mean)
        }


@dataclass
class RelationshipDimensionProfile:
    """Profile of dimensions that encode a specific relationship type."""

    relationship_type: str  # "synonym", "antonym", "hypernym", etc.
    num_pairs: int  # Number of word pairs analyzed

    # Dimensions where the relationship shows up
    discriminative_dims: List[int]  # Dims that vary most between pairs
    similarity_dims: List[int]  # Dims that stay similar between pairs

    # Importance scores for each dimension
    importance_scores: Dict[int, float] = field(default_factory=dict)

    # Statistics
    mean_difference_per_dim: Dict[int, float] = field(default_factory=dict)
    std_difference_per_dim: Dict[int, float] = field(default_factory=dict)

    def get_top_discriminative(self, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get top K most discriminative dimensions with their scores."""
        sorted_dims = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_dims[:top_k]

    def get_dimension_summary(self, dim_idx: int) -> Dict:
        """Get summary for a specific dimension."""
        return {
            'dimension': dim_idx,
            'is_discriminative': dim_idx in self.discriminative_dims,
            'is_similar': dim_idx in self.similarity_dims,
            'importance': self.importance_scores.get(dim_idx, 0.0),
            'mean_difference': self.mean_difference_per_dim.get(dim_idx, 0.0),
            'std_difference': self.std_difference_per_dim.get(dim_idx, 0.0)
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'relationship_type': self.relationship_type,
            'num_pairs': self.num_pairs,
            'discriminative_dimensions': self.discriminative_dims,
            'similarity_dimensions': self.similarity_dims,
            'top_discriminative': [
                {'dim': dim, 'score': float(score)}
                for dim, score in self.get_top_discriminative(10)
            ],
            'statistics': {
                str(dim): {
                    'mean_diff': float(mean),
                    'std_diff': float(self.std_difference_per_dim.get(dim, 0.0)),
                    'importance': float(self.importance_scores.get(dim, 0.0))
                }
                for dim, mean in list(self.mean_difference_per_dim.items())[:20]
            }
        }


@dataclass
class AnchorValidationResult:
    """Result of validating an anchor dimension."""

    anchor_name: str
    dimension_idx: int
    expected_behavior: str  # Description of expected behavior

    # Test results
    passes_validation: bool
    confidence_score: float  # 0-1, how confident we are in the validation

    # Evidence
    high_activation_words: List[Tuple[str, float]]  # Should activate
    low_activation_words: List[Tuple[str, float]]  # Should not activate

    # Statistics
    separation_score: float  # How well it separates expected classes
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'anchor_name': self.anchor_name,
            'dimension': self.dimension_idx,
            'expected_behavior': self.expected_behavior,
            'passes': self.passes_validation,
            'confidence': float(self.confidence_score),
            'separation_score': float(self.separation_score),
            'high_activations': [
                {'word': word, 'value': float(val)}
                for word, val in self.high_activation_words[:10]
            ],
            'low_activations': [
                {'word': word, 'value': float(val)}
                for word, val in self.low_activation_words[:10]
            ],
            'notes': self.notes
        }


@dataclass
class DimensionAnalysisReport:
    """Comprehensive report of dimensional analysis."""

    model_path: str
    vocabulary_size: int
    embedding_dim: int
    num_anchor_dims: int

    # Overall statistics
    dimension_stats: List[DimensionStats]

    # Relationship profiles
    synonym_profile: Optional[RelationshipDimensionProfile] = None
    antonym_profile: Optional[RelationshipDimensionProfile] = None
    hypernym_profile: Optional[RelationshipDimensionProfile] = None

    # Discovered patterns
    semantic_axes: List[SemanticAxis] = field(default_factory=list)

    # Anchor validation
    anchor_validations: List[AnchorValidationResult] = field(default_factory=list)

    # High-level insights
    high_variance_dims: List[int] = field(default_factory=list)
    low_variance_dims: List[int] = field(default_factory=list)

    def get_summary(self) -> Dict:
        """Get high-level summary of analysis."""
        summary = {
            'model': self.model_path,
            'vocabulary_size': self.vocabulary_size,
            'embedding_dimensions': self.embedding_dim,
            'anchor_dimensions': self.num_anchor_dims,
            'learned_dimensions': self.embedding_dim - self.num_anchor_dims,
            'high_variance_count': len(self.high_variance_dims),
            'low_variance_count': len(self.low_variance_dims),
            'semantic_axes_found': len(self.semantic_axes)
        }

        # Relationship summaries
        if self.synonym_profile:
            summary['synonym_analysis'] = {
                'num_pairs': self.synonym_profile.num_pairs,
                'discriminative_dims': len(self.synonym_profile.discriminative_dims),
                'similarity_dims': len(self.synonym_profile.similarity_dims)
            }

        if self.antonym_profile:
            summary['antonym_analysis'] = {
                'num_pairs': self.antonym_profile.num_pairs,
                'discriminative_dims': len(self.antonym_profile.discriminative_dims),
                'similarity_dims': len(self.antonym_profile.similarity_dims),
                'avg_dims_differ': len(self.antonym_profile.discriminative_dims),
                'avg_dims_similar': len(self.antonym_profile.similarity_dims)
            }

        # Anchor validation
        if self.anchor_validations:
            passed = sum(1 for v in self.anchor_validations if v.passes_validation)
            summary['anchor_validation'] = {
                'total_tested': len(self.anchor_validations),
                'passed': passed,
                'failed': len(self.anchor_validations) - passed,
                'pass_rate': passed / len(self.anchor_validations) if self.anchor_validations else 0
            }

        return summary

    def to_dict(self) -> Dict:
        """Convert entire report to dictionary."""
        return {
            'summary': self.get_summary(),
            'dimension_statistics': [stat.to_dict() for stat in self.dimension_stats[:20]],
            'relationships': {
                'synonyms': self.synonym_profile.to_dict() if self.synonym_profile else None,
                'antonyms': self.antonym_profile.to_dict() if self.antonym_profile else None,
                'hypernyms': self.hypernym_profile.to_dict() if self.hypernym_profile else None
            },
            'semantic_axes': [axis.to_dict() for axis in self.semantic_axes],
            'anchor_validations': [val.to_dict() for val in self.anchor_validations],
            'high_variance_dimensions': self.high_variance_dims[:20],
            'low_variance_dimensions': self.low_variance_dims[:20]
        }
