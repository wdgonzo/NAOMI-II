"""
Anchor Dimensions

Defines the predefined semantic/grammatical/logical dimensions that serve
as fixed anchor points in the embedding space.

These dimensions are NOT learned during training - they are predefined based
on linguistic theory. The model learns additional dimensions to capture
more subtle semantic distinctions.

NOTE: Per user feedback, dimensionality is flexible - we start with these
anchors but can add more learned dimensions as needed.
"""

import numpy as np
from enum import Enum, auto
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..parser.enums import SubType, Tag, ConnectionType


class AnchorType(Enum):
    """Types of anchor dimensions."""
    SEMANTIC = auto()
    GRAMMATICAL = auto()
    LOGICAL = auto()


@dataclass
class AnchorDimension:
    """
    A single anchor dimension in the embedding space.

    Attributes:
        name: Human-readable name
        index: Position in the embedding vector
        anchor_type: Type of anchor (semantic/grammatical/logical)
        description: What this dimension represents
        activation_conditions: When this dimension should be activated
    """
    name: str
    index: int
    anchor_type: AnchorType
    description: str
    activation_conditions: List[str] = None

    def __post_init__(self):
        if self.activation_conditions is None:
            self.activation_conditions = []


class AnchorDimensions:
    """
    Collection of all anchor dimensions.

    Based on ARCHITECTURE.md:
    - 27 semantic dimensions (nominals, scopes, roles)
    - 15 grammatical dimensions (tense, aspect, mood, etc.)
    - 9 logical dimensions (AND, OR, NOT, etc.)
    Total: 51 anchor dimensions

    However, this is flexible - we can add more as needed.
    """

    def __init__(self):
        """Initialize anchor dimensions."""
        self.dimensions: List[AnchorDimension] = []
        self._dimension_map: Dict[str, AnchorDimension] = {}
        self._current_index = 0

        self._init_semantic_anchors()
        self._init_grammatical_anchors()
        self._init_logical_anchors()

    def _add_anchor(self, name: str, anchor_type: AnchorType,
                   description: str, activation_conditions: List[str] = None):
        """Helper to add an anchor dimension."""
        anchor = AnchorDimension(
            name=name,
            index=self._current_index,
            anchor_type=anchor_type,
            description=description,
            activation_conditions=activation_conditions or []
        )
        self.dimensions.append(anchor)
        self._dimension_map[name] = anchor
        self._current_index += 1

    def _init_semantic_anchors(self):
        """Initialize semantic anchor dimensions (27 total)."""

        # Nominal properties (6)
        self._add_anchor("determinatory", AnchorType.SEMANTIC,
                        "Whether entity is specific/definite")
        self._add_anchor("personal", AnchorType.SEMANTIC,
                        "Whether entity is person/agent")
        self._add_anchor("living", AnchorType.SEMANTIC,
                        "Whether entity is alive/animate")
        self._add_anchor("permanence", AnchorType.SEMANTIC,
                        "Whether property is permanent vs temporary")
        self._add_anchor("embodiment", AnchorType.SEMANTIC,
                        "Whether entity is concrete vs abstract")
        self._add_anchor("magnitude", AnchorType.SEMANTIC,
                        "Size/scale/importance of entity")

        # Scope dimensions (11)
        self._add_anchor("temporal", AnchorType.SEMANTIC,
                        "Time-related modification")
        self._add_anchor("frequency", AnchorType.SEMANTIC,
                        "How often action occurs")
        self._add_anchor("location", AnchorType.SEMANTIC,
                        "Spatial location/position")
        self._add_anchor("manner", AnchorType.SEMANTIC,
                        "How action is performed")
        self._add_anchor("extent", AnchorType.SEMANTIC,
                        "Degree/amount/extent")
        self._add_anchor("reason", AnchorType.SEMANTIC,
                        "Causation/purpose")
        self._add_anchor("attitude", AnchorType.SEMANTIC,
                        "Speaker's stance/attitude")
        self._add_anchor("relative", AnchorType.SEMANTIC,
                        "Comparative/relative modification")
        self._add_anchor("direction", AnchorType.SEMANTIC,
                        "Directional movement")
        self._add_anchor("spacialExtent", AnchorType.SEMANTIC,
                        "Spatial extent/coverage")
        self._add_anchor("beneficiary", AnchorType.SEMANTIC,
                        "Who benefits from action")

        # Role dimensions (10)
        self._add_anchor("fundamental", AnchorType.SEMANTIC,
                        "Core predicate/action")
        self._add_anchor("subject", AnchorType.SEMANTIC,
                        "Agent/subject role")
        self._add_anchor("subjectComp", AnchorType.SEMANTIC,
                        "Subject complement")
        self._add_anchor("objects", AnchorType.SEMANTIC,
                        "Patient/object role")
        self._add_anchor("results", AnchorType.SEMANTIC,
                        "Result/outcome")
        self._add_anchor("instruments", AnchorType.SEMANTIC,
                        "Instrument/tool used")
        self._add_anchor("sources", AnchorType.SEMANTIC,
                        "Source/origin")
        self._add_anchor("goals", AnchorType.SEMANTIC,
                        "Goal/destination")
        self._add_anchor("experiencer", AnchorType.SEMANTIC,
                        "Experiencer of state/emotion")
        self._add_anchor("nominal", AnchorType.SEMANTIC,
                        "Nominal entity")

    def _init_grammatical_anchors(self):
        """Initialize grammatical anchor dimensions (15 total)."""

        self._add_anchor("tense", AnchorType.GRAMMATICAL,
                        "Past/present/future time reference")
        self._add_anchor("aspect", AnchorType.GRAMMATICAL,
                        "Perfective/imperfective/progressive aspect")
        self._add_anchor("mood", AnchorType.GRAMMATICAL,
                        "Indicative/subjunctive/imperative mood")
        self._add_anchor("voice", AnchorType.GRAMMATICAL,
                        "Active/passive voice")
        self._add_anchor("person", AnchorType.GRAMMATICAL,
                        "1st/2nd/3rd person")
        self._add_anchor("number", AnchorType.GRAMMATICAL,
                        "Singular/plural number")
        self._add_anchor("gender", AnchorType.GRAMMATICAL,
                        "Masculine/feminine/neuter gender")
        self._add_anchor("case", AnchorType.GRAMMATICAL,
                        "Nominative/accusative/etc. case")
        self._add_anchor("definiteness", AnchorType.GRAMMATICAL,
                        "Definite/indefinite")
        self._add_anchor("polarity", AnchorType.GRAMMATICAL,
                        "Affirmative/negative")
        self._add_anchor("animacy", AnchorType.GRAMMATICAL,
                        "Animate/inanimate")
        self._add_anchor("countability", AnchorType.GRAMMATICAL,
                        "Count/mass noun")
        self._add_anchor("degree", AnchorType.GRAMMATICAL,
                        "Comparative/superlative degree")
        self._add_anchor("transitivity", AnchorType.GRAMMATICAL,
                        "Transitive/intransitive verb")
        self._add_anchor("evidentiality", AnchorType.GRAMMATICAL,
                        "Source of information/evidence")

    def _init_logical_anchors(self):
        """Initialize logical operator anchor dimensions (9 total)."""

        self._add_anchor("AND", AnchorType.LOGICAL,
                        "Logical conjunction")
        self._add_anchor("OR", AnchorType.LOGICAL,
                        "Logical disjunction")
        self._add_anchor("XOR", AnchorType.LOGICAL,
                        "Exclusive or")
        self._add_anchor("NAND", AnchorType.LOGICAL,
                        "Not and")
        self._add_anchor("IF", AnchorType.LOGICAL,
                        "Implication/conditional")
        self._add_anchor("XIF", AnchorType.LOGICAL,
                        "Biconditional/if and only if")
        self._add_anchor("NOT", AnchorType.LOGICAL,
                        "Negation")
        self._add_anchor("NOR", AnchorType.LOGICAL,
                        "Not or")
        self._add_anchor("XNOR", AnchorType.LOGICAL,
                        "Equivalence")

    def get_dimension(self, name: str) -> AnchorDimension:
        """Get anchor dimension by name."""
        return self._dimension_map.get(name)

    def get_dimension_index(self, name: str) -> int:
        """Get index of anchor dimension by name."""
        dim = self.get_dimension(name)
        return dim.index if dim else -1

    def num_anchors(self) -> int:
        """Get total number of anchor dimensions."""
        return len(self.dimensions)

    def get_semantic_dimensions(self) -> List[AnchorDimension]:
        """Get all semantic anchor dimensions."""
        return [d for d in self.dimensions if d.anchor_type == AnchorType.SEMANTIC]

    def get_grammatical_dimensions(self) -> List[AnchorDimension]:
        """Get all grammatical anchor dimensions."""
        return [d for d in self.dimensions if d.anchor_type == AnchorType.GRAMMATICAL]

    def get_logical_dimensions(self) -> List[AnchorDimension]:
        """Get all logical anchor dimensions."""
        return [d for d in self.dimensions if d.anchor_type == AnchorType.LOGICAL]

    def create_zero_vector(self, total_dims: int) -> np.ndarray:
        """
        Create a zero vector with specified total dimensions.

        Args:
            total_dims: Total embedding dimensions (anchors + learned)

        Returns:
            Zero vector of shape (total_dims,)
        """
        return np.zeros(total_dims, dtype=np.float32)

    def __repr__(self):
        return f"AnchorDimensions({self.num_anchors()} anchors: " \
               f"{len(self.get_semantic_dimensions())} semantic, " \
               f"{len(self.get_grammatical_dimensions())} grammatical, " \
               f"{len(self.get_logical_dimensions())} logical)"


def get_anchor_vector(word, word_subtypes: List[SubType],
                     anchors: AnchorDimensions,
                     total_dims: int) -> np.ndarray:
    """
    Create an anchor vector based on word properties.

    This activates anchor dimensions based on the word's grammatical features.
    The learned dimensions (beyond anchors) are left as zeros - they will
    be filled in by the learned embeddings.

    Args:
        word: Word object
        word_subtypes: List of SubType features
        anchors: AnchorDimensions object
        total_dims: Total embedding dimensions

    Returns:
        Vector with anchor dimensions activated
    """
    vector = anchors.create_zero_vector(total_dims)

    # Activate grammatical dimensions based on subtypes
    for subtype in word_subtypes:
        if subtype == SubType.SINGULAR:
            idx = anchors.get_dimension_index("number")
            if idx >= 0:
                vector[idx] = 0.0  # Singular = negative
        elif subtype == SubType.PLURAL:
            idx = anchors.get_dimension_index("number")
            if idx >= 0:
                vector[idx] = 1.0  # Plural = positive

        elif subtype == SubType.MASCULINE:
            idx = anchors.get_dimension_index("gender")
            if idx >= 0:
                vector[idx] = -0.5  # Masculine
        elif subtype == SubType.FEMININE:
            idx = anchors.get_dimension_index("gender")
            if idx >= 0:
                vector[idx] = 0.5  # Feminine

        elif subtype == SubType.FIRST_PERSON:
            idx = anchors.get_dimension_index("person")
            if idx >= 0:
                vector[idx] = -1.0  # 1st person
        elif subtype == SubType.SECOND_PERSON:
            idx = anchors.get_dimension_index("person")
            if idx >= 0:
                vector[idx] = 0.0  # 2nd person
        elif subtype == SubType.THIRD_PERSON:
            idx = anchors.get_dimension_index("person")
            if idx >= 0:
                vector[idx] = 1.0  # 3rd person

    # Activate based on POS tag
    if hasattr(word, 'tag'):
        if word.tag == Tag.DET:
            idx = anchors.get_dimension_index("determinatory")
            if idx >= 0:
                vector[idx] = 1.0

        elif word.tag in [Tag.NOUN, Tag.PRON]:
            idx = anchors.get_dimension_index("nominal")
            if idx >= 0:
                vector[idx] = 1.0

    return vector
