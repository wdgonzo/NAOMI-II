"""
WordNet Sense Mapper

Maps word occurrences to specific WordNet senses using parse context.

This solves the Word Sense Disambiguation (WSD) problem by using:
1. Part-of-speech from parse
2. Syntactic role in sentence
3. Neighboring words in parse tree
4. WordNet definitions and examples

Creates sense-tagged vocabulary for training sense-specific embeddings.
"""

from typing import List, Dict, Tuple, Optional, Set
import nltk
from nltk.corpus import wordnet as wn
from dataclasses import dataclass

from ..parser.enums import Tag, ConnectionType
from ..parser.data_structures import Hypothesis


@dataclass
class WordContext:
    """
    Context for a word occurrence in a sentence.

    Attributes:
        sentence_id: Unique sentence identifier
        word: The word itself
        pos_tag: Part-of-speech tag from parser
        syntactic_role: Role in parse tree (SUBJECT, OBJECT, etc.)
        neighbors: Words connected in parse tree
        parse_tree: Full hypothesis (optional, for detailed analysis)
    """
    sentence_id: int
    word: str
    pos_tag: Tag
    syntactic_role: Optional[ConnectionType]
    neighbors: List[str]
    parse_tree: Optional[Hypothesis] = None


class SenseMapper:
    """
    Maps word occurrences to WordNet senses using context.

    Strategy:
    1. Get all WordNet senses for word
    2. Filter by POS tag (hard constraint)
    3. Score each sense based on context overlap
    4. Return best matching sense
    """

    def __init__(self):
        """Initialize sense mapper and ensure WordNet is available."""
        self.ensure_wordnet()
        self._pos_tag_to_wordnet = {
            Tag.NOUN: 'n',
            Tag.VERB: 'v',
            Tag.ADJ: 'a',
            Tag.ADV: 'r',
            Tag.PRON: 'n',  # Treat pronouns as nouns
            Tag.PROPN: 'n',  # Proper nouns as nouns
        }

    def ensure_wordnet(self):
        """Ensure WordNet data is downloaded."""
        try:
            wn.synsets('test')
        except LookupError:
            print("Downloading WordNet data...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def get_senses(self, word: str, pos: Optional[Tag] = None) -> List[Tuple[int, any]]:
        """
        Get all WordNet senses for a word.

        Args:
            word: The word to look up
            pos: Optional POS filter

        Returns:
            List of (sense_index, synset) tuples
        """
        word = word.lower()

        # Get POS filter if provided
        wordnet_pos = None
        if pos and pos in self._pos_tag_to_wordnet:
            wordnet_pos = self._pos_tag_to_wordnet[pos]

        # Get synsets
        synsets = wn.synsets(word, pos=wordnet_pos)

        return list(enumerate(synsets))

    def match_context_to_sense(self, word: str, context: WordContext) -> Tuple[int, float]:
        """
        Match a word context to its most likely WordNet sense.

        Args:
            word: The word to disambiguate
            context: Context information

        Returns:
            (sense_index, confidence_score) tuple
            Returns (-1, 0.0) if no senses found or word not in WordNet
        """
        word = word.lower()

        # Get senses filtered by POS
        senses = self.get_senses(word, context.pos_tag)

        if not senses:
            # Word not in WordNet or wrong POS
            return (-1, 0.0)

        if len(senses) == 1:
            # Only one sense - no ambiguity
            return (0, 1.0)

        # Score each sense
        scores = []
        for sense_idx, synset in senses:
            score = self._score_sense(synset, context)
            scores.append((sense_idx, score))

        # Return best scoring sense
        best_idx, best_score = max(scores, key=lambda x: x[1])
        return (best_idx, best_score)

    def _score_sense(self, synset: any, context: WordContext) -> float:
        """
        Score how well a synset matches a context.

        Uses multiple signals:
        1. Definition overlap with neighbors
        2. Hypernym overlap with neighbors
        3. Example sentence similarity
        4. Frequency (prefer more common senses)

        Args:
            synset: WordNet synset to score
            context: Word context

        Returns:
            Confidence score (0.0 to 1.0)
        """
        score = 0.0
        neighbor_words = set(w.lower() for w in context.neighbors)

        # 1. Definition Overlap (most important)
        definition_words = set(synset.definition().lower().split())
        definition_overlap = len(definition_words & neighbor_words)
        score += definition_overlap * 0.4

        # 2. Hypernym Overlap
        hypernym_words = set()
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                hypernym_words.add(lemma.name().lower().replace('_', ' '))
        hypernym_overlap = len(hypernym_words & neighbor_words)
        score += hypernym_overlap * 0.3

        # 3. Example Sentence Similarity
        for example in synset.examples():
            example_words = set(example.lower().split())
            example_overlap = len(example_words & neighbor_words)
            score += example_overlap * 0.2

        # 4. Frequency Bias (earlier senses are more common in WordNet)
        # This is a weak signal but helps break ties
        # Synsets are ordered by frequency, so lower index = more common
        # We don't have access to the index here, so skip this for now

        # Normalize score
        max_possible = len(neighbor_words) * (0.4 + 0.3 + 0.2)
        if max_possible > 0:
            score = min(score / max_possible, 1.0)

        # Ensure minimum score (avoid zeros that might cause issues)
        score = max(score, 0.01)

        return score

    def create_sense_tag(self, word: str, sense_idx: int, pos_tag: Tag) -> str:
        """
        Create a sense-tagged word identifier.

        Format: word_wn.XX_pos
        Example: "bank_wn.00_n" (financial institution, noun)
                "bank_wn.01_n" (riverbank, noun)

        Args:
            word: The word
            sense_idx: WordNet sense index
            pos_tag: Part of speech

        Returns:
            Sense-tagged identifier
        """
        word = word.lower()
        wordnet_pos = self._pos_tag_to_wordnet.get(pos_tag, 'x')

        if sense_idx < 0:
            # Word not in WordNet - use raw word with POS
            return f"{word}_{wordnet_pos}"
        else:
            return f"{word}_wn.{sense_idx:02d}_{wordnet_pos}"

    def map_word_occurrences(self, word_contexts: List[Tuple[str, WordContext]]) -> Dict[Tuple[int, int], str]:
        """
        Map all word occurrences to sense-tagged identifiers.

        Args:
            word_contexts: List of (word, context) tuples

        Returns:
            Dictionary mapping (sentence_id, word_position) -> sense_tagged_word
        """
        sense_assignments = {}

        for word_pos, (word, context) in enumerate(word_contexts):
            sense_idx, confidence = self.match_context_to_sense(word, context)
            sense_tag = self.create_sense_tag(word, sense_idx, context.pos_tag)

            key = (context.sentence_id, word_pos)
            sense_assignments[key] = sense_tag

        return sense_assignments

    def create_sense_vocabulary(self, word_contexts: List[Tuple[str, WordContext]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Build vocabulary with sense tags.

        Args:
            word_contexts: List of (word, context) tuples

        Returns:
            (word_to_id, id_to_word) dictionaries
        """
        # Get all unique sense-tagged words
        sense_tags = set()

        for word, context in word_contexts:
            sense_idx, _ = self.match_context_to_sense(word, context)
            sense_tag = self.create_sense_tag(word, sense_idx, context.pos_tag)
            sense_tags.add(sense_tag)

        # Create vocabulary
        sorted_tags = sorted(sense_tags)
        word_to_id = {tag: idx for idx, tag in enumerate(sorted_tags)}
        id_to_word = {str(idx): tag for idx, tag in enumerate(sorted_tags)}

        return word_to_id, id_to_word

    def get_sense_info(self, word: str, sense_idx: int) -> Dict:
        """
        Get information about a specific word sense.

        Useful for validation and debugging.

        Args:
            word: The word
            sense_idx: Sense index

        Returns:
            Dictionary with sense information
        """
        synsets = wn.synsets(word.lower())

        if sense_idx < 0 or sense_idx >= len(synsets):
            return {
                'word': word,
                'sense_idx': sense_idx,
                'found': False
            }

        synset = synsets[sense_idx]

        return {
            'word': word,
            'sense_idx': sense_idx,
            'synset_id': synset.name(),
            'definition': synset.definition(),
            'examples': synset.examples(),
            'lemmas': [l.name() for l in synset.lemmas()],
            'hypernyms': [h.name() for h in synset.hypernyms()],
            'found': True
        }


# Helper functions for creating word contexts from parse trees

def extract_word_contexts_from_parse(sentence_id: int, hypothesis: Hypothesis) -> List[Tuple[str, WordContext]]:
    """
    Extract word contexts from a parsed sentence.

    Args:
        sentence_id: Unique sentence identifier
        hypothesis: Parsed hypothesis

    Returns:
        List of (word, context) tuples
    """
    contexts = []

    # Build neighbor map from edges
    neighbors_map = {}
    for edge in hypothesis.edges:
        parent = edge.parent
        child = edge.child

        if parent not in neighbors_map:
            neighbors_map[parent] = []
        if child not in neighbors_map:
            neighbors_map[child] = []

        # Both directions
        neighbors_map[parent].append((child, edge.type))
        neighbors_map[child].append((parent, edge.type))

    # Extract context for each word node
    for node_idx, node in enumerate(hypothesis.nodes):
        if node.value and hasattr(node.value, 'text'):
            word = node.value.text
            pos_tag = node.pos

            # Get neighbors
            neighbor_words = []
            syntactic_role = None

            if node_idx in neighbors_map:
                for neighbor_idx, edge_type in neighbors_map[node_idx]:
                    neighbor_node = hypothesis.nodes[neighbor_idx]
                    if neighbor_node.value and hasattr(neighbor_node.value, 'text'):
                        neighbor_words.append(neighbor_node.value.text)

                    # Track syntactic role
                    if edge_type in [ConnectionType.SUBJECT, ConnectionType.OBJECT]:
                        syntactic_role = edge_type

            context = WordContext(
                sentence_id=sentence_id,
                word=word,
                pos_tag=pos_tag,
                syntactic_role=syntactic_role,
                neighbors=neighbor_words,
                parse_tree=hypothesis
            )

            contexts.append((word, context))

    return contexts
