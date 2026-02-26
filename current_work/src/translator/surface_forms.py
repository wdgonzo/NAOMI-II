"""
Surface form selection — find the correct inflected form for a target language word.

Given a lemma + required morphological features (gender, number, person),
returns the correctly inflected surface form.

Delegates to MorphologyEngine for rule-based inflection. Only definite article
tables remain here (small, correct, closed-class).
"""

from typing import List, Optional
from ..parser.enums import SubType, Tag
from .morphology import MorphologyEngine


# ============================================================================
# Definite article tables: (gender, number) -> article
# ============================================================================

DEFINITE_ARTICLES = {
    'english': lambda g, n: 'the',
    'spanish': {
        (SubType.MASCULINE, SubType.SINGULAR): 'el',
        (SubType.FEMININE, SubType.SINGULAR): 'la',
        (SubType.MASCULINE, SubType.PLURAL): 'los',
        (SubType.FEMININE, SubType.PLURAL): 'las',
    },
    'french': {
        (SubType.MASCULINE, SubType.SINGULAR): 'le',
        (SubType.FEMININE, SubType.SINGULAR): 'la',
        (SubType.PLURAL,): 'les',
        (SubType.MASCULINE, SubType.PLURAL): 'les',
        (SubType.FEMININE, SubType.PLURAL): 'les',
    },
    'german': {
        (SubType.MASCULINE, SubType.SINGULAR): 'der',
        (SubType.FEMININE, SubType.SINGULAR): 'die',
        (SubType.NEUTER, SubType.SINGULAR): 'das',
        (SubType.PLURAL,): 'die',
        (SubType.MASCULINE, SubType.PLURAL): 'die',
        (SubType.FEMININE, SubType.PLURAL): 'die',
        (SubType.NEUTER, SubType.PLURAL): 'die',
    },
    'portuguese': {
        (SubType.MASCULINE, SubType.SINGULAR): 'o',
        (SubType.FEMININE, SubType.SINGULAR): 'a',
        (SubType.MASCULINE, SubType.PLURAL): 'os',
        (SubType.FEMININE, SubType.PLURAL): 'as',
    },
    'japanese': None,  # No articles
}


class SurfaceFormSelector:
    """Select the correct inflected surface form given a lemma + features."""

    def __init__(self, language: str):
        self.language = language
        self.morphology = MorphologyEngine(language)

    def conjugate_verb(self, lemma: str, features: List[SubType]) -> str:
        """Get conjugated verb form."""
        person = None
        number = SubType.SINGULAR  # default

        for f in features:
            if f in (SubType.FIRST_PERSON, SubType.SECOND_PERSON,
                     SubType.THIRD_PERSON):
                person = f
            if f in (SubType.SINGULAR, SubType.PLURAL):
                number = f

        if person is None:
            person = SubType.THIRD_PERSON  # default

        return self.morphology.conjugate_verb(lemma, person, number)

    def inflect_adjective(self, lemma: str, gender: SubType = None,
                          number: SubType = SubType.SINGULAR) -> str:
        """Get correctly inflected adjective form."""
        return self.morphology.inflect_adjective(lemma, gender, number)

    def get_noun_gender(self, lemma: str) -> Optional[SubType]:
        """Get the grammatical gender of a noun in this language."""
        return self.morphology.detect_gender(lemma)

    def get_definite_article(self, gender: SubType = None,
                             number: SubType = SubType.SINGULAR) -> Optional[str]:
        """Get the correct definite article for given gender/number."""
        articles = DEFINITE_ARTICLES.get(self.language)
        if articles is None:
            return None  # Japanese, etc.
        if callable(articles):
            return articles(gender, number)

        # Dict lookup
        keys_to_try = []
        if gender and number:
            keys_to_try.append((gender, number))
        if number:
            keys_to_try.append((number,))
        if gender:
            keys_to_try.append((gender,))

        for key in keys_to_try:
            if key in articles:
                return articles[key]

        return None
