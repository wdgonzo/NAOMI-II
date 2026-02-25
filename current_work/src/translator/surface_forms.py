"""
Surface form selection — find the correct inflected form for a target language word.

Given a lemma + required morphological features (gender, number, person),
returns the correctly inflected surface form.
"""

from typing import List, Optional, Dict, Tuple, FrozenSet
from ..parser.enums import SubType, Tag


# ============================================================================
# Verb conjugation tables: lemma -> {(person, number): surface_form}
# Only 3rd person singular needed for demo scope, but include others for
# completeness.
# ============================================================================

SPANISH_VERBS = {
    'correr': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'corro',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'corres',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'corre',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'corremos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'corren',
    },
    'comer': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'como',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'comes',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'come',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'comemos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'comen',
    },
    'perseguir': {
        (SubType.FIRST_PERSON, SubType.SINGULAR): 'persigo',
        (SubType.SECOND_PERSON, SubType.SINGULAR): 'persigues',
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'persigue',
        (SubType.FIRST_PERSON, SubType.PLURAL): 'perseguimos',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'persiguen',
    },
    'ver': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 've',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'ven',
    },
    'hacer': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'hace',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'hacen',
    },
    'tener': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'tiene',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'tienen',
    },
    'vivir': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'vive',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'viven',
    },
    'leer': {
        (SubType.THIRD_PERSON, SubType.SINGULAR): 'lee',
        (SubType.THIRD_PERSON, SubType.PLURAL): 'leen',
    },
}

FRENCH_VERBS = {
    'courir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'court'},
    'poursuivre': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'poursuit'},
    'manger': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'mange'},
    'voir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'voit'},
}

GERMAN_VERBS = {
    'rennen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'rennt'},
    'jagen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'jagt'},
    'essen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'isst'},
    'sehen': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sieht'},
}

PORTUGUESE_VERBS = {
    'correr': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'corre'},
    'comer': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'come'},
    'perseguir': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'persegue'},
    'ver': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'vê'},
}

ENGLISH_VERBS = {
    'run': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'runs'},
    'chase': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'chases'},
    'eat': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'eats'},
    'see': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'sees'},
    'make': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'makes'},
    'have': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'has'},
    'live': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'lives'},
    'read': {(SubType.THIRD_PERSON, SubType.SINGULAR): 'reads'},
}

VERB_TABLES = {
    'english': ENGLISH_VERBS,
    'spanish': SPANISH_VERBS,
    'french': FRENCH_VERBS,
    'german': GERMAN_VERBS,
    'portuguese': PORTUGUESE_VERBS,
    'japanese': {},  # No conjugation
}


# ============================================================================
# Adjective form tables: lemma -> {(gender, number): surface_form}
# ============================================================================

SPANISH_ADJS = {
    'grande': {
        (): 'grande',
        (SubType.SINGULAR,): 'grande',
        (SubType.PLURAL,): 'grandes',
        (SubType.MASCULINE, SubType.SINGULAR): 'grande',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grandes',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'pequeño': {
        (SubType.MASCULINE, SubType.SINGULAR): 'pequeño',
        (SubType.FEMININE, SubType.SINGULAR): 'pequeña',
        (SubType.MASCULINE, SubType.PLURAL): 'pequeños',
        (SubType.FEMININE, SubType.PLURAL): 'pequeñas',
    },
    'blanco': {
        (SubType.MASCULINE, SubType.SINGULAR): 'blanco',
        (SubType.FEMININE, SubType.SINGULAR): 'blanca',
        (SubType.MASCULINE, SubType.PLURAL): 'blancos',
        (SubType.FEMININE, SubType.PLURAL): 'blancas',
    },
    'negro': {
        (SubType.MASCULINE, SubType.SINGULAR): 'negro',
        (SubType.FEMININE, SubType.SINGULAR): 'negra',
        (SubType.MASCULINE, SubType.PLURAL): 'negros',
        (SubType.FEMININE, SubType.PLURAL): 'negras',
    },
    'rojo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rojo',
        (SubType.FEMININE, SubType.SINGULAR): 'roja',
        (SubType.MASCULINE, SubType.PLURAL): 'rojos',
        (SubType.FEMININE, SubType.PLURAL): 'rojas',
    },
    'azul': {
        (SubType.SINGULAR,): 'azul',
        (SubType.PLURAL,): 'azules',
        (SubType.MASCULINE, SubType.SINGULAR): 'azul',
        (SubType.FEMININE, SubType.SINGULAR): 'azul',
        (SubType.MASCULINE, SubType.PLURAL): 'azules',
        (SubType.FEMININE, SubType.PLURAL): 'azules',
    },
    'verde': {
        (SubType.SINGULAR,): 'verde',
        (SubType.PLURAL,): 'verdes',
        (SubType.MASCULINE, SubType.SINGULAR): 'verde',
        (SubType.FEMININE, SubType.SINGULAR): 'verde',
        (SubType.MASCULINE, SubType.PLURAL): 'verdes',
        (SubType.FEMININE, SubType.PLURAL): 'verdes',
    },
    'bueno': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bueno',
        (SubType.FEMININE, SubType.SINGULAR): 'buena',
        (SubType.MASCULINE, SubType.PLURAL): 'buenos',
        (SubType.FEMININE, SubType.PLURAL): 'buenas',
    },
    'malo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'malo',
        (SubType.FEMININE, SubType.SINGULAR): 'mala',
        (SubType.MASCULINE, SubType.PLURAL): 'malos',
        (SubType.FEMININE, SubType.PLURAL): 'malas',
    },
    'viejo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'viejo',
        (SubType.FEMININE, SubType.SINGULAR): 'vieja',
        (SubType.MASCULINE, SubType.PLURAL): 'viejos',
        (SubType.FEMININE, SubType.PLURAL): 'viejas',
    },
    'nuevo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'nuevo',
        (SubType.FEMININE, SubType.SINGULAR): 'nueva',
        (SubType.MASCULINE, SubType.PLURAL): 'nuevos',
        (SubType.FEMININE, SubType.PLURAL): 'nuevas',
    },
    'bonito': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bonito',
        (SubType.FEMININE, SubType.SINGULAR): 'bonita',
        (SubType.MASCULINE, SubType.PLURAL): 'bonitos',
        (SubType.FEMININE, SubType.PLURAL): 'bonitas',
    },
    'feo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'feo',
        (SubType.FEMININE, SubType.SINGULAR): 'fea',
        (SubType.MASCULINE, SubType.PLURAL): 'feos',
        (SubType.FEMININE, SubType.PLURAL): 'feas',
    },
}

FRENCH_ADJS = {
    'grand': {
        (SubType.MASCULINE, SubType.SINGULAR): 'grand',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grands',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'petit': {
        (SubType.MASCULINE, SubType.SINGULAR): 'petit',
        (SubType.FEMININE, SubType.SINGULAR): 'petite',
        (SubType.MASCULINE, SubType.PLURAL): 'petits',
        (SubType.FEMININE, SubType.PLURAL): 'petites',
    },
    'blanc': {
        (SubType.MASCULINE, SubType.SINGULAR): 'blanc',
        (SubType.FEMININE, SubType.SINGULAR): 'blanche',
        (SubType.MASCULINE, SubType.PLURAL): 'blancs',
        (SubType.FEMININE, SubType.PLURAL): 'blanches',
    },
    'noir': {
        (SubType.MASCULINE, SubType.SINGULAR): 'noir',
        (SubType.FEMININE, SubType.SINGULAR): 'noire',
        (SubType.MASCULINE, SubType.PLURAL): 'noirs',
        (SubType.FEMININE, SubType.PLURAL): 'noires',
    },
    'rouge': {
        (SubType.MASCULINE, SubType.SINGULAR): 'rouge',
        (SubType.FEMININE, SubType.SINGULAR): 'rouge',
        (SubType.MASCULINE, SubType.PLURAL): 'rouges',
        (SubType.FEMININE, SubType.PLURAL): 'rouges',
    },
    'bon': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bon',
        (SubType.FEMININE, SubType.SINGULAR): 'bonne',
        (SubType.MASCULINE, SubType.PLURAL): 'bons',
        (SubType.FEMININE, SubType.PLURAL): 'bonnes',
    },
    'beau': {
        (SubType.MASCULINE, SubType.SINGULAR): 'beau',
        (SubType.FEMININE, SubType.SINGULAR): 'belle',
        (SubType.MASCULINE, SubType.PLURAL): 'beaux',
        (SubType.FEMININE, SubType.PLURAL): 'belles',
    },
    'vieux': {
        (SubType.MASCULINE, SubType.SINGULAR): 'vieux',
        (SubType.FEMININE, SubType.SINGULAR): 'vieille',
        (SubType.MASCULINE, SubType.PLURAL): 'vieux',
        (SubType.FEMININE, SubType.PLURAL): 'vieilles',
    },
    'nouveau': {
        (SubType.MASCULINE, SubType.SINGULAR): 'nouveau',
        (SubType.FEMININE, SubType.SINGULAR): 'nouvelle',
        (SubType.MASCULINE, SubType.PLURAL): 'nouveaux',
        (SubType.FEMININE, SubType.PLURAL): 'nouvelles',
    },
}

GERMAN_ADJS = {
    # Simplified: strong inflection nominative only for demo
    'gross': {
        (SubType.MASCULINE, SubType.SINGULAR): 'grosse',
        (SubType.FEMININE, SubType.SINGULAR): 'grosse',
        (SubType.NEUTER, SubType.SINGULAR): 'grosses',
        (SubType.PLURAL,): 'grossen',
    },
    'klein': {
        (SubType.MASCULINE, SubType.SINGULAR): 'kleine',
        (SubType.FEMININE, SubType.SINGULAR): 'kleine',
        (SubType.NEUTER, SubType.SINGULAR): 'kleines',
    },
    'weiss': {
        (SubType.MASCULINE, SubType.SINGULAR): 'weisser',
        (SubType.FEMININE, SubType.SINGULAR): 'weisse',
        (SubType.NEUTER, SubType.SINGULAR): 'weisses',
    },
    'gut': {
        (SubType.MASCULINE, SubType.SINGULAR): 'guter',
        (SubType.FEMININE, SubType.SINGULAR): 'gute',
        (SubType.NEUTER, SubType.SINGULAR): 'gutes',
    },
}

PORTUGUESE_ADJS = {
    'grande': {
        (): 'grande',
        (SubType.SINGULAR,): 'grande',
        (SubType.PLURAL,): 'grandes',
        (SubType.MASCULINE, SubType.SINGULAR): 'grande',
        (SubType.FEMININE, SubType.SINGULAR): 'grande',
        (SubType.MASCULINE, SubType.PLURAL): 'grandes',
        (SubType.FEMININE, SubType.PLURAL): 'grandes',
    },
    'branco': {
        (SubType.MASCULINE, SubType.SINGULAR): 'branco',
        (SubType.FEMININE, SubType.SINGULAR): 'branca',
        (SubType.MASCULINE, SubType.PLURAL): 'brancos',
        (SubType.FEMININE, SubType.PLURAL): 'brancas',
    },
    'preto': {
        (SubType.MASCULINE, SubType.SINGULAR): 'preto',
        (SubType.FEMININE, SubType.SINGULAR): 'preta',
        (SubType.MASCULINE, SubType.PLURAL): 'pretos',
        (SubType.FEMININE, SubType.PLURAL): 'pretas',
    },
    'pequeno': {
        (SubType.MASCULINE, SubType.SINGULAR): 'pequeno',
        (SubType.FEMININE, SubType.SINGULAR): 'pequena',
        (SubType.MASCULINE, SubType.PLURAL): 'pequenos',
        (SubType.FEMININE, SubType.PLURAL): 'pequenas',
    },
    'bonito': {
        (SubType.MASCULINE, SubType.SINGULAR): 'bonito',
        (SubType.FEMININE, SubType.SINGULAR): 'bonita',
        (SubType.MASCULINE, SubType.PLURAL): 'bonitos',
        (SubType.FEMININE, SubType.PLURAL): 'bonitas',
    },
    'velho': {
        (SubType.MASCULINE, SubType.SINGULAR): 'velho',
        (SubType.FEMININE, SubType.SINGULAR): 'velha',
        (SubType.MASCULINE, SubType.PLURAL): 'velhos',
        (SubType.FEMININE, SubType.PLURAL): 'velhas',
    },
    'novo': {
        (SubType.MASCULINE, SubType.SINGULAR): 'novo',
        (SubType.FEMININE, SubType.SINGULAR): 'nova',
        (SubType.MASCULINE, SubType.PLURAL): 'novos',
        (SubType.FEMININE, SubType.PLURAL): 'novas',
    },
}

ADJ_TABLES = {
    'english': {},  # English adjectives don't inflect
    'spanish': SPANISH_ADJS,
    'french': FRENCH_ADJS,
    'german': GERMAN_ADJS,
    'portuguese': PORTUGUESE_ADJS,
    'japanese': {},  # Japanese adjectives in romaji don't inflect for demo
}


# ============================================================================
# Noun gender registry: lemma -> gender (for target language agreement)
# ============================================================================

NOUN_GENDERS = {
    'spanish': {
        'perro': SubType.MASCULINE, 'gato': SubType.MASCULINE,
        'casa': SubType.FEMININE, 'libro': SubType.MASCULINE,
        'mesa': SubType.FEMININE, 'silla': SubType.FEMININE,
        'hombre': SubType.MASCULINE, 'mujer': SubType.FEMININE,
        'niño': SubType.MASCULINE, 'niña': SubType.FEMININE,
        'pájaro': SubType.MASCULINE, 'ratón': SubType.MASCULINE,
        'coche': SubType.MASCULINE, 'parque': SubType.MASCULINE,
        'ciudad': SubType.FEMININE, 'agua': SubType.FEMININE,
        'comida': SubType.FEMININE, 'día': SubType.MASCULINE,
        'noche': SubType.FEMININE,
    },
    'french': {
        'chien': SubType.MASCULINE, 'chat': SubType.MASCULINE,
        'maison': SubType.FEMININE, 'livre': SubType.MASCULINE,
        'homme': SubType.MASCULINE, 'femme': SubType.FEMININE,
        'enfant': SubType.MASCULINE, 'oiseau': SubType.MASCULINE,
        'souris': SubType.FEMININE,
    },
    'german': {
        'Hund': SubType.MASCULINE, 'Katze': SubType.FEMININE,
        'Haus': SubType.NEUTER, 'Buch': SubType.NEUTER,
        'Mann': SubType.MASCULINE, 'Frau': SubType.FEMININE,
        'Kind': SubType.NEUTER, 'Vogel': SubType.MASCULINE,
        'Maus': SubType.FEMININE,
    },
    'portuguese': {
        'cachorro': SubType.MASCULINE, 'gato': SubType.MASCULINE,
        'casa': SubType.FEMININE, 'livro': SubType.MASCULINE,
        'homem': SubType.MASCULINE, 'mulher': SubType.FEMININE,
        'criança': SubType.FEMININE, 'pássaro': SubType.MASCULINE,
        'rato': SubType.MASCULINE,
    },
}


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
        self.verb_table = VERB_TABLES.get(language, {})
        self.adj_table = ADJ_TABLES.get(language, {})
        self.noun_genders = NOUN_GENDERS.get(language, {})

    def conjugate_verb(self, lemma: str, features: List[SubType]) -> str:
        """Get conjugated verb form."""
        if lemma not in self.verb_table:
            return lemma

        # Extract person and number
        person = None
        number = SubType.SINGULAR  # default
        for f in features:
            if f in (SubType.FIRST_PERSON, SubType.SECOND_PERSON, SubType.THIRD_PERSON):
                person = f
            if f in (SubType.SINGULAR, SubType.PLURAL):
                number = f

        if person is None:
            person = SubType.THIRD_PERSON  # default

        key = (person, number)
        forms = self.verb_table[lemma]
        if key in forms:
            return forms[key]
        return lemma

    def inflect_adjective(self, lemma: str, gender: SubType = None,
                          number: SubType = SubType.SINGULAR) -> str:
        """Get correctly inflected adjective form."""
        if lemma not in self.adj_table:
            return lemma

        forms = self.adj_table[lemma]

        # Try most specific key first, then fall back
        keys_to_try = []
        if gender and number:
            keys_to_try.append((gender, number))
        if number:
            keys_to_try.append((number,))
        keys_to_try.append(())

        for key in keys_to_try:
            if key in forms:
                return forms[key]

        # Return lemma if no match
        return lemma

    def get_noun_gender(self, lemma: str) -> Optional[SubType]:
        """Get the grammatical gender of a noun in this language."""
        return self.noun_genders.get(lemma)

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
