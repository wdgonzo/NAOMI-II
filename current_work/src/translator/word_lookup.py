"""
Bilingual word lookup using manual dictionaries + NLTK OMW fallback.

Provides word equivalency across languages for the translator pipeline.
"""

from typing import Optional
from ..parser.enums import Tag


# NLTK WordNet POS mapping
_TAG_TO_WN_POS = {
    Tag.NOUN: 'n',
    Tag.VERB: 'v',
    Tag.ADJ: 'a',
    Tag.ADV: 'r',
}

# OMW language codes
OMW_CODES = {
    'english': 'eng',
    'spanish': 'spa',
    'french': 'fra',
    'german': 'deu',
    'portuguese': 'por',
    'japanese': 'jpn',
}

# Manual bilingual dictionaries keyed by (source, target) -> {word: translation}
# English is the hub — all pairs go through English lemmas.
# Content words only; function words (determiners, particles) are handled by the linearizer.
ENGLISH_LEMMAS = {
    # Nouns
    'spanish': {
        'dog': 'perro', 'cat': 'gato', 'house': 'casa', 'book': 'libro',
        'table': 'mesa', 'chair': 'silla', 'man': 'hombre', 'woman': 'mujer',
        'child': 'niño', 'boy': 'niño', 'girl': 'niña', 'bird': 'pájaro',
        'mouse': 'ratón', 'car': 'coche', 'park': 'parque', 'city': 'ciudad',
        'water': 'agua', 'food': 'comida', 'day': 'día', 'night': 'noche',
        # Verbs (English base -> Spanish infinitive root)
        'run': 'correr', 'runs': 'correr', 'eat': 'comer', 'eats': 'comer',
        'chase': 'perseguir', 'chases': 'perseguir',
        'see': 'ver', 'sees': 'ver', 'make': 'hacer', 'makes': 'hacer',
        'have': 'tener', 'has': 'tener', 'live': 'vivir', 'lives': 'vivir',
        'read': 'leer', 'reads': 'leer',
        # Adjectives (English -> Spanish masc singular base)
        'big': 'grande', 'small': 'pequeño', 'white': 'blanco', 'black': 'negro',
        'red': 'rojo', 'blue': 'azul', 'green': 'verde',
        'good': 'bueno', 'bad': 'malo', 'old': 'viejo', 'new': 'nuevo',
        'beautiful': 'bonito', 'ugly': 'feo',
        # Adverbs
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muy',
        'well': 'bien', 'badly': 'mal', 'always': 'siempre', 'never': 'nunca',
    },
    'french': {
        'dog': 'chien', 'cat': 'chat', 'house': 'maison', 'book': 'livre',
        'man': 'homme', 'woman': 'femme', 'child': 'enfant',
        'bird': 'oiseau', 'mouse': 'souris',
        'run': 'courir', 'runs': 'courir', 'eat': 'manger', 'eats': 'manger',
        'chase': 'poursuivre', 'chases': 'poursuivre',
        'see': 'voir', 'sees': 'voir',
        'big': 'grand', 'small': 'petit', 'white': 'blanc', 'black': 'noir',
        'red': 'rouge', 'blue': 'bleu', 'green': 'vert',
        'good': 'bon', 'bad': 'mauvais', 'old': 'vieux', 'new': 'nouveau',
        'beautiful': 'beau', 'ugly': 'laid',
        'quickly': 'vite', 'slowly': 'lentement', 'very': 'très',
    },
    'german': {
        'dog': 'Hund', 'cat': 'Katze', 'house': 'Haus', 'book': 'Buch',
        'man': 'Mann', 'woman': 'Frau', 'child': 'Kind',
        'bird': 'Vogel', 'mouse': 'Maus',
        'run': 'rennen', 'runs': 'rennen', 'eat': 'essen', 'eats': 'essen',
        'chase': 'jagen', 'chases': 'jagen',
        'see': 'sehen', 'sees': 'sehen',
        'big': 'gross', 'small': 'klein', 'white': 'weiss', 'black': 'schwarz',
        'red': 'rot', 'blue': 'blau', 'green': 'grün',
        'good': 'gut', 'bad': 'schlecht', 'old': 'alt', 'new': 'neu',
        'quickly': 'schnell', 'slowly': 'langsam', 'very': 'sehr',
    },
    'portuguese': {
        'dog': 'cachorro', 'cat': 'gato', 'house': 'casa', 'book': 'livro',
        'man': 'homem', 'woman': 'mulher', 'child': 'criança',
        'bird': 'pássaro', 'mouse': 'rato',
        'run': 'correr', 'runs': 'correr', 'eat': 'comer', 'eats': 'comer',
        'chase': 'perseguir', 'chases': 'perseguir',
        'see': 'ver', 'sees': 'ver',
        'big': 'grande', 'small': 'pequeno', 'white': 'branco', 'black': 'preto',
        'red': 'vermelho', 'blue': 'azul', 'green': 'verde',
        'good': 'bom', 'bad': 'mau', 'old': 'velho', 'new': 'novo',
        'beautiful': 'bonito', 'ugly': 'feio',
        'quickly': 'rapidamente', 'slowly': 'lentamente', 'very': 'muito',
    },
    'japanese': {
        'dog': 'inu', 'cat': 'neko', 'house': 'ie', 'book': 'hon',
        'man': 'otoko', 'woman': 'onna', 'child': 'kodomo',
        'bird': 'tori', 'mouse': 'nezumi',
        'run': 'hashiru', 'runs': 'hashiru', 'eat': 'taberu', 'eats': 'taberu',
        'chase': 'ou', 'chases': 'ou',
        'see': 'miru', 'sees': 'miru',
        'big': 'ooki', 'small': 'chiisai', 'white': 'shiroi', 'black': 'kuroi',
        'red': 'akai', 'blue': 'aoi', 'green': 'midori',
        'good': 'yoi', 'bad': 'warui', 'old': 'furui', 'new': 'atarashii',
        'quickly': 'hayaku', 'slowly': 'yukkuri', 'very': 'totemo',
    },
}


# Inflected forms -> English base form (for reverse translation)
# These map conjugated/inflected non-English words to their English equivalent.
INFLECTED_TO_ENGLISH = {
    'spanish': {
        # Verb conjugations -> English base
        'corre': 'run', 'corro': 'run', 'corren': 'run',
        'persigue': 'chase', 'persiguen': 'chase',
        'come': 'eat', 'comen': 'eat',
        've': 'see', 'ven': 'see',
        'hace': 'make', 'hacen': 'make',
        'tiene': 'have', 'tienen': 'have',
        'vive': 'live', 'viven': 'live',
        'lee': 'read', 'leen': 'read',
        # Adjective inflected forms
        'blanca': 'white', 'blancos': 'white', 'blancas': 'white',
        'negra': 'black', 'negros': 'black', 'negras': 'black',
        'roja': 'red', 'rojos': 'red', 'rojas': 'red',
        'pequeña': 'small', 'pequeños': 'small', 'pequeñas': 'small',
        'grandes': 'big',
    },
    'french': {
        'court': 'run', 'poursuit': 'chase', 'mange': 'eat', 'voit': 'see',
        'grande': 'big', 'grands': 'big', 'grandes': 'big',
        'petite': 'small', 'petits': 'small', 'petites': 'small',
        'blanche': 'white', 'blancs': 'white', 'blanches': 'white',
    },
    'german': {
        'rennt': 'run', 'jagt': 'chase', 'isst': 'eat', 'sieht': 'see',
        'grosse': 'big', 'grosser': 'big', 'grosses': 'big',
        'kleine': 'small', 'kleiner': 'small', 'kleines': 'small',
        'weisse': 'white', 'weisser': 'white', 'weisses': 'white',
    },
    'portuguese': {
        'corre': 'run', 'persegue': 'chase', 'come': 'eat', 'vê': 'see',
        'branca': 'white', 'brancos': 'white', 'brancas': 'white',
    },
    'japanese': {
        # Romaji forms (same as base in our system)
    },
}


def _build_all_dictionaries():
    """Build full bidirectional dictionary set from English hub."""
    dicts = {}

    for target_lang, en_to_target in ENGLISH_LEMMAS.items():
        # English -> target
        dicts[('english', target_lang)] = dict(en_to_target)

        # Target -> English (reverse): start with lemma mapping
        target_to_en = {}
        for en_word, target_word in en_to_target.items():
            if target_word not in target_to_en:
                target_to_en[target_word] = en_word

        # Add inflected forms
        inflected = INFLECTED_TO_ENGLISH.get(target_lang, {})
        for form, en_base in inflected.items():
            if form not in target_to_en:
                target_to_en[form] = en_base

        dicts[(target_lang, 'english')] = target_to_en

    # Build cross-language pairs via English pivot
    all_langs = list(ENGLISH_LEMMAS.keys())
    for i, lang_a in enumerate(all_langs):
        for lang_b in all_langs[i + 1:]:
            # lang_a -> lang_b: go lang_a -> English -> lang_b
            a_to_b = {}
            b_to_a = {}
            en_to_a = ENGLISH_LEMMAS[lang_a]
            en_to_b = ENGLISH_LEMMAS[lang_b]
            for en_word in en_to_a:
                if en_word in en_to_b:
                    a_to_b[en_to_a[en_word]] = en_to_b[en_word]
                    b_to_a[en_to_b[en_word]] = en_to_a[en_word]
            dicts[(lang_a, lang_b)] = a_to_b
            dicts[(lang_b, lang_a)] = b_to_a

    # Self-translation (identity)
    for lang in ['english'] + all_langs:
        dicts[(lang, lang)] = {}  # Will passthrough

    return dicts


_ALL_DICTS = _build_all_dictionaries()


class WordLookup:
    """Bilingual word lookup with manual dictionary + OMW fallback."""

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.manual_dict = _ALL_DICTS.get((source_lang, target_lang), {})
        self._wn = None
        self._wn_checked = False

    def _ensure_wordnet(self):
        """Lazy-load NLTK WordNet."""
        if self._wn_checked:
            return self._wn is not None
        self._wn_checked = True
        try:
            from nltk.corpus import wordnet as wn
            # Quick test to see if OMW data is available
            wn.synsets('dog', lang='eng')
            self._wn = wn
            return True
        except Exception:
            return False

    def lookup(self, word: str, pos: Tag = None) -> str:
        """
        Find target language equivalent for a source word.

        Returns target word or original word if not found.
        """
        if self.source_lang == self.target_lang:
            return word

        lower = word.lower()

        # 1. Manual dictionary (most reliable)
        if lower in self.manual_dict:
            return self.manual_dict[lower]
        if word in self.manual_dict:
            return self.manual_dict[word]

        # 2. NLTK OMW fallback
        if pos and self._ensure_wordnet():
            result = self._omw_lookup(lower, pos)
            if result:
                return result

        # 3. Passthrough
        return word

    def _omw_lookup(self, word: str, pos: Tag) -> Optional[str]:
        """Try NLTK Open Multilingual WordNet lookup."""
        wn = self._wn
        if wn is None:
            return None

        wn_pos = _TAG_TO_WN_POS.get(pos)
        if not wn_pos:
            return None

        source_code = OMW_CODES.get(self.source_lang, 'eng')
        target_code = OMW_CODES.get(self.target_lang)
        if not target_code:
            return None

        try:
            synsets = wn.synsets(word, pos=wn_pos, lang=source_code)
            for synset in synsets[:3]:
                lemmas = synset.lemma_names(target_code)
                if lemmas:
                    # Return first lemma, replacing underscores with spaces
                    return lemmas[0].replace('_', ' ')
        except Exception:
            pass

        return None
