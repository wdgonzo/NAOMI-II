"""Test structure-based translator."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree
from src.parser.pos_tagger import (
    tag_sentence, tag_spanish_sentence,
    tag_french_sentence, tag_german_sentence,
    tag_portuguese_sentence, tag_japanese_sentence,
)
from src.translator import Translator


GRAMMAR_DIR = os.path.join(_ROOT, "grammars")

TAGGERS = {
    'english': tag_sentence,
    'spanish': tag_spanish_sentence,
    'french': tag_french_sentence,
    'german': tag_german_sentence,
    'portuguese': tag_portuguese_sentence,
    'japanese': tag_japanese_sentence,
}


def parse(sentence: str, language: str):
    """Parse a sentence and return the best hypothesis."""
    tagger = TAGGERS[language]
    words = tagger(sentence)
    grammar_path = os.path.join(GRAMMAR_DIR, f"{language}.json")
    parser = QuantumParser(grammar_path)
    chart = parser.parse(words)
    return chart.best_hypothesis()


def test_en_to_es_intransitive():
    """'The dog runs' -> 'El perro corre'"""
    print("=== EN -> ES: The dog runs ===")
    hyp = parse("The dog runs", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "el perro corre", f"Expected 'El perro corre', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_ja_intransitive():
    """'The dog runs' -> 'inu ga hashiru'"""
    print("=== EN -> JA: The dog runs ===")
    hyp = parse("The dog runs", "english")
    t = Translator("english", "japanese")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result == "inu ga hashiru", f"Expected 'inu ga hashiru', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_es_transitive():
    """'The dog chases the cat' -> 'El perro persigue el gato'"""
    print("=== EN -> ES: The dog chases the cat ===")
    hyp = parse("The dog chases the cat", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "el perro persigue el gato", \
        f"Expected 'El perro persigue el gato', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_ja_transitive():
    """'The dog chases the cat' -> 'inu ga neko wo ou'"""
    print("=== EN -> JA: The dog chases the cat ===")
    hyp = parse("The dog chases the cat", "english")
    t = Translator("english", "japanese")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result == "inu ga neko wo ou", f"Expected 'inu ga neko wo ou', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_de_intransitive():
    """'The dog runs' -> 'Der Hund rennt'"""
    print("=== EN -> DE: The dog runs ===")
    hyp = parse("The dog runs", "english")
    t = Translator("english", "german")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "der hund rennt", \
        f"Expected 'Der Hund rennt', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_fr_intransitive():
    """'The dog runs' -> 'Le chien court'"""
    print("=== EN -> FR: The dog runs ===")
    hyp = parse("The dog runs", "english")
    t = Translator("english", "french")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "le chien court", \
        f"Expected 'Le chien court', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_pt_intransitive():
    """'The dog runs' -> 'O cachorro corre'"""
    print("=== EN -> PT: The dog runs ===")
    hyp = parse("The dog runs", "english")
    t = Translator("english", "portuguese")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "o cachorro corre", \
        f"Expected 'O cachorro corre', got '{result}'"
    print("  [OK]")
    print()


def test_es_to_en_intransitive():
    """'El perro corre' -> 'The dog runs'"""
    print("=== ES -> EN: El perro corre ===")
    hyp = parse("El perro corre", "spanish")
    t = Translator("spanish", "english")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "the dog runs", \
        f"Expected 'The dog runs', got '{result}'"
    print("  [OK]")
    print()


def test_ja_to_en_intransitive():
    """'inu ga hashiru' -> 'The dog runs'"""
    print("=== JA -> EN: inu ga hashiru ===")
    hyp = parse("inu ga hashiru", "japanese")
    t = Translator("japanese", "english")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    assert result.lower() == "the dog runs", \
        f"Expected 'The dog runs', got '{result}'"
    print("  [OK]")
    print()


def test_en_to_es_transitive_full():
    """'The big dog runs quickly' -> 'El perro grande corre rapidamente'"""
    print("=== EN -> ES: The big dog runs quickly ===")
    hyp = parse("The big dog runs quickly", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    lower = result.lower()
    assert "perro" in lower, f"Missing 'perro' in '{result}'"
    assert "corre" in lower, f"Missing 'corre' in '{result}'"
    # Adjective must be AFTER noun in Spanish (post-nominal)
    perro_idx = lower.index("perro")
    if "grande" in lower:
        grande_idx = lower.index("grande")
        assert grande_idx > perro_idx, \
            f"Adjective should be after noun in Spanish: '{result}'"
    print("  [OK]")
    print()


def test_possessive_en_to_es():
    """'My dog runs' -> 'Mi perro corre'"""
    print("=== EN -> ES: My dog runs ===")
    hyp = parse("My dog runs", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    lower = result.lower()
    assert "mi" in lower, f"Missing possessive 'mi' in '{result}'"
    assert "perro" in lower, f"Missing 'perro' in '{result}'"
    assert "corre" in lower, f"Missing 'corre' in '{result}'"
    print("  [OK]")
    print()


def test_new_vocab_en_to_es():
    """'The mother loves the child' -> 'La madre ama el niño'"""
    print("=== EN -> ES: The mother loves the child ===")
    hyp = parse("The mother loves the child", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    lower = result.lower()
    assert "madre" in lower, f"Missing 'madre' in '{result}'"
    assert "ama" in lower, f"Missing 'ama' in '{result}'"
    assert "niño" in lower, f"Missing 'niño' in '{result}'"
    # Feminine article for 'madre'
    assert "la madre" in lower, \
        f"Expected 'la madre' (feminine article) in '{result}'"
    print("  [OK]")
    print()


def test_coordination():
    """'The dog and the cat' -> coordination in other languages"""
    print("=== EN -> ES: The dog and the cat ===")
    hyp = parse("The dog and the cat", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    lower = result.lower()
    assert "perro" in lower, f"Missing 'perro' in '{result}'"
    assert "gato" in lower, f"Missing 'gato' in '{result}'"
    assert "y" in lower, f"Missing 'y' in '{result}'"
    print("  [OK]")
    print()


def test_morphology_verb_conjugation():
    """Test rule-based verb conjugation directly."""
    print("=== Morphology: Verb conjugation ===")
    from src.translator.morphology import MorphologyEngine
    from src.parser.enums import SubType

    # Spanish regular -ar verb
    es = MorphologyEngine('spanish')
    assert es.conjugate_verb('hablar', SubType.THIRD_PERSON, SubType.SINGULAR) == 'habla'
    assert es.conjugate_verb('hablar', SubType.FIRST_PERSON, SubType.SINGULAR) == 'hablo'
    assert es.conjugate_verb('hablar', SubType.FIRST_PERSON, SubType.PLURAL) == 'hablamos'

    # Spanish regular -er verb
    assert es.conjugate_verb('comer', SubType.THIRD_PERSON, SubType.SINGULAR) == 'come'

    # Spanish regular -ir verb
    assert es.conjugate_verb('vivir', SubType.THIRD_PERSON, SubType.SINGULAR) == 'vive'

    # Spanish irregular verb
    assert es.conjugate_verb('ser', SubType.THIRD_PERSON, SubType.SINGULAR) == 'es'
    assert es.conjugate_verb('ir', SubType.THIRD_PERSON, SubType.SINGULAR) == 'va'

    # Spanish stem-changer (e->ie)
    assert es.conjugate_verb('pensar', SubType.THIRD_PERSON, SubType.SINGULAR) == 'piensa'
    assert es.conjugate_verb('pensar', SubType.FIRST_PERSON, SubType.PLURAL) == 'pensamos'

    # Spanish stem-changer (o->ue)
    assert es.conjugate_verb('dormir', SubType.THIRD_PERSON, SubType.SINGULAR) == 'duerme'

    # Spanish stem-changer (e->i)
    assert es.conjugate_verb('perseguir', SubType.THIRD_PERSON, SubType.SINGULAR) == 'persigue'

    # French irregular
    fr = MorphologyEngine('french')
    assert fr.conjugate_verb('être', SubType.THIRD_PERSON, SubType.SINGULAR) == 'est'
    assert fr.conjugate_verb('courir', SubType.THIRD_PERSON, SubType.SINGULAR) == 'court'

    # French regular -er
    assert fr.conjugate_verb('parler', SubType.THIRD_PERSON, SubType.SINGULAR) == 'parle'

    # German irregular
    de = MorphologyEngine('german')
    assert de.conjugate_verb('sein', SubType.THIRD_PERSON, SubType.SINGULAR) == 'ist'

    # German regular
    assert de.conjugate_verb('rennen', SubType.THIRD_PERSON, SubType.SINGULAR) == 'rennt'

    # German stem-vowel change
    assert de.conjugate_verb('sehen', SubType.THIRD_PERSON, SubType.SINGULAR) == 'sieht'
    assert de.conjugate_verb('schlafen', SubType.THIRD_PERSON, SubType.SINGULAR) == 'schläft'

    # Portuguese irregular
    pt = MorphologyEngine('portuguese')
    assert pt.conjugate_verb('ser', SubType.THIRD_PERSON, SubType.SINGULAR) == 'é'

    # Portuguese regular -er
    assert pt.conjugate_verb('correr', SubType.THIRD_PERSON, SubType.SINGULAR) == 'corre'

    # English 3rd person
    en = MorphologyEngine('english')
    assert en.conjugate_verb('run', SubType.THIRD_PERSON, SubType.SINGULAR) == 'runs'
    assert en.conjugate_verb('chase', SubType.THIRD_PERSON, SubType.SINGULAR) == 'chases'
    assert en.conjugate_verb('fly', SubType.THIRD_PERSON, SubType.SINGULAR) == 'flies'
    assert en.conjugate_verb('have', SubType.THIRD_PERSON, SubType.SINGULAR) == 'has'

    print("  [OK]")
    print()


def test_morphology_gender_detection():
    """Test rule-based gender detection."""
    print("=== Morphology: Gender detection ===")
    from src.translator.morphology import MorphologyEngine
    from src.parser.enums import SubType

    es = MorphologyEngine('spanish')
    # -o masculine
    assert es.detect_gender('perro') == SubType.MASCULINE
    assert es.detect_gender('gato') == SubType.MASCULINE
    # -a feminine
    assert es.detect_gender('casa') == SubType.FEMININE
    assert es.detect_gender('mesa') == SubType.FEMININE
    # Exceptions
    assert es.detect_gender('mano') == SubType.FEMININE
    assert es.detect_gender('día') == SubType.MASCULINE
    assert es.detect_gender('madre') == SubType.FEMININE
    assert es.detect_gender('flor') == SubType.FEMININE
    # Suffix heuristics
    assert es.detect_gender('ciudad') == SubType.FEMININE  # -dad ending
    assert es.detect_gender('nación') == SubType.FEMININE  # -ción ending

    # French
    fr = MorphologyEngine('french')
    assert fr.detect_gender('mère') == SubType.FEMININE
    assert fr.detect_gender('maison') == SubType.FEMININE
    assert fr.detect_gender('homme') == SubType.MASCULINE

    # German (manual table)
    de = MorphologyEngine('german')
    assert de.detect_gender('Hund') == SubType.MASCULINE
    assert de.detect_gender('Katze') == SubType.FEMININE
    assert de.detect_gender('Haus') == SubType.NEUTER

    # English/Japanese return None
    en = MorphologyEngine('english')
    assert en.detect_gender('dog') is None

    print("  [OK]")
    print()


def test_morphology_adjective_inflection():
    """Test rule-based adjective inflection."""
    print("=== Morphology: Adjective inflection ===")
    from src.translator.morphology import MorphologyEngine
    from src.parser.enums import SubType

    es = MorphologyEngine('spanish')
    # -o adjective: 4-form
    assert es.inflect_adjective('blanco', SubType.FEMININE) == 'blanca'
    assert es.inflect_adjective('blanco', SubType.MASCULINE, SubType.PLURAL) == 'blancos'
    assert es.inflect_adjective('blanco', SubType.FEMININE, SubType.PLURAL) == 'blancas'
    # -e adjective: invariant gender
    assert es.inflect_adjective('grande', SubType.FEMININE) == 'grande'
    assert es.inflect_adjective('grande', SubType.MASCULINE, SubType.PLURAL) == 'grandes'
    # consonant: invariant gender, +es plural
    assert es.inflect_adjective('feliz', SubType.FEMININE) == 'feliz'
    assert es.inflect_adjective('feliz', SubType.MASCULINE, SubType.PLURAL) == 'felices'

    # French irregular
    fr = MorphologyEngine('french')
    assert fr.inflect_adjective('blanc', SubType.FEMININE) == 'blanche'
    assert fr.inflect_adjective('beau', SubType.FEMININE) == 'belle'

    # English: no inflection
    en = MorphologyEngine('english')
    assert en.inflect_adjective('big', SubType.FEMININE) == 'big'

    print("  [OK]")
    print()


def test_feminine_article_agreement():
    """Test that feminine nouns get correct articles across languages."""
    print("=== Feminine article agreement ===")

    # Spanish: La casa blanca
    hyp = parse("The white house", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  EN->ES 'The white house': '{result}'")
    assert "la casa" in result.lower(), \
        f"Expected feminine 'la casa', got '{result}'"

    # French: La maison blanche
    t = Translator("english", "french")
    result = t.translate(hyp)
    print(f"  EN->FR 'The white house': '{result}'")
    assert "la maison" in result.lower(), \
        f"Expected feminine 'la maison', got '{result}'"

    # German: Das Haus (neuter)
    t = Translator("english", "german")
    result = t.translate(hyp)
    print(f"  EN->DE 'The white house': '{result}'")
    assert "das" in result.lower(), \
        f"Expected neuter 'das' for Haus, got '{result}'"

    print("  [OK]")
    print()


def test_question_sentence():
    """'what is your favorite color' -> Spanish translation."""
    print("=== EN -> ES: what is your favorite color ===")
    hyp = parse("what is your favorite color", "english")
    t = Translator("english", "spanish")
    result = t.translate(hyp)
    print(f"  Result: '{result}'")
    lower = result.lower()
    assert "color" in lower, f"Missing 'color' in '{result}'"
    assert "favorito" in lower, f"Missing 'favorito' in '{result}'"
    print("  [OK]")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSLATOR TESTS")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    errors = []

    tests = [
        test_en_to_es_intransitive,
        test_en_to_ja_intransitive,
        test_en_to_es_transitive,
        test_en_to_ja_transitive,
        test_en_to_de_intransitive,
        test_en_to_fr_intransitive,
        test_en_to_pt_intransitive,
        test_es_to_en_intransitive,
        test_ja_to_en_intransitive,
        test_en_to_es_transitive_full,
        test_possessive_en_to_es,
        test_new_vocab_en_to_es,
        test_coordination,
        test_morphology_verb_conjugation,
        test_morphology_gender_detection,
        test_morphology_adjective_inflection,
        test_feminine_article_agreement,
        test_question_sentence,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  [FAILED] {e}")
            print()

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print()
        print("FAILURES:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)
