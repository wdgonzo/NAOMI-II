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
    # Check key words are present and in right order
    lower = result.lower()
    assert "perro" in lower, f"Missing 'perro' in '{result}'"
    assert "corre" in lower, f"Missing 'corre' in '{result}'"
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
        test_coordination,
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
