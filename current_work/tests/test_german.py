"""Test quantum parser with German grammar."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

GRAMMAR = os.path.join(_ROOT, "grammars", "german.json")

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree


def test_simple_german():
    """Test: Der Hund rennt (The dog runs)"""
    print("=== Test 1: Der Hund rennt ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Der", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("Hund", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("rennt", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print(f"Unconsumed: {len(unconsumed)}")
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    root = best.nodes[list(unconsumed)[0]]
    print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


def test_transitive_german():
    """Test: Der Hund jagt die Katze (The dog chases the cat)"""
    print("=== Test 2: Der Hund jagt die Katze ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Der", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("Hund", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("jagt", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("die", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("Katze", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    print_hypothesis_tree(best)
    print()


def test_prenominal_adjective():
    """Test: Das weisse Haus (The white house) - pre-nominal with neuter"""
    print("=== Test 3: Das weisse Haus ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Das", Tag.DET, [SubType.NEUTER, SubType.SINGULAR]),
        Word("weisse", Tag.ADJ, [SubType.NEUTER, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("Haus", Tag.NOUN, [SubType.NEUTER, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert best.score > 0.5, f"Score too low: {best.score:.3f}"
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    else:
        print(f"Noun phrase: {len(unconsumed)} unconsumed (expected for fragments)")
    print()


def test_masculine_adjective():
    """Test: Der grosse Hund (The big dog) - pre-nominal with masculine"""
    print("=== Test 4: Der grosse Hund ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Der", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grosser", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("Hund", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert best.score > 0.7, f"Score too low: {best.score:.3f}"
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    else:
        print(f"Noun phrase: {len(unconsumed)} unconsumed (expected for fragments)")
    print()


def test_coordination_german():
    """Test: der Hund und die Katze (the dog and the cat)"""
    print("=== Test 5: der Hund und die Katze ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("der", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("Hund", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("und", Tag.CCONJ),
        Word("die", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("Katze", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    root = best.nodes[list(unconsumed)[0]]
    print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


def test_full_sentence():
    """Test: Der grosse Hund rennt schnell (The big dog runs quickly)"""
    print("=== Test 6: Der grosse Hund rennt schnell ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Der", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grosser", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("Hund", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("rennt", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("schnell", Tag.ADV)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print_hypothesis_tree(best)
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    root = best.nodes[list(unconsumed)[0]]
    print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("GERMAN GRAMMAR TESTS (Beta)")
    print("=" * 60)
    print()

    try:
        test_simple_german()
        test_transitive_german()
        test_prenominal_adjective()
        test_masculine_adjective()
        test_coordination_german()
        test_full_sentence()

        print()
        print("=" * 60)
        print("ALL GERMAN TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
