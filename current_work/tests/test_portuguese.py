"""Test quantum parser with Portuguese grammar."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

GRAMMAR = os.path.join(_ROOT, "grammars", "portuguese.json")

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree


def test_simple_portuguese():
    """Test: O cachorro corre (The dog runs)"""
    print("=== Test 1: O cachorro corre ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("O", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("cachorro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("corre", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR])
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


def test_transitive_portuguese():
    """Test: O cachorro persegue o gato (The dog chases the cat)"""
    print("=== Test 2: O cachorro persegue o gato ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("O", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("cachorro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("persegue", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("o", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("gato", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    print_hypothesis_tree(best)
    print()


def test_postnominal_adjective():
    """Test: A casa branca (The white house) - post-nominal adjective"""
    print("=== Test 3: A casa branca ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("A", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("casa", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR]),
        Word("branca", Tag.ADJ, [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL])
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


def test_prenominal_adjective():
    """Test: O grande cachorro (The big dog) - pre-nominal adjective"""
    print("=== Test 4: O grande cachorro ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("O", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grande", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("cachorro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
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


def test_coordination_portuguese():
    """Test: o cachorro e o gato (the dog and the cat)"""
    print("=== Test 5: o cachorro e o gato ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("o", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("cachorro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("e", Tag.CCONJ),
        Word("o", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("gato", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
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
    """Test: O grande cachorro corre rapidamente (The big dog runs quickly)"""
    print("=== Test 6: O grande cachorro corre rapidamente ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("O", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grande", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("cachorro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("corre", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("rapidamente", Tag.ADV)
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
    print("PORTUGUESE GRAMMAR TESTS (Beta)")
    print("=" * 60)
    print()

    try:
        test_simple_portuguese()
        test_transitive_portuguese()
        test_postnominal_adjective()
        test_prenominal_adjective()
        test_coordination_portuguese()
        test_full_sentence()

        print()
        print("=" * 60)
        print("ALL PORTUGUESE TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
