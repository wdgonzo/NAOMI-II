"""Test quantum parser with French grammar."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

GRAMMAR = os.path.join(_ROOT, "grammars", "french.json")

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree


def test_simple_french():
    """Test: Le chien court (The dog runs)"""
    print("=== Test 1: Le chien court ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("chien", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("court", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR])
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


def test_transitive_french():
    """Test: Le chien poursuit le chat (The dog chases the cat)"""
    print("=== Test 2: Le chien poursuit le chat ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("chien", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("poursuit", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("chat", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
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
    """Test: La maison blanche (The white house) - post-nominal adjective"""
    print("=== Test 3: La maison blanche ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("La", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("maison", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR]),
        Word("blanche", Tag.ADJ, [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL])
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
    """Test: Le grand chien (The big dog) - BAGS pre-nominal adjective"""
    print("=== Test 4: Le grand chien ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grand", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("chien", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
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


def test_coordination_french():
    """Test: le chien et le chat (the dog and the cat)"""
    print("=== Test 5: le chien et le chat ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("chien", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("et", Tag.CCONJ),
        Word("le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("chat", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR])
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
    """Test: Le grand chien court vite (The big dog runs quickly)"""
    print("=== Test 6: Le grand chien court vite ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("Le", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grand", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.PRE_NOMINAL]),
        Word("chien", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("court", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        Word("vite", Tag.ADV)
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
    print("FRENCH GRAMMAR TESTS (Beta)")
    print("=" * 60)
    print()

    try:
        test_simple_french()
        test_transitive_french()
        test_postnominal_adjective()
        test_prenominal_adjective()
        test_coordination_french()
        test_full_sentence()

        print()
        print("=" * 60)
        print("ALL FRENCH TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
