"""Test quantum parser with Japanese grammar (romaji)."""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

GRAMMAR = os.path.join(_ROOT, "grammars", "japanese.json")

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree


def test_simple_japanese():
    """Test: inu ga hashiru (The dog runs) - SOV with subject particle"""
    print("=== Test 1: inu ga hashiru ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("inu", Tag.NOUN),
        Word("ga", Tag.ADP),
        Word("hashiru", Tag.VERB)
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


def test_transitive_japanese():
    """Test: inu ga neko wo ou (The dog chases the cat) - SOV"""
    print("=== Test 2: inu ga neko wo ou ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("inu", Tag.NOUN),
        Word("ga", Tag.ADP),
        Word("neko", Tag.NOUN),
        Word("wo", Tag.ADP),
        Word("ou", Tag.VERB)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert len(unconsumed) == 1, f"Expected 1 root, got {len(unconsumed)}"
    print_hypothesis_tree(best)
    print()


def test_adjective_japanese():
    """Test: shiroi ie (white house) - pre-nominal adjective"""
    print("=== Test 3: shiroi ie ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("shiroi", Tag.ADJ),
        Word("ie", Tag.NOUN)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    assert best.score > 0.6, f"Score too low: {best.score:.3f}"
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    else:
        print(f"Noun phrase: {len(unconsumed)} unconsumed (expected for fragments)")
    print()


def test_coordination_japanese():
    """Test: inu to neko (dog and cat) - particle conjunction"""
    print("=== Test 4: inu to neko ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("inu", Tag.NOUN),
        Word("to", Tag.CCONJ),
        Word("neko", Tag.NOUN)
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


def test_adverb_japanese():
    """Test: inu ga hayaku hashiru (The dog runs quickly) - adverb before verb"""
    print("=== Test 5: inu ga hayaku hashiru ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("inu", Tag.NOUN),
        Word("ga", Tag.ADP),
        Word("hayaku", Tag.ADV),
        Word("hashiru", Tag.VERB)
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


def test_full_sentence_japanese():
    """Test: ooki inu ga neko wo ou (The big dog chases the cat) - SOV full"""
    print("=== Test 6: ooki inu ga neko wo ou ===")
    parser = QuantumParser(GRAMMAR)

    words = [
        Word("ooki", Tag.ADJ),
        Word("inu", Tag.NOUN),
        Word("ga", Tag.ADP),
        Word("neko", Tag.NOUN),
        Word("wo", Tag.ADP),
        Word("ou", Tag.VERB)
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
    print("JAPANESE GRAMMAR TESTS (Beta) - Romaji")
    print("=" * 60)
    print()

    try:
        test_simple_japanese()
        test_transitive_japanese()
        test_adjective_japanese()
        test_coordination_japanese()
        test_adverb_japanese()
        test_full_sentence_japanese()

        print()
        print("=" * 60)
        print("ALL JAPANESE TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
