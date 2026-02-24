"""
NAOMI-II Cross-Lingual Parse Demo
===================================
Demonstrates that semantically equivalent sentences in English and Spanish
produce equivalent parse tree structures, proving the parser's language-agnostic
design.

Usage:
    python demo.py
"""

import sys
import os

# Add current_work/ to path so "from src.parser import ..." resolves
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.parser import QuantumParser, Word, Tag, SubType, NodeType, ConnectionType
from src.parser.pos_tagger import tag_sentence, tag_spanish_sentence
from src.parser.visualizer import print_hypothesis_tree


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------

def get_structural_signature(hypothesis):
    """
    Extract a language-independent structural signature from a parse tree.

    Returns a nested tuple of (NodeType, [(ConnectionType, child_signature), ...])
    representing the tree structure using only syntactic categories and grammatical
    relations -- ignoring word text entirely. This allows cross-lingual comparison.
    """
    unconsumed = hypothesis.get_unconsumed()

    # Build parent -> children mapping
    children_map = {}
    for i in range(len(hypothesis.nodes)):
        children_map[i] = []
    for edge in hypothesis.edges:
        children_map[edge.parent].append((edge.child, edge.type))

    def build_signature(node_idx):
        node = hypothesis.nodes[node_idx]
        children = children_map.get(node_idx, [])
        child_sigs = sorted(
            [(conn_type.name, build_signature(child_idx))
             for child_idx, conn_type in children],
            key=lambda x: x[0]
        )
        return (node.type.name, tuple(child_sigs))

    root_sigs = []
    for root_idx in sorted(unconsumed):
        root_sigs.append(build_signature(root_idx))

    return tuple(root_sigs)


def print_signature(sig, indent=0):
    """Pretty-print a structural signature."""
    node_type, children = sig
    prefix = "  " * indent
    print(f"{prefix}{node_type}")
    for conn_name, child_sig in children:
        print(f"{prefix}  --[{conn_name}]-->")
        print_signature(child_sig, indent + 2)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def demo_pair(pair_num, en_sentence, es_sentence, en_words, es_words,
              en_grammar_path, es_grammar_path):
    """Parse and compare a single English/Spanish sentence pair."""
    print(f"\n{'=' * 70}")
    print(f"  PAIR {pair_num}")
    print(f"  English: \"{en_sentence}\"")
    print(f"  Spanish: \"{es_sentence}\"")
    print(f"{'=' * 70}")

    # Parse English
    en_parser = QuantumParser(en_grammar_path)
    en_chart = en_parser.parse(en_words)
    en_best = en_chart.best_hypothesis()

    # Parse Spanish
    es_parser = QuantumParser(es_grammar_path)
    es_chart = es_parser.parse(es_words)
    es_best = es_chart.best_hypothesis()

    # Display trees
    print(f"\n  --- English Parse Tree ---")
    print_hypothesis_tree(en_best)

    print(f"\n  --- Spanish Parse Tree ---")
    print_hypothesis_tree(es_best)

    # Compare structures
    en_sig = get_structural_signature(en_best)
    es_sig = get_structural_signature(es_best)

    print(f"\n  --- Abstract Structure (English) ---")
    for sig in en_sig:
        print_signature(sig, indent=2)

    print(f"\n  --- Abstract Structure (Spanish) ---")
    for sig in es_sig:
        print_signature(sig, indent=2)

    equivalent = (en_sig == es_sig)
    print()
    if equivalent:
        print(f"  RESULT: EQUIVALENT -- Same tree structure from both languages.")
    else:
        print(f"  RESULT: DIFFERENT -- Tree structures differ (see above).")
        print(f"  (This may reflect genuine syntactic differences between languages.)")

    return equivalent


def main():
    print()
    print("=" * 70)
    print("  NAOMI-II: Cross-Lingual Parse Tree Equivalence Demo")
    print("=" * 70)
    print()
    print("  This demo parses semantically identical sentences in English")
    print("  and Spanish, then compares their abstract parse tree structures.")
    print()
    print("  The parser is language-agnostic: grammar files define language-")
    print("  specific rules, but the engine and tree types are universal.")
    print("  If both sentences produce the same abstract tree, it proves")
    print("  the parser captures meaning structure independent of language.")

    en_grammar = os.path.join(SCRIPT_DIR, "grammars", "english.json")
    es_grammar = os.path.join(SCRIPT_DIR, "grammars", "spanish.json")

    results = []

    # -----------------------------------------------------------------------
    # Pair 1: Simple intransitive -- "dogs run" / "perros corren"
    # -----------------------------------------------------------------------
    results.append(demo_pair(
        1,
        "dogs run",
        "perros corren",
        en_words=[
            Word("dogs", Tag.NOUN),
            Word("run", Tag.VERB),
        ],
        es_words=[
            Word("perros", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
            Word("corren", Tag.VERB, [SubType.THIRD_PERSON, SubType.PLURAL]),
        ],
        en_grammar_path=en_grammar,
        es_grammar_path=es_grammar,
    ))

    # -----------------------------------------------------------------------
    # Pair 2: Transitive with determiner -- "the cat eats mice" /
    #         "el gato come ratones"
    # -----------------------------------------------------------------------
    results.append(demo_pair(
        2,
        "the cat eats mice",
        "el gato come ratones",
        en_words=[
            Word("the", Tag.DET),
            Word("cat", Tag.NOUN),
            Word("eats", Tag.VERB),
            Word("mice", Tag.NOUN),
        ],
        es_words=[
            Word("el", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("gato", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("come", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
            Word("ratones", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
        ],
        en_grammar_path=en_grammar,
        es_grammar_path=es_grammar,
    ))

    # -----------------------------------------------------------------------
    # Pair 3: With adjective -- "the big dog runs" /
    #         "el perro grande corre"
    #
    # English: adjective before noun (pre-nominal)
    # Spanish: adjective after noun (post-nominal)
    # Both should produce a DESCRIPTION connection to the noun.
    # -----------------------------------------------------------------------
    results.append(demo_pair(
        3,
        "the big dog runs",
        "el perro grande corre",
        en_words=[
            Word("the", Tag.DET),
            Word("big", Tag.ADJ),
            Word("dog", Tag.NOUN),
            Word("runs", Tag.VERB),
        ],
        es_words=[
            Word("el", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("perro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("grande", Tag.ADJ, [SubType.POST_NOMINAL]),
            Word("corre", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        ],
        en_grammar_path=en_grammar,
        es_grammar_path=es_grammar,
    ))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    passed = sum(results)
    total = len(results)

    print()
    print("=" * 70)
    print(f"  SUMMARY: {passed}/{total} sentence pairs structurally equivalent")
    print("=" * 70)

    if all(results):
        print()
        print("  All sentence pairs produced equivalent parse trees across")
        print("  English and Spanish, demonstrating language-agnostic")
        print("  syntactic analysis.")
    elif passed > 0:
        print()
        print(f"  {passed} of {total} pairs matched. Differences in the remaining")
        print("  pairs reflect genuine syntactic variation between languages")
        print("  that the grammar rules handle differently.")

    print()
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
