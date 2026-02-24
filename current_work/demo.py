"""
NAOMI-II Interactive Parser Demo
=================================
Parse sentences in English or Spanish and visualize the resulting parse trees.
Also demonstrates cross-lingual structural equivalence: semantically identical
sentences in different languages produce equivalent abstract tree structures.

Usage:
    python demo.py                # Run auto-demo (preset sentence pairs)
    python demo.py --interactive  # Interactive mode (type your own sentences)
"""

import sys
import os
import argparse
import tempfile

# Add current_work/ to path so "from src.parser import ..." resolves
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.parser import (
    QuantumParser, Word, Tag, SubType, NodeType, ConnectionType,
    hypothesis_to_dot, print_hypothesis_tree,
)
from src.parser.pos_tagger import tag_sentence, tag_spanish_sentence

# Grammar file paths
EN_GRAMMAR = os.path.join(SCRIPT_DIR, "grammars", "english.json")
ES_GRAMMAR = os.path.join(SCRIPT_DIR, "grammars", "spanish.json")

LANGUAGES = {
    "1": ("English", EN_GRAMMAR, tag_sentence),
    "2": ("Spanish", ES_GRAMMAR, tag_spanish_sentence),
}


# ---------------------------------------------------------------------------
# Graphviz rendering
# ---------------------------------------------------------------------------

def render_tree(hypothesis, title="Parse Tree", view=True, output_dir=None):
    """
    Render a parse tree using Graphviz. Falls back to text if unavailable.

    Returns the path to the rendered file, or None if text fallback was used.
    """
    try:
        import graphviz
    except ImportError:
        print("  [graphviz not installed — showing text tree]")
        print()
        print_hypothesis_tree(hypothesis)
        return None

    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB")  # Top to bottom (root at top)
    dot.attr("node", shape="box", style="filled")

    unconsumed = hypothesis.get_unconsumed()

    # Add nodes
    for i, node in enumerate(hypothesis.nodes):
        word = node.value.text if node.value else "(constructed)"
        node_type = node.type.name

        if i in unconsumed:
            color = "#90EE90"  # light green — root candidates
        else:
            color = "#D3D3D3"  # light gray — consumed

        if node.type != node.original_type:
            label = f"{word}\n{node.original_type.name} → {node_type}"
        else:
            label = f"{word}\n{node_type}"

        dot.node(f"n{i}", label=label, fillcolor=color)

    # Add edges (parent → child, top-down)
    for edge in hypothesis.edges:
        dot.edge(f"n{edge.parent}", f"n{edge.child}", label=edge.type.name)

    # Title
    score_info = f"Score: {hypothesis.score:.3f} | Edges: {len(hypothesis.edges)}"
    dot.attr(label=f"{title}\n{score_info}", labelloc="t", fontsize="14")

    # Output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "_ " else "_" for c in title)
        filepath = os.path.join(output_dir, safe_title.replace(" ", "_"))
    else:
        filepath = os.path.join(tempfile.mkdtemp(), "parse_tree")

    try:
        rendered = dot.render(filepath, cleanup=True, view=view)
        print(f"  Tree saved to: {rendered}")
        return rendered
    except Exception as e:
        print(f"  [Could not render graphviz: {e}]")
        print_hypothesis_tree(hypothesis)
        return None


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------

def get_structural_signature(hypothesis):
    """
    Extract a language-independent structural signature from a parse tree.

    Returns a nested tuple of (NodeType, [(ConnectionType, child_signature), ...])
    using only syntactic categories and grammatical relations — ignoring word text.
    """
    unconsumed = hypothesis.get_unconsumed()

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
# Parse a sentence (shared logic)
# ---------------------------------------------------------------------------

def parse_sentence(sentence, lang_key):
    """Parse a sentence in the given language. Returns (hypothesis, lang_name)."""
    lang_name, grammar_path, tagger_fn = LANGUAGES[lang_key]
    words = tagger_fn(sentence)
    parser = QuantumParser(grammar_path)
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    return best, lang_name


# ---------------------------------------------------------------------------
# Auto-demo mode
# ---------------------------------------------------------------------------

def demo_pair(pair_num, en_sentence, es_sentence, en_words, es_words, view=True):
    """Parse and compare a single English/Spanish sentence pair."""
    print(f"\n{'=' * 70}")
    print(f"  PAIR {pair_num}")
    print(f"  English: \"{en_sentence}\"")
    print(f"  Spanish: \"{es_sentence}\"")
    print(f"{'=' * 70}")

    en_parser = QuantumParser(EN_GRAMMAR)
    en_chart = en_parser.parse(en_words)
    en_best = en_chart.best_hypothesis()

    es_parser = QuantumParser(ES_GRAMMAR)
    es_chart = es_parser.parse(es_words)
    es_best = es_chart.best_hypothesis()

    # Text trees
    print(f"\n  --- English Parse Tree ---")
    print_hypothesis_tree(en_best)
    print(f"\n  --- Spanish Parse Tree ---")
    print_hypothesis_tree(es_best)

    # Graphviz trees
    output_dir = os.path.join(SCRIPT_DIR, "output")
    render_tree(en_best, title=f"Pair {pair_num} English: {en_sentence}",
                view=view, output_dir=output_dir)
    render_tree(es_best, title=f"Pair {pair_num} Spanish: {es_sentence}",
                view=view, output_dir=output_dir)

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


def run_auto_demo(view=True):
    """Run preset sentence pair comparisons."""
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

    results = []

    # Pair 1: Simple intransitive
    results.append(demo_pair(
        1, "dogs run", "perros corren",
        en_words=[Word("dogs", Tag.NOUN), Word("run", Tag.VERB)],
        es_words=[
            Word("perros", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
            Word("corren", Tag.VERB, [SubType.THIRD_PERSON, SubType.PLURAL]),
        ],
        view=view,
    ))

    # Pair 2: Transitive with determiner
    results.append(demo_pair(
        2, "the cat eats mice", "el gato come ratones",
        en_words=[
            Word("the", Tag.DET), Word("cat", Tag.NOUN),
            Word("eats", Tag.VERB), Word("mice", Tag.NOUN),
        ],
        es_words=[
            Word("el", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("gato", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("come", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
            Word("ratones", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
        ],
        view=view,
    ))

    # Pair 3: With adjective (pre-nominal EN vs post-nominal ES)
    results.append(demo_pair(
        3, "the big dog runs", "el perro grande corre",
        en_words=[
            Word("the", Tag.DET), Word("big", Tag.ADJ),
            Word("dog", Tag.NOUN), Word("runs", Tag.VERB),
        ],
        es_words=[
            Word("el", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("perro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
            Word("grande", Tag.ADJ, [SubType.POST_NOMINAL]),
            Word("corre", Tag.VERB, [SubType.THIRD_PERSON, SubType.SINGULAR]),
        ],
        view=view,
    ))

    # Summary
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

    print()
    return passed > 0


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive():
    """Interactive REPL: select language, type sentence, see tree."""
    print()
    print("=" * 70)
    print("  NAOMI-II Interactive Parser")
    print("=" * 70)
    print()
    print("  Type a sentence and see its parse tree visualized.")
    print("  You can also compare trees across languages.")
    print()
    print("  Commands:")
    print("    compare  — parse two sentences and compare structures")
    print("    quit     — exit")
    print()

    output_dir = os.path.join(SCRIPT_DIR, "output")
    prev_hypothesis = None
    prev_lang = None
    prev_sentence = None

    while True:
        # Language selection
        print("  Select language:")
        print("    [1] English")
        print("    [2] Spanish")
        print()

        choice = input("  Language (1/2), 'compare', or 'quit': ").strip().lower()

        if choice in ("quit", "q", "exit"):
            print("\n  Goodbye!\n")
            break

        if choice == "compare":
            run_compare_mode(output_dir)
            continue

        if choice not in LANGUAGES:
            print("  Invalid choice. Please enter 1 or 2.\n")
            continue

        lang_name, _, _ = LANGUAGES[choice]

        # Sentence input
        sentence = input(f"  Enter {lang_name} sentence: ").strip()
        if not sentence:
            print("  Empty sentence. Try again.\n")
            continue

        # Parse
        print(f"\n  Parsing: \"{sentence}\" [{lang_name}]")
        print()

        try:
            hypothesis, lang_name = parse_sentence(sentence, choice)
        except Exception as e:
            print(f"  Error parsing sentence: {e}\n")
            continue

        # Text tree
        print_hypothesis_tree(hypothesis)

        # Graphviz tree
        render_tree(hypothesis, title=f"{lang_name}: {sentence}",
                    view=True, output_dir=output_dir)

        # Compare with previous if available
        if prev_hypothesis is not None:
            print()
            sig_cur = get_structural_signature(hypothesis)
            sig_prev = get_structural_signature(prev_hypothesis)

            if sig_cur == sig_prev:
                print(f"  Compared with previous (\"{prev_sentence}\" [{prev_lang}]):")
                print(f"  EQUIVALENT -- Same abstract tree structure!")
            else:
                print(f"  Compared with previous (\"{prev_sentence}\" [{prev_lang}]):")
                print(f"  DIFFERENT -- Tree structures differ.")

        prev_hypothesis = hypothesis
        prev_lang = lang_name
        prev_sentence = sentence
        print()


def run_compare_mode(output_dir):
    """Compare two sentences side-by-side."""
    print()
    print("  --- Compare Mode ---")
    print("  Parse two sentences and compare their abstract structures.")
    print()

    sentences = []
    for i in range(2):
        print(f"  Sentence {i + 1}:")
        print("    [1] English  [2] Spanish")
        lang = input("    Language: ").strip()
        if lang not in LANGUAGES:
            print("    Invalid. Returning to main menu.\n")
            return
        lang_name, _, _ = LANGUAGES[lang]
        text = input(f"    Enter {lang_name} sentence: ").strip()
        if not text:
            print("    Empty sentence. Returning to main menu.\n")
            return
        sentences.append((text, lang, lang_name))
        print()

    # Parse both
    results = []
    for text, lang_key, lang_name in sentences:
        print(f"  Parsing: \"{text}\" [{lang_name}]")
        try:
            hyp, _ = parse_sentence(text, lang_key)
            results.append((hyp, text, lang_name))
        except Exception as e:
            print(f"  Error: {e}\n")
            return

    # Display trees
    for hyp, text, lang_name in results:
        print(f"\n  --- {lang_name}: \"{text}\" ---")
        print_hypothesis_tree(hyp)
        render_tree(hyp, title=f"{lang_name}: {text}",
                    view=True, output_dir=output_dir)

    # Compare
    sig1 = get_structural_signature(results[0][0])
    sig2 = get_structural_signature(results[1][0])

    print(f"\n  --- Abstract Structure: \"{results[0][1]}\" ---")
    for sig in sig1:
        print_signature(sig, indent=2)
    print(f"\n  --- Abstract Structure: \"{results[1][1]}\" ---")
    for sig in sig2:
        print_signature(sig, indent=2)

    print()
    if sig1 == sig2:
        print("  RESULT: EQUIVALENT -- Same abstract tree structure!")
    else:
        print("  RESULT: DIFFERENT -- Tree structures differ.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NAOMI-II Parser Demo — Cross-lingual parse tree visualization"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Launch interactive mode (type your own sentences)"
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="Don't auto-open tree images (just save them)"
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        success = run_auto_demo(view=not args.no_view)
        print("  Tip: Run with --interactive to parse your own sentences.")
        print("       Run with --no-view to save images without opening them.")
        print()
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
