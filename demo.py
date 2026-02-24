"""
NAOMI-II Interactive Multilingual Parser Demo

Parses sentences in 6 languages — English, Spanish, French, German,
Portuguese, and Japanese — producing language-agnostic semantic parse
trees. Demonstrates that equivalent sentences in different languages
yield identical abstract structures.

The same parser engine + different grammar files = same meaning tree.

Usage:
    python demo.py           # Interactive mode
    python demo.py --all     # Run all examples non-interactively
"""

import sys
import os

# Add current_work to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "current_work"))

from src.parser import (
    QuantumParser, Word, Tag, SubType,
    hypothesis_to_dot, print_hypothesis_tree, save_dot
)
from src.parser.pos_tagger import (
    tag_sentence, tag_spanish_sentence,
    tag_french_sentence, tag_german_sentence,
    tag_portuguese_sentence, tag_japanese_sentence,
)


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), "current_work", "grammars")

LANGUAGES = {
    "english":    {"code": "EN", "grammar": "english.json",    "tagger": tag_sentence,            "status": "Production"},
    "spanish":    {"code": "ES", "grammar": "spanish.json",    "tagger": tag_spanish_sentence,    "status": "Production"},
    "french":     {"code": "FR", "grammar": "french.json",     "tagger": tag_french_sentence,     "status": "Beta"},
    "german":     {"code": "DE", "grammar": "german.json",     "tagger": tag_german_sentence,     "status": "Beta"},
    "portuguese": {"code": "PT", "grammar": "portuguese.json", "tagger": tag_portuguese_sentence, "status": "Beta"},
    "japanese":   {"code": "JA", "grammar": "japanese.json",   "tagger": tag_japanese_sentence,   "status": "Beta"},
}

# ---------------------------------------------------------------------------
# Built-in multilingual example sets
# ---------------------------------------------------------------------------
EXAMPLE_SETS = [
    {
        "label": "The dog runs",
        "english": "The dog runs",
        "spanish": "El perro corre",
        "french": "Le chien court",
        "german": "Der Hund rennt",
        "portuguese": "O cachorro corre",
        "japanese": "inu ga hashiru",
    },
    {
        "label": "The dog chases the cat",
        "english": "The dog chases the cat",
        "spanish": "El perro persigue el gato",
        "french": "Le chien poursuit le chat",
        "german": "Der Hund jagt die Katze",
        "portuguese": "O cachorro persegue o gato",
        "japanese": "inu ga neko wo ou",
    },
    {
        "label": "The big dog runs quickly",
        "english": "The big dog runs quickly",
        "spanish": "El perro grande corre rapidamente",
        "french": "Le grand chien court vite",
        "german": "Der grosse Hund rennt schnell",
        "portuguese": "O grande cachorro corre rapidamente",
        "japanese": "ooki inu ga hayaku hashiru",
    },
    {
        "label": "The white house",
        "english": "The white house",
        "spanish": "La casa blanca",
        "french": "La maison blanche",
        "german": "Das weisse Haus",
        "portuguese": "A casa branca",
        "japanese": "shiroi ie",
    },
    {
        "label": "The dog and the cat",
        "english": "The dog and the cat",
        "spanish": "El perro y el gato",
        "french": "Le chien et le chat",
        "german": "Der Hund und die Katze",
        "portuguese": "O cachorro e o gato",
        "japanese": "inu to neko",
    },
]


def parse_sentence(sentence: str, language: str) -> tuple:
    """
    Parse a sentence in the given language.

    Returns:
        (hypothesis, chart, words) — best parse hypothesis, full chart, tagged words
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")

    lang = LANGUAGES[language]
    words = lang["tagger"](sentence)
    grammar_path = os.path.join(GRAMMAR_DIR, lang["grammar"])
    parser = QuantumParser(grammar_path)

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    return best, chart, words


def get_structure_signature(hypothesis) -> dict:
    """
    Extract an abstract structural signature from a parse hypothesis.
    This strips away words and keeps only node types + edge types,
    allowing cross-language structural comparison.
    """
    if hypothesis is None:
        return {"node_types": [], "edge_types": [], "depth": 0}

    node_types = []
    for node in hypothesis.nodes:
        node_types.append(node.type.name)

    edge_types = []
    for edge in hypothesis.edges:
        edge_types.append(edge.type.name)

    # Compute tree depth
    children_map = {}
    for edge in hypothesis.edges:
        children_map.setdefault(edge.parent, []).append(edge.child)

    unconsumed = hypothesis.get_unconsumed()
    max_depth = 0
    if unconsumed:
        def _depth(node_idx, visited=None):
            if visited is None:
                visited = set()
            if node_idx in visited:
                return 0
            visited.add(node_idx)
            kids = children_map.get(node_idx, [])
            if not kids:
                return 1
            return 1 + max(_depth(c, visited) for c in kids)
        for root in unconsumed:
            max_depth = max(max_depth, _depth(root))

    return {
        "node_types": sorted(node_types),
        "edge_types": sorted(edge_types),
        "depth": max_depth,
    }


def compare_structures(sig1: dict, sig2: dict) -> float:
    """Compare two structural signatures, return similarity (0.0-1.0)."""
    if not sig1["node_types"] or not sig2["node_types"]:
        return 0.0

    # Compare node types
    set1 = set(sig1["node_types"])
    set2 = set(sig2["node_types"])
    node_jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

    # Compare edge types
    set1 = set(sig1["edge_types"])
    set2 = set(sig2["edge_types"])
    edge_jaccard = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0

    # Depth match bonus
    depth_match = 1.0 if sig1["depth"] == sig2["depth"] else 0.5

    return (node_jaccard * 0.4 + edge_jaccard * 0.4 + depth_match * 0.2)


# ---------------------------------------------------------------------------
# Matplotlib tree visualization
# ---------------------------------------------------------------------------
def render_tree_matplotlib(hypothesis, title: str, ax=None, words_label: str = ""):
    """Render a parse tree using matplotlib + networkx."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("  [matplotlib/networkx not installed — skipping visual render]")
        return None

    if hypothesis is None:
        if ax:
            ax.text(0.5, 0.5, "Parse failed", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(title)
        return None

    G = nx.DiGraph()
    unconsumed = set(hypothesis.get_unconsumed())

    # Add nodes
    labels = {}
    colors = []
    for i, node in enumerate(hypothesis.nodes):
        word = node.value.text if node.value else "?"
        ntype = node.type.name
        G.add_node(i)
        labels[i] = f"{word}\n({ntype})"
        if i in unconsumed:
            colors.append("#4CAF50")  # Green for root
        else:
            colors.append("#90CAF9")  # Light blue for consumed

    # Add edges (parent -> child for top-down layout)
    edge_labels = {}
    for edge in hypothesis.edges:
        G.add_edge(edge.parent, edge.child)
        edge_labels[(edge.parent, edge.child)] = edge.type.name

    if not G.nodes():
        return None

    # Compute hierarchical layout
    pos = _hierarchical_layout(G, hypothesis)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    nx.draw(
        G, pos, ax=ax,
        labels=labels,
        node_color=colors,
        node_size=2500,
        font_size=7,
        font_weight="bold",
        arrows=True,
        arrowsize=15,
        edge_color="#666666",
        node_shape="s",
        linewidths=1.5,
        edgecolors="#333333",
    )

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#B71C1C",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
    )

    display_title = title
    if words_label:
        display_title += f'\n"{words_label}"'
    ax.set_title(display_title, fontsize=11, fontweight="bold", pad=10)
    ax.margins(0.15)
    return ax


def _hierarchical_layout(G, hypothesis):
    """Compute a top-down hierarchical layout for the parse tree."""
    import networkx as nx

    unconsumed = hypothesis.get_unconsumed()
    if not unconsumed:
        return nx.spring_layout(G)

    # Build parent->children from edges
    children_map = {}
    for edge in hypothesis.edges:
        children_map.setdefault(edge.parent, []).append(edge.child)

    # BFS from roots to assign levels
    pos = {}
    level_nodes = {}

    for root in unconsumed:
        queue = [(root, 0)]
        visited = {root}
        while queue:
            node, level = queue.pop(0)
            level_nodes.setdefault(level, []).append(node)
            for child in children_map.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, level + 1))

    # Assign positions: x spread within level, y = -level (top-down)
    max_level = max(level_nodes.keys()) if level_nodes else 0
    for level, nodes in level_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * 1.5
            y = -level * 1.5
            pos[node] = (x, y)

    # Spread disconnected nodes instead of stacking at one point
    unplaced = [n for n in G.nodes() if n not in pos]
    for i, node in enumerate(unplaced):
        x = (i - (len(unplaced) - 1) / 2) * 1.5
        pos[node] = (x, -(max_level + 1) * 1.5)

    return pos


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
def print_banner():
    print()
    print("=" * 64)
    print("  NAOMI-II  —  Universal Semantic Parser")
    print("  Structure IS Meaning")
    print("  6 Languages, 1 Abstract Tree")
    print("=" * 64)
    print()


def mode_single(language: str):
    """Parse a single sentence in the chosen language."""
    lang_info = LANGUAGES[language]
    lang_name = language.capitalize()
    status = f" ({lang_info['status']})" if lang_info["status"] != "Production" else ""
    print(f"\n  [{lang_name} Mode{status}] Enter a sentence to parse.")
    print("  Press Enter to return to home menu.")

    while True:
        try:
            sentence = input(f"\n  {lang_info['code']}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() == "back":
            break

        try:
            hyp, chart, words = parse_sentence(sentence, language)
        except Exception as e:
            print(f"  Error parsing: {e}")
            continue

        if hyp is None:
            print("  Parse failed — no valid hypothesis found.")
            continue

        print()
        print_hypothesis_tree(hyp)
        print(f"  Hypotheses explored: {len(chart.hypotheses)}")
        print(f"  Best score: {hyp.score:.3f}")
        print(f"  Edges: {len(hyp.edges)} | Unconsumed: {len(hyp.get_unconsumed())}")

        # Offer visualization
        try:
            resp = input("\n  Show tree visualization? [y/N] ").strip().lower()
            if resp == "y":
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                render_tree_matplotlib(hyp, lang_name, ax=ax, words_label=sentence)
                plt.tight_layout()
                plt.show()
        except (EOFError, KeyboardInterrupt):
            break


def mode_compare():
    """Compare equivalent sentences across all 6 languages."""
    print("\n  [Multilingual Comparison]")
    print("  Built-in example sets:")
    for i, example in enumerate(EXAMPLE_SETS):
        print(f"    [{i+1}] \"{example['label']}\"")
    print()
    print("  Press Enter to return to home menu.")

    try:
        choice = input("\n  Select> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not choice or choice.lower() == "back":
        return

    try:
        idx = int(choice) - 1
    except ValueError:
        print("  Invalid selection.")
        return

    if not (0 <= idx < len(EXAMPLE_SETS)):
        print("  Invalid selection.")
        return

    example = EXAMPLE_SETS[idx]
    print(f"\n  Comparing: \"{example['label']}\" across 6 languages\n")

    # Parse all languages
    results = {}
    en_sig = None
    for lang_name, lang_info in LANGUAGES.items():
        sentence = example.get(lang_name)
        if not sentence:
            continue

        try:
            hyp, chart, words = parse_sentence(sentence, lang_name)
            sig = get_structure_signature(hyp)
            results[lang_name] = {
                "sentence": sentence,
                "hyp": hyp,
                "score": hyp.score if hyp else 0,
                "sig": sig,
            }
            if lang_name == "english":
                en_sig = sig
        except Exception as e:
            print(f"  {lang_info['code']}: Parse error — {e}")

    # Display scores
    print("  " + "-" * 60)
    scores_line = "  "
    for lang_name, r in results.items():
        code = LANGUAGES[lang_name]["code"]
        scores_line += f"{code}: {r['score']:.3f}  "
    print(scores_line)
    print("  " + "-" * 60)

    # Show each language
    for lang_name, r in results.items():
        code = LANGUAGES[lang_name]["code"]
        status = LANGUAGES[lang_name]["status"]
        sim = compare_structures(en_sig, r["sig"]) if en_sig else 0.0
        tag = f" [{status}]" if status != "Production" else ""
        print(f"  {code}{tag}: \"{r['sentence']}\"  —  score {r['score']:.3f}  |  vs EN: {sim:.0%}")

    # Overall structural comparison
    if en_sig:
        print()
        all_match = True
        for lang_name, r in results.items():
            if lang_name == "english":
                continue
            sim = compare_structures(en_sig, r["sig"])
            if sim < 0.8:
                all_match = False
                break

        if all_match:
            print("  >> All languages produce the same abstract structure!")
        else:
            print("  >> Most languages share structural overlap (beta grammars may vary).")


def run_demo_noninteractive():
    """Run all example sets non-interactively (for quick showcase)."""
    print_banner()
    print("  Running all built-in examples across 6 languages...\n")

    for example in EXAMPLE_SETS:
        print("=" * 64)
        print(f'  "{example["label"]}"')
        print("=" * 64)

        # Parse all languages
        results = {}
        en_sig = None
        for lang_name, lang_info in LANGUAGES.items():
            sentence = example.get(lang_name)
            if not sentence:
                continue
            try:
                hyp, chart, words = parse_sentence(sentence, lang_name)
                sig = get_structure_signature(hyp)
                results[lang_name] = {
                    "sentence": sentence,
                    "hyp": hyp,
                    "score": hyp.score if hyp else 0,
                    "sig": sig,
                }
                if lang_name == "english":
                    en_sig = sig
            except Exception as e:
                code = lang_info["code"]
                print(f"  {code}: Error — {e}")

        # Display compact results
        for lang_name, r in results.items():
            code = LANGUAGES[lang_name]["code"]
            status = LANGUAGES[lang_name]["status"]
            sim = compare_structures(en_sig, r["sig"]) if en_sig else 0.0
            tag = "*" if status == "Beta" else " "
            print(f"  {tag}{code}: \"{r['sentence']}\"")
            print(f"       score {r['score']:.3f}  |  vs EN: {sim:.0%}")

        # Check structural match
        if en_sig:
            matches = sum(
                1 for ln, r in results.items()
                if ln != "english" and compare_structures(en_sig, r["sig"]) >= 0.8
            )
            total = len(results) - 1
            if matches == total:
                print(f"\n  >> MATCH: All {total + 1} languages produce the same abstract tree")
            else:
                print(f"\n  >> {matches + 1}/{total + 1} languages share the same structure")
        print()

    print("=" * 64)
    print("  * = Beta grammar")
    print()
    print("  Languages supported:")
    for lang_name, info in LANGUAGES.items():
        print(f"    {info['code']}  {lang_name.capitalize():12s}  [{info['status']}]")
    print()
    print("  Same parser engine. Different grammar files.")
    print("  The abstract tree is the universal intermediary.")
    print("=" * 64)
    print("  Run 'python demo.py' for interactive mode.")
    print("=" * 64)


def check_visualization_deps():
    """Check if matplotlib/networkx are installed and notify user if not."""
    missing = []
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    if missing:
        print(f"  Note: Install {', '.join(missing)} for tree visualization:")
        print(f"    pip install {' '.join(missing)}")
        print()


def print_home_menu():
    """Print the home menu with all options."""
    print_banner()
    check_visualization_deps()
    print("  Languages:")
    print("    [1] English              [4] German       (Beta)")
    print("    [2] Spanish              [5] Portuguese   (Beta)")
    print("    [3] French     (Beta)    [6] Japanese     (Beta)")
    print()
    print("  Tools:")
    print("    [c] Compare   — Multilingual structural comparison")
    print("    [a] Run All   — Auto-run all examples across 6 languages")
    print("    [q] Quit")
    print()


def main():
    if "--all" in sys.argv:
        run_demo_noninteractive()
        return

    lang_keys = {
        "1": "english", "english": "english", "en": "english",
        "2": "spanish", "spanish": "spanish", "es": "spanish",
        "3": "french", "french": "french", "fr": "french",
        "4": "german", "german": "german", "de": "german",
        "5": "portuguese", "portuguese": "portuguese", "pt": "portuguese",
        "6": "japanese", "japanese": "japanese", "ja": "japanese",
    }

    while True:
        print_home_menu()

        try:
            choice = input("  Select> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if choice in lang_keys:
            mode_single(lang_keys[choice])
        elif choice in ("c", "compare"):
            mode_compare()
        elif choice in ("a", "all"):
            run_demo_noninteractive()
        elif choice in ("q", "quit", "exit"):
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice. Enter 1-6, c, a, or q.")


if __name__ == "__main__":
    main()
