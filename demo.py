"""
NAOMI-II Interactive Bilingual Parser Demo

Parses sentences in English and Spanish, producing language-agnostic
semantic parse trees. Demonstrates that equivalent sentences in different
languages yield identical abstract structures.

Usage:
    python demo.py
"""

import sys
import os

# Add current_work to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "current_work"))

from src.parser import (
    QuantumParser, Word, Tag, SubType,
    hypothesis_to_dot, print_hypothesis_tree, save_dot
)
from src.parser.pos_tagger import tag_sentence, tag_spanish_sentence


# ---------------------------------------------------------------------------
# Built-in example sentence pairs (English, Spanish)
# ---------------------------------------------------------------------------
EXAMPLE_PAIRS = [
    ("The dog runs", "El perro corre"),
    ("The big dog runs quickly", "El perro grande corre rápidamente"),
    ("The dog chases the cat", "El perro persigue el gato"),
    ("The white house", "La casa blanca"),
    ("The dog and the cat", "El perro y el gato"),
]

# Grammar file paths (relative to project root)
GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), "current_work", "grammars")
EN_GRAMMAR = os.path.join(GRAMMAR_DIR, "english.json")
ES_GRAMMAR = os.path.join(GRAMMAR_DIR, "spanish.json")


def parse_sentence(sentence: str, language: str) -> tuple:
    """
    Parse a sentence in the given language.

    Returns:
        (hypothesis, chart) — best parse hypothesis and full chart
    """
    if language == "english":
        words = tag_sentence(sentence)
        parser = QuantumParser(EN_GRAMMAR)
    elif language == "spanish":
        words = tag_spanish_sentence(sentence)
        parser = QuantumParser(ES_GRAMMAR)
    else:
        raise ValueError(f"Unsupported language: {language}")

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


def render_comparison(en_hyp, es_hyp, en_sentence: str, es_sentence: str,
                      similarity: float, save_path: str = None):
    """Render side-by-side parse trees with structural similarity."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[matplotlib not installed — skipping visual comparison]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    render_tree_matplotlib(en_hyp, "English", ax=ax1, words_label=en_sentence)
    render_tree_matplotlib(es_hyp, "Spanish", ax=ax2, words_label=es_sentence)

    fig.suptitle(
        f"Cross-Language Structural Comparison — Similarity: {similarity:.0%}",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved visualization to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
def print_banner():
    print()
    print("=" * 64)
    print("  NAOMI-II  —  Language-Agnostic Semantic Parser")
    print("  Structure IS Meaning")
    print("=" * 64)
    print()


def mode_single(language: str):
    """Parse a single sentence in the chosen language."""
    lang_name = language.capitalize()
    print(f"\n  [{lang_name} Mode] Enter a sentence (or 'back' to return):")

    while True:
        try:
            sentence = input(f"\n  {lang_name}> ").strip()
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
    """Compare equivalent sentences across languages."""
    print("\n  [Comparison Mode]")
    print("  Built-in example pairs:")
    for i, (en, es) in enumerate(EXAMPLE_PAIRS):
        print(f"    [{i+1}] \"{en}\"  <->  \"{es}\"")
    print(f"    [{len(EXAMPLE_PAIRS)+1}] Enter custom pair")
    print()

    try:
        choice = input("  Select (number or 'back'): ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if choice.lower() == "back":
        return

    try:
        idx = int(choice) - 1
    except ValueError:
        print("  Invalid selection.")
        return

    if idx == len(EXAMPLE_PAIRS):
        # Custom pair
        try:
            en_sent = input("  English sentence: ").strip()
            es_sent = input("  Spanish sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
    elif 0 <= idx < len(EXAMPLE_PAIRS):
        en_sent, es_sent = EXAMPLE_PAIRS[idx]
    else:
        print("  Invalid selection.")
        return

    print(f"\n  Parsing English: \"{en_sent}\"")
    try:
        en_hyp, en_chart, en_words = parse_sentence(en_sent, "english")
    except Exception as e:
        print(f"  English parse error: {e}")
        return

    print(f"  Parsing Spanish: \"{es_sent}\"")
    try:
        es_hyp, es_chart, es_words = parse_sentence(es_sent, "spanish")
    except Exception as e:
        print(f"  Spanish parse error: {e}")
        return

    # Display text trees
    print("\n" + "=" * 64)
    print("  ENGLISH PARSE TREE")
    print("=" * 64)
    if en_hyp:
        print_hypothesis_tree(en_hyp)
        print(f"  Score: {en_hyp.score:.3f} | Edges: {len(en_hyp.edges)}")

    print("\n" + "=" * 64)
    print("  SPANISH PARSE TREE")
    print("=" * 64)
    if es_hyp:
        print_hypothesis_tree(es_hyp)
        print(f"  Score: {es_hyp.score:.3f} | Edges: {len(es_hyp.edges)}")

    # Structural comparison
    en_sig = get_structure_signature(en_hyp)
    es_sig = get_structure_signature(es_hyp)
    similarity = compare_structures(en_sig, es_sig)

    print("\n" + "=" * 64)
    print("  STRUCTURAL COMPARISON")
    print("=" * 64)
    print(f"  English node types: {en_sig['node_types']}")
    print(f"  Spanish node types: {es_sig['node_types']}")
    print(f"  English edge types: {en_sig['edge_types']}")
    print(f"  Spanish edge types: {es_sig['edge_types']}")
    print(f"  Tree depth: EN={en_sig['depth']}  ES={es_sig['depth']}")
    print(f"\n  Structural similarity: {similarity:.0%}")

    if similarity >= 0.8:
        print("  >> The parse trees share the same abstract structure!")
    elif similarity >= 0.5:
        print("  >> The parse trees share significant structural overlap.")
    else:
        print("  >> The parse trees differ structurally.")

    # Offer visual comparison
    try:
        resp = input("\n  Show side-by-side tree visualization? [y/N] ").strip().lower()
        if resp == "y":
            render_comparison(en_hyp, es_hyp, en_sent, es_sent, similarity)
    except (EOFError, KeyboardInterrupt):
        pass


def run_demo_noninteractive():
    """Run all example pairs non-interactively (for quick showcase)."""
    print_banner()
    print("  Running all built-in example pairs...\n")

    en_parser = QuantumParser(EN_GRAMMAR)
    es_parser = QuantumParser(ES_GRAMMAR)

    for en_sent, es_sent in EXAMPLE_PAIRS:
        print("-" * 64)
        print(f'  EN: "{en_sent}"')
        print(f'  ES: "{es_sent}"')
        print()

        en_words = tag_sentence(en_sent)
        es_words = tag_spanish_sentence(es_sent)

        en_chart = en_parser.parse(en_words)
        es_chart = es_parser.parse(es_words)

        en_hyp = en_chart.best_hypothesis()
        es_hyp = es_chart.best_hypothesis()

        en_sig = get_structure_signature(en_hyp)
        es_sig = get_structure_signature(es_hyp)
        similarity = compare_structures(en_sig, es_sig)

        print(f"  EN score: {en_hyp.score:.3f} | ES score: {es_hyp.score:.3f}")
        print(f"  Structural similarity: {similarity:.0%}")

        if similarity >= 0.8:
            print("  >> MATCH: Same abstract structure across languages")
        print()

    print("=" * 64)
    print("  Demo complete. Run 'python demo.py' for interactive mode.")
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


def main():
    if "--all" in sys.argv:
        run_demo_noninteractive()
        return

    print_banner()
    check_visualization_deps()
    print("  Modes:")
    print("    [1] English   — Parse English sentences")
    print("    [2] Spanish   — Parse Spanish sentences")
    print("    [3] Compare   — Side-by-side bilingual comparison")
    print("    [4] Run All   — Auto-run all built-in examples")
    print("    [q] Quit")
    print()

    while True:
        try:
            choice = input("  Select mode> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if choice in ("1", "english", "en"):
            mode_single("english")
        elif choice in ("2", "spanish", "es"):
            mode_single("spanish")
        elif choice in ("3", "compare", "c"):
            mode_compare()
        elif choice in ("4", "all", "a"):
            run_demo_noninteractive()
        elif choice in ("q", "quit", "exit"):
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice. Enter 1, 2, 3, 4, or q.")

        print()


if __name__ == "__main__":
    main()
