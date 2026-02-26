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
from src.translator import Translator
from src.translator.word_lookup import WordLookup
from src.parser.enums import NodeType, ConnectionType


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


def _hierarchical_layout(G, hypothesis, tree=None):
    """Compute a top-down hierarchical layout for the parse tree.

    Args:
        G: NetworkX graph
        hypothesis: Parse hypothesis
        tree: Optional bidirectional children dict from _build_bidirectional_children().
              If provided, uses this for layout instead of raw hypothesis edges.
              This prevents crossing edges when the graph uses bidirectional traversal.
    """
    import networkx as nx

    unconsumed = hypothesis.get_unconsumed()
    if not unconsumed:
        return nx.spring_layout(G)

    # Build parent->children mapping
    if tree is not None:
        # Use bidirectional tree (matches the graph edges)
        children_map = {idx: [child for child, _ in kids]
                        for idx, kids in tree.items()}
    else:
        # Default: unidirectional from hypothesis edges
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
            x = (i - (n - 1) / 2) * 2.5
            y = -level * 2.0
            pos[node] = (x, y)

    # Spread disconnected nodes instead of stacking at one point
    unplaced = [n for n in G.nodes() if n not in pos]
    for i, node in enumerate(unplaced):
        x = (i - (len(unplaced) - 1) / 2) * 2.5
        pos[node] = (x, -(max_level + 1) * 2.0)

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


def _build_bilingual_labels(hypothesis, word_lookup: WordLookup,
                            surface_forms=None) -> dict:
    """
    Build bilingual labels for every node: {node_idx: (source_word, target_word)}.

    Shows the actual surface form for verbs (conjugated, not infinitive) and
    uses word_lookup for all word types including determiners.
    """
    from src.parser.enums import Tag, SubType

    # Article translation (for determiners that are articles)
    _ARTICLE_MAP = {
        'english': {'el': 'the', 'la': 'the', 'los': 'the', 'las': 'the',
                    'le': 'the', 'les': 'the',
                    'der': 'the', 'die': 'the', 'das': 'the',
                    'o': 'the', 'os': 'the', 'as': 'the'},
        'spanish': {'the': 'el', 'le': 'el', 'der': 'el', 'o': 'el'},
        'french': {'the': 'le', 'el': 'le', 'der': 'le', 'o': 'le'},
        'german': {'the': 'der', 'el': 'der', 'le': 'der', 'o': 'der'},
        'portuguese': {'the': 'o', 'el': 'o', 'le': 'o', 'der': 'o'},
        'japanese': {},
    }

    labels = {}
    for i, node in enumerate(hypothesis.nodes):
        if not node.value:
            labels[i] = (None, None)
            continue

        src_word = node.value.text
        pos = node.value.pos

        # Verbs: show conjugated target form, not infinitive
        if pos == Tag.VERB and surface_forms:
            target_lemma = word_lookup.lookup(src_word, pos)
            features = list(node.value.subtypes) if node.value.subtypes else []
            if not any(f in features for f in
                       (SubType.FIRST_PERSON, SubType.SECOND_PERSON,
                        SubType.THIRD_PERSON)):
                features.append(SubType.THIRD_PERSON)
            if not any(f in features for f in
                       (SubType.SINGULAR, SubType.PLURAL)):
                features.append(SubType.SINGULAR)
            tgt_word = surface_forms.conjugate_verb(target_lemma, features)
            labels[i] = (src_word, tgt_word)
            continue

        # Determiners: try word_lookup first (catches possessives/demonstratives),
        # fall back to article map for basic articles
        if pos == Tag.DET:
            tgt = word_lookup.lookup(src_word, pos)
            if tgt == src_word and word_lookup.source_lang != word_lookup.target_lang:
                # Lookup didn't find it — try article map
                article_map = _ARTICLE_MAP.get(word_lookup.target_lang, {})
                tgt = article_map.get(src_word.lower(), src_word)
            labels[i] = (src_word, tgt)
            continue

        # All other words: use word lookup
        tgt_word = word_lookup.lookup(src_word, pos)
        labels[i] = (src_word, tgt_word)

    return labels


def _build_bidirectional_children(hypothesis):
    """
    Build a tree using bidirectional edge traversal from unconsumed roots.

    NAOMI-II edges use (parent, child) as (from, to) in grammatical direction,
    NOT tree hierarchy. DESCRIPTION edges go FROM modifier TO noun, so the
    modifier is the 'parent' but it's a dependent in the tree. We follow edges
    in both directions to capture all connected nodes.

    Returns: {node_idx: [(child_node_idx, edge_type_name), ...]}
    """
    unconsumed = hypothesis.get_unconsumed()
    tree = {i: [] for i in range(len(hypothesis.nodes))}
    visited = set()

    def _traverse(node_idx):
        visited.add(node_idx)
        for edge in hypothesis.edges:
            # Follow edge in either direction
            if edge.parent == node_idx and edge.child not in visited:
                tree[node_idx].append((edge.child, edge.type.name))
                _traverse(edge.child)
            elif edge.child == node_idx and edge.parent not in visited:
                tree[node_idx].append((edge.parent, edge.type.name))
                _traverse(edge.parent)

    for root_idx in unconsumed:
        _traverse(root_idx)

    return tree


def print_bilingual_tree(hypothesis, word_lookup: WordLookup,
                         src_code: str, tgt_code: str,
                         surface_forms=None):
    """
    Print a bilingual intermediary tree showing source/target words at each node.

    Uses bidirectional edge traversal to capture determiners, adjectives, and
    all other modifiers that connect via DESCRIPTION edges.

    Example output:
        +-- runs/corre (CLAUSE <- VERBAL)
            +--[SUBJECT]
                +-- The/El (DESCRIPTOR)
                +-- dog/perro (NOMINAL <- NOUN)
    """
    labels = _build_bilingual_labels(hypothesis, word_lookup, surface_forms)
    unconsumed = hypothesis.get_unconsumed()
    tree = _build_bidirectional_children(hypothesis)

    print(f"Bilingual Tree ({src_code} -> {tgt_code})")
    print("=" * 60)

    for root_idx in unconsumed:
        _print_bilingual_recursive(hypothesis, root_idx, tree, labels)

    print()


def _print_bilingual_recursive(hyp, node_idx, tree, labels,
                                prefix="", is_last=True):
    """Recursively print bilingual tree structure."""
    node = hyp.nodes[node_idx]
    src_word, tgt_word = labels.get(node_idx, (None, None))

    # Build the node label: source/target (TYPE)
    connector = "+-- " if is_last else "|-- "

    if src_word and tgt_word:
        if src_word.lower() == tgt_word.lower():
            word_label = src_word
        else:
            word_label = f"{src_word}/{tgt_word}"
    elif src_word:
        word_label = src_word
    else:
        word_label = "(constructed)"

    type_str = node.type.name
    if node.type != node.original_type:
        type_str += f" <- {node.original_type.name}"

    print(f"{prefix}{connector}{word_label} ({type_str})")

    # Print children with edge labels
    children = tree.get(node_idx, [])
    for i, (child_idx, edge_type) in enumerate(children):
        is_last_child = (i == len(children) - 1)

        edge_prefix = prefix + ("    " if is_last else "|   ")
        edge_connector = "+--" if is_last_child else "|--"
        print(f"{edge_prefix}{edge_connector}[{edge_type}]")

        child_prefix = prefix + ("    " if is_last else "|   ") + \
                       ("    " if is_last_child else "|   ")
        _print_bilingual_recursive(hyp, child_idx, tree, labels,
                                    child_prefix, is_last_child)


def render_bilingual_tree_matplotlib(hypothesis, word_lookup: WordLookup,
                                     title: str, ax=None,
                                     surface_forms=None):
    """Render a bilingual tree using matplotlib — each node shows source/target."""
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

    bilabels = _build_bilingual_labels(hypothesis, word_lookup, surface_forms)
    tree = _build_bidirectional_children(hypothesis)

    G = nx.DiGraph()
    unconsumed = set(hypothesis.get_unconsumed())

    # Add nodes with bilingual labels
    labels = {}
    colors = []
    for i, node in enumerate(hypothesis.nodes):
        src_w, tgt_w = bilabels.get(i, (None, None))
        if src_w and tgt_w and src_w.lower() != tgt_w.lower():
            word_str = f"{src_w}/{tgt_w}"
        elif src_w:
            word_str = src_w
        else:
            word_str = "?"

        ntype = node.type.name
        G.add_node(i)
        labels[i] = f"{word_str}\n({ntype})"
        if i in unconsumed:
            colors.append("#4CAF50")
        else:
            colors.append("#90CAF9")

    # Add edges from bidirectional tree (parent->child in tree order)
    edge_labels = {}
    for parent_idx, children in tree.items():
        for child_idx, edge_type in children:
            G.add_edge(parent_idx, child_idx)
            edge_labels[(parent_idx, child_idx)] = edge_type

    if not G.nodes():
        return None

    # Use bidirectional tree for layout so edges don't cross
    pos = _hierarchical_layout(G, hypothesis, tree=tree)

    # Dynamic node sizing based on label length
    max_label_len = max(len(l) for l in labels.values()) if labels else 15
    node_size = max(4500, max_label_len * 220)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    nx.draw(
        G, pos, ax=ax,
        labels=labels,
        node_color=colors,
        node_size=node_size,
        font_size=6,
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

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.margins(0.15)
    return ax


def mode_translate():
    """Translate a sentence between any two supported languages."""
    lang_list = list(LANGUAGES.keys())

    print("\n  [Translate Mode]")
    print("  Source language:")
    for i, lang in enumerate(lang_list):
        code = LANGUAGES[lang]["code"]
        status = LANGUAGES[lang]["status"]
        tag = f" ({status})" if status != "Production" else ""
        print(f"    [{i+1}] {lang.capitalize()}{tag}")
    print()
    print("  Press Enter to return to home menu.")

    try:
        src_choice = input("\n  Source> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not src_choice:
        return

    try:
        src_idx = int(src_choice) - 1
        if not (0 <= src_idx < len(lang_list)):
            print("  Invalid selection.")
            return
        source_lang = lang_list[src_idx]
    except ValueError:
        # Try matching by name or code
        src_lower = src_choice.lower()
        source_lang = None
        for lang in lang_list:
            if lang == src_lower or LANGUAGES[lang]["code"].lower() == src_lower:
                source_lang = lang
                break
        if not source_lang:
            print("  Invalid selection.")
            return

    src_code = LANGUAGES[source_lang]["code"]

    # Show examples for source language
    available_examples = []
    for ex in EXAMPLE_SETS:
        if source_lang in ex:
            available_examples.append(ex)

    print(f"\n  Source: {source_lang.capitalize()} ({src_code})")
    print("  Enter a sentence or pick an example:")
    for i, ex in enumerate(available_examples):
        print(f"    [{i+1}] \"{ex[source_lang]}\"")
    print()

    try:
        sentence_choice = input(f"  {src_code}> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not sentence_choice:
        return

    try:
        ex_idx = int(sentence_choice) - 1
        if 0 <= ex_idx < len(available_examples):
            sentence = available_examples[ex_idx][source_lang]
        else:
            sentence = sentence_choice
    except ValueError:
        sentence = sentence_choice

    # Target language
    print("\n  Target language:")
    for i, lang in enumerate(lang_list):
        if lang == source_lang:
            continue
        code = LANGUAGES[lang]["code"]
        status = LANGUAGES[lang]["status"]
        tag = f" ({status})" if status != "Production" else ""
        print(f"    [{i+1}] {lang.capitalize()}{tag}")

    try:
        tgt_choice = input("\n  Target> ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not tgt_choice:
        return

    try:
        tgt_idx = int(tgt_choice) - 1
        if not (0 <= tgt_idx < len(lang_list)):
            print("  Invalid selection.")
            return
        target_lang = lang_list[tgt_idx]
    except ValueError:
        tgt_lower = tgt_choice.lower()
        target_lang = None
        for lang in lang_list:
            if lang == tgt_lower or LANGUAGES[lang]["code"].lower() == tgt_lower:
                target_lang = lang
                break
        if not target_lang:
            print("  Invalid selection.")
            return

    if target_lang == source_lang:
        print("  Source and target are the same language.")
        return

    tgt_code = LANGUAGES[target_lang]["code"]

    # Parse source
    print(f"\n  Translating: {src_code} -> {tgt_code}")
    print(f'  Source: "{sentence}"')
    print()

    try:
        hyp, chart, words = parse_sentence(sentence, source_lang)
    except Exception as e:
        print(f"  Parse error: {e}")
        return

    if hyp is None:
        print("  Parse failed — no valid hypothesis found.")
        return

    # Translate
    try:
        translator = Translator(source_lang, target_lang)
        result = translator.translate(hyp)
    except Exception as e:
        print(f"  Translation error: {e}")
        return

    # Show bilingual intermediary tree
    print_bilingual_tree(hyp, translator.word_lookup, src_code, tgt_code,
                         surface_forms=translator.surface_forms)

    print("  " + "-" * 50)
    print(f'  {src_code}: "{sentence}"')
    print(f'  {tgt_code}: "{result}"')
    print("  " + "-" * 50)

    # Offer matplotlib visualization
    try:
        resp = input("\n  Show bilingual tree visualization? [y/N] ").strip().lower()
        if resp == "y":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            title = f"Translation: {src_code} -> {tgt_code}"
            render_bilingual_tree_matplotlib(
                hyp, translator.word_lookup, title, ax=ax,
                surface_forms=translator.surface_forms)
            plt.tight_layout()
            plt.show()
    except (EOFError, KeyboardInterrupt):
        pass

    print()


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
    print("    [t] Translate — Translate between any two languages")
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
        elif choice in ("t", "translate"):
            mode_translate()
        elif choice in ("c", "compare"):
            mode_compare()
        elif choice in ("a", "all"):
            run_demo_noninteractive()
        elif choice in ("q", "quit", "exit"):
            print("  Goodbye!")
            break
        else:
            print("  Invalid choice. Enter 1-6, t, c, a, or q.")


if __name__ == "__main__":
    main()
