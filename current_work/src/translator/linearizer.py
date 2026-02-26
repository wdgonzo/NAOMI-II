"""
Top-down linearizer — walks a parse tree and produces target language surface text.

Uses the TARGET language's grammar rules to determine word order:
- Each rule's `before[]` and `after[]` encode where constituents go relative to the anchor.
- English predicate1: anchor=VERBAL, after=[NOMINAL] → SVO (object after verb)
- Japanese predicate1: anchor=VERBAL, before=[NOMINAL] → SOV (object before verb)

The grammar rules are bidirectional: they encode both parsing AND generation order.

Edge semantics in NAOMI-II:
- Edges use (parent, child) as (from, to) in grammatical relationship direction.
- SUBJECT edge: parent=CLAUSE/PREDICATE, child=NOMINAL (outgoing from clause)
- DESCRIPTION edge: parent=DESCRIPTOR, child=NOUN (incoming to noun from modifier)
- So traversal must be BIDIRECTIONAL: follow edges in both directions, using a
  visited set to prevent cycles.
"""

import os
from typing import List, Tuple, Optional, Set

from ..parser.data_structures import Hypothesis, Node, Edge
from ..parser.enums import NodeType, ConnectionType, SubType, Tag
from ..parser.dsl import load_grammar, Grammar, Rule, Ruleset, ConnectionSpec

from .word_lookup import WordLookup
from .surface_forms import SurfaceFormSelector


# Grammar directory
_GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'grammars')


class Linearizer:
    """Top-down tree linearizer using target grammar rules."""

    def __init__(self, target_lang: str, word_lookup: WordLookup,
                 surface_forms: SurfaceFormSelector):
        self.target_lang = target_lang
        self.word_lookup = word_lookup
        self.surface_forms = surface_forms
        grammar_path = os.path.join(_GRAMMAR_DIR, f"{target_lang}.json")
        self.grammar = load_grammar(grammar_path)

    def linearize(self, hyp: Hypothesis, node_idx: int,
                  parent_edge_type: ConnectionType = None,
                  visited: Set[int] = None) -> List[str]:
        """
        Recursively linearize a subtree rooted at node_idx.

        Uses bidirectional edge traversal with visited set to prevent cycles.
        Returns a list of surface tokens in target language order.
        """
        if visited is None:
            visited = set()
        if node_idx in visited:
            return []
        visited.add(node_idx)

        node = hyp.nodes[node_idx]
        children = self._get_connected(hyp, node_idx, visited)

        # Leaf node: translate word and inflect
        if not children and node.value:
            return self._translate_leaf(node, parent_edge_type)

        # Constructed node with no children and no word (e.g., implied subject)
        if not children and not node.value:
            return []

        # Internal node: use target grammar to determine child ordering
        return self._linearize_internal(hyp, node_idx, node, children,
                                        parent_edge_type, visited)

    def _get_connected(self, hyp: Hypothesis, node_idx: int,
                       visited: Set[int]) -> List[Tuple[int, ConnectionType]]:
        """
        Get all nodes connected to this node via edges, excluding visited nodes.

        Follows edges bidirectionally: if this node is the parent OR child
        of an edge, the other end is returned.
        """
        connected = []
        for e in hyp.edges:
            if e.parent == node_idx and e.child not in visited:
                connected.append((e.child, e.type))
            elif e.child == node_idx and e.parent not in visited:
                connected.append((e.parent, e.type))
        return connected

    def _translate_leaf(self, node: Node,
                        parent_edge_type: ConnectionType = None) -> List[str]:
        """Translate a leaf node: word lookup + inflection."""
        if not node.value:
            return []

        word = node.value.text
        pos = node.value.pos

        # Articles are re-inserted at the nominal level — skip them here.
        # But possessives, demonstratives, etc. must be translated and kept.
        if pos == Tag.DET:
            articles = {'the', 'a', 'an',
                        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                        'le', 'la', 'les', 'un', 'une', 'des',
                        'der', 'die', 'das', 'ein', 'eine',
                        'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'}
            if word.lower() in articles:
                return []
            # Non-article determiner: translate and keep
            target = self.word_lookup.lookup(word, pos)
            return [target]
        # Skip particles (consumed during parsing; re-inserted for Japanese)
        if pos == Tag.ADP:
            return []
        # Conjunctions
        if pos == Tag.CCONJ:
            return self._translate_conjunction(word)

        # Content word: look up target equivalent
        target_lemma = self.word_lookup.lookup(word, pos)

        # Inflect based on POS
        if pos in (Tag.VERB, Tag.AUX):
            features = list(node.value.subtypes) if node.value.subtypes else []
            if not any(f in features for f in
                       (SubType.FIRST_PERSON, SubType.SECOND_PERSON,
                        SubType.THIRD_PERSON)):
                features.append(SubType.THIRD_PERSON)
            if not any(f in features for f in (SubType.SINGULAR, SubType.PLURAL)):
                features.append(SubType.SINGULAR)
            surface = self.surface_forms.conjugate_verb(target_lemma, features)
            return [surface]

        if pos == Tag.ADJ:
            # Return lemma; inflection is handled by _linearize_nominal
            return [target_lemma]

        if pos == Tag.ADV:
            return [target_lemma]

        if pos in (Tag.NOUN, Tag.PRON, Tag.PROPN):
            return [target_lemma]

        # Fallback
        return [target_lemma]

    def _translate_conjunction(self, word: str) -> List[str]:
        """Translate a coordinating conjunction."""
        conjunctions = {
            'english': {'and': 'and', 'or': 'or', 'but': 'but',
                        'y': 'and', 'o': 'or', 'pero': 'but',
                        'et': 'and', 'ou': 'or', 'mais': 'but',
                        'und': 'and', 'oder': 'or', 'aber': 'but',
                        'e': 'and', 'to': 'and'},
            'spanish': {'and': 'y', 'or': 'o', 'but': 'pero',
                        'et': 'y', 'und': 'y', 'e': 'y', 'to': 'y'},
            'french': {'and': 'et', 'or': 'ou', 'but': 'mais',
                       'y': 'et', 'und': 'et', 'e': 'et', 'to': 'et'},
            'german': {'and': 'und', 'or': 'oder', 'but': 'aber',
                       'y': 'und', 'et': 'und', 'e': 'und', 'to': 'und'},
            'portuguese': {'and': 'e', 'or': 'ou', 'but': 'mas',
                           'y': 'e', 'et': 'e', 'und': 'e', 'to': 'e'},
            'japanese': {'and': 'to', 'or': 'ka', 'but': 'demo',
                         'y': 'to', 'et': 'to', 'und': 'to', 'e': 'to'},
        }
        lang_conj = conjunctions.get(self.target_lang, {})
        return [lang_conj.get(word.lower(), word)]

    def _linearize_internal(self, hyp: Hypothesis, node_idx: int,
                            node: Node,
                            children: List[Tuple[int, ConnectionType]],
                            parent_edge_type: ConnectionType = None,
                            visited: Set[int] = None) -> List[str]:
        """Linearize an internal node using target grammar rules."""

        # Check for coordination (node may be promoted from COORD to NOMINAL)
        has_coordination = any(ct == ConnectionType.COORDINATION
                               for _, ct in children)
        if has_coordination or node.original_type == NodeType.COORD:
            return self._linearize_coordination(hyp, node_idx, node, children,
                                                visited)

        # Route by node type
        if node.type == NodeType.CLAUSE:
            return self._linearize_clause(hyp, node_idx, node, children, visited)
        if node.type == NodeType.PREDICATE:
            return self._linearize_predicate(hyp, node_idx, node, children, visited)
        if node.type in (NodeType.NOMINAL, NodeType.NOUN):
            return self._linearize_nominal(hyp, node_idx, node, children,
                                           parent_edge_type, visited)
        if node.type == NodeType.VERBAL:
            return self._linearize_verbal(hyp, node_idx, node, children, visited)
        if node.type == NodeType.COORD:
            return self._linearize_coordination(hyp, node_idx, node, children,
                                                visited)
        if node.type in (NodeType.DESCRIPTOR, NodeType.SPECIFIER):
            return self._linearize_modifier(hyp, node_idx, node, children, visited)

        # Default
        return self._default_linearize(hyp, node_idx, node, children, visited)

    def _linearize_clause(self, hyp: Hypothesis, node_idx: int,
                          node: Node,
                          children: List[Tuple[int, ConnectionType]],
                          visited: Set[int]) -> List[str]:
        """
        Linearize CLAUSE: determine subject/predicate/object order.

        The clause node often IS the verb (promoted from VERBAL→PREDICATE→CLAUSE),
        with SUBJECT and OBJECT as direct children. We need to use BOTH clause
        rules (for subject placement) and predicate rules (for object placement).
        """
        subject_children = [(ci, ct) for ci, ct in children
                            if ct == ConnectionType.SUBJECT]
        object_children = [(ci, ct) for ci, ct in children
                           if ct == ConnectionType.OBJECT]
        complement_children = [(ci, ct) for ci, ct in children
                               if ct == ConnectionType.COMPLEMENT]
        other_children = [(ci, ct) for ci, ct in children
                          if ct not in (ConnectionType.SUBJECT,
                                        ConnectionType.OBJECT,
                                        ConnectionType.COMPLEMENT)]

        # Handle Japanese parse quirk: intransitive "inu ga hashiru" gets parsed
        # with OBJECT edge (not SUBJECT) because predicate1 runs before clause1.
        # If we have only OBJECT children and no SUBJECT, treat the first OBJECT
        # as a SUBJECT.
        if not subject_children and len(object_children) == 1:
            subject_children = object_children
            object_children = []

        # Determine object placement using predicate rules
        obj_before_verb = False
        if object_children:
            pred_rule = self._find_rule_for('predicate', NodeType.PREDICATE,
                                            {ConnectionType.OBJECT})
            if pred_rule:
                before_types, _ = self._get_rule_placement(pred_rule)
                obj_before_verb = ConnectionType.OBJECT in before_types

        tokens = []

        # 1. Subject (always before in all 6 languages)
        for ci, ct in subject_children:
            subtokens = self.linearize(hyp, ci, ct, visited)
            if self.target_lang == 'japanese':
                subtokens.append('ga')
            tokens.extend(subtokens)

        if obj_before_verb:
            # SOV: objects before verb (Japanese)
            for ci, ct in object_children:
                subtokens = self.linearize(hyp, ci, ct, visited)
                if self.target_lang == 'japanese':
                    subtokens.append('wo')
                tokens.extend(subtokens)

        # 2. Verb (the clause node itself)
        if node.value:
            tokens.extend(self._translate_leaf(node))

        if not obj_before_verb:
            # SVO: objects after verb (English, Spanish, etc.)
            for ci, ct in object_children:
                tokens.extend(self.linearize(hyp, ci, ct, visited))

        # 3. Complements and other children
        for ci, ct in complement_children + other_children:
            tokens.extend(self.linearize(hyp, ci, ct, visited))

        return tokens

    def _linearize_predicate(self, hyp: Hypothesis, node_idx: int,
                             node: Node,
                             children: List[Tuple[int, ConnectionType]],
                             visited: Set[int]) -> List[str]:
        """Linearize PREDICATE: determine verb/object order (SVO vs SOV)."""
        child_edge_types = {ct for _, ct in children}
        rule = self._find_rule_for('predicate', NodeType.PREDICATE,
                                   child_edge_types)

        before_types, after_types = set(), set()
        if rule:
            before_types, after_types = self._get_rule_placement(rule)

        tokens = []

        # BEFORE elements (SOV: object before verb)
        for ci, ct in children:
            if ct in before_types:
                subtokens = self.linearize(hyp, ci, ct, visited)
                if self.target_lang == 'japanese' and ct == ConnectionType.OBJECT:
                    subtokens.append('wo')
                tokens.extend(subtokens)

        # Anchor (verb)
        if node.value:
            tokens.extend(self._translate_leaf(node))

        # AFTER elements (SVO: object after verb)
        for ci, ct in children:
            if ct in after_types:
                tokens.extend(self.linearize(hyp, ci, ct, visited))

        # Unmatched children (fallback)
        matched = before_types | after_types
        for ci, ct in children:
            if ct not in matched:
                tokens.extend(self.linearize(hyp, ci, ct, visited))

        return tokens

    def _linearize_nominal(self, hyp: Hypothesis, node_idx: int,
                           node: Node,
                           children: List[Tuple[int, ConnectionType]],
                           parent_edge_type: ConnectionType = None,
                           visited: Set[int] = None) -> List[str]:
        """
        Linearize NOMINAL/NOUN: handle determiner, adjective placement,
        and gender agreement.
        """
        # Categorize children
        descriptions = [(ci, ct) for ci, ct in children
                        if ct == ConnectionType.DESCRIPTION]
        other = [(ci, ct) for ci, ct in children
                 if ct != ConnectionType.DESCRIPTION]

        # Translate the noun itself
        target_noun = None
        if node.value and node.value.pos in (Tag.NOUN, Tag.PRON, Tag.PROPN):
            target_noun = self.word_lookup.lookup(node.value.text, node.value.pos)
        elif node.value:
            target_noun = self.word_lookup.lookup(node.value.text, node.value.pos)

        # Determine noun gender for agreement
        gender = None
        number = SubType.SINGULAR
        if target_noun:
            gender = self.surface_forms.get_noun_gender(target_noun)
        for f in node.flags:
            if f in (SubType.SINGULAR, SubType.PLURAL):
                number = f
            if f in (SubType.MASCULINE, SubType.FEMININE, SubType.NEUTER) and not gender:
                gender = f

        # Default gender
        if gender is None and self.target_lang in ('spanish', 'portuguese',
                                                    'french'):
            gender = SubType.MASCULINE
        elif gender is None and self.target_lang == 'german':
            gender = SubType.MASCULINE

        # Separate determiners from adjectives in descriptions
        has_article = False
        has_non_article_det = False
        _articles = {'the', 'a', 'an',
                     'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                     'le', 'la', 'les', 'un', 'une', 'des',
                     'der', 'die', 'das', 'ein', 'eine',
                     'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas'}
        adj_indices = []
        non_article_det_indices = []
        for ci, ct in descriptions:
            child_node = hyp.nodes[ci]
            if child_node.value and child_node.value.pos == Tag.DET:
                if child_node.value.text.lower() in _articles:
                    has_article = True
                else:
                    has_non_article_det = True
                    non_article_det_indices.append(ci)
            else:
                adj_indices.append(ci)

        # Translate and inflect adjectives
        adj_tokens = []
        for ci in adj_indices:
            child_node = hyp.nodes[ci]
            if child_node.value:
                adj_lemma = self.word_lookup.lookup(child_node.value.text,
                                                    child_node.value.pos)
                adj_surface = self.surface_forms.inflect_adjective(
                    adj_lemma, gender, number)
                adj_tokens.append(adj_surface)
                visited.add(ci)
            else:
                adj_tokens.extend(self.linearize(hyp, ci, ct, visited))

        # Determine adjective placement
        adj_before = self._adjectives_before_noun()

        # Build determiner
        # Insert definite article ONLY if:
        # 1. Source had an article (not possessive/demonstrative), OR
        # 2. Target language requires articles and source had no determiner at all
        #    (handles JA→EN where source has no articles)
        # Do NOT insert article when source had possessive/demonstrative — those
        # are already translated and emitted by _translate_leaf.
        det_token = None
        if not has_non_article_det:
            needs_article = has_article or (
                self.target_lang in ('english', 'spanish', 'french', 'german',
                                     'portuguese')
                and target_noun is not None
                and node.value and node.value.pos in (Tag.NOUN, Tag.PRON)
            )
            if needs_article and self.target_lang != 'japanese':
                det_token = self.surface_forms.get_definite_article(gender, number)

        # Linearize non-article determiners (possessives, demonstratives)
        poss_tokens = []
        for ci in non_article_det_indices:
            poss_tokens.extend(
                self.linearize(hyp, ci, ConnectionType.DESCRIPTION, visited))

        # Assemble tokens
        tokens = []

        if det_token:
            tokens.append(det_token)
        if poss_tokens:
            tokens.extend(poss_tokens)

        noun_token = [target_noun] if target_noun else []

        if adj_before:
            tokens.extend(adj_tokens)
            tokens.extend(noun_token)
        else:
            tokens.extend(noun_token)
            tokens.extend(adj_tokens)

        # Other children (specifications, modifications, etc.)
        for ci, ct in other:
            tokens.extend(self.linearize(hyp, ci, ct, visited))

        return tokens

    def _linearize_verbal(self, hyp: Hypothesis, node_idx: int,
                          node: Node,
                          children: List[Tuple[int, ConnectionType]],
                          visited: Set[int]) -> List[str]:
        """Linearize VERBAL: verb with adverb modifiers."""
        child_edge_types = {ct for _, ct in children}
        rule = self._find_rule_for('verb', NodeType.VERBAL, child_edge_types)

        tokens = []

        if rule:
            before_types, after_types = self._get_rule_placement(rule)
            for ci, ct in children:
                if ct in before_types:
                    tokens.extend(self.linearize(hyp, ci, ct, visited))
            if node.value:
                tokens.extend(self._translate_leaf(node))
            for ci, ct in children:
                if ct in after_types:
                    tokens.extend(self.linearize(hyp, ci, ct, visited))
        else:
            if node.value:
                tokens.extend(self._translate_leaf(node))
            for ci, ct in children:
                tokens.extend(self.linearize(hyp, ci, ct, visited))

        return tokens

    def _linearize_coordination(self, hyp: Hypothesis, node_idx: int,
                                node: Node,
                                children: List[Tuple[int, ConnectionType]],
                                visited: Set[int]) -> List[str]:
        """Linearize COORD: A and B."""
        coord_children = [(ci, ct) for ci, ct in children
                          if ct == ConnectionType.COORDINATION]

        if not coord_children:
            return self._translate_leaf(node) if node.value else []

        # Translate the conjunction
        conj_tokens = []
        if node.value:
            conj_tokens = self._translate_conjunction(node.value.text)

        # Linearize coordinated elements with conjunction between
        tokens = []
        for i, (ci, ct) in enumerate(coord_children):
            if i > 0:
                tokens.extend(conj_tokens)
            tokens.extend(self.linearize(hyp, ci, ct, visited))

        return tokens

    def _linearize_modifier(self, hyp: Hypothesis, node_idx: int,
                            node: Node,
                            children: List[Tuple[int, ConnectionType]],
                            visited: Set[int]) -> List[str]:
        """Linearize DESCRIPTOR/SPECIFIER with potential children."""
        tokens = []
        # Recurse into any children first (e.g., "very" modifying "big")
        for ci, ct in children:
            tokens.extend(self.linearize(hyp, ci, ct, visited))
        if node.value:
            tokens.extend(self._translate_leaf(node))
        return tokens

    def _default_linearize(self, hyp: Hypothesis, node_idx: int,
                           node: Node,
                           children: List[Tuple[int, ConnectionType]],
                           visited: Set[int]) -> List[str]:
        """Default linearization: anchor word + children in order."""
        tokens = []
        if node.value:
            tokens.extend(self._translate_leaf(node))
        for ci, ct in children:
            tokens.extend(self.linearize(hyp, ci, ct, visited))
        return tokens

    # ------------------------------------------------------------------
    # Grammar rule lookup helpers
    # ------------------------------------------------------------------

    def _find_rule_for(self, ruleset_prefix: str, result_type: NodeType,
                       child_edge_types: set) -> Optional[Rule]:
        """
        Find a matching generation rule in the target grammar.

        Searches rulesets whose name starts with `ruleset_prefix` and
        whose result matches `result_type`. Returns the best rule whose
        connections match the given child edge types.
        """
        best_rule = None
        best_match_size = -1

        for name in reversed(self.grammar.order):
            if not name.startswith(ruleset_prefix):
                continue
            ruleset = self.grammar.rulesets.get(name)
            if not ruleset or ruleset.result != result_type:
                continue

            for rule in ruleset.rules:
                rule_edge_types = {conn.type for conn in rule.connections}
                if not rule_edge_types:
                    continue
                # Match if rule edges are subset of actual edges
                if rule_edge_types.issubset(child_edge_types):
                    # Prefer rules that match MORE edge types
                    if len(rule_edge_types) > best_match_size:
                        best_match_size = len(rule_edge_types)
                        best_rule = rule

        return best_rule

    def _get_rule_placement(self, rule: Rule) -> Tuple[set, set]:
        """
        Determine which connection types go before vs after the anchor.

        Returns (before_types, after_types) sets of ConnectionType.
        """
        before_types = set()
        after_types = set()

        for conn in rule.connections:
            # Determine if the non-anchor end is in before[] or after[]
            for ref in [conn.from_ref, conn.to_ref]:
                if ref == 'anchor':
                    continue
                if ref.startswith('before'):
                    before_types.add(conn.type)
                elif ref.startswith('after'):
                    after_types.add(conn.type)

        return before_types, after_types

    def _adjectives_before_noun(self) -> bool:
        """Check if this target language places adjectives before nouns.

        Grammars have two DESCRIPTION connection rules:
        - noun1 (quantifier='one'): single determiner before noun
        - noun3 (quantifier='all'): adjective list before/after noun
        We only care about the 'all' rules (adjectives), not 'one' (determiners).
        """
        for name in self.grammar.order:
            if not name.startswith('noun'):
                continue
            ruleset = self.grammar.rulesets.get(name)
            if not ruleset:
                continue
            for rule in ruleset.rules:
                # Only look at rules with quantifier='all' (adjective lists)
                has_all_desc_after = any(
                    p.quantifier == 'all' and p.type == NodeType.DESCRIPTOR
                    for p in rule.after
                )
                has_all_desc_before = any(
                    p.quantifier == 'all' and p.type == NodeType.DESCRIPTOR
                    for p in rule.before
                )
                if has_all_desc_after:
                    return False
                if has_all_desc_before:
                    return True

        # Default by language
        return self.target_lang in ('english', 'german', 'japanese')
