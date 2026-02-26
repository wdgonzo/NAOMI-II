"""
Structure-based translator using parse trees and grammar rules.

Pipeline: Parse L1 → abstract tree → word lookup → linearize with L2 grammar → L2 text

This mirrors how a human translator works:
1. Understand the structure (parse)
2. Look up equivalent words (dictionary)
3. Know target grammar rules (linearization)
4. Assemble the output (surface form)
"""

from typing import Optional

from ..parser.data_structures import Hypothesis

from .word_lookup import WordLookup
from .surface_forms import SurfaceFormSelector
from .linearizer import Linearizer


class Translator:
    """
    Translate parsed sentences between languages using grammar rules.

    Usage:
        translator = Translator('english', 'spanish')
        result = translator.translate(hypothesis)
        print(result)  # "El perro corre"
    """

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.word_lookup = WordLookup(source_lang, target_lang)
        self.surface_forms = SurfaceFormSelector(target_lang)
        self.linearizer = Linearizer(target_lang, self.word_lookup,
                                     self.surface_forms)

    def translate(self, hypothesis: Hypothesis) -> str:
        """
        Translate a parsed hypothesis to the target language.

        Args:
            hypothesis: Parsed sentence (output from QuantumParser)

        Returns:
            Translated sentence as a string
        """
        if hypothesis is None:
            return ""

        # Find root node
        unconsumed = hypothesis.get_unconsumed()
        if not unconsumed:
            return ""

        root_idx = unconsumed[0]

        # Linearize from root using target grammar
        tokens = self.linearizer.linearize(hypothesis, root_idx)

        # Post-process
        return self._post_process(tokens)

    def _post_process(self, tokens: list) -> str:
        """Clean up the token list into a final sentence string."""
        if not tokens:
            return ""

        # Filter empty tokens
        tokens = [t for t in tokens if t and t.strip()]

        if not tokens:
            return ""

        # Join tokens, attaching punctuation without leading space
        result = ''
        for t in tokens:
            if t in '.?!,;:' and result:
                result += t
            else:
                if result:
                    result += ' '
                result += t

        # Capitalize first letter (except Japanese which uses romaji lowercase)
        if self.target_lang != 'japanese':
            result = result[0].upper() + result[1:]

        return result
