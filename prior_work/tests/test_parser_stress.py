"""
Parser Stress Tests

Comprehensive test suite to identify parser weaknesses and edge cases.
Tests various grammatical structures, sentence lengths, and edge cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import List, Dict, Tuple

from src.parser.quantum_parser import QuantumParser
from src.parser.pos_tagger import tag_sentence


# Test sentence categories
EDGE_CASES = [
    ("", "Empty string"),
    ("Dog", "Single word"),
    (".", "Punctuation only"),
    ("The", "Single determiner"),
    ("runs quickly", "No subject"),
]

SIMPLE_SENTENCES = [
    ("The dog runs", "Simple SVO"),
    ("Dogs run", "Plural subject"),
    ("The big dog runs quickly", "With adjective and adverb"),
    ("A small cat sleeps", "Indefinite article"),
]

COMPLEX_STRUCTURES = [
    ("The dog that runs is big", "Relative clause"),
    ("The dog which the cat saw runs", "Nested relative clause"),
    ("The dog and the cat run", "Conjunction"),
    ("The dog runs and jumps", "Verb conjunction"),
    ("The big red dog runs", "Multiple adjectives"),
    ("The dog runs very quickly", "Adverb modifier"),
]

LONG_SENTENCES = [
    ("The big brown dog runs quickly through the park", "Long simple"),
    ("The dog and the cat and the bird run", "Multiple conjunctions"),
    ("The very big dog runs extremely quickly", "Multiple modifiers"),
    ("The dog that the cat saw runs through the park quickly", "Nested with PP"),
]

PASSIVE_VOICE = [
    ("The ball was thrown", "Simple passive"),
    ("The dog was seen by the cat", "Passive with agent"),
]

QUESTIONS = [
    ("Does the dog run", "Yes/no question"),
    ("What runs", "Wh-question"),
]

IMPERATIVES = [
    ("Run", "Simple imperative"),
    ("Run quickly", "Imperative with adverb"),
]

NORMALIZATION_TESTS = [
    ("Run", "Imperative should have implied 'you'"),
    ("dog", "Single noun should have EQUIVALENCE"),
    ("cat", "Single noun should have EQUIVALENCE"),
]


class ParserStressTester:
    """Test harness for parser stress testing."""

    def __init__(self, grammar_path: str):
        """Initialize with grammar file."""
        self.parser = QuantumParser(grammar_path)
        self.results = []

    def test_sentence(self, sentence: str, description: str) -> Dict:
        """
        Test a single sentence and record results.

        Returns:
            Dict with parse results and timing
        """
        result = {
            'sentence': sentence,
            'description': description,
            'length': len(sentence.split()) if sentence else 0,
            'success': False,
            'score': 0.0,
            'time_ms': 0.0,
            'error': None
        }

        try:
            # Time the parse
            start = time.time()

            words = tag_sentence(sentence)
            chart = self.parser.parse(words)
            hypothesis = chart.best_hypothesis()

            elapsed = (time.time() - start) * 1000  # Convert to ms

            if hypothesis and hypothesis.score > 0:
                result['success'] = True
                result['score'] = hypothesis.score
                result['num_nodes'] = len(hypothesis.nodes)
                result['num_edges'] = len(hypothesis.edges)

            result['time_ms'] = elapsed

        except Exception as e:
            result['error'] = str(e)
            result['time_ms'] = (time.time() - start) * 1000

        self.results.append(result)
        return result

    def test_category(self, sentences: List[Tuple[str, str]], category_name: str):
        """Test all sentences in a category."""
        print(f"\n{'='*60}")
        print(f"Testing: {category_name}")
        print(f"{'='*60}")

        for sentence, description in sentences:
            result = self.test_sentence(sentence, description)

            status = "PASS" if result['success'] else "FAIL"
            score_str = f"(score={result['score']:.2f})" if result['success'] else ""
            error_str = f"[{result['error']}]" if result['error'] else ""

            print(f"[{status}] {description:30s} | '{sentence:40s}' {score_str} {error_str}")

    def generate_report(self) -> Dict:
        """Generate summary statistics."""
        total = len(self.results)
        successes = sum(1 for r in self.results if r['success'])
        failures = total - successes

        success_rate = (successes / total * 100) if total > 0 else 0
        avg_score = sum(r['score'] for r in self.results if r['success']) / max(successes, 1)
        avg_time = sum(r['time_ms'] for r in self.results) / max(total, 1)

        # Group by sentence length
        by_length = {}
        for r in self.results:
            length = r['length']
            if length not in by_length:
                by_length[length] = {'total': 0, 'success': 0}
            by_length[length]['total'] += 1
            if r['success']:
                by_length[length]['success'] += 1

        return {
            'total_tests': total,
            'successes': successes,
            'failures': failures,
            'success_rate': success_rate,
            'avg_score': avg_score,
            'avg_time_ms': avg_time,
            'by_length': by_length,
            'failed_sentences': [r for r in self.results if not r['success']]
        }

    def print_report(self):
        """Print formatted report."""
        report = self.generate_report()

        print("\n" + "="*60)
        print("PARSER STRESS TEST REPORT")
        print("="*60)
        print(f"\nTotal tests: {report['total_tests']}")
        print(f"Successes: {report['successes']}")
        print(f"Failures: {report['failures']}")
        print(f"Success rate: {report['success_rate']:.1f}%")
        print(f"Average score (successful): {report['avg_score']:.3f}")
        print(f"Average time: {report['avg_time_ms']:.2f}ms")

        print("\nSuccess rate by sentence length:")
        for length in sorted(report['by_length'].keys()):
            stats = report['by_length'][length]
            rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {length:2d} words: {stats['success']:2d}/{stats['total']:2d} ({rate:5.1f}%)")

        if report['failed_sentences']:
            print(f"\nFailed sentences ({len(report['failed_sentences'])}):")
            for r in report['failed_sentences'][:10]:  # Show first 10
                error = f" - {r['error']}" if r['error'] else ""
                print(f"  • {r['description']:30s}: '{r['sentence']}'{error}")
            if len(report['failed_sentences']) > 10:
                print(f"  ... and {len(report['failed_sentences']) - 10} more")


def main():
    """Run all stress tests."""
    print("="*60)
    print("NAOMI-II PARSER STRESS TESTS")
    print("="*60)

    # Initialize tester
    grammar_path = Path(__file__).parent.parent / "grammars" / "english.json"
    tester = ParserStressTester(str(grammar_path))

    # Run all test categories
    tester.test_category(EDGE_CASES, "Edge Cases")
    tester.test_category(SIMPLE_SENTENCES, "Simple Sentences")
    tester.test_category(COMPLEX_STRUCTURES, "Complex Structures")
    tester.test_category(LONG_SENTENCES, "Long Sentences")
    tester.test_category(PASSIVE_VOICE, "Passive Voice")
    tester.test_category(QUESTIONS, "Questions")
    tester.test_category(IMPERATIVES, "Imperatives")
    tester.test_category(NORMALIZATION_TESTS, "Parser Normalizations")

    # Print summary
    tester.print_report()

    return tester


if __name__ == "__main__":
    tester = main()
