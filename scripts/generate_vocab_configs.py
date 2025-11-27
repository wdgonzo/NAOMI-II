"""
Generate Vocabulary-Matched Configuration Files

This script analyzes the actual training vocabulary and generates:
1. config/antonym_types.json - Antonym pairs that exist in vocabulary
2. config/semantic_clusters.json - Semantic clusters from vocabulary words

This ensures polarity and consistency losses can activate during training.

Usage:
    python scripts/generate_vocab_configs.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# Ensure WordNet is downloaded
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class VocabConfigGenerator:
    """Generate configuration files from actual training vocabulary."""

    def __init__(self, vocab_path: str = "checkpoints/vocabulary.json"):
        """
        Initialize generator.

        Args:
            vocab_path: Path to vocabulary JSON file
        """
        self.vocab_path = vocab_path
        self.word_to_id = {}
        self.vocab_words = set()  # Full words with POS tags
        self.base_words = {}  # Mapping from base word to full vocab words

        self._load_vocabulary()

    def _load_vocabulary(self):
        """Load vocabulary and extract base words."""
        print(f"Loading vocabulary from {self.vocab_path}...")

        with open(self.vocab_path, 'r') as f:
            vocab_data = json.load(f)

        self.word_to_id = vocab_data['word_to_id']
        self.vocab_words = set(self.word_to_id.keys())

        print(f"Loaded {len(self.vocab_words)} vocabulary words")

        # Extract base words by removing POS tags and sense numbers
        # Example: "good_wn.00_a" -> "good"
        for word in self.vocab_words:
            # Skip punctuation and numbers
            if word[0] in '$!&(,.-\'\"' or word[0].isdigit():
                continue

            # Extract base word
            if '_wn.' in word:
                # WordNet sense: "good_wn.00_a" -> "good"
                base = word.split('_wn.')[0]
            elif '_' in word:
                # POS tag only: "good_a" -> "good"
                parts = word.rsplit('_', 1)
                if len(parts) == 2 and parts[1] in ['n', 'v', 'a', 'r', 'x']:
                    base = parts[0]
                else:
                    base = word
            else:
                base = word

            if base:
                if base not in self.base_words:
                    self.base_words[base] = []
                self.base_words[base].append(word)

        print(f"Extracted {len(self.base_words)} unique base words")

    def generate_antonym_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate antonym pairs from vocabulary using WordNet.

        Returns:
            Dictionary mapping antonym type to list of (word1, word2) pairs
        """
        print("\nGenerating antonym pairs from vocabulary...")

        # Antonym type assignments (same as before)
        type_assignments = {
            0: "morality",
            1: "temperature",
            2: "size",
            3: "speed",
            4: "emotion_positive_negative",
            5: "light_dark",
            6: "strength",
            7: "difficulty",
            8: "quantity",
            9: "age",
            10: "distance",
            11: "quality",
            12: "wealth",
            13: "knowledge",
            14: "safety",
            15: "truth",
            16: "activity",
            17: "visibility",
            18: "orientation",
        }

        # Seed pairs for each category
        seed_pairs = {
            "morality": [("good", "bad"), ("right", "wrong")],
            "temperature": [("hot", "cold"), ("warm", "cool")],
            "size": [("big", "small"), ("large", "tiny")],
            "speed": [("fast", "slow"), ("quick", "sluggish")],
            "emotion_positive_negative": [("happy", "sad"), ("joy", "sorrow")],
            "light_dark": [("light", "dark"), ("bright", "dim")],
            "strength": [("strong", "weak"), ("powerful", "feeble")],
            "difficulty": [("easy", "hard"), ("simple", "difficult")],
            "quantity": [("many", "few"), ("abundant", "scarce")],
            "age": [("young", "old"), ("new", "ancient")],
            "distance": [("near", "far"), ("close", "distant")],
            "quality": [("beautiful", "ugly"), ("good", "bad")],
            "wealth": [("rich", "poor"), ("wealthy", "impoverished")],
            "knowledge": [("wise", "foolish"), ("intelligent", "stupid")],
            "safety": [("safe", "dangerous"), ("secure", "risky")],
            "truth": [("true", "false"), ("real", "fake")],
            "activity": [("active", "passive"), ("energetic", "lethargic")],
            "visibility": [("visible", "invisible"), ("apparent", "hidden")],
            "orientation": [("vertical", "horizontal"), ("upright", "flat")],
        }

        antonym_pairs = defaultdict(list)

        for antonym_type, seeds in seed_pairs.items():
            print(f"  Processing {antonym_type}...")
            pairs_found = 0

            # Check seed pairs
            for word1, word2 in seeds:
                if word1 in self.base_words and word2 in self.base_words:
                    # Use first vocabulary form of each word
                    vocab_word1 = self.base_words[word1][0]
                    vocab_word2 = self.base_words[word2][0]
                    antonym_pairs[antonym_type].append([vocab_word1, vocab_word2])
                    pairs_found += 1

            # Expand by finding antonyms in WordNet
            for base_word in self.base_words.keys():
                synsets = wn.synsets(base_word)

                for synset in synsets[:3]:  # Check first 3 senses
                    for lemma in synset.lemmas():
                        for antonym_lemma in lemma.antonyms():
                            antonym_base = antonym_lemma.name().lower()

                            # Check if antonym exists in vocabulary
                            if antonym_base in self.base_words and antonym_base != base_word:
                                # Check if this pair fits the category
                                if self._pair_fits_category(base_word, antonym_base, antonym_type):
                                    vocab_word1 = self.base_words[base_word][0]
                                    vocab_word2 = self.base_words[antonym_base][0]

                                    # Avoid duplicates
                                    pair = sorted([vocab_word1, vocab_word2])
                                    if pair not in antonym_pairs[antonym_type]:
                                        antonym_pairs[antonym_type].append(pair)
                                        pairs_found += 1

                # Limit pairs per category
                if pairs_found >= 20:
                    break

            print(f"    Found {len(antonym_pairs[antonym_type])} pairs")

        return dict(antonym_pairs)

    def _pair_fits_category(self, word1: str, word2: str, category: str) -> bool:
        """
        Check if an antonym pair fits a semantic category.

        Args:
            word1: First word
            word2: Second word
            category: Semantic category name

        Returns:
            True if pair fits category
        """
        # Simple heuristic: check if any seed words appear in synset definitions
        category_keywords = {
            "morality": ["good", "bad", "moral", "ethical", "right", "wrong"],
            "temperature": ["hot", "cold", "warm", "cool", "heat", "temperature"],
            "size": ["big", "small", "large", "tiny", "size"],
            "speed": ["fast", "slow", "quick", "speed"],
            "emotion_positive_negative": ["happy", "sad", "joy", "emotion", "feeling"],
            "light_dark": ["light", "dark", "bright", "dim"],
            "strength": ["strong", "weak", "powerful", "strength"],
            "difficulty": ["easy", "hard", "difficult", "simple"],
            "quantity": ["many", "few", "abundant", "scarce", "quantity"],
            "age": ["young", "old", "new", "ancient", "age"],
            "distance": ["near", "far", "close", "distant", "distance"],
            "quality": ["beautiful", "ugly", "quality", "appearance"],
            "wealth": ["rich", "poor", "wealthy", "wealth", "money"],
            "knowledge": ["wise", "foolish", "intelligent", "stupid", "knowledge"],
            "safety": ["safe", "dangerous", "secure", "risky", "safety"],
            "truth": ["true", "false", "real", "fake", "truth"],
            "activity": ["active", "passive", "energetic", "activity"],
            "visibility": ["visible", "invisible", "apparent", "hidden"],
            "orientation": ["vertical", "horizontal", "upright", "flat"],
        }

        keywords = category_keywords.get(category, [])

        # Check if words match keywords
        for keyword in keywords:
            if keyword in word1 or keyword in word2:
                return True

        return False

    def generate_semantic_clusters(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Generate semantic clusters from vocabulary.

        Returns:
            Dictionary mapping dimension to cluster (positive/negative/neutral words)
        """
        print("\nGenerating semantic clusters from vocabulary...")

        clusters = {}

        # Define semantic dimensions
        dimensions = {
            0: ("morality", ["good", "virtue", "moral"], ["bad", "evil", "immoral"]),
            1: ("temperature", ["hot", "warm", "heated"], ["cold", "cool", "chilled"]),
            2: ("size", ["large", "big", "huge"], ["small", "tiny", "minute"]),
            4: ("emotion", ["happy", "joy", "cheerful"], ["sad", "sorrow", "gloomy"]),
            5: ("light", ["light", "bright", "luminous"], ["dark", "dim", "shadowy"]),
            6: ("strength", ["strong", "powerful", "mighty"], ["weak", "feeble", "frail"]),
        }

        for dim_idx, (name, positive_seeds, negative_seeds) in dimensions.items():
            print(f"  Dimension {dim_idx} ({name})...")

            positive_words = self._expand_cluster_from_seeds(positive_seeds, max_words=50)
            negative_words = self._expand_cluster_from_seeds(negative_seeds, max_words=50)
            neutral_words = self._get_neutral_cluster_words(max_words=30)

            clusters[dim_idx] = {
                "positive": sorted(list(positive_words)),
                "negative": sorted(list(negative_words)),
                "neutral": sorted(list(neutral_words))
            }

            print(f"    Positive: {len(positive_words)} words")
            print(f"    Negative: {len(negative_words)} words")
            print(f"    Neutral: {len(neutral_words)} words")

        return clusters

    def _expand_cluster_from_seeds(self, seeds: List[str], max_words: int = 50) -> Set[str]:
        """
        Expand cluster from seed words using WordNet synonyms.

        Args:
            seeds: Seed words
            max_words: Maximum words to return

        Returns:
            Expanded set of vocabulary words
        """
        expanded = set()

        for seed in seeds:
            # Check if seed exists in vocabulary
            if seed in self.base_words:
                expanded.update(self.base_words[seed])

            # Expand using WordNet synonyms
            synsets = wn.synsets(seed)
            for synset in synsets[:3]:
                for lemma in synset.lemmas():
                    word = lemma.name().lower()
                    if word in self.base_words:
                        expanded.update(self.base_words[word])

                # Add similar synsets
                for similar in synset.similar_tos()[:2]:
                    for lemma in similar.lemmas():
                        word = lemma.name().lower()
                        if word in self.base_words:
                            expanded.update(self.base_words[word])

            if len(expanded) >= max_words:
                break

        return set(list(expanded)[:max_words])

    def _get_neutral_cluster_words(self, max_words: int = 30) -> Set[str]:
        """
        Get neutral words (concrete nouns with no semantic bias).

        Args:
            max_words: Maximum words to return

        Returns:
            Set of neutral vocabulary words
        """
        neutral_seeds = ['chair', 'table', 'rock', 'water', 'door', 'window',
                        'number', 'time', 'place', 'thing']

        neutral = set()

        for seed in neutral_seeds:
            if seed in self.base_words:
                neutral.update(self.base_words[seed])

            # Expand with hyponyms (more specific instances)
            synsets = wn.synsets(seed, pos=wn.NOUN)
            for synset in synsets[:2]:
                for lemma in synset.lemmas():
                    word = lemma.name().lower()
                    if word in self.base_words:
                        neutral.update(self.base_words[word])

                for hyponym in synset.hyponyms()[:3]:
                    for lemma in hyponym.lemmas():
                        word = lemma.name().lower()
                        if word in self.base_words:
                            neutral.update(self.base_words[word])

            if len(neutral) >= max_words:
                break

        return set(list(neutral)[:max_words])

    def save_configs(self, antonym_pairs: Dict, clusters: Dict):
        """
        Save configuration files.

        Args:
            antonym_pairs: Antonym pairs by type
            clusters: Semantic clusters by dimension
        """
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        # Save antonym pairs
        antonym_path = config_dir / "antonym_types.json"
        with open(antonym_path, 'w', encoding='utf-8') as f:
            json.dump(antonym_pairs, f, indent=2, ensure_ascii=False)
        print(f"\nSaved antonym pairs to {antonym_path}")

        # Save semantic clusters (convert int keys to strings for JSON)
        clusters_str_keys = {str(k): v for k, v in clusters.items()}
        clusters_path = config_dir / "semantic_clusters.json"
        with open(clusters_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_str_keys, f, indent=2, ensure_ascii=False)
        print(f"Saved semantic clusters to {clusters_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("VOCABULARY-MATCHED CONFIG GENERATION")
    print("=" * 70)
    print()

    # Generate configs
    generator = VocabConfigGenerator()

    antonym_pairs = generator.generate_antonym_pairs()
    clusters = generator.generate_semantic_clusters()

    # Save configs
    generator.save_configs(antonym_pairs, clusters)

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_antonym_pairs = sum(len(pairs) for pairs in antonym_pairs.values())
    print(f"Total antonym pairs: {total_antonym_pairs}")
    print(f"Antonym types: {len(antonym_pairs)}")
    print()
    print(f"Semantic dimensions: {len(clusters)}")
    total_cluster_words = sum(
        len(c['positive']) + len(c['negative']) + len(c['neutral'])
        for c in clusters.values()
    )
    print(f"Total cluster words: {total_cluster_words}")
    print()
    print("=" * 70)
    print("CONFIG GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
