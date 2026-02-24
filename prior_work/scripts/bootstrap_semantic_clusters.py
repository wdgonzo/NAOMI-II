"""
Bootstrap Semantic Clusters from WordNet

Automatically creates semantic clusters by analyzing WordNet synsets and definitions.
These clusters are used to enforce dimensional consistency during training.

Output: config/semantic_clusters.json

Usage:
    python scripts/bootstrap_semantic_clusters.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict
from typing import Dict, List, Set


# Ensure WordNet is downloaded
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')


class SemanticClusterBootstrapper:
    """Bootstrap semantic clusters from WordNet."""

    def __init__(self):
        self.clusters = {}

    def bootstrap_morality_cluster(self) -> Dict[str, List[str]]:
        """
        Bootstrap morality dimension (dim 0).

        Positive: good, virtuous, righteous, moral, ethical
        Negative: bad, evil, wicked, immoral, unethical
        Neutral: chair, table, rock, water (no moral content)
        """
        positive_seeds = ['good', 'virtue', 'righteous', 'moral', 'ethical', 'honest', 'just']
        negative_seeds = ['bad', 'evil', 'wicked', 'immoral', 'unethical', 'dishonest', 'unjust']

        positive_words = self._expand_from_seeds(positive_seeds, max_words=50)
        negative_words = self._expand_from_seeds(negative_seeds, max_words=50)

        # Get neutral words (physical objects with no moral connotations)
        neutral_words = self._get_neutral_words(['chair', 'table', 'rock', 'water', 'door', 'window'], max_words=30)

        return {
            'positive': sorted(list(positive_words)),
            'negative': sorted(list(negative_words)),
            'neutral': sorted(list(neutral_words))
        }

    def bootstrap_temperature_cluster(self) -> Dict[str, List[str]]:
        """Bootstrap temperature dimension (dim 1)."""
        hot_seeds = ['hot', 'warm', 'heated', 'boiling']
        cold_seeds = ['cold', 'cool', 'chilled', 'freezing']

        hot_words = self._expand_from_seeds(hot_seeds, max_words=30)
        cold_words = self._expand_from_seeds(cold_seeds, max_words=30)
        neutral_words = self._get_neutral_words(['chair', 'table', 'number'], max_words=20)

        return {
            'positive': sorted(list(hot_words)),  # hot
            'negative': sorted(list(cold_words)),  # cold
            'neutral': sorted(list(neutral_words))
        }

    def bootstrap_size_cluster(self) -> Dict[str, List[str]]:
        """Bootstrap size dimension (dim 2)."""
        large_seeds = ['large', 'big', 'huge', 'giant', 'massive', 'enormous']
        small_seeds = ['small', 'tiny', 'minuscule', 'miniature', 'minute']

        large_words = self._expand_from_seeds(large_seeds, max_words=40)
        small_words = self._expand_from_seeds(small_seeds, max_words=40)
        neutral_words = self._get_neutral_words(['color', 'idea', 'truth'], max_words=20)

        return {
            'positive': sorted(list(large_words)),
            'negative': sorted(list(small_words)),
            'neutral': sorted(list(neutral_words))
        }

    def bootstrap_emotion_cluster(self) -> Dict[str, List[str]]:
        """Bootstrap emotion dimension (dim 4)."""
        positive_seeds = ['happy', 'joy', 'cheerful', 'delighted', 'pleased']
        negative_seeds = ['sad', 'sorrow', 'gloomy', 'disappointed', 'depressed']

        positive_words = self._expand_from_seeds(positive_seeds, max_words=40)
        negative_words = self._expand_from_seeds(negative_seeds, max_words=40)
        neutral_words = self._get_neutral_words(['rock', 'table', 'number'], max_words=20)

        return {
            'positive': sorted(list(positive_words)),
            'negative': sorted(list(negative_words)),
            'neutral': sorted(list(neutral_words))
        }

    def bootstrap_strength_cluster(self) -> Dict[str, List[str]]:
        """Bootstrap strength dimension (dim 6)."""
        strong_seeds = ['strong', 'powerful', 'mighty', 'robust', 'sturdy']
        weak_seeds = ['weak', 'feeble', 'frail', 'fragile', 'delicate']

        strong_words = self._expand_from_seeds(strong_seeds, max_words=35)
        weak_words = self._expand_from_seeds(weak_seeds, max_words=35)
        neutral_words = self._get_neutral_words(['color', 'shape', 'time'], max_words=20)

        return {
            'positive': sorted(list(strong_words)),
            'negative': sorted(list(weak_words)),
            'neutral': sorted(list(neutral_words))
        }

    def bootstrap_light_cluster(self) -> Dict[str, List[str]]:
        """Bootstrap light/dark dimension (dim 5)."""
        light_seeds = ['light', 'bright', 'luminous', 'radiant', 'brilliant']
        dark_seeds = ['dark', 'dim', 'shadowy', 'murky', 'gloomy']

        light_words = self._expand_from_seeds(light_seeds, max_words=30)
        dark_words = self._expand_from_seeds(dark_seeds, max_words=30)
        neutral_words = self._get_neutral_words(['chair', 'number', 'idea'], max_words=20)

        return {
            'positive': sorted(list(light_words)),
            'negative': sorted(list(dark_words)),
            'neutral': sorted(list(neutral_words))
        }

    def _expand_from_seeds(self, seeds: List[str], max_words: int = 50) -> Set[str]:
        """
        Expand seed words using WordNet synonyms and similar words.

        Args:
            seeds: Initial seed words
            max_words: Maximum words to return

        Returns:
            Expanded set of words
        """
        expanded = set(seeds)

        for seed in seeds:
            # Get all synsets for this seed word
            synsets = wn.synsets(seed)

            for synset in synsets[:3]:  # Limit to top 3 senses
                # Add synonyms (lemmas in the same synset)
                for lemma in synset.lemmas():
                    word = lemma.name().replace('_', ' ').lower()
                    if ' ' not in word:  # Only single words
                        expanded.add(word)

                # Add words from similar synsets
                for similar_synset in synset.similar_tos()[:2]:
                    for lemma in similar_synset.lemmas():
                        word = lemma.name().replace('_', ' ').lower()
                        if ' ' not in word:
                            expanded.add(word)

                if len(expanded) >= max_words:
                    break

            if len(expanded) >= max_words:
                break

        return set(list(expanded)[:max_words])

    def _get_neutral_words(self, seeds: List[str], max_words: int = 30) -> Set[str]:
        """
        Get neutral words (typically concrete nouns with no semantic bias).

        Args:
            seeds: Seed neutral words
            max_words: Maximum words to return

        Returns:
            Set of neutral words
        """
        neutral = set(seeds)

        # Look for concrete nouns
        for seed in seeds:
            synsets = wn.synsets(seed, pos=wn.NOUN)

            for synset in synsets[:2]:
                for lemma in synset.lemmas():
                    word = lemma.name().replace('_', ' ').lower()
                    if ' ' not in word:
                        neutral.add(word)

                # Add hyponyms (more specific instances)
                for hyponym in synset.hyponyms()[:3]:
                    for lemma in hyponym.lemmas():
                        word = lemma.name().replace('_', ' ').lower()
                        if ' ' not in word:
                            neutral.add(word)

                if len(neutral) >= max_words:
                    break

            if len(neutral) >= max_words:
                break

        return set(list(neutral)[:max_words])

    def bootstrap_all_clusters(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Bootstrap all semantic clusters.

        Returns:
            Dictionary mapping dimension index to cluster definition
        """
        print("Bootstrapping semantic clusters from WordNet...")
        print()

        clusters = {}

        print("[1/6] Bootstrapping morality cluster (dim 0)...")
        clusters[0] = self.bootstrap_morality_cluster()
        print(f"  Positive: {len(clusters[0]['positive'])} words")
        print(f"  Negative: {len(clusters[0]['negative'])} words")
        print(f"  Neutral: {len(clusters[0]['neutral'])} words")

        print("[2/6] Bootstrapping temperature cluster (dim 1)...")
        clusters[1] = self.bootstrap_temperature_cluster()
        print(f"  Hot: {len(clusters[1]['positive'])} words")
        print(f"  Cold: {len(clusters[1]['negative'])} words")

        print("[3/6] Bootstrapping size cluster (dim 2)...")
        clusters[2] = self.bootstrap_size_cluster()
        print(f"  Large: {len(clusters[2]['positive'])} words")
        print(f"  Small: {len(clusters[2]['negative'])} words")

        print("[4/6] Bootstrapping emotion cluster (dim 4)...")
        clusters[4] = self.bootstrap_emotion_cluster()
        print(f"  Positive: {len(clusters[4]['positive'])} words")
        print(f"  Negative: {len(clusters[4]['negative'])} words")

        print("[5/6] Bootstrapping strength cluster (dim 6)...")
        clusters[6] = self.bootstrap_strength_cluster()
        print(f"  Strong: {len(clusters[6]['positive'])} words")
        print(f"  Weak: {len(clusters[6]['negative'])} words")

        print("[6/6] Bootstrapping light/dark cluster (dim 5)...")
        clusters[5] = self.bootstrap_light_cluster()
        print(f"  Light: {len(clusters[5]['positive'])} words")
        print(f"  Dark: {len(clusters[5]['negative'])} words")

        print()
        print(f"Bootstrapped {len(clusters)} semantic clusters")

        return clusters


def main():
    """Main execution."""
    print("=" * 70)
    print("SEMANTIC CLUSTER BOOTSTRAPPING")
    print("=" * 70)
    print()

    # Bootstrap clusters
    bootstrapper = SemanticClusterBootstrapper()
    clusters = bootstrapper.bootstrap_all_clusters()

    # Convert integer keys to strings for JSON
    clusters_str_keys = {str(k): v for k, v in clusters.items()}

    # Save to config file
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    output_path = config_dir / "semantic_clusters.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clusters_str_keys, f, indent=2, ensure_ascii=False)

    print(f"Saved semantic clusters to {output_path}")
    print()

    # Print sample
    print("Sample clusters:")
    print()
    for dim_idx, cluster in sorted(clusters.items())[:3]:
        print(f"Dimension {dim_idx}:")
        print(f"  Positive: {', '.join(cluster['positive'][:10])}...")
        print(f"  Negative: {', '.join(cluster['negative'][:10])}...")
        print(f"  Neutral: {', '.join(cluster['neutral'][:5])}...")
        print()

    print("=" * 70)
    print("BOOTSTRAPPING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
