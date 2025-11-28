"""Extract high-quality sentences from Wikipedia articles.

This script processes extracted Wikipedia articles and selects 10 million
high-quality, diverse sentences for training.

Requirements:
    pip install nltk

Usage:
    python scripts/extract_wikipedia_sentences.py

Input:
    data/wikipedia/extracted/ - Extracted articles from download_wikipedia.py

Output:
    data/wikipedia/sentences.txt - 10M high-quality sentences (~5GB)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
import nltk
from tqdm import tqdm


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


def load_articles(wiki_dir: Path, max_articles: int = None) -> List[Dict]:
    """Load extracted Wikipedia articles from JSON files.

    Args:
        wiki_dir: Directory containing extracted wiki files
        max_articles: Maximum number of articles to load (None = all)

    Returns:
        List of article dictionaries with 'title' and 'text' keys
    """
    articles = []

    # Find all wiki files (format: wiki_##)
    wiki_files = sorted(wiki_dir.rglob("wiki_*"))

    print(f"Found {len(wiki_files)} Wikipedia extract files")

    for file_path in tqdm(wiki_files, desc="Loading articles"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        article = json.loads(line)
                        articles.append(article)

                        if max_articles and len(articles) >= max_articles:
                            return articles
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

    return articles


def is_high_quality_sentence(sentence: str) -> bool:
    """Check if a sentence meets quality criteria.

    Quality criteria:
        - Length: 5-50 words
        - Contains at least one verb (heuristic: ends with ed/ing/s)
        - No excessive special characters
        - No URLs
        - No excessive numbers

    Args:
        sentence: Sentence to check

    Returns:
        True if sentence meets quality criteria
    """
    # Basic cleanup
    sentence = sentence.strip()

    if not sentence:
        return False

    words = sentence.split()

    # Length filter: 5-50 words
    if not (5 <= len(words) <= 50):
        return False

    # Must contain at least one verb (simple heuristic)
    # Look for common verb endings
    verb_patterns = ('ed', 'ing', 's', 'es', 'en')
    has_verb = any(
        w.lower().endswith(verb_patterns) and len(w) > 3
        for w in words
    )

    if not has_verb:
        return False

    # No excessive special characters (allow punctuation)
    special_chars = set('[]{}|<>@#$%^&*')
    special_count = sum(1 for c in sentence if c in special_chars)
    if special_count / len(sentence) > 0.03:  # 3% threshold
        return False

    # No URLs
    lower_sent = sentence.lower()
    if 'http' in lower_sent or 'www.' in lower_sent or '.com' in lower_sent:
        return False

    # Not too many numbers (technical spam filter)
    digit_count = sum(1 for c in sentence if c.isdigit())
    if digit_count / len(sentence) > 0.15:  # 15% threshold
        return False

    # Must start with capital letter (proper sentence)
    if not sentence[0].isupper():
        return False

    # Must end with proper punctuation
    if sentence[-1] not in '.!?':
        return False

    return True


def extract_sentences(articles: List[Dict]) -> List[str]:
    """Extract all sentences from articles using NLTK.

    Args:
        articles: List of article dictionaries

    Returns:
        List of sentences
    """
    all_sentences = []

    for article in tqdm(articles, desc="Extracting sentences"):
        text = article.get('text', '')

        if not text:
            continue

        # Tokenize into sentences
        try:
            sentences = nltk.sent_tokenize(text)
            all_sentences.extend(sentences)
        except Exception as e:
            # Skip articles with tokenization errors
            continue

    return all_sentences


def filter_quality(sentences: List[str]) -> List[str]:
    """Filter sentences by quality criteria.

    Args:
        sentences: List of sentences

    Returns:
        Filtered list of high-quality sentences
    """
    return [
        s for s in tqdm(sentences, desc="Filtering quality")
        if is_high_quality_sentence(s)
    ]


def stratified_sample(
    sentences: List[str],
    target: int = 10_000_000,
    window_size: int = 3
) -> List[str]:
    """Sample sentences with diversity across topics.

    Uses first N words as a rough topic proxy to ensure diversity.

    Args:
        sentences: List of sentences to sample from
        target: Target number of sentences
        window_size: Number of words to use for topic grouping

    Returns:
        Sampled sentences with topic diversity
    """
    print(f"Stratified sampling {target:,} sentences from {len(sentences):,}...")

    # If we have fewer sentences than target, return all
    if len(sentences) <= target:
        return sentences

    # Group sentences by first N words (rough topic proxy)
    by_topic = defaultdict(list)

    for sent in tqdm(sentences, desc="Grouping by topic"):
        words = sent.split()
        if len(words) >= window_size:
            # Use first N words as topic key
            topic_key = ' '.join(words[:window_size]).lower()
            by_topic[topic_key].append(sent)

    print(f"Found {len(by_topic)} topic clusters")

    # Sample proportionally from each topic
    sampled = []
    topics = list(by_topic.keys())
    random.shuffle(topics)

    # Calculate samples per topic
    per_topic = max(1, target // len(topics))

    for topic in tqdm(topics, desc="Sampling from topics"):
        available = by_topic[topic]
        sample_size = min(per_topic, len(available))

        if len(available) <= sample_size:
            sampled.extend(available)
        else:
            sampled.extend(random.sample(available, sample_size))

        if len(sampled) >= target:
            break

    # Shuffle final list
    random.shuffle(sampled)

    return sampled[:target]


def deduplicate(sentences: List[str]) -> List[str]:
    """Remove duplicate sentences while preserving order.

    Args:
        sentences: List of sentences

    Returns:
        Deduplicated list
    """
    seen = set()
    unique = []

    for sent in tqdm(sentences, desc="Deduplicating"):
        # Normalize for comparison (lowercase, strip)
        normalized = sent.lower().strip()

        if normalized not in seen:
            seen.add(normalized)
            unique.append(sent)

    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality sentences from Wikipedia"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/wikipedia/extracted"),
        help="Directory with extracted Wikipedia articles"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/wikipedia/sentences.txt"),
        help="Output file for sentences"
    )
    parser.add_argument(
        "--target-sentences",
        type=int,
        default=10_000_000,
        help="Target number of sentences (default: 10M)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to process (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 70)
    print("Wikipedia Sentence Extraction")
    print("=" * 70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Target sentences: {args.target_sentences:,}")
    print("=" * 70)

    # Step 1: Load articles
    print("\n[1/6] Loading articles...")
    articles = load_articles(args.input_dir, args.max_articles)
    print(f"  Loaded {len(articles):,} articles")

    # Step 2: Extract sentences
    print("\n[2/6] Extracting sentences...")
    all_sentences = extract_sentences(articles)
    print(f"  Extracted {len(all_sentences):,} sentences")

    # Step 3: Filter quality
    print("\n[3/6] Filtering quality...")
    quality_sentences = filter_quality(all_sentences)
    print(f"  Retained {len(quality_sentences):,} quality sentences")
    print(f"  Quality rate: {len(quality_sentences)/len(all_sentences)*100:.1f}%")

    # Step 4: Deduplicate
    print("\n[4/6] Deduplicating...")
    unique_sentences = deduplicate(quality_sentences)
    print(f"  After deduplication: {len(unique_sentences):,} sentences")
    print(f"  Duplicate rate: {(1 - len(unique_sentences)/len(quality_sentences))*100:.1f}%")

    # Step 5: Stratified sampling
    print("\n[5/6] Stratified sampling...")
    sampled_sentences = stratified_sample(unique_sentences, args.target_sentences)
    print(f"  Sampled {len(sampled_sentences):,} diverse sentences")

    # Step 6: Save
    print("\n[6/6] Saving to file...")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sent in tqdm(sampled_sentences, desc="Writing"):
            f.write(sent + '\n')

    # Statistics
    file_size_mb = args.output_file.stat().st_size / (1024**2)

    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"  Output file: {args.output_file}")
    print(f"  Sentences: {len(sampled_sentences):,}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Average sentence length: {sum(len(s.split()) for s in sampled_sentences) / len(sampled_sentences):.1f} words")
    print("=" * 70)

    print("\nNext steps:")
    print("  1. Run: python scripts/parse_wikipedia_corpus.py")
    print("  2. This will parse the sentences using the quantum parser")


if __name__ == "__main__":
    main()
