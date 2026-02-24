"""
Random Wikipedia Sentence Sampler

Uniformly samples N sentences from Wikipedia corpus for bootstrap training.

Usage:
    # Sample 300K sentences for bootstrap
    python scripts/sample_wikipedia.py --num-samples 300000 --output-file data/wikipedia_300k.txt

    # Sample 10K sentences, excluding those in another file
    python scripts/sample_wikipedia.py --num-samples 10000 --exclude-file data/wikipedia_300k.txt --output-file data/wikipedia_10k_batch1.txt

Features:
    - Uniform random sampling (each sentence has equal probability)
    - Reproducible (seeded random number generator)
    - Memory-efficient (reservoir sampling for large corpora)
    - Can exclude sentences from a previous sample (for incremental batches)
"""

import argparse
import random
from pathlib import Path
from typing import Set, Optional
from tqdm import tqdm


def load_exclude_set(exclude_file: Optional[Path]) -> Set[str]:
    """
    Load sentences to exclude from sampling.

    Args:
        exclude_file: Path to file containing sentences to exclude (one per line)

    Returns:
        Set of sentences to exclude
    """
    if exclude_file is None or not exclude_file.exists():
        return set()

    print(f"Loading exclusion set from {exclude_file}...")
    exclude_set = set()
    with open(exclude_file, 'r', encoding='utf-8') as f:
        for line in f:
            exclude_set.add(line.strip())
    print(f"  Loaded {len(exclude_set):,} sentences to exclude")
    return exclude_set


def count_lines(file_path: Path) -> int:
    """
    Count total lines in file for progress bar.

    Args:
        file_path: Path to file

    Returns:
        Number of lines
    """
    print(f"Counting total sentences in {file_path}...")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    print(f"  Found {count:,} total sentences")
    return count


def reservoir_sample(input_file: Path, num_samples: int,
                     exclude_set: Set[str], seed: int) -> list[str]:
    """
    Uniformly sample N sentences using reservoir sampling algorithm.

    This is memory-efficient for large corpora - uses O(N) memory regardless of corpus size.

    Algorithm (Reservoir Sampling):
        1. Keep first K sentences in reservoir
        2. For sentence i (where i > K):
           - Generate random number r in [0, i]
           - If r < K, replace reservoir[r] with sentence i
        3. Each sentence has equal probability K/total of being selected

    Args:
        input_file: Path to Wikipedia corpus (one sentence per line)
        num_samples: Number of sentences to sample
        exclude_set: Set of sentences to exclude
        seed: Random seed for reproducibility

    Returns:
        List of sampled sentences
    """
    random.seed(seed)

    total_lines = count_lines(input_file)

    print(f"\nSampling {num_samples:,} sentences using reservoir sampling...")
    print(f"  Random seed: {seed}")
    print(f"  Exclusions: {len(exclude_set):,} sentences")
    print()

    reservoir = []
    candidates_seen = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Sampling", unit="sent")):
            sentence = line.strip()

            # Skip empty lines or excluded sentences
            if not sentence or sentence in exclude_set:
                continue

            candidates_seen += 1

            # Reservoir sampling algorithm
            if len(reservoir) < num_samples:
                # Fill reservoir until we have num_samples
                reservoir.append(sentence)
            else:
                # Randomly replace element with decreasing probability
                j = random.randint(0, candidates_seen - 1)
                if j < num_samples:
                    reservoir[j] = sentence

    print()
    print(f"Sampling complete!")
    print(f"  Candidates seen: {candidates_seen:,}")
    print(f"  Samples selected: {len(reservoir):,}")

    if len(reservoir) < num_samples:
        print(f"  WARNING: Only found {len(reservoir):,} eligible sentences (requested {num_samples:,})")

    return reservoir


def save_samples(samples: list[str], output_file: Path):
    """
    Save sampled sentences to file (one per line).

    Args:
        samples: List of sampled sentences
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {len(samples):,} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in samples:
            f.write(sentence + '\n')

    print(f"  Saved successfully!")
    print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Uniformly sample N sentences from Wikipedia corpus'
    )
    parser.add_argument('--input-file', type=str,
                       default='notebooks/data/extracted_articles.txt',
                       help='Path to Wikipedia corpus (one sentence per line)')
    parser.add_argument('--num-samples', type=int, required=True,
                       help='Number of sentences to sample')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Path to output file')
    parser.add_argument('--exclude-file', type=str, default=None,
                       help='Path to file with sentences to exclude (for incremental batches)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    exclude_file = Path(args.exclude_file) if args.exclude_file else None

    # Validate input
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    if output_file.exists():
        response = input(f"WARNING: Output file {output_file} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print("=" * 70)
    print("WIKIPEDIA SENTENCE SAMPLER")
    print("=" * 70)
    print()
    print(f"Input corpus: {input_file}")
    print(f"Sample size: {args.num_samples:,} sentences")
    print(f"Output file: {output_file}")
    if exclude_file:
        print(f"Exclusion file: {exclude_file}")
    print(f"Random seed: {args.seed}")
    print()

    # Load exclusion set (if any)
    exclude_set = load_exclude_set(exclude_file)

    # Sample sentences
    samples = reservoir_sample(input_file, args.num_samples, exclude_set, args.seed)

    # Save to file
    save_samples(samples, output_file)

    print()
    print("=" * 70)
    print("SAMPLING COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. Parse corpus:")
    print(f"     python scripts/stream_parse_corpus.py --corpus {output_file} --num-workers 6 --output-dir data/parsed")
    print(f"  2. Build knowledge graph:")
    print(f"     python scripts/build_sense_graph.py --corpus data/parsed/parsed_corpus.pkl --output-dir data/graph")
    print(f"  3. Train embeddings:")
    print(f"     python scripts/train_embeddings.py --graph-dir data/graph --epochs 100")
    print()


if __name__ == "__main__":
    main()
