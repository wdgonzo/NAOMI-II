"""
Corpus Loader

Loads text corpora for training. Supports multiple sources:
- Brown Corpus (balanced, linguistic annotations)
- Project Gutenberg (books, literary texts)
- Wikipedia (encyclopedic, clean)
- Plain text files

Yields sentences in batches for efficient processing.
"""

import re
from typing import List, Iterator, Tuple
from pathlib import Path


def load_brown_corpus(num_sentences: int = None) -> Iterator[str]:
    """
    Load sentences from Brown Corpus via NLTK.

    Brown Corpus: ~57K sentences, balanced across genres
    (news, fiction, academic, etc.)

    Args:
        num_sentences: Maximum number of sentences to load (None = all)

    Yields:
        Individual sentences as strings
    """
    try:
        import nltk
        from nltk.corpus import brown
    except ImportError:
        raise ImportError("NLTK required for Brown Corpus. Install: pip install nltk")

    # Download if needed
    try:
        brown.sents()
    except LookupError:
        print("Downloading Brown Corpus...")
        nltk.download('brown')

    count = 0
    for sent_tokens in brown.sents():
        # Brown corpus comes pre-tokenized
        sentence = ' '.join(sent_tokens)
        yield sentence

        count += 1
        if num_sentences and count >= num_sentences:
            break


def load_gutenberg_corpus(num_books: int = 10) -> Iterator[str]:
    """
    Load sentences from Project Gutenberg via NLTK.

    Gutenberg: Classic literature, ~18 books in NLTK corpus

    Args:
        num_books: Maximum number of books to load

    Yields:
        Individual sentences as strings
    """
    try:
        import nltk
        from nltk.corpus import gutenberg
    except ImportError:
        raise ImportError("NLTK required for Gutenberg. Install: pip install nltk")

    # Download if needed
    try:
        gutenberg.fileids()
    except LookupError:
        print("Downloading Gutenberg Corpus...")
        nltk.download('gutenberg')

    fileids = gutenberg.fileids()[:num_books]

    for fileid in fileids:
        # Get raw text and split into sentences ourselves
        text = gutenberg.raw(fileid)
        for sentence in split_into_sentences(text):
            yield sentence


def load_text_file(file_path: str, sentence_per_line: bool = False) -> Iterator[str]:
    """
    Load sentences from a plain text file.

    Args:
        file_path: Path to text file
        sentence_per_line: If True, each line is a sentence.
                          If False, split paragraphs into sentences.

    Yields:
        Individual sentences as strings
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        if sentence_per_line:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield line
        else:
            # Read full file and split into sentences
            text = f.read()
            for sentence in split_into_sentences(text):
                yield sentence


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.

    Splits on . ! ? followed by whitespace and capital letter.
    Not perfect but good enough for training data.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Basic sentence boundary detection
    # Handles: "Hello. World" but not "Dr. Smith" perfectly
    sentences = []

    # Split on sentence boundaries
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    chunks = re.split(pattern, text)

    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and len(chunk) > 1:  # Skip single characters
            sentences.append(chunk)

    return sentences


def load_corpus_batch(source: str, batch_size: int = 1000,
                      max_sentences: int = None, **kwargs) -> Iterator[List[str]]:
    """
    Load corpus in batches for memory-efficient processing.

    Args:
        source: Corpus source ('brown', 'gutenberg', or file path)
        batch_size: Number of sentences per batch
        max_sentences: Maximum total sentences to load
        **kwargs: Additional arguments for specific loaders

    Yields:
        Batches of sentences (lists of strings)
    """
    # Select loader
    if source == 'brown':
        sentence_iter = load_brown_corpus(num_sentences=max_sentences)
    elif source == 'gutenberg':
        sentence_iter = load_gutenberg_corpus(**kwargs)
    elif Path(source).exists():
        sentence_iter = load_text_file(source, **kwargs)
    else:
        raise ValueError(f"Unknown corpus source: {source}")

    # Yield in batches
    batch = []
    total_count = 0

    for sentence in sentence_iter:
        batch.append(sentence)
        total_count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

        if max_sentences and total_count >= max_sentences:
            break

    # Yield final partial batch
    if batch:
        yield batch


def get_corpus_statistics(source: str, sample_size: int = 1000) -> dict:
    """
    Get statistics about a corpus.

    Args:
        source: Corpus source
        sample_size: Number of sentences to sample

    Returns:
        Dictionary with corpus statistics
    """
    sentences = []
    word_counts = []

    for batch in load_corpus_batch(source, batch_size=100, max_sentences=sample_size):
        sentences.extend(batch)

    for sentence in sentences:
        word_counts.append(len(sentence.split()))

    return {
        'num_sentences': len(sentences),
        'avg_sentence_length': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_sentence_length': min(word_counts) if word_counts else 0,
        'max_sentence_length': max(word_counts) if word_counts else 0,
        'sample_sentences': sentences[:5]
    }


# Example usage
if __name__ == "__main__":
    print("=== Brown Corpus Sample ===")
    brown_stats = get_corpus_statistics('brown', sample_size=100)
    print(f"Sentences: {brown_stats['num_sentences']}")
    print(f"Avg length: {brown_stats['avg_sentence_length']:.1f} words")
    print(f"Sample: {brown_stats['sample_sentences'][0]}")

    print("\n=== Gutenberg Corpus Sample ===")
    gutenberg_stats = get_corpus_statistics('gutenberg', sample_size=100)
    print(f"Sentences: {gutenberg_stats['num_sentences']}")
    print(f"Avg length: {gutenberg_stats['avg_sentence_length']:.1f} words")
    print(f"Sample: {gutenberg_stats['sample_sentences'][0]}")
