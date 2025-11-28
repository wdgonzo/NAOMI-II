"""
Extract plain text from Wikipedia XML dump.

This is a simple alternative to wikiextractor that works with Python 3.12+.
Extracts article text from Wikipedia XML dumps and outputs clean sentences.

Usage:
    python scripts/extract_wiki_dump.py \
        --input data/wikipedia/enwiki-latest-pages-articles.xml.bz2 \
        --output data/wikipedia/extracted_articles.txt \
        --max-articles 100000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bz2
import re
import argparse
from typing import Iterator
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    Clean Wikipedia markup from text.

    Args:
        text: Raw Wikipedia text with markup

    Returns:
        Cleaned plain text
    """
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)

    # Remove file references [[File:...]]
    text = re.sub(r'\[\[File:[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Image:[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # Remove categories [[Category:...]]
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # Convert internal links [[Link|Text]] -> Text or [[Link]] -> Link
    text = re.sub(r'\[\[([^\|\]]*\|)?([^\]]*)\]\]', r'\2', text)

    # Remove external links [http://... Text] -> Text
    text = re.sub(r'\[http[^\s]*\s+([^\]]*)\]', r'\1', text)

    # Remove bare URLs
    text = re.sub(r'http[s]?://[^\s]+', '', text)

    # Remove HTML comments <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove references <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*\/>', '', text, flags=re.IGNORECASE)

    # Remove bold and italic markup
    text = re.sub(r"'{2,}", '', text)

    # Remove section headers == Header ==
    text = re.sub(r'={2,}[^=]*={2,}', '', text)

    # Remove bullet points and numbering
    text = re.sub(r'^\s*[\*#:;]+', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_articles(dump_path: Path, max_articles: int = None) -> Iterator[tuple[str, str]]:
    """
    Extract articles from Wikipedia XML dump.

    Args:
        dump_path: Path to .xml.bz2 dump file
        max_articles: Maximum number of articles to extract (None = all)

    Yields:
        (title, text) tuples for each article
    """
    # Open bz2 file
    if str(dump_path).endswith('.bz2'):
        f = bz2.open(dump_path, 'rt', encoding='utf-8', errors='ignore')
    else:
        f = open(dump_path, 'r', encoding='utf-8', errors='ignore')

    current_title = None
    current_text = []
    in_text = False
    article_count = 0

    try:
        for line in f:
            # Start of article
            if '<title>' in line:
                current_title = re.search(r'<title>(.*?)</title>', line)
                if current_title:
                    current_title = current_title.group(1)

            # Start of text content
            elif '<text' in line:
                in_text = True
                # Extract text on same line if present
                text_match = re.search(r'<text[^>]*>(.*)', line)
                if text_match:
                    current_text.append(text_match.group(1))

            # End of text content
            elif '</text>' in line:
                in_text = False
                # Extract remaining text
                text_match = re.search(r'(.*)</text>', line)
                if text_match:
                    current_text.append(text_match.group(1))

                # Process article
                if current_title and current_text:
                    # Skip redirects and special pages
                    text = ''.join(current_text)
                    if not text.strip().lower().startswith('#redirect') and \
                       not current_title.startswith('Wikipedia:') and \
                       not current_title.startswith('Talk:') and \
                       not current_title.startswith('User:') and \
                       not current_title.startswith('Template:'):

                        # Clean the text
                        cleaned_text = clean_text(text)

                        # Only yield if we have substantial content
                        if len(cleaned_text) > 100:
                            yield (current_title, cleaned_text)
                            article_count += 1

                            if max_articles and article_count >= max_articles:
                                return

                # Reset for next article
                current_title = None
                current_text = []

            # Collect text content
            elif in_text:
                current_text.append(line)

    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Extract Wikipedia articles from XML dump')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to Wikipedia XML dump (.xml.bz2)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for extracted text')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Maximum number of articles to extract (default: all)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WIKIPEDIA DUMP EXTRACTION")
    print("=" * 70)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    if args.max_articles:
        print(f"Max articles: {args.max_articles:,}")
    else:
        print(f"Max articles: unlimited")
    print()

    # Extract articles
    article_count = 0
    sentence_count = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Use tqdm for progress if max_articles specified
        if args.max_articles:
            articles = tqdm(
                extract_articles(input_path, args.max_articles),
                total=args.max_articles,
                desc="Extracting articles"
            )
        else:
            articles = extract_articles(input_path, args.max_articles)
            print("Extracting articles (this may take a while)...")

        for title, text in articles:
            article_count += 1

            # Write title
            out_f.write(f"=== {title} ===\n")

            # Write text
            out_f.write(text)
            out_f.write("\n\n")

            # Count sentences (approximate)
            sentence_count += text.count('.') + text.count('!') + text.count('?')

            # Progress update every 1000 articles (if not using tqdm)
            if not args.max_articles and article_count % 1000 == 0:
                print(f"  Processed {article_count:,} articles, ~{sentence_count:,} sentences")

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nArticles extracted: {article_count:,}")
    print(f"Approximate sentences: {sentence_count:,}")
    print(f"Output file: {output_path}")
    print(f"Output size: {output_path.stat().st_size / 1e6:.1f} MB")
    print()


if __name__ == "__main__":
    main()
