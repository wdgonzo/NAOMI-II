"""Download English Wikipedia dump and extract clean articles.

This script downloads the latest English Wikipedia XML dump and extracts
clean article text using WikiExtractor.

Requirements:
    pip install wikiextractor

Usage:
    python scripts/download_wikipedia.py

Output:
    data/wikipedia/extracted/ - Clean article text (~20GB)
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
import argparse


def download_wikipedia_dump(output_dir: Path, url: Optional[str] = None) -> Path:
    """Download latest English Wikipedia XML dump.

    Args:
        output_dir: Directory to save the dump
        url: Optional custom URL. If None, uses latest dump.

    Returns:
        Path to downloaded dump file
    """
    if url is None:
        url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

    output_file = output_dir / "enwiki-latest.xml.bz2"

    # Check if already downloaded
    if output_file.exists():
        print(f"Dump already exists: {output_file}")
        print("Skipping download. Delete file to re-download.")
        return output_file

    print(f"Downloading Wikipedia dump from:")
    print(f"  {url}")
    print(f"To: {output_file}")
    print("This will take 1-2 hours depending on connection speed...")

    try:
        # Use wget with resume support (-c flag)
        subprocess.run([
            "wget",
            "-c",  # Resume if interrupted
            url,
            "-O", str(output_file)
        ], check=True)
    except FileNotFoundError:
        print("\nERROR: wget not found. Please install wget:")
        print("  Windows: choco install wget")
        print("  Mac: brew install wget")
        print("  Linux: apt-get install wget")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Download failed: {e}")
        sys.exit(1)

    print(f"\nDownload complete: {output_file}")
    return output_file


def extract_articles(dump_path: Path, output_path: Path, num_processes: int = 8):
    """Extract clean text from Wikipedia XML using WikiExtractor.

    Args:
        dump_path: Path to Wikipedia XML dump
        output_path: Directory to save extracted articles
        num_processes: Number of parallel processes for extraction
    """
    print(f"\nExtracting articles from {dump_path}")
    print(f"Output directory: {output_path}")
    print(f"Using {num_processes} parallel processes...")
    print("This will take 1-2 hours...")

    # Check if wikiextractor is installed
    try:
        result = subprocess.run(
            ["wikiextractor", "--help"],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        print("\nERROR: wikiextractor not found. Please install:")
        print("  pip install wikiextractor")
        sys.exit(1)

    try:
        subprocess.run([
            "wikiextractor",
            str(dump_path),
            "--output", str(output_path),
            "--bytes", "100M",  # Split into 100MB chunks
            "--filter_disambig_pages",  # Remove disambiguation pages
            "--no-templates",  # Remove templates
            "--processes", str(num_processes),  # Parallel extraction
            "--json"  # Output as JSON for easier parsing
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Extraction failed: {e}")
        sys.exit(1)

    print(f"\nExtraction complete: {output_path}")


def get_directory_size(path: Path) -> float:
    """Get total size of directory in GB."""
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024**3)  # Convert to GB


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Wikipedia dump"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/wikipedia"),
        help="Output directory (default: data/wikipedia)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom Wikipedia dump URL (default: latest English dump)"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=8,
        help="Number of parallel extraction processes (default: 8)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only extract (assumes dump already exists)"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction, only download"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Wikipedia Download and Extraction")
    print("=" * 70)

    # Step 1: Download
    if not args.skip_download:
        print("\n[1/2] Downloading Wikipedia dump...")
        dump_path = download_wikipedia_dump(args.output_dir, args.url)
    else:
        dump_path = args.output_dir / "enwiki-latest.xml.bz2"
        if not dump_path.exists():
            print(f"ERROR: Dump not found at {dump_path}")
            print("Remove --skip-download flag to download.")
            sys.exit(1)
        print(f"\n[1/2] Using existing dump: {dump_path}")

    # Step 2: Extract
    if not args.skip_extract:
        print("\n[2/2] Extracting articles...")
        extracted_path = args.output_dir / "extracted"
        extract_articles(dump_path, extracted_path, args.processes)

        # Report statistics
        size_gb = get_directory_size(extracted_path)
        num_files = len(list(extracted_path.rglob("wiki_*")))

        print("\n" + "=" * 70)
        print("Extraction Statistics:")
        print(f"  Output directory: {extracted_path}")
        print(f"  Total size: {size_gb:.2f} GB")
        print(f"  Number of files: {num_files}")
        print("=" * 70)
    else:
        print("\n[2/2] Skipping extraction (--skip-extract)")

    print("\nDone! Next steps:")
    print("  1. Run: python scripts/extract_wikipedia_sentences.py")
    print("  2. This will extract 10M high-quality sentences for training")


if __name__ == "__main__":
    main()
