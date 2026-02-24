"""
Streaming Corpus Parser - Memory-Safe for Large Corpora

Designed for parsing millions of sentences (e.g., 12.4M Wikipedia sentences)
without accumulating results in memory.

Key Features:
- Streams results to disk immediately (constant RAM usage)
- Lightweight progress checkpoints (just counters, not full results)
- Resumable after crash/OOM/reboot
- Real-time progress: "X/12,377,687" in terminal
- Windows-safe multiprocessing (conservative worker count)

Output Structure:
  data/wikipedia_parsed/
  ├── batches/
  │   ├── batch_0000000.jsonl  # 10K sentences each
  │   ├── batch_0010000.jsonl
  │   └── ...
  ├── progress.json            # Lightweight checkpoint
  └── parse_stats.json         # Statistics

Usage:
  # Test with 10K sentences
  python scripts/stream_parse_corpus.py \\
      --corpus notebooks/data/extracted_articles.txt \\
      --max-sentences 10000 \\
      --num-workers 4 \\
      --output-dir data/wikipedia_parsed_test

  # Full 12.4M parse (run for ~29 hours)
  python scripts/stream_parse_corpus.py \\
      --corpus notebooks/data/extracted_articles.txt \\
      --max-sentences 12400000 \\
      --num-workers 4 \\
      --output-dir data/wikipedia_parsed \\
      --resume

  # After completion, consolidate to pickle (optional)
  python scripts/stream_parse_corpus.py \\
      --consolidate-only \\
      --output-dir data/wikipedia_parsed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pickle
import json
import time
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import glob

from src.parser.quantum_parser import QuantumParser
from src.parser.chart_parser import ChartParser
from src.parser.pos_tagger import tag_sentence
from src.graph.triple_extractor import extract_triples
from src.embeddings.sense_mapper import extract_word_contexts_from_parse
from src.data_pipeline.corpus_loader import load_corpus_batch


# Global variables for multiprocessing workers
_worker_parser = None
_parser_type = None
_grammar_path = None


def _init_worker(parser_type: str, grammar_path: str):
    """
    Initialize parser in worker process.

    Args:
        parser_type: 'quantum' or 'chart'
        grammar_path: Path to grammar file
    """
    global _worker_parser, _parser_type, _grammar_path
    _parser_type = parser_type
    _grammar_path = grammar_path

    if parser_type == 'chart':
        _worker_parser = ChartParser(grammar_path)
    else:
        _worker_parser = QuantumParser(grammar_path)


def _parse_sentence_worker(sentence_and_id: Tuple[str, int]) -> Dict:
    """
    Worker function for parallel parsing.

    Args:
        sentence_and_id: Tuple of (sentence, sentence_id)

    Returns:
        Dict with parse result (serialized to reduce pickle overhead)
    """
    sentence, sentence_id = sentence_and_id
    result = parse_sentence_with_context(_worker_parser, sentence, sentence_id)
    # Serialize in worker to reduce pickle overhead (80-90% reduction)
    # Remove Hypothesis object before sending back to main process
    return serialize_parse_result(result)


def parse_sentence_with_context(parser, sentence: str, sentence_id: int) -> Dict:
    """
    Parse a single sentence and extract context.

    Args:
        parser: QuantumParser or ChartParser instance
        sentence: Sentence text
        sentence_id: Sentence ID for tracking

    Returns:
        Dict with parse result, triples, and word contexts
    """
    result = {
        'sentence_id': sentence_id,
        'sentence': sentence,
        'success': False,
        'parse_score': 0.0,
        'hypothesis': None,
        'triples': [],
        'word_contexts': {},
        'num_triples': 0,
        'error': None
    }

    try:
        # Tag sentence (convert string to List[Word])
        words = tag_sentence(sentence)

        # Parse sentence
        chart = parser.parse(words)
        hypothesis = chart.best_hypothesis()

        if hypothesis and hypothesis.score > 0:
            result['success'] = True
            result['parse_score'] = hypothesis.score
            result['hypothesis'] = hypothesis

            # Extract triples
            triples = extract_triples(hypothesis)
            result['triples'] = triples
            result['num_triples'] = len(triples)

            # Extract word contexts for WSD
            word_contexts = extract_word_contexts_from_parse(sentence_id, hypothesis)
            result['word_contexts'] = word_contexts

    except Exception as e:
        result['error'] = str(e)

    return result


def serialize_parse_result(result: Dict) -> Dict:
    """
    Serialize parse result for JSON output.

    Removes non-serializable objects (e.g., Hypothesis, SemanticTriple) and converts to basic types.

    Args:
        result: Parse result dict

    Returns:
        Serializable dict
    """
    # Serialize triples (convert SemanticTriple objects to dicts)
    serialized_triples = []
    for triple in result.get('triples', []):
        if hasattr(triple, '__dict__'):
            # Convert object to dict (handle enums by converting to string/value)
            serialized_triples.append({
                'relation': str(triple.relation) if hasattr(triple.relation, 'name') else triple.relation,
                'subject': str(triple.subject),
                'object': str(triple.object) if hasattr(triple, 'object') else None
            })
        else:
            serialized_triples.append(triple)  # Already a dict

    # Serialize word_contexts (convert WordContext objects to dicts)
    word_contexts = result.get('word_contexts', [])

    if isinstance(word_contexts, dict):
        # It's a dict - serialize each entry
        serialized_contexts = {}
        for word, contexts in word_contexts.items():
            if isinstance(contexts, list):
                serialized_contexts[word] = [
                    {
                        'word': ctx.word if hasattr(ctx, 'word') else str(ctx),
                        'pos': ctx.pos if hasattr(ctx, 'pos') else None,
                        'context': ctx.context if hasattr(ctx, 'context') else str(ctx)
                    } if hasattr(ctx, '__dict__') else str(ctx)
                    for ctx in contexts
                ]
            else:
                serialized_contexts[word] = str(contexts)
    elif isinstance(word_contexts, list):
        # It's a list of (word, WordContext) tuples
        serialized_contexts = []
        for item in word_contexts:
            if isinstance(item, tuple) and len(item) == 2:
                word, ctx = item
                serialized_contexts.append({
                    'word': ctx.word if hasattr(ctx, 'word') else word,
                    'pos': str(ctx.pos_tag) if hasattr(ctx, 'pos_tag') else None,
                    'context': ctx.context if hasattr(ctx, 'context') else None,
                    'syntactic_role': str(ctx.syntactic_role) if hasattr(ctx, 'syntactic_role') and ctx.syntactic_role else None,
                    'neighbors': ctx.neighbors if hasattr(ctx, 'neighbors') else []
                })
            elif hasattr(item, '__dict__'):
                # It's a WordContext object directly
                serialized_contexts.append({
                    'word': item.word if hasattr(item, 'word') else str(item),
                    'pos': str(item.pos_tag) if hasattr(item, 'pos_tag') else None,
                    'context': item.context if hasattr(item, 'context') else None,
                    'syntactic_role': str(item.syntactic_role) if hasattr(item, 'syntactic_role') and item.syntactic_role else None,
                    'neighbors': item.neighbors if hasattr(item, 'neighbors') else []
                })
            else:
                # Fallback - skip malformed items
                continue
    else:
        serialized_contexts = []

    serializable = {
        'sentence_id': result['sentence_id'],
        'sentence': result['sentence'],
        'success': result['success'],
        'parse_score': float(result['parse_score']),
        'num_triples': result['num_triples'],
        'error': result['error'],
        'triples': serialized_triples,
        'word_contexts': serialized_contexts
    }

    # Skip hypothesis for now (not serializable)
    # Could implement hypothesis.to_dict() if needed

    return serializable


def write_batch_jsonl(batch_results: List[Dict], batch_id: int, output_dir: Path):
    """
    Write batch results to JSONL file (one result per line).

    This is APPEND-ONLY and never loads existing data into memory.

    Args:
        batch_results: List of parse results for this batch
        batch_id: Batch identifier (based on sentence number)
        output_dir: Output directory (must exist)
    """
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_file = batch_dir / f"batch_{batch_id:07d}.jsonl"

    with open(batch_file, 'w') as f:
        for result in batch_results:
            # Result is already serialized by worker (no double serialization)
            f.write(json.dumps(result) + '\n')


def load_progress(progress_file: Path) -> Dict:
    """
    Load lightweight progress checkpoint.

    Args:
        progress_file: Path to progress.json

    Returns:
        Progress dict with counters
    """
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress: Dict, progress_file: Path):
    """
    Save lightweight progress checkpoint.

    Only saves counters, not full results.

    Args:
        progress: Progress dict with counters
        progress_file: Path to progress.json
    """
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def update_stats(cumulative_stats: Dict, batch_stats: Dict) -> Dict:
    """
    Update cumulative statistics with batch statistics.

    Args:
        cumulative_stats: Cumulative statistics dict
        batch_stats: Batch statistics dict

    Returns:
        Updated cumulative statistics
    """
    if not cumulative_stats:
        cumulative_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_triples': 0,
            'total_score': 0.0,
            'errors': []
        }

    cumulative_stats['total'] += batch_stats.get('total', 0)
    cumulative_stats['success'] += batch_stats.get('success', 0)
    cumulative_stats['failed'] += batch_stats.get('failed', 0)
    cumulative_stats['total_triples'] += batch_stats.get('total_triples', 0)
    cumulative_stats['total_score'] += batch_stats.get('total_score', 0.0)

    # Keep sample errors (limit to 100)
    errors = cumulative_stats['errors']
    errors.extend(batch_stats.get('errors', []))
    cumulative_stats['errors'] = errors[:100]

    return cumulative_stats


def parse_corpus_batch(parser, sentences: List[str],
                       start_id: int, pool=None, chunksize: int = 100) -> Tuple[List[Dict], Dict]:
    """
    Parse a batch of sentences.

    Args:
        parser: Parser instance (used if pool is None)
        sentences: List of sentence strings
        start_id: Starting sentence ID
        pool: Optional multiprocessing pool for parallel parsing
        chunksize: Number of tasks per IPC transfer (default: 100)

    Returns:
        Tuple of (results, statistics)
    """
    if pool is not None:
        # Parallel processing
        sentence_and_ids = [(sent, start_id + i) for i, sent in enumerate(sentences)]
        # Higher chunksize reduces IPC overhead
        results = list(pool.imap(_parse_sentence_worker, sentence_and_ids, chunksize=chunksize))
    else:
        # Sequential processing
        results = []
        for i, sentence in enumerate(sentences):
            result = parse_sentence_with_context(parser, sentence, start_id + i)
            results.append(result)

    # Compute batch statistics
    stats = {
        'total': len(results),
        'success': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'total_triples': sum(r['num_triples'] for r in results),
        'total_score': sum(r['parse_score'] for r in results),
        'errors': [
            {'sentence': r['sentence'], 'error': r['error']}
            for r in results if r['error']
        ][:10]  # First 10 errors
    }

    return results, stats


def count_lines(file_path: Path) -> int:
    """
    Count lines in a file (for progress bar total).

    Args:
        file_path: Path to file

    Returns:
        Number of lines
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def consolidate_batches_to_pickle(output_dir: Path):
    """
    Consolidate all batch JSONL files into a single pickle file.

    This is OPTIONAL and only done at the very end for backward compatibility.

    Args:
        output_dir: Output directory with batches/ subdirectory
    """
    print("\nConsolidating batch files to pickle...")
    batch_dir = output_dir / "batches"

    if not batch_dir.exists():
        print("  No batch directory found!")
        return

    # Load all batch files in order
    all_results = []
    batch_files = sorted(batch_dir.glob("batch_*.jsonl"))

    print(f"  Found {len(batch_files)} batch files")

    for batch_file in tqdm(batch_files, desc="  Loading batches"):
        with open(batch_file, 'r') as f:
            for line in f:
                all_results.append(json.loads(line))

    # Save consolidated pickle
    output_file = output_dir / "parsed_corpus.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"  Saved {len(all_results)} results to {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")


def main():
    arg_parser = argparse.ArgumentParser(
        description='Stream-parse large corpus without accumulating results in memory'
    )
    arg_parser.add_argument('--corpus', type=str, default='brown',
                       help='Corpus source (brown, gutenberg, or file path)')
    arg_parser.add_argument('--parser-type', type=str, default='chart', choices=['quantum', 'chart'],
                       help='Parser type: quantum (faster) or chart (more robust)')
    arg_parser.add_argument('--max-sentences', type=int, default=10000,
                       help='Maximum sentences to parse')
    arg_parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for parsing')
    arg_parser.add_argument('--checkpoint-every', type=int, default=10000,
                       help='Save progress checkpoint every N sentences')
    arg_parser.add_argument('--output-dir', type=str, default='data/wikipedia_parsed',
                       help='Output directory')
    arg_parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if exists')
    arg_parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4, conservative for Windows)')
    arg_parser.add_argument('--maxtasksperchild', type=int, default=1000,
                       help='Restart workers after N tasks to free memory (default: 1000)')
    arg_parser.add_argument('--chunksize', type=int, default=100,
                       help='Number of tasks per IPC transfer (default: 100, higher = less overhead)')
    arg_parser.add_argument('--consolidate', action='store_true',
                       help='Consolidate batch files to pickle at end')
    arg_parser.add_argument('--consolidate-only', action='store_true',
                       help='Only consolidate existing batches (skip parsing)')

    args = arg_parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "progress.json"
    stats_file = output_dir / "parse_stats.json"

    # Handle consolidate-only mode
    if args.consolidate_only:
        consolidate_batches_to_pickle(output_dir)
        return

    print("=" * 70)
    print("STREAMING CORPUS PARSER (MEMORY-SAFE)")
    print("=" * 70)

    # [1/6] Initialize parser
    print("\n[1/6] Initializing parser...")
    print(f"  Parser type: {args.parser_type}")
    print(f"  Num workers: {args.num_workers}")

    grammar_path = Path('grammars/english.json')
    if not grammar_path.exists():
        print(f"ERROR: Grammar file not found: {grammar_path}")
        return

    # Warning if too many workers
    cpu_count = mp.cpu_count()
    if args.num_workers > cpu_count - 2:
        print(f"  WARNING: {args.num_workers} workers may be too many for {cpu_count} cores")
        print(f"  Consider reducing to {cpu_count - 2} workers (leave 2 cores free)")

    # Create parser for main process (used if num_workers=1)
    if args.parser_type == 'chart':
        parser = ChartParser(str(grammar_path))
    else:
        parser = QuantumParser(str(grammar_path))

    # Create multiprocessing pool if needed
    pool = None
    if args.num_workers > 1:
        print(f"  Creating multiprocessing pool with {args.num_workers} workers...")
        pool = mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.parser_type, str(grammar_path)),
            maxtasksperchild=args.maxtasksperchild  # Restart workers to free leaked memory
        )

    # [2/6] Load progress (if resuming)
    progress = {}
    cumulative_stats = {}
    start_sentence_id = 0

    if args.resume and progress_file.exists():
        print("\n[2/6] Resuming from checkpoint...")
        progress = load_progress(progress_file)
        start_sentence_id = progress.get('sentences_processed', 0)
        cumulative_stats = progress.get('cumulative_stats', {})
        print(f"  Resuming from sentence {start_sentence_id}")
    else:
        print("\n[2/6] Starting fresh parse...")

    # [3/6] Count total sentences (for progress bar)
    print("\n[3/6] Counting total sentences...")
    if isinstance(args.corpus, str) and Path(args.corpus).exists():
        total_sentences = min(count_lines(Path(args.corpus)), args.max_sentences)
    else:
        total_sentences = args.max_sentences
    print(f"  Total sentences to parse: {total_sentences:,}")

    # [4/6] Parse corpus
    print(f"\n[4/6] Parsing corpus: {args.corpus}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Checkpoint every: {args.checkpoint_every} sentences")
    print()

    batch_num = 0
    sentences_processed = start_sentence_id
    start_time = time.time()

    # Progress bar
    pbar = tqdm(
        total=total_sentences,
        initial=start_sentence_id,
        desc="Parsing",
        unit="sent",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    try:
        for batch in load_corpus_batch(args.corpus, batch_size=args.batch_size,
                                        max_sentences=args.max_sentences,
                                        sentence_per_line=True):  # Wikipedia has one sentence per line
            # Check if batch already processed (skip if resuming)
            batch_start = sentences_processed
            if batch_start < start_sentence_id:
                pbar.update(len(batch))
                sentences_processed += len(batch)
                batch_num += 1
                continue

            # Parse batch
            batch_results, batch_stats = parse_corpus_batch(
                parser, batch, batch_start, pool, chunksize=args.chunksize
            )

            # CRITICAL: Write to disk immediately (don't accumulate in memory)
            write_batch_jsonl(batch_results, batch_start, output_dir)

            # Update statistics
            cumulative_stats = update_stats(cumulative_stats, batch_stats)
            sentences_processed += len(batch)
            batch_num += 1

            # Update progress bar
            pbar.update(len(batch))
            success_rate = cumulative_stats.get('success', 0) / max(cumulative_stats.get('total', 1), 1) * 100
            pbar.set_postfix({
                'success': f"{success_rate:.1f}%",
                'batch': batch_num
            })

            # Checkpoint progress (lightweight - just counters)
            if sentences_processed % args.checkpoint_every == 0:
                progress_data = {
                    'sentences_processed': sentences_processed,
                    'last_batch_id': batch_start,
                    'cumulative_stats': cumulative_stats,
                    'timestamp': time.time()
                }
                save_progress(progress_data, progress_file)
                pbar.write(f"  [OK] Checkpoint saved at {sentences_processed:,} sentences")

            # Stop if we've reached max
            if sentences_processed >= args.max_sentences:
                break

    finally:
        pbar.close()

        # Clean up multiprocessing pool
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start_time
    print()
    print(f"Parsing completed in {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")
    print(f"  Sentences per second: {sentences_processed/elapsed:.1f}")
    print()

    # [5/6] Save final statistics
    print("[5/6] Saving final statistics...")

    avg_score = cumulative_stats.get('total_score', 0) / max(cumulative_stats.get('success', 1), 1)

    final_stats = {
        'total': cumulative_stats.get('total', 0),
        'success': cumulative_stats.get('success', 0),
        'failed': cumulative_stats.get('failed', 0),
        'success_rate': cumulative_stats.get('success', 0) / max(cumulative_stats.get('total', 1), 1) * 100,
        'avg_score': avg_score,
        'total_triples': cumulative_stats.get('total_triples', 0),
        'avg_triples_per_sentence': cumulative_stats.get('total_triples', 0) / max(cumulative_stats.get('success', 1), 1),
        'sentences_per_second': sentences_processed / elapsed,
        'total_time_seconds': elapsed,
        'total_time_hours': elapsed / 3600,
        'sample_errors': cumulative_stats.get('errors', [])[:100]
    }

    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"  Saved statistics: {stats_file}")
    print()

    # [6/6] Print summary
    print("[6/6] Parse Summary")
    print("=" * 70)
    print(f"Total sentences: {final_stats['total']:,}")
    print(f"Successful parses: {final_stats['success']:,}")
    print(f"Failed parses: {final_stats['failed']:,}")
    print(f"Success rate: {final_stats['success_rate']:.1f}%")
    print(f"Average parse score: {final_stats['avg_score']:.3f}")
    print(f"Total triples extracted: {final_stats['total_triples']:,}")
    print(f"Average triples per sentence: {final_stats['avg_triples_per_sentence']:.1f}")
    print(f"Parsing speed: {final_stats['sentences_per_second']:.1f} sent/sec")
    print("=" * 70)
    print()

    # Consolidate if requested
    if args.consolidate:
        consolidate_batches_to_pickle(output_dir)

    print("Corpus parsing complete!")
    print(f"Results saved to: {output_dir}/batches/")
    print(f"Progress checkpoint: {progress_file}")
    print(f"Statistics: {stats_file}")
    print()
    print("To consolidate batch files to pickle (for backward compatibility):")
    print(f"  python scripts/stream_parse_corpus.py --consolidate-only --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
