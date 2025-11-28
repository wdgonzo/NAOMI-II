"""
Batch Corpus Parser with Context Tracking

Parses large corpora in batches with:
- Progress tracking and checkpointing
- Error handling (skip unparseable, log errors)
- Context extraction for word sense disambiguation
- Quality statistics

Saves:
- Parsed hypotheses with contexts
- Parse quality metrics
- Error logs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pickle
import json
import time
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.parser.quantum_parser import QuantumParser
from src.parser.chart_parser import ChartParser
from src.parser.pos_tagger import tag_sentence
from src.graph.triple_extractor import extract_triples
from src.embeddings.sense_mapper import extract_word_contexts_from_parse
from src.data_pipeline.corpus_loader import load_corpus_batch


def parse_sentence_with_context(parser, sentence: str, sentence_id: int) -> Dict:
    """
    Parse a sentence and extract contexts.

    Args:
        parser: QuantumParser instance
        sentence: Sentence text
        sentence_id: Unique sentence identifier

    Returns:
        Dictionary with parse results and contexts
    """
    result = {
        'sentence_id': sentence_id,
        'sentence': sentence,
        'success': False,
        'parse_score': 0.0,
        'num_triples': 0,
        'error': None,
        'hypothesis': None,
        'triples': [],
        'word_contexts': []
    }

    try:
        # Parse sentence
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        hypothesis = chart.best_hypothesis()

        if hypothesis and hypothesis.score > 0:
            result['success'] = True
            result['parse_score'] = hypothesis.score
            result['hypothesis'] = hypothesis

            # Extract triples
            triples = extract_triples(hypothesis)
            result['num_triples'] = len(triples)
            result['triples'] = triples

            # Extract word contexts for WSD
            word_contexts = extract_word_contexts_from_parse(sentence_id, hypothesis)
            result['word_contexts'] = word_contexts

    except Exception as e:
        result['error'] = str(e)

    return result


def parse_corpus_batch(parser, sentences: List[str],
                       start_id: int) -> Tuple[List[Dict], Dict]:
    """
    Parse a batch of sentences.

    Args:
        parser: QuantumParser instance
        sentences: List of sentence strings
        start_id: Starting sentence ID for this batch

    Returns:
        (parsed_results, batch_stats) tuple
    """
    results = []
    stats = {
        'total': len(sentences),
        'success': 0,
        'failed': 0,
        'total_triples': 0,
        'avg_score': 0.0,
        'errors': []
    }

    for i, sentence in enumerate(sentences):
        sentence_id = start_id + i
        result = parse_sentence_with_context(parser, sentence, sentence_id)
        results.append(result)

        if result['success']:
            stats['success'] += 1
            stats['total_triples'] += result['num_triples']
            stats['avg_score'] += result['parse_score']
        else:
            stats['failed'] += 1
            if result['error']:
                stats['errors'].append({
                    'sentence_id': sentence_id,
                    'sentence': sentence[:100],  # Truncate long sentences
                    'error': result['error']
                })

    if stats['success'] > 0:
        stats['avg_score'] /= stats['success']

    return results, stats


def save_checkpoint(results: List[Dict], stats: Dict, checkpoint_path: Path):
    """
    Save parsing checkpoint.

    Args:
        results: Parsed results
        stats: Cumulative statistics
        checkpoint_path: Where to save
    """
    checkpoint = {
        'results': results,
        'stats': stats,
        'num_sentences': len(results)
    }

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path: Path) -> Tuple[List[Dict], Dict, int]:
    """
    Load parsing checkpoint.

    Args:
        checkpoint_path: Checkpoint file

    Returns:
        (results, stats, num_sentences) tuple
    """
    if not checkpoint_path.exists():
        return [], {}, 0

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    return checkpoint['results'], checkpoint['stats'], checkpoint['num_sentences']


def merge_stats(stats1: Dict, stats2: Dict) -> Dict:
    """Merge two statistics dictionaries."""
    merged = {
        'total': stats1.get('total', 0) + stats2.get('total', 0),
        'success': stats1.get('success', 0) + stats2.get('success', 0),
        'failed': stats1.get('failed', 0) + stats2.get('failed', 0),
        'total_triples': stats1.get('total_triples', 0) + stats2.get('total_triples', 0),
        'errors': stats1.get('errors', []) + stats2.get('errors', [])
    }

    # Recalculate average score
    total_success = merged['success']
    if total_success > 0:
        # Weighted average
        avg1 = stats1.get('avg_score', 0) * stats1.get('success', 0)
        avg2 = stats2.get('avg_score', 0) * stats2.get('success', 0)
        merged['avg_score'] = (avg1 + avg2) / total_success
    else:
        merged['avg_score'] = 0.0

    return merged


def main():
    arg_parser = argparse.ArgumentParser(description='Parse corpus in batches with context tracking')
    arg_parser.add_argument('--corpus', type=str, default='brown',
                       help='Corpus source (brown, gutenberg, or file path)')
    arg_parser.add_argument('--parser-type', type=str, default='quantum', choices=['quantum', 'chart'],
                       help='Parser type: quantum (faster) or chart (more robust, evaluates all options)')
    arg_parser.add_argument('--max-sentences', type=int, default=10000,
                       help='Maximum sentences to parse')
    arg_parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for parsing')
    arg_parser.add_argument('--checkpoint-every', type=int, default=1000,
                       help='Save checkpoint every N sentences')
    arg_parser.add_argument('--output-dir', type=str, default='data/parsed_corpus',
                       help='Output directory')
    arg_parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if exists')

    args = arg_parser.parse_args()

    print("=" * 70)
    print("BATCH CORPUS PARSER WITH CONTEXT TRACKING")
    print("=" * 70)
    print()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.pkl"
    final_output = output_dir / "parsed_corpus.pkl"
    stats_output = output_dir / "parse_stats.json"
    error_log = output_dir / "parse_errors.json"

    # Initialize parser
    print("[1/5] Initializing parser...")
    print(f"  Parser type: {args.parser_type}")
    grammar_path = Path(__file__).parent.parent / "grammars" / "english.json"

    if args.parser_type == 'chart':
        parser = ChartParser(str(grammar_path))
        print("  Using ChartParser (evaluates all parse options - more robust)")
    else:
        parser = QuantumParser(str(grammar_path))
        print("  Using QuantumParser (smart branching - faster)")
    print()

    # Check for resume
    all_results = []
    cumulative_stats = {}
    start_sentence_id = 0

    if args.resume and checkpoint_path.exists():
        print("[2/5] Resuming from checkpoint...")
        all_results, cumulative_stats, start_sentence_id = load_checkpoint(checkpoint_path)
        print(f"  Loaded {len(all_results)} previously parsed sentences")
        print(f"  Resuming from sentence {start_sentence_id}")
        print()
    else:
        print("[2/5] Starting fresh parse...")
        print()

    # Parse corpus
    print(f"[3/5] Parsing corpus: {args.corpus}")
    print(f"  Max sentences: {args.max_sentences}")
    print(f"  Batch size: {args.batch_size}")
    print()

    batch_num = 0
    sentences_processed = start_sentence_id
    start_time = time.time()

    # Progress bar
    pbar = tqdm(total=args.max_sentences, initial=start_sentence_id,
                desc="Parsing", unit="sent")

    for batch in load_corpus_batch(args.corpus, batch_size=args.batch_size,
                                    max_sentences=args.max_sentences):
        # Parse batch
        batch_results, batch_stats = parse_corpus_batch(
            parser, batch, sentences_processed
        )

        # Accumulate results
        all_results.extend(batch_results)
        cumulative_stats = merge_stats(cumulative_stats, batch_stats)
        sentences_processed += len(batch)
        batch_num += 1

        # Update progress bar
        pbar.update(len(batch))
        pbar.set_postfix({
            'success_rate': f"{cumulative_stats.get('success', 0) / max(cumulative_stats.get('total', 1), 1) * 100:.1f}%",
            'avg_score': f"{cumulative_stats.get('avg_score', 0):.2f}"
        })

        # Checkpoint
        if sentences_processed % args.checkpoint_every == 0:
            save_checkpoint(all_results, cumulative_stats, checkpoint_path)
            pbar.write(f"  Checkpoint saved at {sentences_processed} sentences")

        # Stop if we've reached max
        if sentences_processed >= args.max_sentences:
            break

    pbar.close()

    elapsed = time.time() - start_time
    print()
    print(f"Parsing completed in {elapsed/60:.1f} minutes")
    print(f"  Sentences per second: {sentences_processed/elapsed:.1f}")
    print()

    # Save final results
    print("[4/5] Saving results...")

    # Save parsed corpus
    with open(final_output, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  Saved parsed corpus: {final_output}")

    # Save statistics
    with open(stats_output, 'w') as f:
        json.dump(cumulative_stats, f, indent=2)
    print(f"  Saved statistics: {stats_output}")

    # Save error log (first 100 errors)
    error_data = {
        'total_errors': cumulative_stats.get('failed', 0),
        'sample_errors': cumulative_stats.get('errors', [])[:100]
    }
    with open(error_log, 'w') as f:
        json.dump(error_data, f, indent=2)
    print(f"  Saved error log: {error_log}")
    print()

    # Print summary
    print("[5/5] Parse Summary")
    print("=" * 70)
    print(f"Total sentences: {cumulative_stats.get('total', 0)}")
    print(f"Successful parses: {cumulative_stats.get('success', 0)}")
    print(f"Failed parses: {cumulative_stats.get('failed', 0)}")
    print(f"Success rate: {cumulative_stats.get('success', 0) / max(cumulative_stats.get('total', 1), 1) * 100:.1f}%")
    print(f"Average parse score: {cumulative_stats.get('avg_score', 0):.3f}")
    print(f"Total triples extracted: {cumulative_stats.get('total_triples', 0)}")
    print(f"Average triples per sentence: {cumulative_stats.get('total_triples', 0) / max(cumulative_stats.get('success', 1), 1):.1f}")
    print("=" * 70)
    print()

    print("Corpus parsing complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
