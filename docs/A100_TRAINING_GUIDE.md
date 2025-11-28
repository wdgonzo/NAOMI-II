# NAOMI-II Large-Scale Training on A100 GPU
## Master Reference: Goals, Strategy, and Implementation

---

## EXECUTIVE SUMMARY

This document outlines the complete plan for scaling NAOMI-II embeddings from the current WordNet-only baseline (157K vocab, 4 semantic dimensions) to a large-scale Wikipedia-augmented model (500K-1M vocab, 100-200 semantic dimensions).

**Key Innovation:** Cycled training alternating between structured knowledge (WordNet) and contextual diversity (Wikipedia) to discover rich semantic structure.

**Hardware:** Google Colab Pro High-RAM A100 (80GB GPU, 170GB CPU RAM, 7 credits/hour)

**Target:** Train in 4-8 hours for $2.80-$5.60 to discover 100+ interpretable semantic dimensions.

---

## TABLE OF CONTENTS

1. [Project Goals & Motivation](#project-goals--motivation)
2. [Current Baseline](#current-baseline)
3. [Technical Strategy](#technical-strategy)
4. [Hardware Specifications](#hardware-specifications)
5. [Implementation Plan](#implementation-plan)
6. [Expected Outcomes](#expected-outcomes)
7. [Cost & Timeline](#cost--timeline)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)
10. [Execution Checklist](#execution-checklist)

---

## PROJECT GOALS & MOTIVATION

### Primary Objective

**Discover 100-200 interpretable semantic dimensions** through large-scale unsupervised training on combined WordNet + Wikipedia corpus.

### Current Problem

**WordNet-only training discovered only 4 semantic dimensions:**
- Size (big ↔ small)
- Morality (good ↔ bad)
- Temperature (hot ↔ cold)
- Speed (fast ↔ slow)

**Why so few?**
1. **Limited lexical diversity:** WordNet has only 157K vocabulary
2. **Narrow context types:** Only formal definitions and taxonomic relations
3. **Insufficient training signal:** 15.67M edges is sparse for 512 dimensions
4. **Single epoch convergence:** Model stopped learning after 1 epoch (7 minutes)

### Root Cause Analysis

WordNet provides excellent **structured semantic knowledge** (hypernyms, antonyms, meronyms) but lacks:
- **Contextual diversity:** Real-world usage patterns
- **Domain specialization:** Technical, medical, legal, colloquial vocabulary
- **Polysemy resolution:** Words used in different contexts
- **Subtle distinctions:** Near-synonyms that differ by context
- **Scale:** Enough training signal to saturate 512+ dimensions

### Solution

**Add Wikipedia corpus** to provide:
- **10M+ diverse sentences** from real-world articles
- **400K+ additional vocabulary** (domain-specific terms)
- **484M+ contextual edges** (30x more training data)
- **Balanced learning:** Cycle between WordNet structure and Wikipedia context

### Why Wikipedia?

1. **Coverage:** 6.8M articles covering all human knowledge domains
2. **Quality:** Collaboratively edited, relatively clean text
3. **Diversity:** Science, history, arts, culture, technology, etc.
4. **Scale:** Billions of words of high-quality prose
5. **Availability:** Free, downloadable dumps updated regularly

### Expected Impact

**With 500M+ combined edges and cycled training:**
- **More dimensions will activate:** Training signal strong enough to use 1000+ of 4096 dims
- **Finer semantic distinctions:** Capture subtle differences (e.g., "large" vs "big" vs "huge")
- **Domain-specific axes:** Medical terminology, legal jargon, technical concepts
- **Contextual polysemy:** Same word in different contexts (e.g., "bank" financial vs "bank" river)
- **Richer taxonomy:** 100-200 interpretable dimensions spanning all semantic categories

---

## CURRENT BASELINE

### WordNet Training Results (Completed November 2025)

**Hardware:** Google Colab T4 GPU (15GB VRAM)

**Dataset:**
- Vocabulary: 157,306 sense-tagged words
- Training edges: 15.67 million
- Relation types: hypernym, hyponym, meronym, holonym, antonym, similar

**Training Configuration:**
```bash
--embedding-dim 512
--batch-size 262144
--patience 5
--min-delta 0.0002
--lr 0.01
```

**Training Performance:**
- Total epochs: 6 (early stopping)
- Best model: Epoch 1 (val_loss: 0.0011)
- Training time: 7 minutes
- GPU utilization: 98% (14.6GB / 15GB)
- Sparsity: 70.8% (target: 40-70%)

**Discovered Dimensions:** 4 semantic axes
1. **Size** (dim 180): big/large/huge ↔ small/tiny/little
2. **Morality** (dim 229): good/virtuous ↔ bad/evil
3. **Temperature** (dim 226): hot/warm ↔ cold/cool
4. **Speed** (dim 434): fast/quick/rapid ↔ slow

**Key Observation:**
Model converged in 1 epoch, then started overfitting. This indicates:
- WordNet data is too homogeneous for deeper learning
- Need more diverse training signal for dimension discovery
- 512 dimensions vastly underutilized (only 4 interpretable axes)

---

## TECHNICAL STRATEGY

### 1. Cycled Training Approach

**Instead of merging WordNet + Wikipedia into one dataset, we alternate:**

```
Epoch 1: WordNet (15.67M edges) - Learn structured relations
Epoch 2: Wikipedia (484M edges) - Learn contextual usage
Epoch 3: WordNet - Reinforce structure
Epoch 4: Wikipedia - Reinforce context
... continues for 10-20 epochs
```

**Advantages:**
1. **Prevents overfitting to either source:** Balanced exposure
2. **Structured learning:** WordNet anchors semantic structure
3. **Contextual refinement:** Wikipedia adds nuance and diversity
4. **Better gradient dynamics:** Switching datasets acts like regularization
5. **Interpretable dimensions:** Structure from WordNet + context from Wikipedia

**Implementation:**
- Load both datasets at training start
- Alternate DataLoader each epoch
- Track validation loss separately for each source
- Early stopping when both plateau

### 2. Dynamic Dimension Expansion

**Start small, grow as needed:**
- **Initial:** 512 dimensions
- **Expand by:** 256 dimensions every 5 epochs
- **Maximum:** 4,096 dimensions
- **Trigger:** When >70% of current dimensions are active (low sparsity)

**Rationale:**
- Efficient: Don't waste compute on unused dimensions early
- Adaptive: Add capacity when model needs it
- Discovery-driven: More dimensions → more semantic axes

**Adaptive batch sizing:**
As dimensions expand, reduce batch size to maintain memory:
```
512 dims @ 4M batch = 75GB
1024 dims @ 2M batch = 75GB
2048 dims @ 1M batch = 75GB
4096 dims @ 500K batch = 75GB
```

### 3. A100-Specific Optimizations

**Hardware:** Colab Pro High-RAM A100
- 80GB GPU VRAM (5.3x more than T4)
- 170GB CPU RAM (11x more than T4)
- 5-10x faster computation

**Optimizations:**

**a) Massive Batch Sizes (4M samples)**
- Utilize full 80GB VRAM
- More stable gradients
- Faster convergence
- Better generalization

**b) Mixed Precision Training (float16)**
- 2x speed improvement
- 50% memory reduction
- Same accuracy with automatic loss scaling
- PyTorch AMP (Automatic Mixed Precision)

**c) Gradient Accumulation (8 steps)**
- Simulate 32M effective batch size
- Even smoother gradients
- Better optimization landscape

**d) Parallel Data Loading (16 workers)**
- Utilize 170GB CPU RAM
- Prefetch batches during GPU compute
- Eliminate data loading bottleneck
- Keep GPU saturated at 100%

**e) Aggressive Learning Rate Schedule**
- Start high (0.05) for fast convergence
- Cosine annealing for smooth decay
- Warmup (3 epochs) to prevent instability

### 4. Cost Optimization Strategy

**Goal:** Train in 4-8 hours (28-56 credits = $2.80-$5.60)

**Speed optimizations:**
1. Mixed precision: 2x speedup
2. Large batches: Fewer iterations per epoch
3. Aggressive LR: Faster convergence
4. Parallel loading: No data bottleneck
5. Early stopping: Avoid wasted epochs

**Cost breakdown:**
- Data preparation: Free (run on local CPU)
- Training: $2.80-$5.60 (A100 time only)
- Analysis: Free (run locally)

**Budget safety:**
- Monitor credit usage every hour
- Save checkpoints every epoch
- Can resume if interrupted
- Early stopping prevents runaway training

---

## HARDWARE SPECIFICATIONS

### Google Colab Pro High-RAM A100

**GPU:**
- Model: NVIDIA A100 (PCIe or SXM4)
- VRAM: 80GB GDDR6
- Compute: 19.5 TFLOPS (FP32), 312 TFLOPS (FP16 with Tensor Cores)
- Memory Bandwidth: 1.6-2 TB/s

**CPU:**
- RAM: 170GB
- Cores: ~16 vCPUs
- Use: Data loading, preprocessing, CPU-intensive parsing

**Storage:**
- Disk: ~100GB available
- Speed: Fast SSD for data loading

**Cost:**
- 7 compute credits per hour
- ~$0.10 per credit (varies by region/plan)
- ~$0.70/hour

**Comparison to T4:**
| Metric | T4 | A100 | Improvement |
|--------|-----|------|-------------|
| GPU RAM | 15GB | 80GB | 5.3x |
| CPU RAM | 15GB | 170GB | 11.3x |
| Speed | 1x | 5-10x | 5-10x |
| Cost | Free | $0.70/hr | - |

**Why A100 is necessary:**
- WordNet alone used 14.6GB / 15GB on T4 (98% utilization)
- Wikipedia adds 30x more data + 8x more dimensions
- Estimated requirement: 60-75GB GPU RAM
- T4 cannot handle this scale

---

## IMPLEMENTATION PLAN

### Phase 1: Wikipedia Data Pipeline

#### Script 1: `scripts/download_wikipedia.py`

**Purpose:** Download and preprocess Wikipedia dump

**Implementation:**
```python
"""Download English Wikipedia dump and extract clean articles."""

import bz2
import subprocess
from pathlib import Path

def download_wikipedia_dump():
    """Download latest English Wikipedia XML dump."""
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    output = Path("data/wikipedia/enwiki-latest.xml.bz2")

    # Download using wget (resume support)
    subprocess.run(["wget", "-c", url, "-O", str(output)])

    return output

def extract_articles(dump_path: Path, output_path: Path):
    """Extract clean text from Wikipedia XML using WikiExtractor."""
    # Install: pip install wikiextractor
    # Removes: templates, infoboxes, references, navigation
    # Keeps: Article text only

    subprocess.run([
        "wikiextractor",
        str(dump_path),
        "--output", str(output_path),
        "--bytes", "100M",  # Split into 100MB chunks
        "--filter_disambig_pages",  # Remove disambiguation
        "--no-templates",  # Remove templates
        "--processes", "8"  # Parallel extraction
    ])

def main():
    output_dir = Path("data/wikipedia")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Downloading Wikipedia dump...")
    dump_path = download_wikipedia_dump()

    print("[2/2] Extracting articles...")
    extract_articles(dump_path, output_dir / "extracted")

    print(f"Done! Articles saved to {output_dir / 'extracted'}")
```

**Output:** `data/wikipedia/extracted/` (~20GB clean text)

**Time:** 2 hours (download + extraction)

**Cost:** Free (run on local machine or Colab CPU)

---

#### Script 2: `scripts/extract_wikipedia_sentences.py`

**Purpose:** Extract high-quality sentences from articles

**Implementation:**
```python
"""Extract 10M high-quality sentences from Wikipedia articles."""

import json
import nltk
from pathlib import Path
from typing import List, Set
from collections import defaultdict
import random

nltk.download('punkt')

def load_articles(wiki_dir: Path) -> List[str]:
    """Load all extracted Wikipedia articles."""
    articles = []
    for file in wiki_dir.rglob("wiki_*"):
        with open(file, 'r', encoding='utf-8') as f:
            articles.extend(json.load(f))
    return articles

def is_high_quality(sentence: str) -> bool:
    """Check if sentence meets quality criteria."""
    words = sentence.split()

    # Length filter: 5-50 words
    if not (5 <= len(words) <= 50):
        return False

    # Must contain at least one verb (simple heuristic)
    # Check for common verb patterns
    has_verb = any(w.endswith(('ed', 'ing', 's')) for w in words)

    # No excessive special characters
    special_count = sum(1 for c in sentence if c in '[]{}|<>@#$%^&*')
    if special_count / len(sentence) > 0.05:
        return False

    # No URLs
    if 'http' in sentence.lower() or 'www.' in sentence.lower():
        return False

    return has_verb

def stratified_sample(sentences: List[str], target: int = 10_000_000) -> List[str]:
    """Sample sentences with diversity across topics."""
    # Group by first word (rough topic proxy)
    by_topic = defaultdict(list)
    for sent in sentences:
        first_word = sent.split()[0].lower()
        by_topic[first_word].append(sent)

    # Sample proportionally from each topic
    sampled = []
    topics = list(by_topic.keys())
    random.shuffle(topics)

    per_topic = target // len(topics)

    for topic in topics:
        available = by_topic[topic]
        sample_size = min(per_topic, len(available))
        sampled.extend(random.sample(available, sample_size))

        if len(sampled) >= target:
            break

    return sampled[:target]

def main():
    wiki_dir = Path("data/wikipedia/extracted")
    output_file = Path("data/wikipedia/sentences.txt")

    print("[1/4] Loading articles...")
    articles = load_articles(wiki_dir)
    print(f"  Loaded {len(articles)} articles")

    print("[2/4] Extracting sentences...")
    all_sentences = []
    for article in articles:
        sentences = nltk.sent_tokenize(article['text'])
        all_sentences.extend(sentences)
    print(f"  Extracted {len(all_sentences)} sentences")

    print("[3/4] Filtering quality...")
    quality_sentences = [s for s in all_sentences if is_high_quality(s)]
    print(f"  Retained {len(quality_sentences)} quality sentences")

    print("[4/4] Stratified sampling...")
    sampled = stratified_sample(quality_sentences, target=10_000_000)
    print(f"  Sampled {len(sampled)} diverse sentences")

    # Deduplicate
    unique = list(set(sampled))
    print(f"  After deduplication: {len(unique)} sentences")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique))

    print(f"Done! Sentences saved to {output_file}")
```

**Output:** `data/wikipedia/sentences.txt` (~5GB, 10M sentences)

**Time:** 3 hours

**Cost:** Free (CPU only)

---

#### Script 3: `scripts/parse_wikipedia_corpus.py`

**Purpose:** Parse sentences using quantum parser

**Implementation:**
```python
"""Parse Wikipedia sentences using NAOMI-II quantum parser."""

from pathlib import Path
from typing import List
import pickle
from multiprocessing import Pool
from tqdm import tqdm

# Import existing parser infrastructure
from src.parser.quantum_parser import QuantumParser
from src.data_pipeline.corpus_loader import load_sentences

def parse_batch(batch: List[str]) -> List[dict]:
    """Parse a batch of sentences (for parallel processing)."""
    parser = QuantumParser()
    results = []

    for sentence in batch:
        try:
            parse_tree = parser.parse(sentence)
            results.append({
                'sentence': sentence,
                'parse_tree': parse_tree,
                'success': True
            })
        except Exception as e:
            results.append({
                'sentence': sentence,
                'error': str(e),
                'success': False
            })

    return results

def main():
    sentences_file = Path("data/wikipedia/sentences.txt")
    output_file = Path("data/wikipedia_parsed/parsed_corpus.pkl")
    checkpoint_dir = Path("data/wikipedia_parsed/checkpoints")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading sentences...")
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(sentences)} sentences")

    print("[2/3] Parsing corpus (parallel)...")
    batch_size = 1000
    num_workers = 16  # Utilize A100's CPU RAM

    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

    all_results = []
    with Pool(num_workers) as pool:
        with tqdm(total=len(batches)) as pbar:
            for i, batch_results in enumerate(pool.imap(parse_batch, batches)):
                all_results.extend(batch_results)
                pbar.update(1)

                # Checkpoint every 100 batches (100K sentences)
                if (i + 1) % 100 == 0:
                    checkpoint_file = checkpoint_dir / f"checkpoint_{i+1}.pkl"
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(all_results, f)
                    print(f"  Checkpoint saved: {checkpoint_file}")

    print("[3/3] Saving final parsed corpus...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)

    # Statistics
    successful = sum(1 for r in all_results if r['success'])
    failed = len(all_results) - successful
    print(f"\nParsing complete:")
    print(f"  Successful: {successful} ({successful/len(all_results)*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/len(all_results)*100:.1f}%)")
    print(f"  Output: {output_file}")
```

**Output:** `data/wikipedia_parsed/parsed_corpus.pkl` (~50GB)

**Time:** 6-8 hours on A100 (or 20 hours on local CPU)

**Cost:** $4.20-$5.60 (can run on local CPU overnight for free)

---

#### Script 4: `scripts/generate_wikipedia_training_data.py`

**Purpose:** Convert parsed corpus to training edges

**Implementation:**
```python
"""Generate training edges from parsed Wikipedia corpus."""

import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm

# Import existing sense mapping infrastructure
from src.embeddings.sense_mapper import SenseMapper

def extract_edges_from_parse(parse_tree: dict, sense_mapper: SenseMapper) -> List[Tuple[str, str, str, float]]:
    """Extract semantic edges from a parse tree.

    Returns:
        List of (source_word, relation_type, target_word, confidence)
    """
    edges = []

    # Extract co-occurrence edges (words in same sentence)
    words = parse_tree.get('words', [])
    for i, word1 in enumerate(words):
        for word2 in words[i+1:i+6]:  # Within 5-word window
            # Map to senses
            sense1 = sense_mapper.get_sense(word1, context=words)
            sense2 = sense_mapper.get_sense(word2, context=words)

            edges.append((sense1, 'co-occur', sense2, 0.5))

    # Extract syntactic edges (parse tree relations)
    relations = parse_tree.get('relations', [])
    for rel in relations:
        source_sense = sense_mapper.get_sense(rel['source'], context=words)
        target_sense = sense_mapper.get_sense(rel['target'], context=words)

        edges.append((source_sense, f"syntax_{rel['type']}", target_sense, 0.8))

    return edges

def build_vocabulary(edges: List[Tuple[str, str, str, float]]) -> Dict:
    """Build vocabulary from edges."""
    words = set()
    for source, _, target, _ in edges:
        words.add(source)
        words.add(target)

    word_to_id = {word: idx for idx, word in enumerate(sorted(words))}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'vocab_size': len(words)
    }

def main():
    parsed_file = Path("data/wikipedia_parsed/parsed_corpus.pkl")
    output_dir = Path("data/wikipedia_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading parsed corpus...")
    with open(parsed_file, 'rb') as f:
        parsed_corpus = pickle.load(f)
    successful_parses = [p for p in parsed_corpus if p['success']]
    print(f"  Loaded {len(successful_parses)} successful parses")

    print("[2/4] Extracting edges...")
    sense_mapper = SenseMapper()
    all_edges = []

    for parse_result in tqdm(successful_parses):
        edges = extract_edges_from_parse(parse_result['parse_tree'], sense_mapper)
        all_edges.extend(edges)

    print(f"  Extracted {len(all_edges)} raw edges")

    print("[3/4] Filtering by confidence...")
    # Keep only edges with confidence >= 0.3
    filtered_edges = [(s, r, t, c) for s, r, t, c in all_edges if c >= 0.3]
    print(f"  Retained {len(filtered_edges)} edges after filtering")

    print("[4/4] Building vocabulary and saving...")
    vocab = build_vocabulary(filtered_edges)

    # Convert to ID-based edges
    id_edges = []
    for source, relation, target, confidence in filtered_edges:
        source_id = vocab['word_to_id'][source]
        target_id = vocab['word_to_id'][target]
        id_edges.append((source_id, relation, target_id, confidence))

    # Save
    with open(output_dir / "training_edges.pkl", 'wb') as f:
        pickle.dump(id_edges, f)

    with open(output_dir / "vocabulary.json", 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\nTraining data generated:")
    print(f"  Vocabulary size: {vocab['vocab_size']}")
    print(f"  Training edges: {len(id_edges)}")
    print(f"  Output: {output_dir}")
```

**Output:**
- `data/wikipedia_training/training_edges.pkl` (~2GB, 484M edges)
- `data/wikipedia_training/vocabulary.json` (400K words)

**Time:** 4 hours

**Cost:** Free (CPU only)

---

#### Script 5: `scripts/merge_datasets.py`

**Purpose:** Merge WordNet + Wikipedia into unified dataset

**Implementation:**
```python
"""Merge WordNet and Wikipedia training datasets."""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_dataset(data_dir: Path) -> Tuple[List, Dict]:
    """Load training edges and vocabulary."""
    with open(data_dir / "training_edges.pkl", 'rb') as f:
        edges = pickle.load(f)

    with open(data_dir / "vocabulary.json", 'r') as f:
        vocab = json.load(f)

    return edges, vocab

def merge_vocabularies(wordnet_vocab: Dict, wikipedia_vocab: Dict) -> Dict:
    """Merge two vocabularies, creating unified ID space."""
    # Union of all words
    all_words = set(wordnet_vocab['word_to_id'].keys()) | set(wikipedia_vocab['word_to_id'].keys())

    # Create new unified mapping
    unified_word_to_id = {word: idx for idx, word in enumerate(sorted(all_words))}
    unified_id_to_word = {idx: word for word, idx in unified_word_to_id.items()}

    return {
        'word_to_id': unified_word_to_id,
        'id_to_word': unified_id_to_word,
        'vocab_size': len(all_words)
    }

def remap_edges(edges: List[Tuple], old_vocab: Dict, new_vocab: Dict, source_tag: str) -> List[Tuple]:
    """Remap edge IDs from old vocabulary to new unified vocabulary."""
    remapped = []

    for source_id, relation, target_id, confidence in edges:
        # Get words from old vocab
        source_word = old_vocab['id_to_word'][str(source_id)]
        target_word = old_vocab['id_to_word'][str(target_id)]

        # Map to new IDs
        new_source_id = new_vocab['word_to_id'][source_word]
        new_target_id = new_vocab['word_to_id'][target_word]

        # Add source tag to relation type
        tagged_relation = f"{source_tag}:{relation}"

        remapped.append((new_source_id, tagged_relation, new_target_id, confidence))

    return remapped

def main():
    wordnet_dir = Path("data/wordnet_training")
    wikipedia_dir = Path("data/wikipedia_training")
    output_dir = Path("data/combined_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading WordNet dataset...")
    wordnet_edges, wordnet_vocab = load_dataset(wordnet_dir)
    print(f"  WordNet: {len(wordnet_edges)} edges, {wordnet_vocab['vocab_size']} words")

    print("[2/5] Loading Wikipedia dataset...")
    wikipedia_edges, wikipedia_vocab = load_dataset(wikipedia_dir)
    print(f"  Wikipedia: {len(wikipedia_edges)} edges, {wikipedia_vocab['vocab_size']} words")

    print("[3/5] Merging vocabularies...")
    unified_vocab = merge_vocabularies(wordnet_vocab, wikipedia_vocab)
    print(f"  Unified vocabulary: {unified_vocab['vocab_size']} words")

    print("[4/5] Remapping edges...")
    wordnet_remapped = remap_edges(wordnet_edges, wordnet_vocab, unified_vocab, 'wordnet')
    wikipedia_remapped = remap_edges(wikipedia_edges, wikipedia_vocab, unified_vocab, 'wikipedia')

    combined_edges = wordnet_remapped + wikipedia_remapped
    print(f"  Combined: {len(combined_edges)} edges")

    print("[5/5] Saving merged dataset...")
    with open(output_dir / "training_edges.pkl", 'wb') as f:
        pickle.dump(combined_edges, f)

    with open(output_dir / "vocabulary.json", 'w') as f:
        json.dump(unified_vocab, f, indent=2)

    # Save separate lists for cycled training
    with open(output_dir / "wordnet_edges.pkl", 'wb') as f:
        pickle.dump(wordnet_remapped, f)

    with open(output_dir / "wikipedia_edges.pkl", 'wb') as f:
        pickle.dump(wikipedia_remapped, f)

    print(f"\nDatasets merged successfully:")
    print(f"  Vocabulary: {unified_vocab['vocab_size']} words")
    print(f"  Total edges: {len(combined_edges)}")
    print(f"  WordNet edges: {len(wordnet_remapped)}")
    print(f"  Wikipedia edges: {len(wikipedia_remapped)}")
    print(f"  Output: {output_dir}")
```

**Output:**
- `data/combined_training/training_edges.pkl` (500M edges)
- `data/combined_training/vocabulary.json` (500K words)
- `data/combined_training/wordnet_edges.pkl` (for cycled training)
- `data/combined_training/wikipedia_edges.pkl` (for cycled training)

**Time:** 30 minutes

**Cost:** Free (CPU only)

---

### Phase 2: Training Infrastructure Modifications

**(Continued in next section due to length...)**

---

## EXPECTED OUTCOMES

### Vocabulary Growth
- **Before:** 157,306 words (WordNet only)
- **After:** 500,000-1,000,000 words (WordNet + Wikipedia)
- **Growth:** 3.2-6.4x increase

### Training Data Growth
- **Before:** 15.67M edges (WordNet only)
- **After:** 500M-1B edges (combined)
- **Growth:** 32-64x increase

### Dimensional Capacity
- **Before:** 512 dimensions max
- **After:** 4,096 dimensions max
- **Growth:** 8x increase

### Semantic Axes Discovered

**Current (WordNet only):** 4 dimensions
- Size, morality, temperature, speed

**Expected (WordNet + Wikipedia):** 100-200 dimensions

**Projected breakdown:**

**1. Spatial Dimensions (15-20 axes):**
- Size: big/large/huge ↔ small/tiny/little
- Height: tall/high ↔ short/low
- Depth: deep/profound ↔ shallow/superficial
- Width: wide/broad ↔ narrow/thin
- Distance: far/distant ↔ near/close
- Length: long ↔ short
- Volume: spacious/voluminous ↔ cramped/confined
- Position: above/over ↔ below/under
- Direction: forward/ahead ↔ backward/behind
- Proximity: adjacent/neighboring ↔ remote/isolated

**2. Temporal Dimensions (10-15 axes):**
- Speed: fast/quick/rapid ↔ slow/sluggish
- Duration: long/prolonged ↔ brief/momentary
- Frequency: frequent/common ↔ rare/infrequent
- Age: old/ancient ↔ young/new
- Recentness: recent/fresh ↔ old/dated
- Permanence: eternal/permanent ↔ temporary/transient
- Sequence: before/prior ↔ after/subsequent
- Regularity: periodic/regular ↔ sporadic/irregular

**3. Perceptual Dimensions (20-25 axes):**

Visual:
- Brightness: bright/luminous ↔ dark/dim
- Color saturation: vivid/vibrant ↔ pale/faded
- Clarity: clear/transparent ↔ opaque/murky
- Size (visual): large/big ↔ small/tiny

Auditory:
- Loudness: loud/noisy ↔ quiet/silent
- Pitch: high/shrill ↔ low/deep
- Tone: melodious/harmonious ↔ discordant/harsh

Tactile:
- Temperature: hot/warm ↔ cold/cool
- Texture: smooth/soft ↔ rough/coarse
- Hardness: hard/solid ↔ soft/malleable
- Wetness: wet/moist ↔ dry/arid

Olfactory:
- Pleasantness: fragrant/aromatic ↔ stinky/foul
- Intensity: strong/pungent ↔ mild/faint

Gustatory:
- Sweetness: sweet/sugary ↔ bitter/sour
- Saltiness: salty/briny ↔ bland/tasteless

**4. Emotional Dimensions (20-25 axes):**
- Happiness: happy/joyful/elated ↔ sad/miserable/depressed
- Anger: angry/furious/enraged ↔ calm/peaceful/serene
- Fear: afraid/terrified/scared ↔ brave/fearless/confident
- Love: loving/affectionate/adoring ↔ hateful/hostile/contemptuous
- Excitement: excited/thrilled/exhilarated ↔ bored/apathetic/indifferent
- Anxiety: anxious/worried/stressed ↔ relaxed/calm/tranquil
- Surprise: surprised/astonished/shocked ↔ unsurprised/expected
- Disgust: disgusted/repulsed/revolted ↔ attracted/drawn
- Pride: proud/confident ↔ ashamed/embarrassed
- Jealousy: jealous/envious ↔ content/satisfied

**5. Social Dimensions (15-20 axes):**
- Formality: formal/official ↔ informal/casual
- Politeness: polite/courteous ↔ rude/impolite
- Intimacy: intimate/close ↔ distant/aloof
- Authority: authoritative/commanding ↔ submissive/obedient
- Power: powerful/dominant ↔ weak/powerless
- Status: high-status/prestigious ↔ low-status/humble
- Cooperation: cooperative/collaborative ↔ competitive/antagonistic
- Friendliness: friendly/warm ↔ hostile/cold
- Trustworthiness: trustworthy/reliable ↔ untrustworthy/unreliable

**6. Cognitive Dimensions (15-20 axes):**
- Complexity: complex/complicated ↔ simple/straightforward
- Difficulty: difficult/hard/challenging ↔ easy/simple/effortless
- Clarity: clear/obvious/apparent ↔ unclear/ambiguous/vague
- Certainty: certain/sure/definite ↔ uncertain/doubtful/questionable
- Knowledge: knowledgeable/informed ↔ ignorant/uninformed
- Intelligence: intelligent/smart/clever ↔ stupid/dumb/foolish
- Understanding: understandable/comprehensible ↔ incomprehensible/baffling
- Logic: logical/rational ↔ illogical/irrational
- Creativity: creative/original ↔ uncreative/conventional

**7. Evaluative Dimensions (15-20 axes):**
- Quality: high-quality/excellent ↔ low-quality/poor
- Morality: good/virtuous/moral ↔ bad/evil/immoral
- Beauty: beautiful/attractive ↔ ugly/unattractive
- Utility: useful/practical ↔ useless/impractical
- Importance: important/significant ↔ unimportant/trivial
- Value: valuable/precious ↔ worthless/cheap
- Correctness: correct/right ↔ incorrect/wrong
- Appropriateness: appropriate/suitable ↔ inappropriate/unsuitable
- Effectiveness: effective/efficient ↔ ineffective/inefficient

**8. Abstract Dimensions (10-15 axes):**
- Causality: causal/causing ↔ consequent/resulting
- Possibility: possible/feasible ↔ impossible/infeasible
- Necessity: necessary/required ↔ unnecessary/optional
- Generality: general/universal ↔ specific/particular
- Abstractness: abstract/theoretical ↔ concrete/practical
- Reality: real/actual ↔ imaginary/fictional
- Existence: existent/present ↔ nonexistent/absent

---

## COST & TIMELINE

### Development Phase (3-5 days, $0)

**Day 1: Wikipedia Download & Extraction**
- Script: `download_wikipedia.py`
- Script: `extract_wikipedia_sentences.py`
- Time: 5 hours total
- Cost: Free (local machine or Colab CPU)

**Day 2: Corpus Parsing**
- Script: `parse_wikipedia_corpus.py`
- Time: 6-8 hours development + testing
- Cost: Free (development only)

**Day 3: Training Data Generation**
- Script: `generate_wikipedia_training_data.py`
- Script: `merge_datasets.py`
- Time: 6-8 hours development + testing
- Cost: Free (development only)

**Day 4: Training Script Modifications**
- Modify: `train_embeddings.py`
- Add: Cycled training, A100 optimizations, mixed precision
- Time: 8 hours
- Cost: Free (development only)

**Day 5: Analysis & Notebook**
- Modify: `analyze_dimensions.py`
- Create: `A100_Training_Colab.ipynb`
- Update: Documentation
- Time: 6 hours
- Cost: Free

### Data Generation Phase (15.5-17.5 hours total)

| Task | Time | Hardware | Cost |
|------|------|----------|------|
| Download Wikipedia dump | 2h | Local/CPU | Free |
| Extract 10M sentences | 3h | Local/CPU | Free |
| Parse corpus | 6-8h | A100 or local | $4-6 or free |
| Generate training data | 4h | Local/CPU | Free |
| Merge datasets | 0.5h | Local/CPU | Free |
| **Subtotal** | **15.5-17.5h** | Mixed | **$4-6 or free** |

**Cost optimization:** Run parsing on local CPU overnight (20 hours) instead of A100 to save $4-6.

### Training Phase (4-8 hours)

| Task | Time | Hardware | Credits | Cost |
|------|------|----------|---------|------|
| Upload data to Drive | 1h | - | 0 | Free |
| Train embeddings | 4-8h | A100 | 28-56 | $2.80-$5.60 |
| **Subtotal** | **5-9h** | A100 | **28-56** | **$2.80-$5.60** |

### Analysis Phase (1 hour, $0)
- Run dimension discovery locally
- Generate taxonomy
- Cross-validate results

### TOTAL PROJECT COST

**Maximum (all on A100):** $7-12
**Optimized (parsing on local CPU):** $2.80-$5.60

**Budget breakdown:**
- Development: $0
- Data preparation: $0-6 (free if run locally)
- Training: $2.80-$5.60
- Analysis: $0

---

## SUCCESS CRITERIA

### Minimum Success (30+ dimensions)
✅ Training completes in <12 hours (84 credits = $8.40)
✅ Discovers 30+ interpretable semantic dimensions
✅ Covers at least 5 major semantic categories
✅ Validation loss improves over WordNet-only baseline
✅ No OOM errors or crashes

### Target Success (100+ dimensions)
✅ Training completes in 4-8 hours (28-56 credits = $2.80-$5.60)
✅ Discovers 100+ interpretable semantic dimensions
✅ Covers 8+ major semantic categories (spatial, temporal, perceptual, emotional, social, cognitive, evaluative, abstract)
✅ Clear improvements in antonym discrimination
✅ Dimensions show cross-corpus consistency (WordNet + Wikipedia)
✅ Sparsity maintained at 60-70%
✅ Cost under $10 per training run

### Stretch Success (200+ dimensions)
✅ Training completes in 4-6 hours (28-42 credits = $2.80-$4.20)
✅ Discovers 200+ interpretable semantic dimensions
✅ Covers all major semantic categories with multiple axes per category
✅ Dimensions cluster into clear taxonomic groups
✅ Domain-specific dimensions emerge (medical, legal, technical)
✅ Near-perfect antonym separation on all tested pairs
✅ Synonym pairs show context-dependent subtle differences
✅ Cost under $7 per training run

---

## RISK MITIGATION

### Risk 1: A100 Credits Run Out During Training
**Probability:** Low
**Impact:** High (lost progress, wasted credits)

**Mitigation:**
- Monitor credit balance before starting (need 84+ credits for safety)
- Save checkpoint every epoch (resume capability)
- Early stopping prevents runaway training (patience=15)
- Budget alerts if available in Colab Pro

**Backup plan:**
- Resume from last checkpoint if disconnected
- Reduce batch size if memory issues arise
- Can complete on T4 if necessary (slower but free)

### Risk 2: Training Exceeds 12 Hours
**Probability:** Medium
**Impact:** Medium (higher cost, timeout risk)

**Mitigation:**
- Aggressive LR (0.05) for faster convergence
- Massive batches (4M) for fewer iterations
- Mixed precision for 2x speedup
- Early stopping when plateau detected

**Backup plan:**
- If approaching 8 hours with no convergence, reduce Wikipedia to 5M sentences
- Lower patience to 10 instead of 15
- Increase learning rate to 0.08

### Risk 3: Dimension Discovery Plateaus <50 Axes
**Probability:** Medium
**Impact:** Medium (didn't meet target goals)

**Mitigation:**
- Start with 4096 max dims (plenty of capacity)
- Train longer if needed (patience=15)
- Cycled training ensures diverse signal
- Can increase to 8192 dims if needed

**Backup plan:**
- Add more Wikipedia data (20M sentences instead of 10M)
- Longer training (patience=25)
- Different expansion strategy (every 3 epochs instead of 5)
- Lower sparsity threshold for expansion (60% instead of 70%)

### Risk 4: Out of Memory on A100
**Probability:** Very Low (80GB is huge)
**Impact:** High (training fails)

**Mitigation:**
- Conservative batch size (4M) leaves 5GB headroom
- Adaptive batch sizing when dims expand
- Mixed precision reduces memory 50%
- Gradient checkpointing available if needed

**Backup plan:**
- Reduce batch to 2M → 1M → 500K
- Reduce max_dims to 2048
- Enable gradient checkpointing
- Disable prefetching (num_workers=0)

### Risk 5: Low-Quality Wikipedia Edges
**Probability:** Medium
**Impact:** Medium (noisy training signal)

**Mitigation:**
- Quality filtering (confidence >= 0.3)
- Stratified sentence sampling
- Word sense disambiguation
- WordNet anchoring via cycled training

**Backup plan:**
- Increase confidence threshold to 0.5
- Weight WordNet edges 2x higher
- Monitor separate validation for each corpus
- Can retrain with adjusted weighting

### Risk 6: Parsing Fails on Many Sentences
**Probability:** Medium
**Impact:** Medium (less training data)

**Mitigation:**
- Robust error handling (skip unparseable sentences)
- Quality pre-filtering of sentences
- Expect 80-90% success rate (8-9M successful parses from 10M sentences)
- Checkpointing every 100K sentences

**Backup plan:**
- If success rate <70%, improve sentence filtering
- Add more sentences to corpus (12-15M total)
- Simplify parser for Wikipedia (less strict constraints)

---

## EXECUTION CHECKLIST

### Pre-Training Checklist

**Environment Setup:**
- [ ] Python 3.10+ installed
- [ ] PyTorch 2.0+ with CUDA support
- [ ] Required packages: nltk, tqdm, wikiextractor
- [ ] 100GB free disk space (local or Drive)
- [ ] Colab Pro subscription active
- [ ] 84+ compute credits available

**Data Pipeline:**
- [ ] `scripts/download_wikipedia.py` created
- [ ] `scripts/extract_wikipedia_sentences.py` created
- [ ] `scripts/parse_wikipedia_corpus.py` created
- [ ] `scripts/generate_wikipedia_training_data.py` created
- [ ] `scripts/merge_datasets.py` created

**Training Infrastructure:**
- [ ] `scripts/train_embeddings.py` modified (cycled training)
- [ ] Mixed precision support added
- [ ] A100 optimizations implemented
- [ ] Checkpoint/resume capability verified

**Analysis:**
- [ ] `scripts/analyze_dimensions.py` updated (4096 dims)
- [ ] Taxonomy generation implemented

**Notebooks:**
- [ ] `notebooks/A100_Training_Colab.ipynb` created
- [ ] Tested on sample data

**Documentation:**
- [ ] `docs/A100_TRAINING_GUIDE.md` (this file) complete
- [ ] `docs/WIKIPEDIA_INTEGRATION.md` created
- [ ] `docs/TRAINING_FEATURES.md` updated
- [ ] `docs/SCALING_PLAN.md` updated

### Data Generation Checklist

- [ ] Wikipedia dump downloaded (20GB)
- [ ] Articles extracted (20GB clean text)
- [ ] 10M sentences extracted (5GB)
- [ ] Sentences parsed (50GB parsed corpus)
- [ ] Training edges generated (2GB)
- [ ] Datasets merged (2.5GB combined)
- [ ] Data uploaded to Google Drive
- [ ] Data integrity verified (file sizes, sample checks)

### Training Checklist

**Pre-flight:**
- [ ] Colab Pro High-RAM A100 runtime selected
- [ ] GPU verified (80GB VRAM)
- [ ] Training data mounted from Drive
- [ ] Training command prepared
- [ ] Credit balance checked (>84 credits)

**During Training:**
- [ ] Monitor GPU utilization (should be 75-90GB)
- [ ] Monitor training loss (should decrease)
- [ ] Check dimension expansion events (every 5 epochs)
- [ ] Verify checkpoints saving (every epoch)
- [ ] Track credit usage (every hour)

**Post-Training:**
- [ ] Best model checkpoint saved
- [ ] Final embeddings exported
- [ ] Checkpoints downloaded/backed up to Drive
- [ ] Training logs saved

### Analysis Checklist

- [ ] Embeddings loaded successfully
- [ ] Dimension discovery run
- [ ] 100+ semantic axes discovered
- [ ] Taxonomy generated
- [ ] Cross-corpus validation completed
- [ ] Results documented
- [ ] Analysis report saved

---

## APPENDIX: Training Command Reference

### Full Training Command

```bash
python scripts/train_embeddings.py \
    --training-data-wordnet data/combined_training/wordnet_edges.pkl \
    --training-data-wikipedia data/combined_training/wikipedia_edges.pkl \
    --vocabulary data/combined_training/vocabulary.json \
    --cycled-training \
    --dynamic-dims \
    --embedding-dim 512 \
    --max-dims 4096 \
    --expand-interval 5 \
    --expand-by 256 \
    --batch-size 4194304 \
    --gradient-accumulation-steps 8 \
    --mixed-precision \
    --num-workers 16 \
    --prefetch-factor 4 \
    --epochs 200 \
    --patience 15 \
    --min-delta 0.0001 \
    --lr 0.05 \
    --lr-scheduler cosine \
    --warmup-epochs 3 \
    --output-dir data/checkpoints \
    --device cuda \
    --save-every 1
```

### Parameter Explanations

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--cycled-training` | flag | Alternate WordNet/Wikipedia each epoch |
| `--embedding-dim` | 512 | Starting dimension count |
| `--max-dims` | 4096 | Maximum capacity (8x baseline) |
| `--expand-by` | 256 | Add 256 dims when expanding |
| `--batch-size` | 4194304 | 4M samples (utilize 80GB VRAM) |
| `--gradient-accumulation-steps` | 8 | Simulate 32M effective batch |
| `--mixed-precision` | flag | Float16 for 2x speed |
| `--num-workers` | 16 | Parallel data loading |
| `--lr` | 0.05 | Aggressive for fast convergence |
| `--lr-scheduler` | cosine | Smooth decay |
| `--warmup-epochs` | 3 | Prevent instability |
| `--patience` | 15 | Early stopping tolerance |

---

**END OF GUIDE**

This document serves as the master reference for all large-scale Wikipedia training efforts. Update as implementation details change or new optimizations are discovered.
