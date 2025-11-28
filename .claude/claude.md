# NAOMI-II - Claude Code Configuration

## Project Overview

**NAOMI-II (New Age of Machine Intelligence - Version II)** is a universal semantic representation system building AGI through language-agnostic parsing, WordNet-guided embedding learning, and graph-based reasoning. The system operates on the principle that **structure IS meaning** - using explicit parse trees and semantic relationships rather than opaque transformer models.

**Core Innovation**: Parse sentences into trees → Encode to semantic vectors → Reason via knowledge graphs

## Documentation Quick Reference

### 📚 Essential Reading (Start Here)

| Document | Purpose | Read Time | Priority |
|----------|---------|-----------|----------|
| [README.md](../README.md) | Vision, overview, quick start | 10 min | **START HERE** |
| [QUICK_START.md](../QUICK_START.md) | Get running in 5 minutes | 5 min | **ESSENTIAL** |
| [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) | Current state & achievements | 15 min | **CURRENT** |
| [STATUS.md](../STATUS.md) | Development status tracker | 5 min | Reference |

### 🏗️ Architecture & Design

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Complete technical design (79KB) | 60 min | Deep dive |
| [TRANSLATION_DESIGN.md](../TRANSLATION_DESIGN.md) | Structure-based translation system | 30 min | For translation |
| [DUAL_GRAPH_ARCHITECTURE.md](../DUAL_GRAPH_ARCHITECTURE.md) | Dual knowledge graph design | 20 min | For training |
| [SEMANTIC_VECTOR_SPACE_GOALS.md](../SEMANTIC_VECTOR_SPACE_GOALS.md) | Embedding space philosophy | 15 min | For embeddings |

### 📖 Grammar & Language

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| [GRAMMAR_DESIGN.md](../GRAMMAR_DESIGN.md) | How to write grammars | 10 min | Adding languages |
| [GRAMMAR_UPDATES.md](../GRAMMAR_UPDATES.md) | Grammar changelog | 2 min | Reference |
| [POLARITY_CONSTRAINTS.md](../POLARITY_CONSTRAINTS.md) | Polarity system design | 10 min | For constraints |
| [POLARITY_INTEGRATION_DESIGN.md](../POLARITY_INTEGRATION_DESIGN.md) | Polarity integration | 15 min | Advanced |

### 🎓 Training & Deployment

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| **[Wikipedia Training (below)](#wikipedia-corpus-training-with-transparent-dimensions)** | **Wikipedia + A100 guide** | **20 min** | **LARGE-SCALE** |
| [docs/MULTILINGUAL_TRAINING.md](../docs/MULTILINGUAL_TRAINING.md) | Bootstrap new languages | 20 min | Multilingual |
| [docs/TRAINING_DEPLOYMENT.md](../docs/TRAINING_DEPLOYMENT.md) | Cloud training guide | 15 min | Scaling up |
| [docs/TRAINING_FEATURES.md](../docs/TRAINING_FEATURES.md) | Training system features | 10 min | Training |
| [docs/COLAB_TRAINING_RESULTS.md](../docs/COLAB_TRAINING_RESULTS.md) | Training benchmarks | 5 min | Results |
| [docs/A100_TRAINING_GUIDE.md](../docs/A100_TRAINING_GUIDE.md) | A100 GPU training | 10 min | GPU training |

### 🎯 Vision & Future

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| [docs/MEMORY_GRAPH_VISION.md](../docs/MEMORY_GRAPH_VISION.md) | Reasoning & knowledge graphs | 25 min | Future vision |
| [SCALING_PLAN.md](../SCALING_PLAN.md) | Scaling to production | 15 min | Planning |
| [LEGACY_CONCEPTS.md](../LEGACY_CONCEPTS.md) | Original NAOMI design | 20 min | Historical |

### 🔄 Recent Updates

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| [RECENT_UPDATES.md](../RECENT_UPDATES.md) | Latest changes (Nov 2025) | 15 min | **RECENT** |

### 🛠️ Setup Guides

| Document | Purpose | Read Time | When to Read |
|----------|---------|-----------|--------------|
| [docs/VSCODE_COLAB_SETUP.md](../docs/VSCODE_COLAB_SETUP.md) | VSCode + Colab setup | 10 min | Development |
| [docs/GETTING_STARTED_COLAB.md](../docs/GETTING_STARTED_COLAB.md) | Colab quick start | 5 min | Colab users |

## Current Status (as of 2025-11-27)

### ✅ **COMPLETED: Full Corpus Training Pipeline**

The system has successfully completed end-to-end corpus training with Word Sense Disambiguation (WSD):

- **Corpus**: 1000 sentences from Brown Corpus
- **Parse Success**: 100% (1000/1000 sentences at 17.7 sent/sec)
- **Vocabulary**: 5,290 sense-tagged words (e.g., `bank_wn.01_n`, `run_wn.02_v`)
- **Knowledge Graph**: 35,554 triples (parse-derived + WordNet relations)
- **Model**: Trained 128-dim embeddings (51 anchor + 77 learned)
- **Training**: 100 epochs in 1.6 minutes (Loss: 0.1167 → 0.1013)
- **Artifacts**: Saved embeddings, vocabulary, checkpoints ✅

**Key Achievement**: Solved the WSD chicken-and-egg problem using context-based sense mapping with parse trees and WordNet semantics - no circular dependency on embeddings!

### 🎯 Next Steps

1. **Immediate**: Run embedding quality tests
   - Word similarity benchmarks
   - Sense separation analysis
   - t-SNE visualization

2. **Short-term**: Scale up to 10K sentences
   - Larger vocabulary (15-20K words)
   - Deeper embeddings (256 dims)
   - Better coverage of rare senses

3. **Medium-term**: Evaluation metrics
   - SimLex-999 correlation
   - SemEval WSD accuracy
   - Compositional phrase tests

## Development Guidelines

### Code Style
- Python 3.12+
- Type hints for all functions
- Docstrings (Google style)
- Max line length: 100 characters

### Architecture Principles
1. **Structure IS Meaning**: Parse trees encode semantic relationships
2. **Fixed anchors**: First 51 embedding dimensions are predefined semantic/grammatical/logical features
3. **Language-agnostic**: Grammar DSL works for any language (English ✅, Spanish planned)
4. **Sense-tagged vocabulary**: Each word sense gets separate embedding (prevents conflation)
5. **No transformers**: Graph composition, not attention mechanisms

### Testing
- Unit tests for all core functions
- Integration tests for pipelines
- Regression tests for consistency
- Comprehensive test suites (40+ tests per language)

### When Making Changes

**Grammar Files ([grammars/](../grammars/)):**
- Document all changes in [GRAMMAR_UPDATES.md](../GRAMMAR_UPDATES.md)
- Test with example sentences
- Validate against existing parse outputs
- Check both English and Spanish grammars

**Parser Core ([src/parser/](../src/parser/)):**
- Maintain hypothesis tracking integrity
- Ensure backward compatibility with grammar files
- Update tests (see [tests/test_comprehensive.py](../tests/test_comprehensive.py))
- Test normalization pipeline

**Embedding Model ([src/embeddings/](../src/embeddings/)):**
- **NEVER** modify anchor dimensions (first 51 dims)
- Test on validation set after changes
- Save checkpoints every 10 epochs
- Preserve sense-tagged vocabulary format

**Data Pipeline ([src/data_pipeline/](../src/data_pipeline/)):**
- Handle errors gracefully (skip, don't crash)
- Log all processing steps
- Use multiprocessing for batch operations
- Checkpoint for resumability

**Word Sense Disambiguation ([src/embeddings/sense_mapper.py](../src/embeddings/sense_mapper.py)):**
- Use parse context + WordNet definitions
- No dependency on learned embeddings
- High-confidence scoring for disambiguation
- Fallback for OOV words

## Common Commands

### Setup
```bash
# Create environment
python3.12 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WordNet
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Testing
```bash
# Run all tests
pytest tests/

# Run comprehensive tests
python tests/test_comprehensive.py

# Run graph tests
python tests/test_graph.py

# Run embedding integration tests
python tests/test_embeddings_integration.py
```

### Parsing
```bash
# Parse a single sentence
python -m src.parser.quantum_parser "The dog runs quickly"

# Batch parse corpus
python scripts/batch_parse_corpus.py \
  --corpus brown \
  --max-sentences 1000 \
  --batch-size 100 \
  --output-dir data/parsed_corpus_1k
```

### Training Pipeline

**Step 1: Parse Corpus**
```bash
python scripts/batch_parse_corpus.py \
  --corpus brown \
  --max-sentences 1000 \
  --output-dir data/parsed_corpus_1k
```

**Step 2: Build Knowledge Graph**
```bash
python scripts/build_sense_graph.py \
  --corpus data/parsed_corpus_1k/parsed_corpus.pkl \
  --output-dir data/sense_graph \
  --add-wordnet
```

**Step 3: Train Embeddings**
```bash
python scripts/train_embeddings.py \
  --epochs 100 \
  --lr 0.001 \
  --batch-size 32 \
  --embedding-dim 128
```

**Step 4: Test Quality**
```bash
python scripts/test_embeddings.py \
  --checkpoint-dir checkpoints
```

### Analysis
```bash
# Test sense mapping
python scripts/test_sense_mapping.py

# Analyze dimensions
python scripts/analyze_dimensions.py \
  --embeddings checkpoints/embeddings.npy \
  --vocab checkpoints/vocabulary.json

# Visualize embeddings
python scripts/visualize_embeddings.py \
  --embeddings checkpoints/embeddings.npy \
  --output results/visualizations/
```

## File Organization

```
NAOMI-II/
├── .claude/                    # Claude Code configuration
│   └── CLAUDE.md              # This file
├── src/
│   ├── parser/                # Quantum parser implementation
│   │   ├── quantum_parser.py  # Main parsing engine
│   │   ├── normalizer.py      # Parse tree normalization
│   │   ├── pos_tagger.py      # POS tagging
│   │   └── enums.py           # Type definitions
│   ├── graph/                 # Knowledge graph operations
│   │   ├── triple_extractor.py # Parse tree → triples
│   │   └── knowledge_graph.py  # Graph data structure
│   ├── embeddings/            # Embedding system
│   │   ├── sense_mapper.py    # WSD core algorithm ✨ NEW
│   │   ├── model.py           # Embedding model
│   │   ├── anchors.py         # Predefined dimensions
│   │   └── training.py        # Training loop
│   ├── data_pipeline/         # Data processing
│   │   └── corpus_loader.py   # Corpus loading utilities
│   └── utils/                 # Utilities
├── scripts/                   # Executable scripts
│   ├── batch_parse_corpus.py  # Batch corpus parser ✨ NEW
│   ├── build_sense_graph.py   # Graph builder ✨ NEW
│   ├── train_embeddings.py    # Training script ✨ NEW
│   └── test_embeddings.py     # Quality tests ✨ NEW
├── grammars/                  # Language grammar files
│   └── english.json           # English grammar (85KB)
├── data/                      # Generated data (gitignored)
│   ├── parsed_corpus_1k/      # Parsed Brown corpus
│   └── sense_graph/           # Knowledge graph
├── checkpoints/               # Model checkpoints
│   ├── embeddings.npy         # Trained embeddings ✅
│   └── vocabulary.json        # Sense-tagged vocabulary ✅
├── tests/                     # Unit tests
│   ├── test_comprehensive.py  # 40+ parsing tests
│   ├── test_graph.py          # Graph operations
│   └── test_embeddings_integration.py # WSD + training
├── docs/                      # Additional documentation
│   ├── MULTILINGUAL_TRAINING.md
│   ├── MEMORY_GRAPH_VISION.md
│   ├── TRAINING_DEPLOYMENT.md
│   └── ...
└── README.md                  # Main project README
```

## Important Notes

### Critical Constraints
- **Anchor dimensions**: Always preserve first 51 dims in embeddings (semantic/grammatical/logical)
- **Sense tagging**: Use `word_wn.XX_pos` format (e.g., `bank_wn.01_n` for financial bank)
- **Grammar changes**: Must update [GRAMMAR_UPDATES.md](../GRAMMAR_UPDATES.md)
- **Large files**: Never commit data/ contents (use .gitignore)
- **Checkpoints**: Save every 10 epochs during training
- **Parse normalization**: Always normalize before triple extraction

### Key Design Principles
1. **No Transformers**: Structure-based composition, not attention
2. **Interpretable**: Every dimension has semantic meaning
3. **Compositional**: Meaning from grammatical relationships
4. **Bootstrapped WSD**: Parse context + WordNet, no circular dependencies
5. **Incremental learning**: Store derived facts, never forget

### Performance Characteristics
- **Parsing**: 17.7 sentences/second (CPU)
- **Training**: ~240 iterations/second (CPU)
- **Model size**: 677K parameters (5,290 words × 128 dims)
- **Memory**: 4-8 GB RAM for full pipeline
- **Scalability**: Tested up to 1000 sentences, ready for 10K+

## Wikipedia Corpus Training with Transparent Dimensions

### Overview

**Goal**: Train interpretable semantic embeddings on Wikipedia corpus using chart parser, achieving transparent dimensional structure where each dimension represents one consistent semantic axis across all words.

**Key Innovation**: Unlike Word2Vec/BERT with opaque dimensions, NAOMI-II discovers 10+ interpretable semantic axes (morality, size, temperature, etc.) through:
- Chart parser (evaluates all parse options for maximum robustness)
- Word Sense Disambiguation (separate embeddings for each word sense)
- Polarity constraints (antonyms oppose on 1-5 specific dimensions)
- Sparsity targets (40-70% - words are zero on irrelevant dimensions)
- Anchor preservation (first 51 dims are predefined, never trained)

**Notebook**: [colab-results/NAOMI_A100_Training.ipynb](../colab-results/NAOMI_A100_Training.ipynb)

### Complete Workflow

#### Strategy 1: Hybrid (RECOMMENDED - ~$5 total cost)

**Phase 1: Parse Wikipedia Locally (24-48 hours, FREE)**
```bash
# Download Wikipedia dump
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Extract articles
wikiextractor --json --processes 16 enwiki-latest-pages-articles.xml.bz2 -o wikipedia/extracted

# Extract quality sentences (NEW SCRIPT - needs creation)
python scripts/extract_wikipedia_sentences.py \
    --input-dir wikipedia/extracted \
    --output-file data/wikipedia/sentences_10M.txt \
    --target-sentences 10000000 \
    --min-words 5 \
    --max-words 50 \
    --seed 42

# Parse with CHART PARSER (more robust than quantum for complex Wikipedia text)
python scripts/batch_parse_corpus.py \
    --corpus data/wikipedia/sentences_10M.txt \
    --parser-type chart \
    --max-sentences 10000000 \
    --batch-size 1000 \
    --checkpoint-every 100000 \
    --output-dir data/wikipedia_parsed \
    --resume
```

**Phase 2: Upload to Google Drive**
- Upload `parsed_corpus.pkl` (large file, ~10-20GB)
- Upload `parse_stats.json`

**Phase 3: Train on A100 (6 hours, $4.20)**

Open [colab-results/NAOMI_A100_Training.ipynb](../colab-results/NAOMI_A100_Training.ipynb) in Google Colab:

1. Mount Google Drive
2. Load pre-parsed corpus
3. Build knowledge graph with WSD (~3 hours, CPU)
4. Train embeddings with transparent dimensions (~6 hours, GPU)
5. Validate quality (dimension analysis, polarity tests)
6. Download results

**Total**: ~30-50 hours wall-clock, ~6 hours active work, **~$5 cost**

#### Strategy 2: All on A100 (~$40-50 total cost)

Run entire pipeline in Colab notebook:
- Download Wikipedia: 2-3 hours ($1.40-2.10)
- Parse corpus: 48-72 hours ($33.60-50.40)
- Train embeddings: 6 hours ($4.20)

**Total**: ~60 hours, **$40-55 cost**

#### Strategy 3: Validation First (1M sentences, ~$1)

Test the full pipeline at small scale:
```bash
# Parse 1M sentences first
python scripts/batch_parse_corpus.py \
    --corpus data/wikipedia/sentences_10M.txt \
    --parser-type chart \
    --max-sentences 1000000 \
    --batch-size 1000 \
    --output-dir data/wikipedia_1M_parsed
```

Train on A100 (1 hour, $0.70), validate quality, then scale up if good.

### Chart Parser vs Quantum Parser

**Chart Parser** (recommended for Wikipedia):
- Evaluates ALL possible parse options (exhaustive exploration)
- More robust for complex/ambiguous Wikipedia sentences
- Better for structural ambiguity (PP attachment, coordination, etc.)
- ~2-3x slower (runs multiple rule application schedules)
- Use: `--parser-type chart`

**Quantum Parser** (faster, good for clean text):
- Smart branching (only branches on true ambiguity)
- Faster parsing (~17.7 sent/sec)
- Proven 100% success on Brown Corpus
- Use: `--parser-type quantum` (default)

**For Wikipedia**: Use **chart parser** due to sentence complexity and variety.

### Transparent Dimension Training

**Key Configuration Parameters:**

```python
CONFIG = {
    # Model architecture
    'embedding_dim': 128,          # Start conservative (51 anchor + 77 learned)
    'preserve_anchors': True,      # CRITICAL: Don't train first 51 dims

    # Transparent dimension constraints
    'polarity_weight': 1.0,        # Weight for antonym opposition
    'sparsity_weight': 0.001,      # Weight for sparsity penalty
    'sparsity_target': 0.55,       # Target 55% sparsity (40-70% range)

    # Training
    'batch_size': 262144,          # 256K samples (good for 80GB VRAM)
    'epochs': 100,
    'patience': 15,

    # A100 optimizations
    'mixed_precision': True,       # FP16 for 2x speedup
    'num_workers': 16,
}
```

**What Makes Dimensions Transparent:**

1. **Anchor Preservation** (51 dims)
   - First 51 dimensions predefined (semantic/grammatical/logical features)
   - Never modified during training
   - Provides baseline interpretability

2. **Polarity Constraints**
   - Antonyms oppose on 1-5 specific dimensions (selective polarity)
   - good/bad oppose on "morality" dimension
   - hot/cold oppose on "temperature" dimension
   - Different antonym pairs use different dimensions

3. **Sparsity Targets** (40-70%)
   - Words are zero on irrelevant dimensions
   - "dog" doesn't activate "temperature" dimension
   - Encourages dimensional specialization
   - Each dimension = one consistent semantic axis

4. **Semantic Distance Constraints**
   - Synonyms: close (distance < 0.2)
   - Antonyms: far (distance > 0.7)
   - Hypernyms: moderate (distance ~0.3-0.5)

**Training Command:**

```bash
python scripts/train_embeddings.py \
    --graph-dir data/wikipedia_graph \
    --embedding-dim 128 \
    --preserve-anchors \
    --polarity-weight 1.0 \
    --sparsity-weight 0.001 \
    --sparsity-target 0.55 \
    --batch-size 262144 \
    --epochs 100 \
    --patience 15 \
    --lr 0.001 \
    --mixed-precision \
    --output-dir checkpoints
```

### Expected Outcomes

**Data Statistics:**
- Vocabulary: 300K-500K sense-tagged words (e.g., `bank_wn.01_n` for financial institution)
- Knowledge graph: 50M+ triples (Wikipedia parse-derived + WordNet semantic relations)
- Training examples: 30M+ edges

**Model Quality:**
- 10+ interpretable semantic dimensions discovered
- Sparsity 40-70% (dimensional specialization working)
- Polarity structure validated (NOT operation works)
- Anchor dimensions preserved (first 51 dims unchanged)

**Success Criteria:**
- ✓ Parse success rate >80% on Wikipedia
- ✓ WSD confidence >0.6 average
- ✓ Training converges (loss decreases steadily)
- ✓ Sparsity in 40-70% range
- ✓ Can identify 10+ semantic axes (morality, size, temperature, etc.)
- ✓ Antonyms show selective polarity (oppose on 1-5 dims, not all)
- ✓ NOT(good) ≈ bad (compositional semantics works)

### Scripts & Files

**✅ Completed:**
- [colab-results/NAOMI_A100_Training.ipynb](../colab-results/NAOMI_A100_Training.ipynb) - Full training notebook
- [scripts/batch_parse_corpus.py](../scripts/batch_parse_corpus.py) - Parser with chart option
- [scripts/build_sense_graph.py](../scripts/build_sense_graph.py) - Knowledge graph builder with WSD

**⏳ To Be Created:**
- `scripts/extract_wikipedia_sentences.py` - Quality sentence extraction
- `scripts/analyze_dimensions.py` - Post-training dimension discovery
- `scripts/test_polarity.py` - Polarity structure validation
- Updates to `scripts/train_embeddings.py` - Add transparent dimension constraints

### Quality Validation

**Post-Training Analysis:**

```bash
# Analyze discovered dimensions
python scripts/analyze_dimensions.py \
    --embeddings checkpoints/embeddings.npy \
    --vocab checkpoints/vocabulary.json \
    --output-dir results/dimension_analysis

# Test polarity structure
python scripts/test_polarity.py \
    --embeddings checkpoints/embeddings.npy \
    --vocab checkpoints/vocabulary.json
```

**Expected Output:**
- Dimension taxonomy (list of discovered semantic axes with examples)
- Polarity structure report (which antonym pairs use which dimensions)
- Sparsity statistics (per-dimension activation patterns)
- t-SNE visualization (2D projection of embedding space)

### Cost Breakdown

**Hybrid Approach (RECOMMENDED):**
| Phase | Time | Active Work | Cost | Where |
|-------|------|-------------|------|-------|
| Parse locally | 24-48 hrs | 2 hrs | $0 | Your machine |
| Upload to Drive | 30 min | 30 min | $0 | Local |
| Build graph (Colab) | 3 hrs | 15 min | $0 | Colab CPU |
| Train (A100) | 6 hrs | 30 min | $4.20 | Colab GPU |
| **TOTAL** | **30-50 hrs** | **3-4 hrs** | **$4.20** | Mixed |

**All-on-A100 Approach:**
| Phase | Time | Cost | Where |
|-------|------|------|-------|
| Download Wikipedia | 2-3 hrs | $1.40-2.10 | Colab |
| Parse corpus | 48-72 hrs | $33.60-50.40 | Colab CPU |
| Build graph | 3 hrs | $0 | Colab CPU |
| Train | 6 hrs | $4.20 | Colab GPU |
| **TOTAL** | **60-85 hrs** | **$40-55** | Colab |

**Validation Run (1M sentences):**
| Phase | Time | Cost |
|-------|------|------|
| Parse 1M | 2 hrs | $1.40 |
| Build graph | 20 min | $0 |
| Train | 1 hr | $0.70 |
| Analyze | 30 min | $0 |
| **TOTAL** | **4 hrs** | **$2.10** |

### Troubleshooting

**Parse failures (>20% fail rate):**
- Check grammar rules for missing constructions
- Review failed sentence patterns in `parse_errors.json`
- Consider fallback to simpler sentences
- Quantum parser may work better for specific constructions

**Training not converging:**
- Reduce learning rate (try 0.0001)
- Reduce sparsity weight (try 0.0001)
- Check for NaN/Inf values in loss
- Verify data quality (parse success rate, WSD confidence)

**Sparsity outside 40-70% range:**
- Too sparse (>70%): Reduce sparsity weight
- Too dense (<40%): Increase sparsity weight
- Monitor per-dimension activation patterns

**Polarity structure not emerging:**
- Increase polarity weight (try 2.0 or 5.0)
- Verify antonym pairs in knowledge graph
- Check WordNet relations were added correctly
- May need more training epochs

**A100 out of memory:**
- Reduce batch size (try 131072 or 65536)
- Reduce gradient accumulation steps
- Disable mixed precision (slower but uses less memory)

### Next Steps After Training

1. **Download Results** from Google Drive
2. **Dimension Discovery** - Run comprehensive analysis locally
3. **Visualization** - t-SNE plots of embedding space
4. **Compositional Tests** - Test NOT, AND, VERY operations
5. **Benchmarking** - SimLex-999, WordSim-353 correlations
6. **Translation** - Use discovered dimensions for English → Spanish

### Key Innovations Summary

1. **Chart Parser** - Evaluates all parse options (more robust than quantum for Wikipedia)
2. **Word Sense Disambiguation** - Each sense gets separate embedding (no conflation)
3. **Transparent Dimensions** - Interpretable semantic axes (not opaque like Word2Vec)
4. **Polarity Structure** - Selective opposition (different dims for different antonym types)
5. **Anchor Preservation** - First 51 dims provide baseline interpretability
6. **Sparsity** - Words activate only relevant dimensions (specialization)

**This is NOT Word2Vec - every dimension has semantic meaning!**

---

## Roadmap Summary

### ✅ Phase 1: Parser Foundation (COMPLETED)
- Quantum parser with hypothesis exploration
- English & Spanish grammar support
- Comprehensive test suites
- Parse tree normalization

### ✅ Phase 2: Embedding Training (COMPLETED)
- Word sense disambiguation (WSD)
- Corpus parsing pipeline
- Knowledge graph construction
- Embedding model training
- 100% parse success on 1K sentences

### 🔄 Phase 3: Quality & Scale (IN PROGRESS)
- Embedding quality evaluation
- Scale to 10K sentences
- t-SNE visualization
- Benchmark metrics (SimLex-999, WordSim-353)

### 📋 Phase 4: Translation (PLANNED)
- Structure-based English → Spanish translation
- Template retrieval system
- Surface realization
- Evaluation metrics

### 🔮 Phase 5: Memory & Reasoning (FUTURE)
- Knowledge graph reasoning
- Inference engine
- Self-teaching system
- Multilingual extension

## Getting Help

### Quick Links
- **Start**: [README.md](../README.md) → [QUICK_START.md](../QUICK_START.md)
- **Current Work**: [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md)
- **Deep Dive**: [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Training**: [docs/MULTILINGUAL_TRAINING.md](../docs/MULTILINGUAL_TRAINING.md)
- **Vision**: [docs/MEMORY_GRAPH_VISION.md](../docs/MEMORY_GRAPH_VISION.md)

### Common Issues
- **Parse failures**: Check grammar rules in [grammars/english.json](../grammars/english.json)
- **WSD errors**: Review [src/embeddings/sense_mapper.py](../src/embeddings/sense_mapper.py)
- **Training slow**: See [docs/TRAINING_DEPLOYMENT.md](../docs/TRAINING_DEPLOYMENT.md) for cloud options
- **Memory issues**: Reduce batch size or use smaller corpus subset

### Documentation Navigation
Use the **Documentation Quick Reference** table above to find what you need. Documents are organized by:
- **Priority**: Essential vs. Reference
- **Topic**: Architecture, Training, Grammar, etc.
- **Read Time**: 5-60 minutes
- **When to Read**: Context for when each doc is most useful

---

**Last Updated**: 2025-11-27
**Status**: ✅ End-to-end pipeline operational + Wikipedia training notebook ready
**Next Milestone**: Wikipedia corpus training → Transparent dimensions → Translation system

**New Additions (2025-11-27):**
- ✅ Wikipedia training workflow with chart parser
- ✅ Transparent dimension training configuration
- ✅ A100 Colab notebook updated ([NAOMI_A100_Training.ipynb](../colab-results/NAOMI_A100_Training.ipynb))
- ✅ Chart parser integration in batch_parse_corpus.py
- ⏳ Remaining scripts to create (extract_wikipedia_sentences.py, analyze_dimensions.py, test_polarity.py)
