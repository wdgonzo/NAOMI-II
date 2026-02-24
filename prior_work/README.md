# NAOMI-II: Prior Work & Ongoing Research

This directory contains the full development history of the NAOMI project, from the original Go implementation through the current semantic embedding research.

## Project Evolution

### Phase 1: NAOMI v1 -- Go Parser (`naomi_v1/`)

The original NAOMI project, written in Go. Key components:

- **Bucket Parser** (`parser/cores/parser.go`): Iteratively combines adjacent word "buckets" using grammar rules until a single parse tree remains
- **Masher** (`parser/cores/masher.go`): Tree combination logic that merges partial parse results
- **Grammar Rules** (`assignments.json`, `rules.json`): Relationship definitions using a custom before/after assignment table
- **WordNet Drill** (`drill/`): Go tool for exploring WordNet relationships and building vocabulary databases
- **Lemmatizer**: English lemmatization model (`parser/en.lmm`)

This phase established the core design principles: structure-based parsing, grammatical relationships as first-class entities, and the vision of "meaning-axes" for semantic vectors.

### Phase 2: NAOMI-II Parser -- Python Rewrite (see `../current_work/`)

Complete rewrite from Go to Python with major architectural improvements:

- **Quantum Hypothesis Exploration**: Maintains multiple parse interpretations simultaneously (vs. sequential backtracking in v1)
- **JSON Grammar DSL**: Declarative grammar format with explicit patterns, connections, and subcategory agreement
- **Multi-language Support**: English + Spanish grammars with gender/number agreement
- **Comprehensive Scoring**: Structural + semantic scoring (coverage, connectivity, projectivity, balance)

### Phase 3: Semantic Embeddings (`src/`)

Extension of the parser into a full semantic representation system:

- **51 Anchor Dimensions** (`src/embeddings/anchors.py`): Fixed semantic, grammatical, and logical basis vectors
- **Dual-Source Training** (`src/embeddings/training.py`): Combines WordNet expert knowledge (60%) with parser-derived relationships (40%)
- **Word Sense Disambiguation** (`src/embeddings/sense_mapper.py`): Context-based WSD using parse tree structure
- **Knowledge Graph** (`src/graph/`): Triple extraction from parse trees, WordNet integration
- **Polarity Discovery** (`src/embeddings/polarity_discovery.py`): Automatic discovery of semantic axes (big/small, good/bad)

**Results achieved:**
- 1000 Brown Corpus sentences parsed with 100% success rate
- 5,290 sense-tagged vocabulary words
- 35,554 knowledge graph triples
- 128-dim embeddings trained (51 anchor + 77 learned)
- Training loss reduced from 0.1167 to 0.0879 (24.7% reduction)

## Directory Structure

```
prior_work/
├── naomi_v1/              # Original Go implementation
│   ├── parser/            # Go parser source
│   ├── drill/             # WordNet integration tool
│   └── ref/               # Reference materials
├── src/
│   ├── embeddings/        # Semantic embedding system (24 files)
│   ├── graph/             # Knowledge graph operations (3 files)
│   ├── data_pipeline/     # Corpus processing
│   └── utils/             # Visualization, evaluation, logging
├── scripts/               # Training and analysis scripts (44 files)
├── notebooks/             # Jupyter notebooks for exploration
├── config/                # Embedding configuration files
├── data/                  # Training artifacts and results
├── tests/                 # Embedding/graph test files
└── deprecated/            # Old training checkpoints
```

## Next Steps

The next phase of development focuses on:
1. Scaling embedding training to full WordNet + Wikipedia corpora
2. Implementing the structure-based translation system (designed in `../references/docs/TRANSLATION_DESIGN.md`)
3. Building the 3-graph incremental learning architecture (designed in `../references/docs/INCREMENTAL_LEARNING_DESIGN.md`)
