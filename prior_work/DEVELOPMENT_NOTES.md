# Prior Work: Research & Experimental Code

This directory contains the research exploration and experimental code developed as part of NAOMI-II's larger vision. While the core parser (in `current_work/`) is production-ready, this code represents the next phases of the project.

## What's Here

### Semantic Embeddings (`embeddings/`)
Training pipeline for 128-dimensional semantic word embeddings with:
- **Word Sense Disambiguation** (`sense_mapper.py`): Context-based WSD using parse trees + WordNet, solving the chicken-and-egg problem without depending on learned embeddings
- **Anchor dimensions** (`anchors.py`): 51 predefined semantic/grammatical/logical dimensions that are never modified during training
- **Polarity constraints**: Antonyms oppose on specific dimensions (e.g., good/bad on "morality")
- **Sparsity targets**: Words activate only relevant dimensions (40-70% sparsity)

### Knowledge Graphs (`graph/`)
- Triple extraction from parse trees (subject-verb-object, modifications, etc.)
- WordNet relation integration (synonyms, hypernyms, antonyms)
- Built and validated graphs with 35,000+ triples from 1,000 Brown Corpus sentences

### Training Scripts (`scripts/`)
45 scripts for the full pipeline:
- Corpus parsing (`batch_parse_corpus.py`) — 100% parse success on 1K sentences at 17.7 sent/sec
- Graph construction (`build_sense_graph.py`)
- Embedding training (`train_embeddings.py`) — 100 epochs in 1.6 minutes
- Dimension analysis and polarity discovery

### Notebooks (`notebooks/`)
Jupyter notebooks for Colab training and Wikipedia parsing experiments.

### Original Prototype (`Previous Manual Work/`)
The original NAOMI codebase including:
- Go-based parser implementations (`parser/`, `webber/`)
- Early Python quantum parser port (`quantum_parser/`)
- Language drill materials

## Key Achievement
Trained 128-dim embeddings on 5,290 sense-tagged words from Brown Corpus:
- Loss: 0.1167 -> 0.1013 over 100 epochs
- Vocabulary includes sense-tagged entries (e.g., `bank_wn.01_n` for financial bank)
- Knowledge graph: 35,554 triples

## How to Explore
```bash
# The scripts assume the old directory structure (src/ at root).
# To run them, you'll need to adjust import paths.
# See references/ARCHITECTURE.md for the full technical design.
```
