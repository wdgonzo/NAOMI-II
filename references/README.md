# NAOMI-II: Reference Materials

Documentation, grammar files, and design specifications for the NAOMI-II project.

## Grammar Files (`grammars/`)

| File | Description | Size |
|------|-------------|------|
| `english.json` | Complete English grammar (40+ rulesets) | ~85 KB |
| `spanish.json` | Complete Spanish grammar with gender/number agreement | ~80 KB |
| `test_grammar.json` | Minimal grammar for parser testing | ~2 KB |

## Documentation (`docs/`)

### Architecture & Design
| Document | Description |
|----------|-------------|
| `ARCHITECTURE.md` | Complete technical design of the parser and embedding system (79 KB) |
| `TRANSLATION_DESIGN.md` | Structure-based translation system design (28 KB) |
| `DUAL_GRAPH_ARCHITECTURE.md` | Dual knowledge graph design |
| `SEMANTIC_VECTOR_SPACE_GOALS.md` | Embedding space philosophy and goals |
| `GRAMMAR_DESIGN.md` | Grammar DSL development guide |
| `GRAMMAR_UPDATES.md` | Changelog for grammar rule changes |

### Implementation
| Document | Description |
|----------|-------------|
| `IMPLEMENTATION_STATUS.md` | Detailed progress report (38 KB) |
| `IMPLEMENTATION_OUTLINE.md` | Implementation roadmap |
| `SCALING_PLAN.md` | Plan for scaling to production |
| `ANTONYM_CLUSTERING_IMPLEMENTATION.md` | Polarity dimension discovery |
| `REFACTORING_SUMMARY.md` | Code refactoring notes |

### Status & History
| Document | Description |
|----------|-------------|
| `STATUS.md` | Current development status |
| `RECENT_UPDATES.md` | November 2025 updates |
| `QUICK_START.md` | Getting started guide |
| `LEGACY_CONCEPTS.md` | Evolution from original NAOMI (Go) design |

### Vision & Future
| Document | Description |
|----------|-------------|
| `INCREMENTAL_LEARNING_DESIGN.md` | 3-graph incremental learning architecture |
| `MEMORY_GRAPH_VISION.md` | Knowledge reasoning system design |
| `SEMANTIC_AXIS_BOOTSTRAP_DESIGN.md` | Semantic axis discovery from large corpora |

### Training Guides
| Document | Description |
|----------|-------------|
| `A100_TRAINING_GUIDE.md` | GPU training setup |
| `GETTING_STARTED_COLAB.md` | Google Colab setup |
| `MULTILINGUAL_TRAINING.md` | Adding new languages |
| `TRAINING_FEATURES.md` | Training system features |
| `TRAINING_DEPLOYMENT.md` | Deployment guide |
| `COLAB_TRAINING_RESULTS.md` | Training results analysis |
| `ANTONYM_CLUSTERING_GUIDE.md` | Polarity clustering guide |
| `VSCODE_COLAB_SETUP.md` | VS Code + Colab integration |
