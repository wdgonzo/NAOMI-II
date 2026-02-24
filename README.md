# NAOMI-II: Language-Agnostic Semantic Parser

**Structure IS meaning** — equivalent sentences in different languages produce identical abstract parse trees.

NAOMI-II is a universal semantic parser that demonstrates a fundamental insight: the structure of meaning is independent of language. By defining grammars in a custom DSL and applying parallel hypothesis exploration, the parser produces the same abstract semantic tree from "The dog runs" (English) and "El perro corre" (Spanish).

## Quick Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
python demo.py

# Or auto-run all built-in examples
python demo.py --all
```

### What you'll see

```
  EN: "The dog runs"
  ES: "El perro corre"

  EN score: 0.847 | ES score: 0.847
  Structural similarity: 100%
  >> MATCH: Same abstract structure across languages

  EN: "The dog chases the cat"
  ES: "El perro persigue el gato"

  EN score: 0.894 | ES score: 0.894
  Structural similarity: 100%
  >> MATCH: Same abstract structure across languages
```

The demo includes an interactive mode where you can type your own sentences, select between English and Spanish, view parse trees as text or visual matplotlib graphs, and compare equivalent sentence pairs side-by-side.

## How It Works

### 1. Grammar DSL

Each language has a grammar file (JSON) defining syntactic rules. Rules specify patterns to match (anchor + before/after context), connections to create (SUBJECT, OBJECT, MODIFICATION), and consumption rules (what gets grouped into constituents).

```
grammars/english.json   — 85KB, 40+ rulesets
grammars/spanish.json   — 82KB, handles gender/number agreement, post-nominal adjectives
```

### 2. Quantum Parser

The parser maintains multiple parse interpretations (hypotheses) simultaneously, exploring them in parallel:

1. **Initialize**: Tag words with POS tags, create initial nodes
2. **Apply rulesets** in order: for each hypothesis, find all matching rules
   - Independent matches (different words) → apply all in one hypothesis
   - Conflicting matches (same word) → branch into separate hypotheses
3. **Score & prune**: Rank by semantic coherence, keep top-K
4. **Return**: Best hypothesis as the parse tree

This "quantum" approach (parallel exploration, scoring, collapse) handles genuine ambiguity while keeping the search space manageable.

### 3. Language-Agnostic Output

The parser outputs abstract parse trees using universal node types and edge types:

| Node Types | Edge Types |
|-----------|------------|
| CLAUSE, PREDICATE, NOMINAL | SUBJECT, OBJECT, COMPLEMENT |
| VERBAL, DESCRIPTOR, SPECIFIER | MODIFICATION, DESCRIPTION, SPECIFICATION |
| PREP, COORD | COORDINATION |

These are the same regardless of input language. Different languages just have different grammar rules that map their syntax to this shared abstract structure.

## Languages Supported

| Language | Grammar | POS Tagger | Status |
|----------|---------|------------|--------|
| English | `english.json` | Dictionary + heuristics + spaCy fallback | Complete |
| Spanish | `spanish.json` | Dictionary + gender/number agreement | Complete |

Adding a new language requires only a grammar JSON file and POS tag dictionary — no changes to the parser engine.

## Project Structure

```
NAOMI-II/
├── demo.py                  # Interactive bilingual demo
├── README.md                # This file
├── requirements.txt         # Python dependencies
│
├── current_work/            # Active, functional code
│   ├── src/parser/          # Parser engine (language-agnostic)
│   ├── grammars/            # Language grammar files
│   └── tests/               # Parser test suites
│
├── prior_work/              # Research & experimental code
│   ├── embeddings/          # Semantic embedding experiments
│   ├── graph/               # Knowledge graph construction
│   ├── scripts/             # Training & analysis pipelines
│   └── Previous Manual Work/  # Original Go prototype
│
└── references/              # Design documents & specs
    ├── docs/                # Training guides, architecture docs
    ├── ARCHITECTURE.md      # Full technical design (80KB)
    └── GRAMMAR_DESIGN.md    # How to write grammars
```

## Running Tests

```bash
# Run core parser tests
pytest current_work/tests/test_english.py current_work/tests/test_spanish.py -v

# Run comprehensive test suite (40+ tests)
python current_work/tests/test_comprehensive.py
```

## Technical Details

- **Python 3.12+**, no heavy ML frameworks required for the parser
- **Dependencies**: matplotlib, networkx (for visualization), nltk (for WordNet data)
- **Parse speed**: ~17.7 sentences/second on CPU
- **Parse success rate**: 100% on Brown Corpus test set (1,000 sentences)

For the full technical design, see [references/ARCHITECTURE.md](references/ARCHITECTURE.md).

## Background

NAOMI-II evolved from a Go-based parser prototype (2024) into a Python system with a custom grammar DSL, parallel hypothesis exploration, and bilingual support. The project explores the hypothesis that meaning has structure independent of any particular language — and that this structure can be captured algorithmically.

The `prior_work/` directory contains extensive research into semantic embeddings, knowledge graphs, and training pipelines that represent the next phase of this work: encoding these parse trees into continuous vector spaces where logical operations (NOT, AND, OR) operate directly on meaning.
