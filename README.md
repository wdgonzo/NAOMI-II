# NAOMI-II: Universal Semantic Parser

**Structure IS meaning** — the same abstract tree from any language.

NAOMI-II is a deterministic semantic parser that captures *what is said*, not *how it's said*. The same sentence in English, Spanish, French, German, Portuguese, or Japanese produces the same abstract parse tree. One parser engine, different grammar files, identical meaning representation.

This isn't translation. It's a universal intermediary for language itself.

## Quick Demo

```bash
pip install -r requirements.txt
python demo.py --all
```

### What you'll see

```
  "The dog runs"
================================================================
   EN: "The dog runs"              score 0.847  |  vs EN: 100%
   ES: "El perro corre"            score 0.847  |  vs EN: 100%
  *FR: "Le chien court"            score 0.847  |  vs EN: 100%
  *DE: "Der Hund rennt"            score 0.847  |  vs EN: 100%
  *PT: "O cachorro corre"          score 0.847  |  vs EN: 100%
  *JA: "inu ga hashiru"            score 0.847  |  vs EN: 100%

  "The dog chases the cat"
================================================================
   EN: "The dog chases the cat"    score 0.894  |  vs EN: 100%
   ES: "El perro persigue el gato" score 0.894  |  vs EN: 100%
  *FR: "Le chien poursuit le chat" score 0.894  |  vs EN: 100%
  *DE: "Der Hund jagt die Katze"   score 0.894  |  vs EN: 100%
  *PT: "O cachorro persegue o gato"score 0.894  |  vs EN: 100%
  *JA: "inu ga neko wo ou"         score 0.894  |  vs EN: 100%
```

Six languages. SVO and SOV word orders. Gendered and genderless systems. Pre-nominal and post-nominal adjectives. Particles and articles. All producing the same abstract tree:

```
CLAUSE
  +--[SUBJECT]
  |   +-- NOMINAL (dog/perro/chien/Hund/cachorro/inu)
  +--[PREDICATE]
      +-- VERBAL (runs/corre/court/rennt/corre/hashiru)
```

## Why This Matters

**Semantic Translation**: Not word-for-word, but meaning-for-meaning. Parse in one language, generate in another — the abstract tree is the bridge.

**NLP Data Processing**: Every human text, in every language, reduced to one regular format. Structured, deterministic, and queryable.

**ML Training Data**: Train models on structured meaning rather than surface syntax. Every sentence ever written, in any language, normalized into the same representation.

**Language Research**: Isolate what's universal across languages vs. what's language-specific — algorithmically, not by hand.

## How It Works

### 1. Grammar DSL

Each language has a JSON grammar file defining syntactic rules. Rules specify patterns to match (anchor + before/after context), connections to create (SUBJECT, OBJECT, MODIFICATION), and consumption rules (what gets grouped into constituents).

```
grammars/english.json      — Production (85KB, 40+ rulesets)
grammars/spanish.json      — Production (82KB, gender/number agreement)
grammars/french.json       — Beta (post-nominal adj, BAGS pre-nominal)
grammars/german.json       — Beta (three-gender, pre-nominal adj)
grammars/portuguese.json   — Beta (post-nominal adj, like Spanish)
grammars/japanese.json     — Beta (SOV, particles, no articles)
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

The parser outputs abstract trees using universal node types and edge types:

| Node Types | Edge Types |
|-----------|------------|
| CLAUSE, PREDICATE, NOMINAL | SUBJECT, OBJECT, COMPLEMENT |
| VERBAL, DESCRIPTOR, SPECIFIER | MODIFICATION, DESCRIPTION, SPECIFICATION |
| PREP, COORD | COORDINATION |

These are the same regardless of input language. Different languages just have different grammar rules that map their syntax to this shared structure.

## Languages

| Language | Grammar | Status | Key Features |
|----------|---------|--------|-------------|
| English | `english.json` | Production | SVO, NLTK fallback POS tagger |
| Spanish | `spanish.json` | Production | SVO, gender/number agreement, post-nominal adj |
| French | `french.json` | Beta | SVO, gender agreement, BAGS pre-nominal adj |
| German | `german.json` | Beta | V2, three-gender (der/die/das), pre-nominal adj |
| Portuguese | `portuguese.json` | Beta | SVO, gender/number agreement, like Spanish |
| Japanese | `japanese.json` | Beta | SOV, particles (ga/wo/wa), no articles, romaji |

**Adding a new language** = one JSON grammar file + word dictionary. The parser engine never changes.

## Project Structure

```
NAOMI-II/
├── demo.py                  # Interactive 6-language demo
├── README.md                # This file
├── requirements.txt         # Python dependencies
│
├── current_work/            # Active, functional code
│   ├── src/parser/          # Parser engine (language-agnostic)
│   ├── grammars/            # 6 language grammar files
│   └── tests/               # Parser test suites (EN, ES, FR, DE, PT, JA)
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
# Run all language tests
python current_work/tests/test_english.py
python current_work/tests/test_spanish.py
python current_work/tests/test_french.py
python current_work/tests/test_german.py
python current_work/tests/test_portuguese.py
python current_work/tests/test_japanese.py

# Comprehensive test suite (40+ tests)
python current_work/tests/test_comprehensive.py
```

## Technical Details

- **Python 3.12+**, no heavy ML frameworks required for the parser
- **Dependencies**: matplotlib, networkx (visualization), nltk (POS tagging fallback)
- **Parse speed**: ~17.7 sentences/second on CPU
- **Parse success rate**: 100% on Brown Corpus test set (1,000 sentences)

For the full technical design, see [references/ARCHITECTURE.md](references/ARCHITECTURE.md).

## Background

NAOMI-II evolved from a Go-based parser prototype (2024) into a Python system with a custom grammar DSL, parallel hypothesis exploration, and 6-language support. The project demonstrates that meaning has structure independent of any particular language — and that this structure can be captured algorithmically with a single deterministic engine.

The `prior_work/` directory contains research into semantic embeddings, knowledge graphs, and training pipelines — encoding parse trees into continuous vector spaces where logical operations (NOT, AND, OR) operate directly on meaning.
