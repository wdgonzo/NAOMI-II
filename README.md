# NAOMI-II: Language-Agnostic Semantic Parser

A multilingual parsing system that demonstrates language-agnostic syntactic analysis. Given semantically equivalent sentences in different languages, NAOMI-II produces equivalent parse tree structures -- proving that deep syntactic structure can be separated from surface-level language differences.

## Quick Demo

```bash
cd current_work
python demo.py
```

**Output**: Parses English/Spanish sentence pairs and shows they produce identical abstract tree structures:

```
"the cat eats mice"    -->  CLAUSE(SUBJECT->NOMINAL, OBJECT->NOMINAL)
"el gato come ratones" -->  CLAUSE(SUBJECT->NOMINAL, OBJECT->NOMINAL)

RESULT: EQUIVALENT
```

## What This Demonstrates

1. **Language-agnostic parsing**: One parser engine handles any language via swappable grammar files
2. **Structural equivalence**: "dogs run" and "perros corren" produce the same tree shape
3. **Grammar DSL**: Declarative JSON grammar format defines each language's syntax rules, including word order, morphological agreement, and phrase structure
4. **Quantum hypothesis exploration**: The parser maintains multiple parse interpretations simultaneously, selecting the best by structural scoring

## Key Innovation

The parser uses a "quantum" approach -- instead of committing to a single parse path and backtracking on failure, it maintains all viable interpretations in parallel, prunes low-scoring hypotheses, and returns the best parse. This handles ambiguity naturally and produces robust results for both English and Spanish.

Different languages have different surface rules (English: "the big dog", Spanish: "el perro grande") but the parser maps both to the same abstract structure: a NOMINAL node with DESCRIPTION and DETERMINATION connections. The universal type system (NodeType, ConnectionType) is shared across all languages.

## Repository Structure

```
NAOMI-II/
├── current_work/          Working parser with cross-lingual demo
│   ├── demo.py            Run this to see the equivalence proof
│   ├── grammars/          English + Spanish grammar files
│   ├── src/parser/        Parser engine source code
│   └── tests/             80+ test cases (40+ per language)
│
├── prior_work/            Full project history and ongoing research
│   ├── naomi_v1/          Original Go implementation
│   ├── src/               Semantic embedding system (Python)
│   ├── scripts/           Training and analysis scripts
│   └── data/              Training artifacts
│
└── references/            Documentation and grammar reference
    ├── grammars/          Grammar file archive
    └── docs/              Architecture, design, and training docs
```

## Architecture Overview

```
Input: "the big dog runs"  (English)
       "el perro grande corre"  (Spanish)
                    |
          +-------------------+
          |   POS Tagger      |   Tags words with parts of speech
          |   (per-language    |   using lexicon + morphological rules
          |    dictionaries)   |
          +-------------------+
                    |
          +-------------------+
          |   Grammar Rules   |   Language-specific JSON files define
          |   (english.json / |   how words combine into phrases
          |    spanish.json)  |
          +-------------------+
                    |
          +-------------------+
          |  Quantum Parser   |   Explores multiple parse hypotheses
          |  (language-        |   simultaneously, scores by structure
          |   agnostic engine) |
          +-------------------+
                    |
          +-------------------+
          |  Parse Tree       |   Universal node types: CLAUSE,
          |  (universal types) |   NOMINAL, PREDICATE, VERBAL
          +-------------------+   Universal connections: SUBJECT,
                                  OBJECT, DESCRIPTION, SPECIFICATION
```

## Requirements

- Python 3.10+
- No external packages required for the parser demo

## Project History

NAOMI (New Age of Machine Intelligence) began as a Go implementation exploring structure-based parsing. NAOMI-II is a complete rewrite in Python that adds:

- Quantum hypothesis exploration (parallel parse interpretations)
- Multi-language support via grammar DSL
- Semantic embedding research (51 interpretable anchor dimensions)
- Knowledge graph construction from parse trees

See `prior_work/README.md` for the full evolution from NAOMI v1 through the current research.
