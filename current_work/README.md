# NAOMI-II Parser: Cross-Lingual Demo

A working demonstration of language-agnostic syntactic parsing. The parser produces equivalent parse tree structures from semantically identical English and Spanish sentences.

## Running the Demo

```bash
cd current_work
python demo.py
```

The demo parses three sentence pairs and shows that both languages produce the same abstract tree structure:

| English | Spanish | Structure |
|---------|---------|-----------|
| "dogs run" | "perros corren" | CLAUSE(SUBJECT->NOMINAL) |
| "the cat eats mice" | "el gato come ratones" | CLAUSE(SUBJECT->NOMINAL, OBJECT->NOMINAL) |
| "the big dog runs" | "el perro grande corre" | CLAUSE(SUBJECT->NOMINAL) |

## Running Tests

```bash
cd current_work

# Run individual test suites
python tests/test_english.py
python tests/test_spanish.py
python tests/test_comprehensive.py
python tests/test_spanish_comprehensive.py
```

80+ test cases cover basic structures, modification, coordination, prepositional phrases, and complex sentences in both languages.

## Architecture

```
current_work/
├── demo.py                # Cross-lingual equivalence demo
├── grammars/
│   ├── english.json       # English grammar rules (~2600 lines)
│   └── spanish.json       # Spanish grammar rules (~2200 lines)
├── src/parser/            # Language-agnostic parser engine
│   ├── quantum_parser.py  # Main parsing engine
│   ├── dsl.py             # Grammar DSL loader
│   ├── data_structures.py # Word, Node, Edge, Hypothesis
│   ├── enums.py           # Universal type system (NodeType, ConnectionType)
│   ├── matcher.py         # Pattern matching
│   ├── scorer.py          # Hypothesis scoring
│   ├── pos_tagger.py      # POS tagging (English + Spanish lexicons)
│   ├── visualizer.py      # Parse tree visualization
│   └── ...
└── tests/                 # Test suites (80+ cases)
```

## How It Works

1. **POS Tagging**: Words are tagged with parts of speech (NOUN, VERB, ADJ, DET, etc.)
2. **Grammar Rules**: Language-specific JSON grammars define how words combine into phrases
3. **Quantum Parsing**: The parser explores multiple parse hypotheses simultaneously, scoring each by structural coherence
4. **Universal Types**: All trees use the same node types (NOMINAL, PREDICATE, CLAUSE) and connection types (SUBJECT, OBJECT, DESCRIPTION) regardless of input language

The key insight: English "the big dog runs" and Spanish "el perro grande corre" use different word orders (adjective before/after noun), but both produce the same abstract tree because the grammatical relationships are universal.

## Dependencies

The parser uses only the Python standard library. No external packages required.
