# Multilingual Training Strategy

## Overview

This document outlines the strategy for bootstrapping new languages into the Semantic Vector Space (SVS) using English as the foundation. The key insight is that we don't need to rediscover the entire semantic structure for each language - we can transfer knowledge from English and refine it based on language-specific patterns.

## Core Philosophy

**Semantic universality**: While languages differ in grammar and vocabulary, core semantic concepts are largely universal. "Big" vs "small", "good" vs "bad", "past" vs "future" exist across languages, even if encoded differently.

**English as bootstrap**: Since we've already discovered the SVS structure through English WordNet and corpus training, we can use this as a starting point for other languages rather than learning from scratch.

## Four-Phase Training Pipeline

### Phase 1: Dictionary-Based Similarity Scoring

**Goal**: Identify which English and target-language words are semantically related.

**Method**:
1. Use bilingual dictionaries (e.g., English-Spanish translation dictionaries)
2. For each translation pair, compute how well their definitions match
3. High match score → words are close in SVS
4. Low match score → words are distant (false friends, polysemy issues)

**Example**:
```
English "embarrassed" → Spanish "embarazada" (dictionary says they translate)
English definition: "feeling shame or awkwardness"
Spanish definition: "pregnant"
Match score: 0.02 (very low!)
Result: Don't treat these as semantically similar despite dictionary translation
```

**Implementation**:
- Use cross-lingual WordNet (e.g., Open Multilingual WordNet)
- Compare synset definitions across languages
- Weight by definition overlap, example sentence similarity

### Phase 2: Direct One-to-One Word Translations

**Goal**: Get rough initial SVS encodings for target language words.

**Method**:
1. For high-confidence translation pairs (from Phase 1), initialize target word with English word's SVS vector
2. This gives immediate semantic positioning without training
3. Polysemous words get multiple initializations (one per sense)

**Example**:
```
Spanish "perro" → English "dog"
Initialize: SVS("perro") = SVS("dog") = [0.23, -0.45, 0.0, 0.87, ...]
```

**Benefits**:
- Instant semantic space for new language
- Can immediately compute similarity in target language
- Provides strong prior for fine-tuning

**Limitations**:
- Assumes 1:1 semantic mapping (not always true)
- Misses language-specific nuances
- May not handle cultural concepts unique to target language

### Phase 3: Definition Similarity Refinement

**Goal**: Adjust SVS vectors based on how definitions differ between languages.

**Method**:
1. Parse definitions in both languages using respective grammars
2. Extract semantic relationships from definitions
3. Compute divergence between English and target language definitions
4. Adjust target word's SVS vector to minimize definition mismatch

**Example**:
```
English "blue" definition: "color of sky/ocean"
Spanish "azul" definition: "color of sky/ocean"
→ High overlap, minimal adjustment

English "brother" definition: "male sibling"
Spanish "hermano" definition: "male sibling or close male friend"
→ Semantic space for "hermano" should be broader, adjust accordingly
```

**Implementation**:
- Use parse tree relationships as distance constraints
- Train to minimize divergence between:
  - English definition's parse → English word's SVS
  - Target definition's parse → Target word's SVS
- Let language-specific nuances emerge

### Phase 4: Corpus-Based Usage Adjustment

**Goal**: Capture how words are actually used in the target language, beyond dictionary definitions.

**Method**:
1. Parse large corpus in target language (e.g., Wikipedia, news, literature)
2. Extract co-occurrence patterns, syntactic contexts, semantic relationships
3. Fine-tune SVS vectors to match observed usage
4. Let language-specific semantic drift happen naturally

**Example**:
```
English "friend" often co-occurs with: companion, buddy, pal
Spanish "amigo" often co-occurs with: novio (boyfriend), compadre (godfather relation)
→ "amigo" has broader semantic field, adjust SVS accordingly
```

**Implementation**:
- Use quantum parser to parse target language corpus
- Extract relationships: hypernyms, antonyms, co-occurrence
- Train embeddings with distance constraints from parse trees
- Unsupervised dimension discovery for language-specific axes

## Grammar Metadata: Language-Specific Tagging

**Critical insight from user**: Grammar features should be stored as **categorical metadata**, not learned dimensions.

**Why this matters for multilingual**:
- Spanish has gendered nouns (masculine/feminine)
- English doesn't gender most nouns
- This is metadata, not semantic content!

**Implementation**:
```python
# English "dog"
semantic_vector = [0.23, -0.45, 0.0, 0.87, ...]  # 128-512 dims
grammar_metadata = {
    'pos': 'noun',
    'number': 'singular',
    'gender': 'none',  # English doesn't gender "dog"
    'animacy': 'animate'
}

# Spanish "perro" (masculine)
semantic_vector = [0.23, -0.45, 0.0, 0.87, ...]  # Same as English "dog"
grammar_metadata = {
    'pos': 'noun',
    'number': 'singular',
    'gender': 'masculine',  # Spanish genders all nouns
    'animacy': 'animate'
}

# Spanish "perra" (feminine, female dog)
semantic_vector = [0.23, -0.42, 0.0, 0.87, ...]  # Very similar to "perro"
grammar_metadata = {
    'pos': 'noun',
    'number': 'singular',
    'gender': 'feminine',  # Grammatical gender
    'animacy': 'animate'
}
```

**Interesting Analysis Opportunity**:
Do feminine and masculine words cluster differently in semantic space?
- Compare SVS vectors for all masculine vs feminine Spanish nouns
- Check if there's systematic semantic bias correlated with grammatical gender
- Example hypothesis: Feminine nouns tend toward abstract/emotional, masculine toward concrete/physical?
- This would be discoverable through unsupervised training!

## Benefits of This Approach

### Speed
- Phase 1-2: Instant bootstrapping (hours, not weeks)
- Phase 3-4: Refinement on target corpus (days, not months)
- Compared to: Training from scratch (weeks to months)

### Accuracy
- Leverages high-quality English WordNet and training
- Avoids rediscovering universal semantic relationships
- Focuses learning on language-specific nuances

### Scalability
- Once English SVS is stable, can bootstrap dozens of languages
- Parallel training: Spanish, French, German, Chinese simultaneously
- Shared semantic space enables cross-lingual reasoning

### Multilingual Reasoning
All languages share the same 128-512 dimensional SVS:
```python
SVS("dog") ≈ SVS("perro") ≈ SVS("chien") ≈ SVS("犬")
```

This enables:
- Cross-lingual question answering
- Zero-shot translation (map parse tree from L1 to SVS to L2)
- Multilingual knowledge graph (facts in any language)

## Handling Language-Specific Concepts

**Problem**: Some concepts exist in one language but not others.

**Examples**:
- German "Schadenfreude" (pleasure at others' misfortune) - no English equivalent
- Japanese "積ん読" (tsundoku) (buying books and not reading them) - no English equivalent
- Portuguese "saudade" (melancholic longing) - no English equivalent

**Solution**: Let them discover new regions of SVS
- Initialize near semantically related English concepts
- Let corpus training push them to unique position
- This discovers new semantic dimensions English doesn't have!

**Example**:
```
German "Schadenfreude":
  Phase 1: No direct English translation → initialize near "joy" + "malice" average
  Phase 2: Definition says "joy at others' pain" → adjust toward negative emotion
  Phase 3: Corpus shows usage near "glee", "smugness", "vindication"
  Phase 4: Settles into unique SVS position that's distinct from English concepts
```

This naturally extends the semantic space to cover language-specific concepts.

## Grammar-Specific Training

Some languages require additional grammar metadata not used in English:

**Spanish**:
- Gender (masculine/feminine/neuter)
- Formal/informal address (tú vs usted)

**Japanese**:
- Honorific levels (plain, polite, respectful, humble)
- Counter words (classifiers for different object types)

**Russian**:
- Aspect (perfective/imperfective)
- 6 grammatical cases

**Arabic**:
- Dual number (singular/dual/plural)
- Verb patterns (Form I-X)

**Implementation**: Extend `GRAMMAR_DIMENSIONS` in `grammar_extraction.py` with language-specific features.

## Future: Cross-Lingual Word Sense Disambiguation

**Problem**: "bank" in English can mean:
1. Financial institution
2. River edge

Different languages split these differently:
- Spanish: "banco" (financial), "orilla" (river edge)
- French: "banque" (financial), "rive" (river edge)

**Solution**: Use cross-lingual evidence for WSD
1. Parse English sentence: "I went to the bank to deposit money"
2. Translation suggests Spanish "banco" not "orilla"
3. Semantic context (deposit, money) confirms financial sense
4. Select English sense accordingly

This uses multilingual knowledge to improve monolingual disambiguation!

## Implementation Roadmap

### Short Term (Current Phase)
1. ✅ Establish stable English SVS (128-512 dims, 150K+ words)
2. ✅ Separate grammar metadata from semantic vectors
3. 🔄 Unsupervised dimension discovery on English

### Medium Term (Next 3-6 months)
1. Implement Phase 1-2: Dictionary bootstrapping for Spanish
2. Extract Spanish WordNet definitions
3. Initialize Spanish vocabulary with English SVS vectors
4. Test semantic similarity on Spanish word pairs

### Long Term (6-12 months)
1. Implement Phase 3-4: Corpus-based refinement
2. Parse Spanish Wikipedia (1M+ sentences)
3. Fine-tune Spanish embeddings with distance constraints
4. Analyze gender/semantic correlations in Spanish
5. Extend to French, German, Chinese

### Ultimate Goal (1-2 years)
- 50+ languages sharing universal SVS
- Cross-lingual reasoning and question answering
- Zero-shot translation via semantic space
- Discovery of language-specific semantic dimensions

## Summary

**Key Innovation**: Don't rediscover semantics for each language - bootstrap from English SVS and refine based on language-specific usage.

**Four Phases**:
1. Dictionary similarity → identify translation pairs
2. Direct translation → initialize SVS vectors
3. Definition comparison → adjust for nuances
4. Corpus usage → capture language-specific patterns

**Grammar as Metadata**: Store grammatical features separately from semantic vectors, enabling analysis of grammar/semantic correlations.

**Result**: Fast, accurate multilingual semantic space that captures both universal concepts and language-specific nuances.
