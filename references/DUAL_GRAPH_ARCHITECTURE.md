# Dual-Graph Architecture Reference

## Overview

This document describes an alternative architecture where WordNet semantic relationships and corpus-based parse relationships are maintained as two separate but linked graphs. This is for future reference - **not currently implemented**.

## Current Architecture (Option C - Tagged Single Graph)

**What we have now:**
- Single unified graph with all nodes and edges
- Sense-tagged nodes: `good_wn.01_a`, `bad_wn.01_a`, `dog_wn.01_n`
- Natural language nodes: `good`, `bad`, `dog` (implicitly in parse triples)
- WordNet relations: SYNONYM, HYPERNYM, HYPONYM, ANTONYM, meronyms, etc.
- Parse relations: describes, is-agent-of, modifies-manner, etc.
- All relations trained together with different distance weights

**Benefits:**
- Simple: one embedding matrix, one training loop
- Pragmatic: builds on existing sense-tagging infrastructure
- Sufficient: achieves the goal of learning semantic space

## Alternative Architecture (Option A - True Dual Graph)

### Graph Structure

```
┌─────────────────────────────────────────────────────────┐
│                 WORDNET SEMANTIC GRAPH                  │
│              (Pure Semantic Relationships)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Nodes: sense-tagged only (good_wn.01_a, bad_wn.01_a)  │
│                                                         │
│  Edges: WordNet relations only                         │
│    • SYNONYM                                            │
│    • ANTONYM ←── Critical for polarity!                │
│    • HYPERNYM / HYPONYM                                │
│    • MERONYM / HOLONYM                                 │
│    • ENTAILMENT / CAUSE                                │
│    • SIMILAR_TO                                        │
│    • ATTRIBUTE                                         │
│    • DERIVATIONALLY_RELATED                            │
│                                                         │
│  Purpose: Define "what words MEAN" (denotation)        │
│                                                         │
└───────────────────┬─────────────────────────────────────┘
                    │
                    │ Cross-Links (IS_SENSE_OF)
                    │ Low weight, allows polysemy
                    │
┌───────────────────┴─────────────────────────────────────┐
│              CORPUS PARSE/USAGE GRAPH                   │
│             (Contextual Relationships)                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Nodes: natural language words (good, bad, dog, runs)  │
│                                                         │
│  Edges: Parse-based relations from corpus              │
│    • describes                                         │
│    • is-agent-of / is-patient-of                       │
│    • modifies-manner / modifies-degree                 │
│    • coordinates-with                                  │
│    • related-to                                        │
│                                                         │
│  Purpose: Define "how words are USED" (connotation,    │
│           distribution, context)                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Training Architecture

#### Embedding Structure
```python
# Two embedding matrices (could be same dimension or different)
semantic_embeddings = nn.Embedding(num_senses, embedding_dim)
word_embeddings = nn.Embedding(num_words, embedding_dim)

# Cross-link matrix: which senses belong to which words
sense_to_word_links = {
    'good_wn.01_a': 'good',
    'good_wn.02_a': 'good',  # Multiple senses → same word
    'bank_wn.01_n': 'bank',  # Financial institution
    'bank_wn.09_n': 'bank',  # River bank
}
```

#### Loss Function
```python
def compute_total_loss(semantic_emb, word_emb, batch):
    # 1. WordNet semantic loss (on sense embeddings)
    semantic_loss = 0
    for (sense1, relation, sense2) in wordnet_edges:
        emb1 = semantic_emb[sense1]
        emb2 = semantic_emb[sense2]

        # Distance-based constraints
        target_dist = relation_distances[relation]
        actual_dist = distance(emb1, emb2)
        semantic_loss += (actual_dist - target_dist) ** 2

        # Polarity constraints (antonyms only)
        if relation == 'ANTONYM':
            for dim in polarity_dims:
                semantic_loss += polarity_penalty(emb1[dim], emb2[dim])

    # 2. Parse contextual loss (on word embeddings)
    contextual_loss = 0
    for (word1, relation, word2) in parse_edges:
        emb1 = word_emb[word1]
        emb2 = word_emb[word2]

        # Distance-based constraints
        target_dist = relation_distances[relation]
        actual_dist = distance(emb1, emb2)
        contextual_loss += (actual_dist - target_dist) ** 2

    # 3. Cross-link grounding loss (weak constraints)
    grounding_loss = 0
    for (sense, word) in sense_to_word_links:
        sense_emb = semantic_emb[sense]
        word_emb_vec = word_emb[word]

        # Encourage similarity but don't force exact match
        grounding_loss += weak_similarity_penalty(sense_emb, word_emb_vec)

    # Combined loss with tunable weights
    total_loss = (
        α * semantic_loss +      # Strong (e.g., α=2.0)
        β * contextual_loss +    # Moderate (e.g., β=1.0)
        γ * grounding_loss       # Weak (e.g., γ=0.1)
    )

    return total_loss
```

### Key Benefits

#### 1. **Clean Polarity Training**
Polarity constraints only apply to semantic graph (antonym pairs from WordNet). No noise from contextual co-occurrence.

```python
# Discovery is cleaner
for (sense1, sense2) in antonym_pairs:
    # These embeddings are trained ONLY on semantic relations
    # No interference from distributional similarity
    diff = semantic_emb[sense1] - semantic_emb[sense2]
    # Clearer signal for finding polarity dimensions
```

#### 2. **Handles Polysemy Naturally**
```python
# "bank" has two senses in WordNet
bank_financial = semantic_emb['bank_wn.01_n']  # Near 'institution'
bank_river = semantic_emb['bank_wn.09_n']      # Near 'slope'

# Surface form learns weighted combination
bank_word = word_emb['bank']
# Influenced by both senses via grounding loss
# But also learns from actual corpus usage
```

#### 3. **Separate Analysis**
Can analyze contributions independently:
```python
# Which dimensions are semantic vs distributional?
semantic_variance = variance(semantic_embeddings, axis=0)
word_variance = variance(word_embeddings, axis=0)

# Test polarity on clean semantic space
NOT(good_wn.01_a) ≈ bad_wn.01_a  # Clean test

# Test distributional similarity on word space
similar_to('bank', top_k=10)  # Based on usage patterns
```

#### 4. **Flexible Weight Tuning**
Can experiment with different balances:
```python
# Scenario 1: Emphasize semantics (for clean logical structure)
α=3.0, β=1.0, γ=0.1

# Scenario 2: Emphasize distribution (for better word similarity)
α=1.0, β=3.0, γ=0.1

# Scenario 3: Tight grounding (force word ≈ sense)
α=2.0, β=2.0, γ=1.0
```

### Implementation Sketch

#### Data Structure
```python
class DualGraph:
    def __init__(self):
        # WordNet semantic graph
        self.semantic_nodes = {}  # sense_tag → node_id
        self.semantic_edges = []  # (sense1, relation, sense2)

        # Corpus parse graph
        self.word_nodes = {}      # word → node_id
        self.parse_edges = []     # (word1, relation, word2)

        # Cross-links
        self.sense_to_word = {}   # sense_tag → word
        self.word_to_senses = {}  # word → [sense_tags]
```

#### Training Loop
```python
def train_dual_graph(semantic_emb, word_emb, epochs):
    for epoch in range(epochs):
        # Sample from both graphs
        semantic_batch = sample_edges(semantic_edges)
        parse_batch = sample_edges(parse_edges)
        grounding_batch = sample_cross_links(sense_to_word)

        # Compute separate losses
        loss_sem = compute_semantic_loss(semantic_emb, semantic_batch)
        loss_ctx = compute_contextual_loss(word_emb, parse_batch)
        loss_gnd = compute_grounding_loss(
            semantic_emb, word_emb, grounding_batch
        )

        # Weighted combination
        total_loss = α * loss_sem + β * loss_ctx + γ * loss_gnd

        # Update both embedding matrices
        total_loss.backward()
        optimizer.step()
```

### When to Consider This Architecture

Consider refactoring to dual-graph if:

1. **Polarity results are noisy**: Current single-graph approach produces poor NOT operation results
2. **Need interpretability**: Want to separate semantic vs distributional contributions
3. **Experimentation needed**: Want to tune relative importance of WordNet vs corpus
4. **Scaling up**: Moving to larger corpus where distributional signal might overwhelm semantic signal

### Migration Path

If we decide to implement this later:

1. **Preserve current vocabulary**: Keep sense tags as-is
2. **Split edges by type**: Separate WordNet edges from parse edges
3. **Create two embedding matrices**: Initialize from current single matrix
4. **Add cross-link edges**: Extract sense-to-word mappings
5. **Update training loop**: Implement dual-loss computation
6. **Tune weights**: Experiment with α, β, γ

Estimated effort: 2-3 days of implementation + testing

## Why We're Not Doing This Now

1. **Current approach works**: Tagged single-graph achieves the same end goal
2. **Less complexity**: Simpler to implement and debug
3. **Incremental improvement**: Can add later if needed
4. **Sufficient for hypothesis testing**: Polarity constraints will work with current approach

## References

- This document: Future reference for architectural improvements
- Current implementation: [scripts/build_sense_graph.py](scripts/build_sense_graph.py)
- Polarity training: [POLARITY_CONSTRAINTS.md](POLARITY_CONSTRAINTS.md)
- Training script: [scripts/train_pytorch_polarity.py](scripts/train_pytorch_polarity.py)

---

**Status**: REFERENCE ONLY - Not currently implemented
**Date**: 2025-11-26
**Author**: Claude Code
**Purpose**: Document alternative architecture for future consideration
