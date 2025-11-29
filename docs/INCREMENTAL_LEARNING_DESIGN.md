# Incremental Learning Design: 3-Graph Architecture

**Status**: Design Document
**Last Updated**: 2025-11-29
**Author**: NAOMI-II Team

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [The 3-Graph System](#the-3-graph-system)
3. [Bootstrap Strategy](#bootstrap-strategy)
4. [Incremental Learning Workflow](#incremental-learning-workflow)
5. [Compositional Embeddings](#compositional-embeddings)
6. [Edge Decision Model](#edge-decision-model)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Use Cases](#use-cases)

---

## Executive Summary

NAOMI-II implements **incremental, never-forgetting AGI** through a 3-graph architecture that separates innate knowledge (WordNet), learned facts (long-term memory), and working context (short-term memory). The system bootstraps semantic dimensions from a 300K sentence corpus, then learns continuously on new data without retraining from scratch.

**Key Innovation**: After bootstrap, the neural network is **no longer needed** - embeddings are composed from graph structure and semantic relationships, enabling true lifelong learning.

---

## The 3-Graph System

### Graph 1: WordNet Reference (Immutable, Lazy-Loaded)

**Purpose**: Provides innate semantic knowledge for word sense disambiguation.

**Characteristics**:
- **Immutable**: Never changes, acts as ground truth
- **Lazy-loaded**: Only load senses that appear in corpus (not all 117K WordNet synsets)
- **Scope**: Semantic relationships (synonyms, antonyms, hypernyms, meronyms, etc.)

**Example**:
```python
wordnet.get_relations("bank_wn.01_n")  # Financial institution
# Returns:
# - Hypernym: financial_institution
# - Antonym: None
# - Meronym: bank_account, teller, vault
```

**Implementation**: `src/graph/wordnet_reference.py`

**Design Principle**: WordNet is like human "innate knowledge" - a semantic scaffold we're born with, refined through language exposure.

---

### Graph 2: Long-Term Memory (Mutable, Persistent)

**Purpose**: Stores all learned facts from parsed sentences. This is the AGI's "knowledge base".

**Characteristics**:
- **Mutable**: Facts can be updated, refined, or contradicted
- **Persistent**: Saved to disk, never forgotten
- **Incremental**: Grows continuously as new sentences are processed
- **Entity resolution**: Same entity referenced across multiple sentences gets merged

**Example**:
```python
# From sentence 1: "Banks are financial institutions."
ltm.add_triple(("bank_wn.01_n", "IS_A", "financial_institution"))

# From sentence 2: "Communists hate banks."
ltm.add_triple(("communists", "HATE", "bank_wn.01_n"))

# From sentence 3: "The bank collapsed in 2008."
ltm.add_triple(("bank_wn.01_n", "EVENT", "collapse_2008"))

# Result: Single "bank_wn.01_n" node with 3 edges
```

**Triple Structure**:
```python
{
    "subject": "bank_wn.01_n",  # Sense-tagged word
    "relation": ConnectionType.IS_A,  # Enum from enums.py
    "object": "financial_institution",
    "confidence": 0.95,  # WSD confidence
    "source_sentence_id": 1042,
    "mutable": True  # Can be updated
}
```

**Implementation**: `src/graph/long_term_memory.py`

**Design Principle**: Long-term memory is like human declarative memory - accumulated facts that can be recalled, updated, or recontextualized.

---

### Graph 3: Short-Term Memory (Mutable, Ephemeral)

**Purpose**: Working context for reasoning, hypotheticals, and conversational state. Facts are tentative until promoted to long-term.

**Characteristics**:
- **Ephemeral**: Cleared after conversation/task completion
- **Tentative**: Facts are unvalidated, may be hypothetical
- **Promotable**: Validated facts move to long-term memory
- **Small**: Size-limited (e.g., last 100 triples)

**Example Use Cases**:

**1. Conversational Context**:
```python
# User: "I have a dog named Max."
stm.add_triple(("user", "HAS", "dog_named_Max"))

# User: "He likes to swim."
stm.resolve_pronoun("he", context=stm.get_recent(5))
# Resolves "he" → "dog_named_Max" using short-term context

# After conversation ends: Promote to long-term
ltm.merge_from_short_term(stm, validated=True)
```

**2. Hypothetical Reasoning**:
```python
# "What if banks didn't exist?"
stm.add_triple(("bank_wn.01_n", "EXISTS", False), hypothetical=True)
stm.infer_consequences()  # Chain reasoning in working memory
# Result: Hypothetical triples, NOT promoted to long-term
```

**3. Web-Scraped Learning** (Teaching Mode):
```python
# User scrapes "How to make sourdough bread" tutorial
for sentence in tutorial:
    parse_and_add_to_stm(sentence)

# User validates: "This tutorial is accurate"
ltm.merge_from_short_term(stm, source="sourdough_tutorial")
# Now NAOMI-II "knows" how to make sourdough
```

**Implementation**: `src/graph/short_term_memory.py`

**Design Principle**: Short-term memory is like human working memory - holds active context, hypotheticals, and tentative facts that may or may not be committed to long-term storage.

---

## Bootstrap Strategy

### Phase 1: Initial Dimension Discovery (300K Sentences)

**Goal**: Discover the primary semantic dimensions that represent most of language.

**Process**:
1. **Random sampling**: Uniformly sample 300K sentences from Wikipedia
2. **Parse**: Chart parser → semantic triples
3. **WSD**: Map words to WordNet senses
4. **Graph construction**: Build long-term memory graph
5. **Neural net training**: Discover dimensions via matrix factorization
6. **Convergence**: Train until sparsity stabilizes (< 0.01 change for 50 epochs)

**Expected Output**:
- **Dimensions discovered**: 50-150 learned dimensions (+ 51 predefined anchors = 101-201 total)
- **Vocabulary**: 50K-100K sense-tagged words
- **Sparsity**: 40-70% (words activate only relevant dimensions)
- **Triples**: 5M-10M semantic relationships

**Why 300K?**
- Zipf's law: Top 10K words cover 90% of usage
- 300K sentences ensure:
  - Common words: 100+ occurrences each (good statistics)
  - Rare words: At least 1-2 occurrences (bootstrap embedding)
  - Diverse contexts: Multiple senses per polysemous word

**Uniform sampling** (vs. quality-weighted):
- Large sample size naturally filters low-quality articles (noise averages out)
- Avoids selection bias toward specific topics
- Simple to implement and reproduce

---

### Phase 2: Sparsity Stabilization

**What is sparsity?**
- Each word activates only a subset of dimensions (e.g., "dog" uses [animal, size, domestication] but not [temperature, morality, finance])
- Sparsity = % of dimensions that are zero for each word
- Target: 40-70% sparsity

**Why does sparsity matter?**
- **Interpretability**: Each dimension represents ONE semantic axis
- **Compositional**: Can combine dimensions via logical operations (NOT, AND, VERY)
- **Efficiency**: Sparse vectors = faster computation

**Convergence criteria**:
```python
# Pseudo-code
for epoch in range(500):
    train_one_epoch()
    current_sparsity = compute_sparsity(embeddings)

    if abs(current_sparsity - previous_sparsity) < 0.01:
        patience_counter += 1
    else:
        patience_counter = 0

    if patience_counter >= 50:
        print("Sparsity stabilized! Bootstrap complete.")
        break
```

**Example stabilization**:
- Epoch 100: Sparsity = 0.45
- Epoch 150: Sparsity = 0.52
- Epoch 200: Sparsity = 0.56
- Epoch 250: Sparsity = 0.58
- Epoch 300: Sparsity = 0.585
- Epoch 350: Sparsity = 0.587 ← Change < 0.01
- **Bootstrap complete at epoch 350**

---

## Incremental Learning Workflow

### Adding New Data (10K Batches)

**Process**:
1. **Parse new batch**: 10K sentences → triples
2. **Update long-term memory**: Add triples to existing graph
3. **Vocabulary expansion**: Identify new words not in bootstrap
4. **Embedding update**: Choose strategy (retrain, compose, or edge decision)

### Strategy A: Continue Training (Allow New Dimensions)

**When to use**: Early incremental batches (10K-100K new sentences)

**Process**:
```bash
# Load bootstrap checkpoint
python scripts/train_embeddings.py \
    --resume-from-checkpoint checkpoints/bootstrap_300k/best_model.pt \
    --graph-dir data/graph_310k \
    --epochs 50 \
    --allow-dimension-growth
```

**What happens**:
- Existing words: Refine embeddings (small gradient updates)
- New words: Initialize randomly, train from scratch
- New dimensions: If sparsity saturates, add new dimension

**Pro**: Can discover dimensions not found in bootstrap
**Con**: Requires GPU/compute

---

### Strategy B: Pure Compositional (No Training)

**When to use**: After sufficient dimensions discovered (>100K new sentences)

**Process**:
```bash
# Compose embeddings from graph structure
python scripts/compose_embeddings.py \
    --base-embeddings checkpoints/bootstrap_300k/embeddings.npy \
    --graph data/graph_310k \
    --new-words-only
```

**Algorithm** (for new word `w`):
```python
def compose_embedding(word: str, graph: LongTermMemory) -> np.ndarray:
    # 1. Get neighbors from graph
    neighbors = graph.get_neighbors(word, max_hops=2)

    # 2. Get their embeddings (known from bootstrap)
    neighbor_vecs = [embeddings[n] for n in neighbors if n in vocabulary]

    # 3. Weight by edge type
    weights = [edge_weight(word, n) for n in neighbors]

    # 4. Compose via weighted average
    composed = np.average(neighbor_vecs, axis=0, weights=weights)

    # 5. Enforce sparsity (zero out low-activation dimensions)
    composed[np.abs(composed) < threshold] = 0

    return composed
```

**Example**:
```
New word: "cryptocurrency"
Neighbors in graph:
  - "currency_wn.01_n" (IS_A relation, weight=1.0)
  - "digital_wn.02_adj" (DESCRIPTION relation, weight=0.8)
  - "bank_wn.01_n" (RELATED_TO relation, weight=0.5)

Composed embedding:
  = 1.0 * embed("currency") + 0.8 * embed("digital") + 0.5 * embed("bank")
  = Normalize and enforce sparsity
```

**Pro**: No compute needed, instant
**Con**: Quality depends on graph connectivity

---

### Strategy C: Edge Decision Model (Hybrid)

**When to use**: Always (future implementation)

**Concept**: Lightweight classifier decides **per-word** whether to retrain or compose.

**Features**:
- Graph connectivity (how many known neighbors?)
- Sense ambiguity (polysemous words need more context)
- Bootstrap coverage (is word semantically similar to bootstrap vocab?)

**Decision tree**:
```
if connectivity > 5:
    if ambiguity < 0.3:
        return "compose"  # Well-connected, unambiguous
    else:
        return "retrain"  # Polysemous, needs disambiguation
else:
    return "retrain"  # Sparse graph, need data
```

**Implementation** (future):
```python
# Lightweight neural net (3 layers, <10K parameters)
class EdgeDecisionModel:
    def forward(self, word_features):
        # Returns: "retrain" or "compose"
        pass
```

**Pro**: Best of both worlds - retrain when needed, compose when possible
**Con**: Requires training edge model on bootstrap data

---

## Compositional Embeddings

### Why Composition Over Training?

After bootstrap, **graph structure encodes semantics**. The neural net was just a tool to discover dimensions - now we can compose directly.

**Analogy**:
- **Bootstrap**: Discovering the periodic table (what are the fundamental elements?)
- **Incremental**: Combining elements to form molecules (composition, not discovery)

### How Humans Learn New Words

Humans don't "retrain neural nets" when learning new words. They:
1. Hear word in context: "Bitcoin is a cryptocurrency"
2. Connect to known concepts: crypto=hidden, currency=money
3. Compose meaning: "Hidden money? Probably digital money that's encrypted"
4. Refine through usage: Multiple exposures refine understanding

NAOMI-II does the same:
1. Parse sentence: Extract semantic relationships
2. Connect to graph: Find neighbors (currency, digital, cryptography)
3. Compose embedding: Weight by relationship strength
4. Refine through usage: Update edge weights as more contexts appear

### Compositional Operations

**NOT** (antonym):
```python
embed("good") = [0.8, 0.1, 0.0, ...]  # morality=0.8
embed("bad") = NOT(embed("good"))
            = [-0.8, 0.1, 0.0, ...]  # morality=-0.8 (flip polarity dims)
```

**AND** (intersection):
```python
embed("red") = [0.0, 0.0, 0.9, ...]  # color=red
embed("car") = [0.0, 0.3, 0.0, ...]  # size=medium
embed("red car") = AND(embed("red"), embed("car"))
                 = [0.0, 0.3, 0.9, ...]  # color=red, size=medium
```

**VERY** (intensification):
```python
embed("good") = [0.6, 0.1, 0.0, ...]
embed("very good") = VERY(embed("good"))
                   = [0.9, 0.1, 0.0, ...]  # Scale up polarity dims
```

---

## Edge Decision Model

### Architecture

**Input features** (per new word):
1. **Graph connectivity**: Number of neighbors within 2 hops
2. **Neighbor quality**: Average embedding confidence of neighbors
3. **Sense ambiguity**: Number of WordNet senses for this word
4. **Semantic similarity**: Cosine similarity to closest bootstrap word
5. **Frequency**: Occurrence count in new corpus batch

**Output**: Binary decision (`retrain` or `compose`)

**Model**:
```python
class EdgeDecisionModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(5, 16)  # 5 input features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)   # Binary output

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Probability of "retrain"
        return x > 0.5  # Threshold
```

**Training data**:
- Bootstrap 300K: For each word, compute features and try both strategies
- Label: Which strategy produced better embedding? (measured by downstream task)

**Size**: ~5K parameters (tiny model, runs in milliseconds)

---

## Implementation Roadmap

### Week 1: Foundation
- [x] Document 3-graph architecture (this file)
- [ ] Add resume training support
- [ ] Create Wikipedia sampling script
- [ ] Parse 100K bootstrap sample
- [ ] Train 100K bootstrap model

### Week 2: Incremental Infrastructure
- [ ] Implement `WordNetReference` class (lazy loading)
- [ ] Implement `LongTermMemory` class (incremental updates)
- [ ] Implement `ShortTermMemory` class (working context)
- [ ] Add incremental vocabulary expansion
- [ ] Modify `build_sense_graph.py` for incremental mode

### Week 3: Full Bootstrap
- [ ] Parse 300K Wikipedia sample
- [ ] Train until sparsity stabilizes
- [ ] Analyze dimension discovery
- [ ] Save bootstrap checkpoint

### Week 4: First Incremental Batch
- [ ] Parse 10K new sentences
- [ ] Update long-term memory graph
- [ ] Test Strategy B (pure compositional)
- [ ] Validate embedding quality

### Future Work
- [ ] Implement edge decision model
- [ ] Add short-term memory reasoning
- [ ] Scale to full Wikipedia (12.4M sentences)
- [ ] Test on downstream tasks (translation, QA, reasoning)

---

## Use Cases

### 1. Wikipedia Learning (Current)

**Goal**: Learn factual knowledge from Wikipedia

**Workflow**:
1. Bootstrap 300K → discover dimensions
2. Add 10K batches → grow knowledge base
3. All facts → long-term memory (auto-validated)

**Short-term memory**: Not used (Wikipedia is trusted source)

---

### 2. Conversational AI (Future)

**Goal**: Chat with user, remember context

**Workflow**:
```python
# User: "I have a dog named Max."
stm.add_triple(("user", "HAS_PET", "Max"))
stm.add_triple(("Max", "IS_A", "dog"))

# User: "He's 5 years old."
# Resolve pronoun using short-term context
stm.resolve("he") → "Max"
stm.add_triple(("Max", "AGE", 5))

# User: "Remember that Max likes swimming."
# Promote to long-term (user explicitly requested)
ltm.merge_from_short_term(stm, validated=True)
```

**Short-term memory**: Active conversational context
**Long-term memory**: User profile, preferences

---

### 3. Web-Scraped Learning (Future)

**Goal**: Learn specific skill from online tutorial

**Workflow**:
```python
# User: "Teach yourself how to code in Rust from this tutorial"
tutorial_url = "https://doc.rust-lang.org/book/"

for page in scrape(tutorial_url):
    for sentence in parse(page):
        stm.add_triples(sentence)

# User reviews: "This tutorial is accurate"
ltm.merge_from_short_term(stm, source="rust_tutorial", confidence=0.9)

# Now NAOMI-II can answer: "How do I create a vector in Rust?"
# Retrieves from long-term memory: Vec::new() or vec![] macro
```

**Short-term memory**: Tutorial content (tentative)
**Long-term memory**: Validated knowledge after review

---

### 4. Hypothetical Reasoning (Future)

**Goal**: Answer "what if" questions

**Workflow**:
```python
# User: "What would happen if banks didn't exist?"
stm.add_triple(("bank_wn.01_n", "EXISTS", False), hypothetical=True)

# Chain reasoning in short-term
stm.infer([
    ("people", "CANNOT", "store_money"),
    ("loans", "BECOME", "difficult"),
    ("economy", "REQUIRES", "alternative_systems")
])

# User: "Interesting. What about barter systems?"
# Continue reasoning in same short-term context

# After conversation: Discard short-term (hypotheticals don't go to long-term)
stm.clear()
```

**Short-term memory**: Hypothetical premises + inferences
**Long-term memory**: Not affected (hypotheticals stay local)

---

## Key Principles

1. **Incremental, never forgetting**: Facts accumulate, never deleted
2. **Compositional semantics**: After bootstrap, no neural net needed
3. **Interpretable dimensions**: Each dimension = one semantic axis
4. **Mutable knowledge**: Facts can be updated/contradicted/refined
5. **Safe learning**: Short-term validates before long-term promotion
6. **Grounded in structure**: Parse trees + WordNet = meaning

---

## References

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Overall system design
- [DUAL_GRAPH_ARCHITECTURE.md](../DUAL_GRAPH_ARCHITECTURE.md) - Original 2-graph design
- [MEMORY_GRAPH_VISION.md](../docs/MEMORY_GRAPH_VISION.md) - Reasoning with knowledge graphs
- [SEMANTIC_VECTOR_SPACE_GOALS.md](../SEMANTIC_VECTOR_SPACE_GOALS.md) - Embedding space philosophy

---

**Last Updated**: 2025-11-29
**Status**: Design phase - implementation starting Week 1
