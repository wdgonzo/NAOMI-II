# NAOMI-II Scaling Plan: Large Vocabulary + Unsupervised Dimension Discovery

**Date**: 2025-11-26
**Goal**: Scale from 5K to 500K+ vocabulary with unsupervised semantic dimension discovery

---

## Executive Summary

We're moving from a **proof-of-concept** (5K words, manual dimension assignments) to a **production-scale system** (500K+ words, unsupervised discovery). This requires:

1. **Remove manual dimension assignments** - Let the model discover semantic axes naturally
2. **Scale to full WordNet** - 150K sense-tagged words, 3,989 antonym pairs
3. **Add Wikipedia corpus** - Millions of sentences for richer context
4. **Implement iterative WSD** - Reparse sentences as embeddings improve for true word sense disambiguation

---

## Current State vs Target State

| Aspect | Current (Proof of Concept) | Target (Production) |
|--------|---------------------------|---------------------|
| **Vocabulary** | 5,290 words | 500K+ words |
| **Data Source** | 1,000 Brown Corpus sentences | Full WordNet + Wikipedia |
| **Dimension Assignment** | Manual config files | Unsupervised discovery |
| **Antonym Pairs** | 47 (matched to vocab) | 3,989 (all from WordNet) |
| **WSD Method** | Single-pass context-based | Iterative embedding-based |
| **Training Time** | 15 minutes | Hours to days |
| **Interpretability** | High (manually assigned) | High (discovered post-training) |

---

## Phase 1: Unsupervised Dimension Discovery (1-2 Days)

### Goal
Prove that interpretable semantic dimensions emerge WITHOUT manual assignment.

### What We're Removing
- `config/dimension_assignments.json` - Manual "morality=dim0, size=dim2" assignments
- `config/semantic_clusters.json` - Manual positive/negative word lists
- `config/antonym_types.json` - Manual antonym categorization
- `SelectivePolarityLoss` - Forces specific antonym pairs onto specific dimensions
- `DimensionalConsistencyLoss` - Forces specific words onto specific poles

### What We're Keeping
- ✅ **Distance Loss** - Parse tree relationships = geometric distances
- ✅ **Sparsity Loss** - L1 regularization for sparse encodings (40-70%)
- ✅ **Batch parsing pipeline** - Already scales to arbitrary corpus size
- ✅ **Sense-tagged vocabulary** - Prevents polysemy confusion

### New Unsupervised Training Approach

**Training constraints:**
```python
total_loss = distance_loss + sparsity_loss
```

**No dimension pre-assignment!** The model learns:
- Which dimensions to use for which semantic properties
- Which antonym pairs oppose on which dimensions
- How to allocate sparse capacity across dimensions

### New Post-Training Discovery Script

**File**: `scripts/discover_polarity_dimensions.py`

**What it does:**
1. Loads trained embeddings (no manual assignments used in training)
2. For each dimension (0-127):
   - Computes opposite-sign consistency for all antonym pairs
   - Identifies pairs with >70% opposite-sign rate on this dimension
   - Clusters antonym pairs by semantic type (size, emotion, morality, etc.)
3. Outputs discovered dimension meanings:
   ```
   Dimension 42:
     - 89% opposite-sign consistency
     - Antonym types: size (big/small, large/tiny)
     - Interpretation: SIZE dimension (discovered!)
   ```

### Expected Results

**If successful:**
- 5-15 dimensions show clear polarity patterns (>70% opposite-sign)
- Different antonym types cluster on different dimensions
- Dimensions have interpretable meanings (size, emotion, morality)
- **Without any manual pre-assignment!**

**If unsuccessful:**
- Fallback: Hybrid approach (some manual initialization, then discovery)
- Analysis: What constraints are needed for emergence?

### Implementation Steps

1. **Modify `scripts/train_embeddings.py`**:
   - Make polarity/consistency losses truly optional (already are)
   - Add `--unsupervised` flag to skip config loading
   - Confirm training works with distance + sparsity only

2. **Create `scripts/discover_polarity_dimensions.py`**:
   ```python
   def analyze_dimension_polarity(embeddings, antonym_pairs, dim_idx):
       """Check if antonyms have opposite signs on this dimension."""
       opposite_count = 0
       for word1, word2 in antonym_pairs:
           vec1, vec2 = embeddings[word1], embeddings[word2]
           if np.sign(vec1[dim_idx]) != np.sign(vec2[dim_idx]):
               opposite_count += 1
       return opposite_count / len(antonym_pairs)

   def discover_semantic_dimensions(embeddings, antonym_pairs, threshold=0.7):
       """Find dimensions where antonyms consistently oppose."""
       discovered_dims = {}
       for dim in range(embeddings.shape[1]):
           consistency = analyze_dimension_polarity(embeddings, antonym_pairs, dim)
           if consistency > threshold:
               discovered_dims[dim] = {
                   'consistency': consistency,
                   'antonym_pairs': [...],  # Pairs that oppose on this dim
                   'interpretation': infer_semantic_type(...)
               }
       return discovered_dims
   ```

3. **Test on current 5K vocabulary**:
   ```bash
   # Train unsupervised
   python scripts/train_embeddings.py --unsupervised --epochs 50

   # Discover dimensions
   python scripts/discover_polarity_dimensions.py \
     --embeddings checkpoints/embeddings.npy \
     --vocabulary checkpoints/vocabulary.json \
     --antonyms config/antonym_types.json \
     --output checkpoints/discovered_dimensions.json
   ```

4. **Validate discovery**:
   - Compare discovered dimensions to manual assignments
   - Check if interpretable dimensions emerged
   - Measure overlap: Did model choose similar dims as manual assignment?

---

## Phase 2: Scale to Full WordNet (3-5 Days)

### Goal
Train on ALL of WordNet's semantic knowledge (150K words, 3,989 antonym pairs).

### WordNet Statistics

- **Total synsets**: 117,659
- **Total unique words**: 148,730
- **Antonym pairs**: 3,989 (automatically extracted)
- **Hypernym relations**: ~80,000
- **Meronym relations**: ~13,000
- **All semantic relations**: ~500,000 edges

### New Script: Extract Full WordNet

**File**: `scripts/extract_full_wordnet.py`

**Purpose**: Create training data from complete WordNet without parsing sentences.

**Algorithm**:
```python
from nltk.corpus import wordnet as wn

def extract_full_wordnet():
    """Extract all WordNet synsets and relations."""
    graph = KnowledgeGraph()

    # Add all synsets as nodes
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name()
            sense_tag = f"{word}_wn.{synset.offset():02d}_{synset.pos()}"
            graph.add_node(sense_tag, 'en')

    # Add all semantic relations
    for synset in wn.all_synsets():
        # Antonyms (3,989 pairs)
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                graph.add_edge(source, target, 'ANTONYM', confidence=1.0)

        # Hypernyms (IS-A relations)
        for hypernym in synset.hypernyms():
            graph.add_edge(source, target, 'HYPERNYM', confidence=1.0)

        # Meronyms (PART-OF relations)
        for meronym in synset.part_meronyms():
            graph.add_edge(source, target, 'PART_OF', confidence=1.0)

        # ... all other relations ...

    return graph.to_training_data()
```

**Output**: `data/wordnet_full/training_data.pkl`
- ~500,000 training edges
- 150,000 sense-tagged words
- All WordNet semantic relations

### Training at Scale

**Configuration**:
```bash
python scripts/train_embeddings.py \
  --data-dir data/wordnet_full \
  --embedding-dim 256 \
  --epochs 200 \
  --lr 0.001 \
  --batch-size 512 \
  --unsupervised \
  --output-dir checkpoints_wordnet_full
```

**Expected training time**: 6-12 hours (CPU)

**Why 256 dimensions?**
- 150K vocabulary needs more capacity than 5K vocabulary
- Rule of thumb: log₂(vocab_size) × 20 ≈ 256 for 150K vocab
- Can experiment with 512 if 256 shows saturation

### Validation

**After training:**
```bash
# Discover polarity dimensions
python scripts/discover_polarity_dimensions.py \
  --embeddings checkpoints_wordnet_full/embeddings.npy \
  --vocabulary checkpoints_wordnet_full/vocabulary.json \
  --output checkpoints_wordnet_full/discovered_dimensions.json

# Analyze all dimensions
python scripts/analyze_embedding_dimensions.py
```

**Success criteria:**
- 10-20 interpretable polarity dimensions discovered
- All 3,989 antonym pairs show polarity on ≥1 dimension
- Sparsity maintained at 40-70%
- Dimension discovery matches semantic expectations

---

## Phase 3: Add Wikipedia Corpus (1-2 Weeks)

### Goal
Combine WordNet semantic knowledge with real-world usage from Wikipedia.

### Why Wikipedia + WordNet?

| Data Source | Provides | Example |
|-------------|----------|---------|
| **WordNet** | Semantic relations (clean, structured) | "dog" IS-A "animal", "good" ANTONYM "bad" |
| **Wikipedia** | Usage patterns (real, contextual) | "The dog chased the cat" → dog SUBJECT chase |
| **Combined** | Best of both worlds | Semantic structure + real usage = robust embeddings |

### Wikipedia Data Pipeline

#### Step 1: Download & Preprocess

**File**: `scripts/download_wikipedia.py`

```python
import wikipediaapi
from nltk.tokenize import sent_tokenize

def download_wikipedia_corpus(num_articles=10000):
    """Download and preprocess Wikipedia articles."""
    wiki = wikipediaapi.Wikipedia('en')

    sentences = []
    for article_title in get_random_articles(num_articles):
        page = wiki.page(article_title)
        if page.exists():
            # Sentence segmentation
            text = page.text
            article_sentences = sent_tokenize(text)
            sentences.extend(article_sentences)

    # Filter quality
    filtered = [
        sent for sent in sentences
        if is_quality_sentence(sent)  # Length, grammar, etc.
    ]

    return filtered
```

**Output**: `data/wikipedia/raw_sentences.txt`
- Target: 100K-1M sentences
- ~10-50M words

#### Step 2: Parse Wikipedia Sentences

**Use existing batch parser:**
```bash
python scripts/batch_parse_corpus.py \
  --input data/wikipedia/raw_sentences.txt \
  --output-dir data/wikipedia_parsed \
  --batch-size 1000 \
  --checkpoint-interval 10000
```

**Expected**:
- 100K sentences → ~500K unique words
- Parse time: 2-4 hours (17 sent/sec)

#### Step 3: Word Sense Disambiguation

**File**: `src/embeddings/sense_mapper.py` (extend)

**Current WSD**: Context-based (uses parse tree neighbors)
```python
def map_word_to_sense(word, context_words):
    """Map word to WordNet sense using context."""
    synsets = wn.synsets(word)

    # Score each sense by context overlap
    scores = []
    for synset in synsets:
        context_score = compute_context_similarity(synset, context_words)
        scores.append((synset, context_score))

    # Return highest-scoring sense
    best_synset, _ = max(scores, key=lambda x: x[1])
    return format_sense_tag(best_synset)
```

**New**: Embedding-based WSD (Phase 4)

#### Step 4: Build Combined Graph

**File**: `scripts/build_combined_graph.py`

```python
def build_combined_graph(wordnet_graph, wikipedia_parsed):
    """Combine WordNet + Wikipedia into single training graph."""
    combined = KnowledgeGraph()

    # Add WordNet semantic edges
    combined.add_edges_from(wordnet_graph.edges)

    # Add Wikipedia parse-derived edges
    for sentence_parse in wikipedia_parsed:
        triples = extract_triples(sentence_parse)
        for subject, relation, object in triples:
            combined.add_edge(subject, relation, object)

    return combined.to_training_data()
```

**Output**: `data/combined_wordnet_wikipedia/training_data.pkl`
- WordNet edges: ~500K (semantic relations)
- Wikipedia edges: ~5M (usage patterns)
- **Total**: ~5.5M training edges
- **Vocabulary**: 500K+ sense-tagged words

### Training on Combined Data

```bash
python scripts/train_embeddings.py \
  --data-dir data/combined_wordnet_wikipedia \
  --embedding-dim 512 \
  --epochs 500 \
  --lr 0.0001 \
  --batch-size 1024 \
  --unsupervised \
  --output-dir checkpoints_combined
```

**Training time**: 12-48 hours (may need GPU)

**Considerations**:
- Large batch size (1024) for stability
- Lower learning rate (0.0001) for convergence
- More dimensions (512) for capacity
- Gradient clipping to prevent explosions

### Validation at Scale

**Quantitative metrics**:
- Sparsity: 40-70%
- Antonym polarization: >90% on discovered dimensions
- Training loss convergence

**Qualitative analysis**:
- Discovered dimension interpretability
- Word similarity rankings (vs human judgments)
- Analogy tasks (king - man + woman ≈ queen)

---

## Phase 4: Iterative WSD Reparsing (2-3 Weeks)

### The WSD Chicken-and-Egg Problem

**Current limitation**: WSD is single-pass
```
Parse sentences → WSD (context-based) → Build graph → Train embeddings
                  ↑                                         ↓
                  └─────────── No feedback loop ────────────┘
```

**The problem**:
- WSD needs good embeddings to disambiguate accurately
- But embeddings depend on accurate WSD
- Circular dependency!

### Solution: Iterative Refinement

**Iterative approach**:
```
Iteration 0: Parse → WSD (context-based) → Train embeddings₀

Iteration 1: Use embeddings₀ for WSD → Reparse → Train embeddings₁

Iteration 2: Use embeddings₁ for WSD → Reparse → Train embeddings₂

...

Iteration N: Convergence (WSD stops changing)
```

**Key insight**: Break circularity by bootstrapping with context-based WSD, then iterate.

### Embedding-Based WSD Algorithm

**File**: `src/embeddings/sense_mapper.py` (extend)

```python
class EmbeddingBasedWSD:
    """Word Sense Disambiguation using trained embeddings."""

    def __init__(self, embeddings, word_to_id):
        self.embeddings = embeddings
        self.word_to_id = word_to_id

    def disambiguate(self, word, context_words):
        """Select best sense for word given context."""
        # Get all possible senses
        synsets = wn.synsets(word)

        if len(synsets) == 1:
            return format_sense_tag(synsets[0])  # Unambiguous

        # Compute context embedding (average of context words)
        context_embedding = np.zeros(self.embeddings.shape[1])
        num_context = 0
        for context_word in context_words:
            if context_word in self.word_to_id:
                context_embedding += self.embeddings[self.word_to_id[context_word]]
                num_context += 1

        if num_context > 0:
            context_embedding /= num_context
        else:
            # Fallback to context-based WSD
            return self.context_based_wsd(word, context_words)

        # Score each sense by similarity to context
        best_sense = None
        best_similarity = -1

        for synset in synsets:
            sense_tag = format_sense_tag(synset)
            if sense_tag not in self.word_to_id:
                continue

            sense_embedding = self.embeddings[self.word_to_id[sense_tag]]
            similarity = cosine_similarity(sense_embedding, context_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_sense = sense_tag

        return best_sense or self.context_based_wsd(word, context_words)
```

### Iterative Training Loop

**File**: `scripts/train_iterative_wsd.py`

```python
def iterative_wsd_training(corpus_sentences, max_iterations=5):
    """Train with iterative WSD refinement."""

    # Iteration 0: Bootstrap with context-based WSD
    print("Iteration 0: Context-based WSD...")
    parsed_corpus = parse_corpus_with_context_wsd(corpus_sentences)
    graph = build_graph(parsed_corpus)
    embeddings_0, vocab_0 = train_embeddings(graph, epochs=100)

    wsd_changes = []

    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}: Embedding-based WSD...")

        # Use previous embeddings for WSD
        wsd_mapper = EmbeddingBasedWSD(embeddings_0, vocab_0)

        # Reparse with improved WSD
        reparsed_corpus = parse_corpus_with_embedding_wsd(
            corpus_sentences, wsd_mapper
        )

        # Count WSD changes
        num_changes = count_wsd_differences(parsed_corpus, reparsed_corpus)
        wsd_changes.append(num_changes)
        print(f"  WSD changes: {num_changes} / {len(corpus_sentences)}")

        # Check convergence
        if num_changes < 0.01 * len(corpus_sentences):  # <1% changed
            print(f"  Converged after {iteration} iterations!")
            break

        # Rebuild graph with new WSD
        graph = build_graph(reparsed_corpus)

        # Retrain embeddings
        embeddings_0, vocab_0 = train_embeddings(graph, epochs=100)
        parsed_corpus = reparsed_corpus

    return embeddings_0, vocab_0, wsd_changes
```

### Convergence Criteria

**Stop iterating when:**
1. WSD changes <1% of sentences
2. Embedding distance between iterations <0.01
3. Reached max iterations (5-10)

**Expected**:
- Convergence in 3-5 iterations
- WSD accuracy improves each iteration
- Final WSD better than context-based baseline

### Validation: SemCor Benchmark

**Dataset**: SemCor (manually annotated WordNet senses)
- ~200K word occurrences
- Gold-standard sense annotations
- Standard WSD benchmark

**Evaluation**:
```python
def evaluate_wsd_accuracy(wsd_mapper, semcor_dataset):
    """Measure WSD accuracy vs gold annotations."""
    correct = 0
    total = 0

    for sentence, gold_senses in semcor_dataset:
        predicted_senses = wsd_mapper.disambiguate_sentence(sentence)
        for word_idx, (predicted, gold) in enumerate(zip(predicted_senses, gold_senses)):
            if predicted == gold:
                correct += 1
            total += 1

    return correct / total
```

**Target**: >80% accuracy (state-of-the-art WSD systems: 75-85%)

### Challenges & Solutions

**Challenge 1**: Bootstrapping (no embeddings for iteration 0)
- **Solution**: Use context-based WSD for first iteration
- **Validation**: Iteration 0 should match current system

**Challenge 2**: Computational cost (full reparse each iteration)
- **Solution**: Only reparse sentences where WSD changed
- **Optimization**: Track ambiguous words, skip unambiguous

**Challenge 3**: Circular dependency trap (embeddings degrade)
- **Solution**: Early stopping if WSD accuracy decreases
- **Monitoring**: Track WSD accuracy on held-out validation set

**Challenge 4**: Sense drift (training shifts sense meanings)
- **Solution**: Fix WordNet sense definitions (anchor points)
- **Constraint**: WordNet antonyms must maintain polarity

---

## Phase 5: Dynamic Dimension Management (Future Work)

### Motivation

**Current**: Fixed 128 or 256 dimensions
**Problem**:
- Too few → saturation (all dims used, low sparsity)
- Too many → waste (unused dims, overfitting)

**Solution**: Dynamically add dimensions as needed.

### Dimension Addition Algorithm

```python
def should_add_dimension(embeddings, sparsity_threshold=0.3):
    """Check if we need more dimensions."""
    # Compute per-dimension sparsity
    dim_sparsity = compute_dimension_sparsity(embeddings)

    # If all dimensions are <30% sparse, we need more capacity
    if np.max(dim_sparsity) < sparsity_threshold:
        return True
    return False

def add_dimension(model, embeddings):
    """Add a new dimension to the model."""
    vocab_size, current_dim = embeddings.shape

    # Initialize new dimension near zero
    new_dim_values = np.random.randn(vocab_size) * 0.001

    # Expand embeddings
    new_embeddings = np.concatenate([
        embeddings,
        new_dim_values.reshape(-1, 1)
    ], axis=1)

    # Update model parameters
    model.embeddings = nn.Parameter(torch.from_numpy(new_embeddings))

    return model
```

### Training Loop with Dynamic Dims

```python
for epoch in range(max_epochs):
    train_epoch(model, data)

    # Every 10 epochs, check if we need more dimensions
    if epoch % 10 == 0:
        embeddings = model.embeddings.detach().numpy()
        if should_add_dimension(embeddings):
            print(f"Adding dimension at epoch {epoch}...")
            model = add_dimension(model, embeddings)
```

### Dimension Pruning (Future)

**Not recommended initially** - removing dimensions loses information.

**Alternative**: Mark dimensions as "dormant"
- Low-variance dimensions get flagged
- Not removed (backward compatibility)
- Can be reactivated if needed

---

## Technical Implementation Details

### Files to Create

#### Phase 1: Unsupervised Discovery
- `scripts/discover_polarity_dimensions.py` - Post-training dimension analysis
- Modify `scripts/train_embeddings.py` - Add `--unsupervised` flag

#### Phase 2: Full WordNet
- `scripts/extract_full_wordnet.py` - Extract all WordNet data
- `data/wordnet_full/` - Output directory for full WordNet graph

#### Phase 3: Wikipedia
- `scripts/download_wikipedia.py` - Download & preprocess Wikipedia
- `scripts/build_combined_graph.py` - Merge WordNet + Wikipedia
- `data/wikipedia/` - Wikipedia raw sentences
- `data/wikipedia_parsed/` - Parsed Wikipedia with WSD
- `data/combined_wordnet_wikipedia/` - Final training data

#### Phase 4: Iterative WSD
- `scripts/train_iterative_wsd.py` - Iterative training loop
- Extend `src/embeddings/sense_mapper.py` - Add embedding-based WSD
- `scripts/evaluate_wsd.py` - Benchmark on SemCor

#### Phase 5: Dynamic Dimensions
- Extend `scripts/train_embeddings.py` - Add dimension addition logic

### Files to Remove (After Phase 1)

**Manual configuration files** (no longer needed):
- `config/dimension_assignments.json`
- `config/semantic_clusters.json`
- `config/antonym_types.json`
- `scripts/generate_vocab_configs.py`

**Keep for reference**: Move to `archive/proof_of_concept/`

### Training Hardware Requirements

| Phase | Vocabulary | Edges | Training Time | Hardware |
|-------|-----------|-------|---------------|----------|
| **Phase 1** | 5K | 23K | 15 min | CPU (current) |
| **Phase 2** | 150K | 500K | 6-12 hrs | CPU or GPU |
| **Phase 3** | 500K | 5.5M | 12-48 hrs | GPU recommended |
| **Phase 4** | 500K | 5.5M × 5 iterations | 2-10 days | GPU required |

**GPU Recommendation**:
- Phase 2: Optional (CPU works, just slower)
- Phase 3+: Highly recommended (10-20x speedup)
- Cloud options: AWS p3.2xlarge, GCP n1-highmem-8 + T4 GPU

---

## Success Metrics

### Phase 1: Unsupervised Discovery
- ✅ 5-15 interpretable polarity dimensions discovered without manual assignment
- ✅ >70% opposite-sign consistency for antonym pairs on discovered dims
- ✅ Discovered dimensions match semantic expectations (size, emotion, etc.)
- ✅ Maintained 40-70% sparsity

### Phase 2: Full WordNet
- ✅ Successfully trained on 150K vocabulary
- ✅ All 3,989 antonym pairs polarized on discovered dimensions
- ✅ 10-20 interpretable dimensions emerged
- ✅ Training converged (<6% change over 10 epochs)

### Phase 3: Wikipedia Integration
- ✅ Processed 100K+ Wikipedia sentences
- ✅ Scaled to 500K+ vocabulary
- ✅ Training time <24 hours per epoch (with GPU)
- ✅ Word similarity rankings correlate with human judgments (>0.6 Spearman ρ)

### Phase 4: Iterative WSD
- ✅ WSD accuracy improves each iteration
- ✅ Convergence in 3-5 iterations
- ✅ Final WSD accuracy >80% on SemCor benchmark
- ✅ Parse quality improves (fewer WSD errors)

### Phase 5: Dynamic Dimensions
- ✅ Dimensions added when sparsity drops below threshold
- ✅ No dimension pruning needed (sparsity maintained)
- ✅ Final embedding dim adapts to vocabulary size

---

## Risks & Mitigation Strategies

### Risk 1: Unsupervised Discovery Fails
**Symptom**: No interpretable dimensions emerge, random polarity patterns

**Root causes**:
- Insufficient training data (5K vocab too small)
- Missing constraints (need weak initialization?)
- Sparsity too aggressive (killed useful signal)

**Mitigation**:
- ✅ Scale to Phase 2 (150K vocab) - more data helps emergence
- ✅ Try weak initialization (initialize dims near manual assignments, let drift)
- ✅ Tune sparsity weight (try 0.005, 0.01, 0.02)
- ✅ Hybrid approach: Manual init for 10 dims, discover rest

**Fallback**: Keep manual assignments for critical dimensions (size, morality), discover others

---

### Risk 2: Wikipedia Scale Too Large
**Symptom**: Training doesn't converge, loss oscillates, OOM errors

**Root causes**:
- Too many edges (>10M) → batch sampling issues
- Vocabulary too large (>1M) → memory issues
- Learning rate too high for scale

**Mitigation**:
- ✅ Start with 10K Wikipedia sentences, scale gradually (10K → 50K → 100K → 1M)
- ✅ Use gradient accumulation (simulate large batches without OOM)
- ✅ Lower learning rate (0.0001 or 0.00001)
- ✅ Cloud GPU with large RAM (AWS p3.8xlarge: 64GB RAM)

**Fallback**: Cap Wikipedia at 100K sentences if larger scale unstable

---

### Risk 3: Iterative WSD Doesn't Improve
**Symptom**: WSD accuracy stays flat or decreases across iterations

**Root causes**:
- Circular dependency trap (bad embeddings → bad WSD → bad embeddings)
- Sense drift (embedding meanings shift, invalidating WSD)
- Context embeddings too noisy (averaging context words ineffective)

**Mitigation**:
- ✅ Careful initialization (context-based WSD for iteration 0)
- ✅ Track WSD accuracy on held-out validation set (stop if degrading)
- ✅ Fix WordNet antonym constraints (prevent sense drift)
- ✅ Better context representation (weighted average, attention mechanism)

**Fallback**: Use best WSD from any iteration, don't iterate further

---

### Risk 4: Compute Resources Insufficient
**Symptom**: Training takes weeks, crashes due to memory, can't scale

**Mitigation**:
- ✅ Batch processing with checkpointing (resume after crashes)
- ✅ Incremental scaling (don't jump from 5K → 500K, go 5K → 50K → 150K → 500K)
- ✅ Cloud compute for training phases (AWS, GCP, Lambda Labs)
- ✅ Mixed precision training (float16 for speed, float32 for stability)

**Cost estimate**:
- AWS p3.2xlarge (V100 GPU): $3.06/hr
- Phase 3 training (48 hrs): ~$150
- Phase 4 training (10 days): ~$750
- **Total cloud cost**: <$1,000 for all phases

**Fallback**: Use smaller vocab (150K instead of 500K) to fit on local hardware

---

## Timeline & Milestones

### Week 1: Unsupervised Discovery Proof
- **Days 1-2**: Implement unsupervised training + discovery script
- **Day 3**: Test on 5K vocab, validate emergence
- **Days 4-5**: Extract full WordNet, prepare Phase 2

**Deliverable**: Proof that unsupervised discovery works + full WordNet dataset ready

---

### Week 2: Full WordNet Training
- **Days 1-2**: Train on 150K WordNet vocabulary
- **Day 3**: Analyze discovered dimensions (detailed report)
- **Days 4-5**: Wikipedia download + preprocessing pipeline

**Deliverable**: 150K vocab embeddings with discovered dimensions + Wikipedia corpus ready

---

### Weeks 3-4: Wikipedia Integration
- **Week 3**:
  - Parse 100K Wikipedia sentences with WSD
  - Build combined WordNet + Wikipedia graph
  - Start training on combined dataset
- **Week 4**:
  - Complete training (may take days)
  - Analyze results at scale
  - Validate on word similarity benchmarks

**Deliverable**: 500K vocab embeddings trained on WordNet + Wikipedia

---

### Weeks 5-6: Iterative WSD
- **Week 5**:
  - Implement embedding-based WSD
  - Implement iterative training loop
  - Test on small corpus (1K sentences)
- **Week 6**:
  - Run full iterative training (3-5 iterations)
  - Evaluate on SemCor benchmark
  - Compare to baseline (context-based WSD)

**Deliverable**: Iterative WSD system with >80% SemCor accuracy

---

### Total Timeline: 4-6 Weeks

**Critical path**:
```
Phase 1 (2 days) → Phase 2 (5 days) → Phase 3 (2 weeks) → Phase 4 (2 weeks)
```

**Parallelizable**:
- Wikipedia download can happen during Phase 2 training
- WSD implementation can start during Phase 3 training

---

## Next Steps After Approval

1. **Immediate** (Today):
   - Create `scripts/discover_polarity_dimensions.py`
   - Add `--unsupervised` flag to training script
   - Test unsupervised training on current 5K vocab

2. **This Week**:
   - Validate unsupervised discovery works
   - Extract full WordNet data
   - Train on 150K vocabulary

3. **Next 2 Weeks**:
   - Download Wikipedia corpus
   - Build combined training data
   - Train at scale (may need cloud GPU)

4. **Following 2 Weeks**:
   - Implement iterative WSD
   - Benchmark on SemCor
   - Validate final system

---

## Open Questions for Discussion

1. **Embedding dimensionality**:
   - 256 or 512 for 500K vocab?
   - Start low and use dynamic dimension addition?

2. **Wikipedia scope**:
   - 100K sentences or 1M sentences?
   - Full Wikipedia dump (60M articles) or subset?

3. **GPU access**:
   - Use cloud GPU (AWS/GCP) or local hardware?
   - Budget for cloud compute ($500-$1000)?

4. **Validation metrics**:
   - What benchmarks matter most (word similarity, analogies, WSD accuracy)?
   - Need custom evaluation for parser-specific tasks?

5. **Phase 5 (dynamic dims)**:
   - Implement now or defer to later?
   - More important than iterative WSD?

---

## Conclusion

This plan transforms NAOMI-II from a **proof-of-concept** (5K words, manual dimensions) to a **production-scale system** (500K+ words, unsupervised discovery). The key innovations:

1. ✅ **Unsupervised dimension discovery** - No manual assignments, emergent structure
2. ✅ **Full WordNet integration** - All 150K words, 3,989 antonym pairs
3. ✅ **Wikipedia corpus** - Real-world usage patterns, millions of sentences
4. ✅ **Iterative WSD** - True word sense disambiguation via embedding feedback
5. ✅ **Scalable architecture** - Batch processing, checkpointing, cloud-ready

**This is ambitious but achievable.** Each phase builds on proven components (parser, WSD, training pipeline) and scales incrementally. The unsupervised discovery is the riskiest piece, but we validate it early (Phase 1) before committing to large-scale training.

**Ready to proceed?** Let's start with Phase 1 and prove unsupervised discovery works! 🚀
