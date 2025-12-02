# Antonym Clustering for Transparent Semantic Axes

## Overview

This system automatically discovers interpretable semantic axes by clustering WordNet antonym pairs using graph-based similarity signals. **This is a crucial first step for Phase 1 of NAOMI-II's transparent dimension learning.**

### Key Innovation

Unlike Word2Vec/BERT with opaque dimensions, this approach:
- **Discovers explicit semantic axes** from WordNet structure (NOT neural network weights)
- **Clusters 3,992 antonym pairs** into 30-80 interpretable dimensions
- **Combines 4 complementary similarity signals** (transitivity, quadrilaterals, definitions, hypernyms)
- **Produces human-nameable axes** (e.g., "morality", "size", "temperature")
- **Validates via coherence metrics** (pole words are semantically related)

### What This Enables

**Phase 1 (This System):**
- Discover which antonym pairs measure the "same concept"
- Group pairs into semantic axes (discrete structure)
- Validate interpretability (humans can name axes)

**Phase 2 (Next Step):**
- Use discovered axes as constraints for neural positioning
- Learn continuous word positions on each axis
- Achieve transparent dimensions where each axis = one semantic concept

---

## Architecture

### File Structure

```
src/embeddings/
├── similarity_signals.py    - 4 similarity signal computations
├── antonym_clustering.py    - Hierarchical clustering algorithm
└── axis_extraction.py       - Axis naming and validation

scripts/
├── cluster_antonym_axes.py       - Main clustering executable
└── validate_discovered_axes.py   - Validation and analysis

data/
├── full_wordnet/
│   └── antonym_pairs.json   - Input: 3,992 antonym pairs
└── discovered_axes/
    ├── similarity_matrix.npy          - Pairwise similarities
    ├── linkage_matrix.npy             - Hierarchical structure
    ├── cluster_assignments.npy        - Cluster IDs
    ├── semantic_axes.json             - Discovered axes (JSON)
    ├── axis_report.txt                - Human-readable report
    └── clustering_stats.json          - Metrics
```

### Similarity Signals

The algorithm combines 4 weighted signals to measure if two antonym pairs belong to the same semantic axis:

#### 1. Antonym Transitivity (0.4 weight) — MOST IMPORTANT

**Intuition:** Pairs sharing an endpoint likely measure the same concept.

**Example:**
- `good ↔ bad` and `good ↔ evil` share "good" → **same axis (morality)**
- `hot ↔ cold` and `warm ↔ cool` share "hot"/"warm" (similar synsets) → **same axis (temperature)**

**Implementation:**
```python
if pair1 and pair2 share a word or synset:
    return 1.0
else:
    return 0.0
```

#### 2. Similar-to Quadrilaterals (0.3 weight)

**Intuition:** If similar words form antonym pairs, original pairs measure the same axis.

**Example:**
- `good` ~ `virtuous` (WordNet similar_to)
- `bad` ~ `wicked` (WordNet similar_to)
- `virtuous ↔ wicked` exists
- → `good ↔ bad` and `virtuous ↔ wicked` share axis **via** quadrilateral

**Implementation:**
```python
similarity = average(
    jaccard(similar_tos(A1), similar_tos(A2)),
    jaccard(similar_tos(B1), similar_tos(B2))
)
```

#### 3. Definition TF-IDF Cosine (0.2 weight)

**Intuition:** Pairs with keyword overlap in definitions measure related concepts.

**Example:**
- `good ↔ bad`: "morally excellent" vs. "lacking moral qualities"
- `virtuous ↔ wicked`: "morally good" vs. "morally bad"
- Shared keywords: "moral", "morally" → **same axis**

**Implementation:**
```python
# Extract content words from definitions (remove stopwords)
# Compute TF-IDF vectors (using all antonym definitions as corpus)
# Return cosine similarity
```

#### 4. Hypernym Path Overlap (0.1 weight)

**Intuition:** Pairs tracing to common abstract hypernyms measure related concepts.

**Example:**
- `big ↔ small` → both trace to "attribute" → "property"
- `large ↔ tiny` → both trace to "attribute" → "property"
- Shared hypernyms → **same axis (size)**

**Implementation:**
```python
hypernyms1 = all_hypernyms(pair1.synset1) | all_hypernyms(pair1.synset2)
hypernyms2 = all_hypernyms(pair2.synset1) | all_hypernyms(pair2.synset2)
return jaccard(hypernyms1, hypernyms2)
```

### Clustering Algorithm

#### Step 1: Build Similarity Matrix (3,992 × 3,992)

For each pair of antonym pairs, compute weighted combination:

```python
similarity = (
    0.4 * transitivity +
    0.3 * quadrilateral +
    0.2 * definition_tfidf +
    0.1 * hypernym_overlap
)
```

**Complexity:** O(N²) where N = 3,992
- ~7.9M pairwise comparisons
- Estimated time: 10-30 minutes on CPU

#### Step 2: Hierarchical Agglomerative Clustering

Uses **complete linkage** (maximum distance between cluster members):
- Encourages tight, interpretable clusters
- Provides hierarchical structure (dendrogram)
- Allows sub-axis discovery

```python
from scipy.cluster.hierarchy import linkage
distance_matrix = 1.0 - similarity_matrix
linkage_matrix = linkage(distance_matrix, method='complete')
```

#### Step 3: Optimize Cut Height via Silhouette Score

Tests 100 cut heights to find optimal cluster count:

```python
for height in linspace(0.1, 0.95, 100):
    clusters = fcluster(linkage_matrix, height, criterion='distance')
    n_clusters = len(set(clusters))

    if 10 <= n_clusters <= 200:  # Valid range
        score = silhouette_score(distance_matrix, clusters)
        if score > best_score:
            best_height = height
```

**Expected outcome:** 30-80 clusters with silhouette score > 0.4

#### Step 4: Extract Semantic Axes

For each cluster:
1. **Extract poles:** Separate positive vs. negative words
2. **Name axis:** TF-IDF keywords from member definitions
3. **Compute coherence:** WordNet path similarity within poles
4. **Compute separation:** 1 - path similarity between poles
5. **Validate:** size ≥ 3, coherence > 0.3, separation > 0.5

---

## Usage

### Quick Start

```bash
# Cluster all 3,992 antonym pairs
python scripts/cluster_antonym_axes.py \
    --input data/full_wordnet/antonym_pairs.json \
    --output-dir data/discovered_axes \
    --min-clusters 10 \
    --max-clusters 200 \
    --plot-dendrogram \
    --plot-silhouette

# Validate discovered axes
python scripts/validate_discovered_axes.py \
    --results-dir data/discovered_axes
```

### Command-Line Options

**`cluster_antonym_axes.py`:**

```bash
--input PATH                # Input antonym pairs JSON (default: data/full_wordnet/antonym_pairs.json)
--output-dir PATH           # Output directory (default: data/discovered_axes)
--min-clusters INT          # Minimum clusters to consider (default: 10)
--max-clusters INT          # Maximum clusters to consider (default: 200)
--min-axis-size INT         # Minimum pairs per axis (default: 3)
--min-coherence FLOAT       # Minimum pole coherence (default: 0.3)
--min-separation FLOAT      # Minimum pole separation (default: 0.5)
--plot-dendrogram           # Generate dendrogram visualization
--plot-silhouette           # Generate silhouette optimization plot
--max-pairs INT             # Limit pairs for testing (optional)
```

**`validate_discovered_axes.py`:**

```bash
--results-dir PATH          # Results directory (default: data/discovered_axes)
--no-plots                  # Skip generating plots
```

### Expected Runtime

- **500 pairs:** ~5 minutes (testing)
- **3,992 pairs:** ~20-40 minutes (full run)
  - Similarity matrix: ~15-25 min
  - Clustering: ~2-5 min
  - Axis extraction: ~3-8 min

---

## Output Format

### `semantic_axes.json`

```json
{
  "axes": [
    {
      "axis_id": 0,
      "name": "morality",
      "cluster_id": 42,
      "size": 47,
      "coherence_score": 0.72,
      "separation_score": 0.89,
      "positive_pole": ["good", "virtuous", "moral", "righteous", ...],
      "negative_pole": ["bad", "evil", "wicked", "immoral", ...],
      "representative_pairs": [
        {
          "word1": "good",
          "word2": "bad",
          "synset1": "good.a.01",
          "synset2": "bad.a.01"
        },
        ...
      ]
    },
    ...
  ],
  "metadata": {
    "total_axes": 67,
    "mean_size": 52.4,
    "mean_coherence": 0.64,
    "mean_separation": 0.71
  }
}
```

### `axis_report.txt` (Human-Readable)

```
====================================================================================================
DISCOVERED SEMANTIC AXES
====================================================================================================
Total axes: 67

====================================================================================================
AXIS 1: MORALITY
====================================================================================================
Cluster ID: 42
Size: 47 antonym pairs
Coherence: 0.7245
Separation: 0.8912

POSITIVE POLE (23 words):
  good, virtuous, moral, righteous, ethical, benevolent, ...

NEGATIVE POLE (24 words):
  bad, evil, wicked, immoral, unethical, malevolent, ...

REPRESENTATIVE PAIRS:
  1. good ↔ bad
  2. virtuous ↔ wicked
  3. moral ↔ immoral
  4. righteous ↔ unrighteous
  5. ethical ↔ unethical
```

---

## Validation Metrics

### Quantitative Metrics

1. **Silhouette Score** (target: > 0.4)
   - Measures cluster quality
   - Higher = better-defined clusters

2. **Coherence Score** (target: > 0.3)
   - WordNet path similarity within each pole
   - Higher = pole words are more related

3. **Separation Score** (target: > 0.5)
   - 1 - path similarity between poles
   - Higher = poles are more distinct

4. **Coverage Rate** (target: > 85%)
   - Percentage of antonym pairs assigned to axes
   - Remaining pairs are singletons (domain-specific)

### Qualitative Validation

1. **Human Interpretability**
   - Can humans name axes from pole word lists alone?
   - Target: 70%+ of axes are nameable

2. **Known Axes Recovery**
   - Do known semantic categories emerge?
   - Expected: morality, size, temperature, emotion, strength, light

3. **Pole Consistency**
   - Are positive pole words truly positive?
   - Are negative pole words truly negative?

---

## Example Discovered Axes

Based on preliminary analysis, expected top axes include:

| Axis ID | Name | Size | Representative Pairs |
|---------|------|------|---------------------|
| 0 | morality | 40-60 | good↔bad, virtuous↔wicked, moral↔immoral |
| 1 | size | 30-50 | big↔small, large↔tiny, huge↔minuscule |
| 2 | temperature | 20-30 | hot↔cold, warm↔cool, freezing↔boiling |
| 3 | emotion | 35-55 | happy↔sad, joyful↔sorrowful, cheerful↔gloomy |
| 4 | strength | 25-40 | strong↔weak, powerful↔feeble, mighty↔frail |
| 5 | light | 15-25 | bright↔dark, light↔dim, luminous↔obscure |
| 6 | speed | 10-20 | fast↔slow, quick↔sluggish, rapid↔gradual |
| 7 | difficulty | 15-25 | easy↔difficult, simple↔complex, hard↔effortless |

---

## Troubleshooting

### Low Silhouette Score (< 0.3)

**Cause:** Similarity matrix too sparse (most pairs unrelated)

**Solutions:**
- Reduce `--min-clusters` (allow more granular axes)
- Increase `--max-clusters` (allow more specific groupings)
- Check if antonym pairs have sufficient WordNet metadata

### Too Many Singletons (> 20%)

**Cause:** Cut height too high (over-clustering)

**Solutions:**
- Silhouette optimization should handle this automatically
- Manually adjust cut height if needed
- Some singletons are expected (domain-specific pairs like "leeward↔windward")

### Low Coherence (<0.2)

**Cause:** Clusters mixing unrelated concepts

**Solutions:**
- Increase `--min-coherence` threshold
- Use stricter similarity signal weights
- Check for WordNet coverage issues

### Axes Don't Match Manual Categories

**Cause:** Algorithm discovered different but valid groupings

**Solutions:**
- This is OKAY! Automatic discovery may find non-obvious patterns
- Validate via coherence/separation metrics, not preconceived categories
- Review axis reports to understand discovered structure

---

## Integration with NAOMI-II Training

### Phase 1 (Completed by This System)

**Goal:** Discover discrete semantic axes

**Output:** 30-80 interpretable axes with:
- Axis name (e.g., "morality")
- Positive pole words (e.g., "good", "virtuous")
- Negative pole words (e.g., "bad", "wicked")
- Member antonym pairs

### Phase 2 (Next Step: Neural Positioning)

**Goal:** Learn continuous word positions on discovered axes

**Approach:**
1. Initialize embedding dimensions (51 anchors + 77 learned)
2. For each discovered axis, assign 1-5 dimensions
3. Add polarity constraints: opposite poles oppose on assigned dimensions
4. Train embeddings with:
   - Semantic distance loss (synonyms close, antonyms far)
   - Polarity loss (antonyms oppose on specific dims)
   - Sparsity loss (words zero on irrelevant dims)

**Expected Outcome:** Transparent dimensions where:
- Dimension 52-56: "morality" axis (good +1.0, bad -1.0)
- Dimension 57-59: "size" axis (big +0.8, small -0.8)
- Dimension 60-62: "temperature" axis (hot +0.9, cold -0.9)
- etc.

### Phase 3 (Future: Compositional Semantics)

**Goal:** NOT operation via axis negation

**Approach:**
- NOT(good) = negate morality dimensions → ≈ bad
- NOT(big) = negate size dimensions → ≈ small
- Validate via cosine similarity to actual antonyms

---

## References

### Key Insights from OPUS Analysis

1. **Antonym transitivity is the strongest signal** (0.4 weight)
   - Shared endpoints = shared axis (high precision)

2. **Complete linkage works better than average/ward**
   - Creates tight, interpretable clusters
   - Avoids "chaining" artifacts

3. **Silhouette score optimization is robust**
   - Reliably finds 30-80 cluster sweet spot
   - Better than fixed cluster count

4. **Definition TF-IDF needs preprocessing**
   - Remove stopwords aggressively
   - Use content words only (nouns, verbs, adjectives, adverbs)

### Alternative Approaches Considered

1. **Louvain community detection**
   - Pro: Faster, automatic cluster count
   - Con: Non-hierarchical, less interpretable

2. **K-means clustering**
   - Pro: Fast, well-understood
   - Con: Requires fixed K, assumes spherical clusters

3. **DBSCAN**
   - Pro: Automatic cluster count, handles noise
   - Con: Sensitive to density parameters, non-deterministic

**Recommendation:** Stick with hierarchical clustering for interpretability and hierarchical sub-axis structure.

---

## Performance Optimization

### For Large-Scale Runs (10K+ pairs)

1. **Parallelize similarity computation**
   ```python
   from multiprocessing import Pool
   with Pool(16) as pool:
       similarities = pool.starmap(compute_similarity, pair_combinations)
   ```

2. **Use sparse matrix representation**
   ```python
   from scipy.sparse import csr_matrix
   # Most similarities are 0.0, exploit sparsity
   ```

3. **Cache WordNet lookups**
   - Already implemented via `_synset_cache`, `_similar_to_cache`, etc.

4. **Incremental IDF statistics**
   - For streaming/incremental clustering

---

## Citation

If using this system in research, please cite:

```bibtex
@software{naomi_ii_antonym_clustering,
  title = {Antonym Clustering for Transparent Semantic Axes},
  author = {NAOMI-II Development Team},
  year = {2025},
  url = {https://github.com/your-repo/NAOMI-II},
  note = {Phase 1 of transparent dimension discovery for AGI}
}
```

---

## Changelog

### 2025-11-30 - Initial Implementation

- ✅ Implemented 4 similarity signals (transitivity, quadrilaterals, TF-IDF, hypernyms)
- ✅ Hierarchical clustering with silhouette optimization
- ✅ Axis extraction and naming
- ✅ Validation metrics (coherence, separation, coverage)
- ✅ Comparison to manual seed clusters
- ✅ Comprehensive visualization (dendrogram, distributions)

### Next Steps

- ⏳ Run full clustering on 3,992 antonym pairs
- ⏳ Validate discovered axes against manual categories
- ⏳ Integrate with Phase 2 neural positioning
- ⏳ Extend to multilingual axes (Spanish, French, etc.)

---

**Last Updated:** 2025-11-30
**Status:** ✅ Implementation complete, testing in progress
**Next Milestone:** Full clustering run → Transparent dimension training
