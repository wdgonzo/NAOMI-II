# Semantic Axis Bootstrap Design

**Status:** Implementation In Progress
**Date:** 2025-12-01
**Phase:** Phase 1 - Core Axis Discovery
**Replaces:** Previous loss-based approach (2025-11-29)

---

## Overview

This document describes the complete architecture for **Phase 1: Semantic Axis Discovery**, which automatically discovers 100-120 interpretable semantic dimensions from WordNet antonym pairs and adjectives.

**Key Innovation:** Unlike Word2Vec/BERT with opaque dimensions, this system discovers explicit semantic axes (morality, size, temperature, states_of_matter, etc.) and represents them with clean geometry (binary = 1D, ternary = 2D simplex, n-ary = (n-1)D simplex).

---

## Goals

### Primary Goal
Discover **100-120 core semantic axes** to serve as the bootstrap foundation for transparent dimension learning.

### Success Criteria
- ✅ N-ary pole structure detected (binary, ternary, quaternary, etc.)
- ✅ Simplex geometry computed for all n-ary axes
- ✅ Clean metadata separation (axis definitions vs. word assignments)
- ✅ 200-300 total dimensions allocated
- ✅ Singletons deferred to Phase 3 (compositional inference)
- ✅ Ready for Phase 2 neural positioning

---

## Architecture

### Three-Phase Discovery Process

```
Phase 1: Antonym Pair Clustering
├─ Input: 3,992 WordNet antonym pairs
├─ Method: Hierarchical clustering + silhouette optimization
├─ Output: ~68 multi-pair axes (≥3 pairs each)
└─ Singletons: 3,330 pairs (saved for Phase 3)

Phase 1.5: Adjective Clustering
├─ Input: Adjectives from singleton pairs + WordNet similar_to closure
├─ Method: Louvain community detection on similar_to + hypernym graphs
├─ Output: ~20-40 evaluative axes (morality, size, temperature, etc.)
└─ Pole detection: Split communities using antonym links

Phase 1 Final: Merge & Allocate
├─ Input: Antonym axes + Adjective axes
├─ Method: Resolve overlaps, allocate dimensions
├─ Output: dimension_metadata.json, word_axis_assignments.json, dimension_names.json
└─ Total: ~100-120 axes, ~200-300 dimensions
```

---

## Metadata Structure

### Core Principle
**Separate concerns:** Axis definitions (geometry, poles, metrics) live in `dimension_metadata.json`. Word→axis assignments live in `word_axis_assignments.json`. Dimension labels live in `dimension_names.json`.

### 1. dimension_metadata.json (Axis Definitions)

**Purpose:** Comprehensive axis catalog with geometry, poles, provenance, and metrics.

**Structure:**
```json
{
  "metadata": {
    "total_axes": 112,
    "total_dimensions": 237,
    "anchor_dimensions": 51,
    "learned_dimensions": 186,
    "generation_date": "2025-12-01",
    "phase": "phase_1_bootstrap"
  },
  "axes": [
    {
      "axis_id": 0,
      "name": "morality",
      "type": "binary",
      "source": "adjective_clustering",
      "dimensions": {
        "start": 51,
        "end": 52,
        "count": 1,
        "labels": ["morality"]
      },
      "poles": {
        "count": 2,
        "names": ["good", "bad"],
        "geometry": {
          "good": [1.0],
          "bad": [-1.0]
        }
      },
      "metrics": {
        "size": 47,
        "coherence": 0.72,
        "separation": 0.89
      },
      "representative_pairs": [
        {"word1": "good", "word2": "bad"},
        {"word1": "virtuous", "word2": "wicked"}
      ]
    },
    {
      "axis_id": 1,
      "name": "states_of_matter",
      "type": "ternary",
      "source": "antonym_clustering",
      "dimensions": {
        "start": 52,
        "end": 54,
        "count": 2,
        "labels": ["states_of_matter_0", "states_of_matter_1"]
      },
      "poles": {
        "count": 3,
        "names": ["solid", "liquid", "gas"],
        "geometry": {
          "solid": [1.0, 0.0],
          "liquid": [-0.5, 0.866],
          "gas": [-0.5, -0.866]
        }
      },
      "metrics": {
        "size": 3,
        "coherence": 0.65,
        "separation": 0.71
      },
      "representative_pairs": [
        {"word1": "solid", "word2": "liquid"},
        {"word1": "liquid", "word2": "gas"}
      ]
    },
    {
      "axis_id": 2,
      "name": "color",
      "type": "5-ary",
      "source": "adjective_clustering",
      "dimensions": {
        "start": 54,
        "end": 58,
        "count": 4,
        "labels": ["color_0", "color_1", "color_2", "color_3"]
      },
      "poles": {
        "count": 5,
        "names": ["red", "blue", "green", "yellow", "purple"],
        "geometry": {
          "red": [1.0, 0.0, 0.0, 0.0],
          "blue": [-0.25, 0.968, 0.0, 0.0],
          "green": [-0.25, -0.323, 0.913, 0.0],
          "yellow": [-0.25, -0.323, -0.304, 0.866],
          "purple": [-0.25, -0.323, -0.304, -0.289]
        }
      },
      "metrics": {
        "size": 12,
        "coherence": 0.58,
        "separation": 0.64
      },
      "representative_pairs": [
        {"word1": "red", "word2": "blue"},
        {"word1": "green", "word2": "yellow"}
      ]
    }
  ],
  "dimension_index": {
    "morality": [51],
    "states_of_matter": [52, 53],
    "color": [54, 55, 56, 57],
    "size": [58],
    "temperature": [59]
  }
}
```

**Key Fields:**

- **axis_id**: Unique integer ID (0-indexed)
- **name**: Human-readable axis name (e.g., "morality", "states_of_matter")
- **type**: "binary", "ternary", "quaternary", "5-ary", etc.
- **source**: "antonym_clustering" or "adjective_clustering"
- **dimensions.start/end**: Range of embedding dimensions (half-open interval [start, end))
- **dimensions.count**: Number of dimensions (= n_poles - 1 for simplex)
- **dimensions.labels**: List of dimension names (e.g., ["color_0", "color_1", "color_2", "color_3"])
- **poles.count**: Number of poles (2 for binary, 3 for ternary, etc.)
- **poles.names**: Semantic labels for each pole (e.g., ["red", "blue", "green", "yellow", "purple"])
- **poles.geometry**: Simplex vertex coordinates for each pole (normalized to unit sphere)
- **metrics.size**: Number of antonym pairs or adjectives in cluster
- **metrics.coherence**: WordNet path similarity within poles (0-1)
- **metrics.separation**: 1 - path similarity between poles (0-1)
- **dimension_index**: Quick lookup dict mapping axis_name → dimension_indices

### 2. word_axis_assignments.json (Word→Axis Mapping)

**Purpose:** Which words belong to which axes and poles.

**Structure:**
```json
{
  "metadata": {
    "total_words": 8742,
    "total_axes": 112,
    "generation_date": "2025-12-01"
  },
  "assignments": [
    {
      "axis_id": 0,
      "axis_name": "morality",
      "pole_assignments": {
        "good": ["good", "virtuous", "moral", "righteous", "ethical", "benevolent"],
        "bad": ["bad", "evil", "wicked", "immoral", "unethical", "malevolent"]
      },
      "total_words": 12
    },
    {
      "axis_id": 1,
      "axis_name": "states_of_matter",
      "pole_assignments": {
        "solid": ["solid", "frozen", "crystalline"],
        "liquid": ["liquid", "fluid", "molten"],
        "gas": ["gas", "vapor", "gaseous"]
      },
      "total_words": 9
    }
  ],
  "word_index": {
    "good": [{"axis_id": 0, "axis_name": "morality", "pole": "good"}],
    "bad": [{"axis_id": 0, "axis_name": "morality", "pole": "bad"}],
    "solid": [{"axis_id": 1, "axis_name": "states_of_matter", "pole": "solid"}]
  }
}
```

**Key Fields:**

- **pole_assignments**: Dict mapping pole_name → list of words
- **word_index**: Reverse index for quick lookup (word → axes it belongs to)

### 3. dimension_names.json (Simple List)

**Purpose:** Ordered list of dimension labels for indexing embedding matrix.

**Structure:**
```json
[
  "anchor_0_entity",
  "anchor_1_action",
  "anchor_2_property",
  ...
  "anchor_50_negation",
  "morality",
  "states_of_matter_0",
  "states_of_matter_1",
  "color_0",
  "color_1",
  "color_2",
  "color_3",
  "size",
  "temperature",
  ...
]
```

**Usage:**
```python
# Load dimension names
with open('dimension_names.json') as f:
    dim_names = json.load(f)

# Lookup dimension index by name
morality_idx = dim_names.index("morality")  # → 51
color_dims = [dim_names.index(f"color_{i}") for i in range(4)]  # → [54, 55, 56, 57]

# Access embedding value
embeddings = np.load('embeddings.npy')  # (N_words × N_dims)
word_idx = vocabulary['good']
morality_value = embeddings[word_idx, morality_idx]  # Scalar value on morality axis
```

---

## N-ary Pole Detection

### Algorithm: Graph Community Detection

**Problem:** Antonym pairs may form triangular/n-ary structures, not just binary oppositions.

**Example:**
- early ↔ late ↔ middle (temporal ordering)
- left ↔ center ↔ right (political spectrum)
- solid ↔ liquid ↔ gas (states of matter)

**Solution:** Build word co-occurrence graph from antonym pairs, run Louvain community detection.

**Steps:**

1. **Build graph:**
   ```python
   G = nx.Graph()
   for pair in member_pairs:
       G.add_edge(pair['word1'], pair['word2'], weight=1.0)
   ```

2. **Run Louvain:**
   ```python
   communities = nx.community.louvain_communities(G, resolution=1.0)
   # Each community = one pole
   ```

3. **Assign semantic labels:**
   ```python
   # Use TF-IDF on definitions to name each pole
   for i, community in enumerate(communities):
       pole_name = extract_pole_name(community)  # "solid", "liquid", "gas"
   ```

**Output:** List of pole word sets with semantic labels.

---

## Simplex Geometry

### Regular Simplex Embedding

**Goal:** Represent n poles as vertices of a regular simplex in (n-1)D space.

**Properties:**
- All poles are equidistant (captures mutual opposition)
- Centered at origin (irrelevant = zero vector)
- Normalized to unit sphere (cosine similarity works)

**Formulas:**

**Binary (n=2):**
```python
pole_0 = [+1.0]
pole_1 = [-1.0]
```

**Ternary (n=3):**
```python
pole_0 = [+1.0, 0.0]
pole_1 = [-0.5, +0.866]  # 120° rotation
pole_2 = [-0.5, -0.866]  # 240° rotation
```

**Quaternary (n=4):**
```python
pole_0 = [+1.0, 0.0, 0.0]
pole_1 = [-0.333, +0.943, 0.0]  # 109.47° (tetrahedral angle)
pole_2 = [-0.333, -0.471, +0.816]
pole_3 = [-0.333, -0.471, -0.816]
```

**General (n poles):**
```python
def compute_simplex_vertices(n: int) -> np.ndarray:
    """Compute vertices of regular simplex in (n-1)D space."""
    # Use Gram-Schmidt orthogonalization
    # See implementation in simplex_geometry.py
```

---

## Adjective Clustering (Phase 1.5)

### Goal
Discover evaluative/scalar axes (morality, size, temperature, etc.) that antonym pair clustering missed.

### Why Needed?
- Antonym pair clustering only finds axes with ≥3 **pairs**
- Many important axes have few pairs but many **individual adjectives**
- Example: "morality" axis has 47 adjectives but only 3-4 explicit antonym pairs in WordNet

### Algorithm

**Step 1: Extract adjectives from singleton pairs**
```python
adjectives = set()
for pair in singleton_pairs:
    if is_adjective(pair['word1']):
        adjectives.add(pair['word1'])
    if is_adjective(pair['word2']):
        adjectives.add(pair['word2'])
# ~6,600 adjectives
```

**Step 2: Expand via WordNet similar_to closure**
```python
expanded = set(adjectives)
for adj in adjectives:
    for synset in wn.synsets(adj, pos='a'):  # Adjectives only
        for similar in synset.similar_tos():
            expanded.add(similar.lemmas()[0].name())
# ~18,000 total adjectives
```

**Step 3: Build similarity graph**
```python
G = nx.Graph()
for adj1 in expanded:
    for adj2 in expanded:
        # Compute similarity via:
        # - Jaccard(similar_tos)
        # - Jaccard(hypernym_paths)
        sim = compute_adjective_similarity(adj1, adj2)
        if sim > 0.2:  # Threshold
            G.add_edge(adj1, adj2, weight=sim)
```

**Step 4: Louvain community detection**
```python
communities = nx.community.louvain_communities(G, resolution=1.0)
# Each community = potential axis
```

**Step 5: Detect poles within each community**
```python
for community in communities:
    # Build subgraph of antonym links within community
    antonym_graph = build_antonym_subgraph(community)

    # Run community detection on antonym graph
    # Nodes connected by antonyms belong to OPPOSITE poles
    poles = detect_poles_from_antonyms(antonym_graph)

    # Name each pole via TF-IDF on definitions
    pole_names = [extract_pole_name(pole) for pole in poles]
```

**Output:** List of adjective-based axes with pole assignments.

---

## Dimension Allocation

### Goal
Assign contiguous dimension ranges to each axis, maintaining clean indexing.

### Algorithm

**Step 1: Merge axes from all sources**
```python
all_axes = antonym_axes + adjective_axes
```

**Step 2: Resolve overlaps**
```python
# If two axes share ≥50% of words, merge them
# Keep the one with higher coherence score
merged_axes = resolve_overlaps(all_axes, threshold=0.5)
```

**Step 3: Sort by priority**
```python
# Priority: size (descending), coherence (descending), source (adjective > antonym)
sorted_axes = sorted(merged_axes, key=lambda a: (-a['size'], -a['coherence'], a['source']))
```

**Step 4: Allocate dimensions**
```python
current_dim = 51  # After 51 anchor dimensions
dimension_names = []

for axis in sorted_axes:
    n_poles = len(axis['poles'])
    n_dims = n_poles - 1  # Simplex in (n-1)D space

    axis['dim_start'] = current_dim
    axis['dim_end'] = current_dim + n_dims

    # Generate dimension labels
    if n_dims == 1:
        dimension_names.append(axis['name'])  # "morality"
    else:
        for i in range(n_dims):
            dimension_names.append(f"{axis['name']}_{i}")  # "color_0", "color_1", ...

    current_dim += n_dims
```

**Output:**
- `dimension_metadata.json` with allocated ranges
- `dimension_names.json` with full list
- `word_axis_assignments.json` with pole mappings

---

## Singleton Handling

### Decision: Defer to Phase 3

**Rationale:**
- Most singletons are compositional (leeward = sheltered ∧ from(wind) ∧ side(ship))
- Adding 3,330 singleton dimensions → sparsity nightmare (99.5% sparse)
- Better approach: Infer positions via compositional semantics after bootstrap training

**Phase 3 Approach:**
```python
# After Phase 2 bootstrap training completes
for singleton_pair in singletons:
    # Parse definition
    parse = parser.parse(singleton_pair['definition'])

    # Extract semantic components
    components = extract_components(parse)

    # Compose embedding using trained model
    embedding = compose(
        embeddings[components[0]],
        embeddings[components[1]],
        parse.relations
    )

    # Add to Memory Graph (incremental, no retraining)
    memory_graph.add_word(singleton_pair['word1'], embedding)
```

---

## File Structure

```
data/discovered_axes/
├── dimension_metadata.json        # Axis definitions (this is the CORE file)
├── word_axis_assignments.json     # Word→axis mappings
├── dimension_names.json            # Dimension labels list
├── singleton_pairs.json            # 3,330 singletons (deferred to Phase 3)
├── axis_report.txt                 # Human-readable summary
├── similarity_matrix.npy           # Pairwise similarities (debugging)
├── linkage_matrix.npy              # Hierarchical structure (debugging)
└── cluster_assignments.npy         # Cluster IDs (debugging)
```

---

## Usage Examples

### Example 1: Load Axis Metadata

```python
import json
import numpy as np

# Load dimension metadata
with open('data/discovered_axes/dimension_metadata.json') as f:
    metadata = json.load(f)

# Get axis by name
morality_axis = next(a for a in metadata['axes'] if a['name'] == 'morality')
print(f"Morality axis: dimensions {morality_axis['dimensions']['labels']}")
# → "Morality axis: dimensions ['morality']"

# Get dimension range for n-ary axis
color_axis = next(a for a in metadata['axes'] if a['name'] == 'color')
print(f"Color axis: {color_axis['poles']['count']} poles, {color_axis['dimensions']['count']} dimensions")
print(f"Dimensions: {color_axis['dimensions']['labels']}")
# → "Color axis: 5 poles, 4 dimensions"
# → "Dimensions: ['color_0', 'color_1', 'color_2', 'color_3']"
```

### Example 2: Check Word Assignment

```python
# Load word assignments
with open('data/discovered_axes/word_axis_assignments.json') as f:
    assignments = json.load(f)

# Look up which axes a word belongs to
word = "good"
word_axes = assignments['word_index'][word]
for assignment in word_axes:
    print(f"{word} → {assignment['axis_name']} (pole: {assignment['pole']})")
# → "good → morality (pole: good)"
```

### Example 3: Get Embedding Subspace for Axis

```python
# Load embeddings
embeddings = np.load('checkpoints/embeddings.npy')  # (N_words × N_dims)

# Load dimension names
with open('data/discovered_axes/dimension_names.json') as f:
    dim_names = json.load(f)

# Get color axis dimensions
color_dims = [dim_names.index(f"color_{i}") for i in range(4)]
# → [54, 55, 56, 57]

# Extract color subspace for a word
vocabulary = json.load(open('checkpoints/vocabulary.json'))
word_idx = vocabulary['red']
color_embedding = embeddings[word_idx, color_dims]
# → array([0.92, 0.15, -0.03, 0.08])  # 4D color subspace
```

### Example 4: Compute Pole Coordinates

```python
# Load metadata
metadata = json.load(open('data/discovered_axes/dimension_metadata.json'))

# Get political spectrum axis (ternary)
political_axis = next(a for a in metadata['axes'] if a['name'] == 'political_spectrum')

# Get pole coordinates
left_coords = political_axis['poles']['geometry']['left']
center_coords = political_axis['poles']['geometry']['center']
right_coords = political_axis['poles']['geometry']['right']

print(f"Left: {left_coords}")
print(f"Center: {center_coords}")
print(f"Right: {right_coords}")
# → Left: [1.0, 0.0]
# → Center: [-0.5, 0.866]
# → Right: [-0.5, -0.866]

# Verify equidistant (should all be ≈ 1.5 apart)
from scipy.spatial.distance import cosine
print(cosine(left_coords, center_coords))  # ≈ 1.5
print(cosine(center_coords, right_coords))  # ≈ 1.5
print(cosine(left_coords, right_coords))    # ≈ 1.5
```

---

## Expected Outcomes

### Quantitative

| Metric | Target | Actual (TBD) |
|--------|--------|--------------|
| Total axes | 100-120 | - |
| Antonym axes | 60-80 | - |
| Adjective axes | 20-40 | - |
| Total dimensions | 200-300 | - |
| Binary axes | ~70% | - |
| Ternary axes | ~20% | - |
| 4+ ary axes | ~10% | - |
| Singletons (deferred) | ~3,330 | - |

### Qualitative

✅ **Axis interpretability:** Humans can name 90%+ of axes from pole word lists alone
✅ **Known axes recovered:** morality, size, temperature, emotion, strength, light
✅ **N-ary structure validated:** ternary+ axes make semantic sense (temporal, political, states_of_matter)
✅ **Clean separation:** dimension_metadata doesn't contain word lists (in separate file)
✅ **Robust tracking:** Can easily look up axis→dimensions, word→axis, dimension→axis

---

## Implementation Modules

### New Files

1. **src/embeddings/pole_detection.py** (~200 lines)
   - Graph community detection for n-ary poles
   - Louvain algorithm integration
   - Pole naming via TF-IDF

2. **src/embeddings/simplex_geometry.py** (~150 lines)
   - Compute regular simplex vertices
   - Normalize to unit sphere
   - Handle 2 to 10+ poles

3. **src/embeddings/adjective_clustering.py** (~300 lines)
   - Extract adjectives from singletons
   - Expand via WordNet similar_to
   - Cluster and detect poles

4. **src/embeddings/dimension_allocator.py** (~200 lines)
   - Merge axes from multiple sources
   - Resolve overlaps
   - Allocate dimension indices
   - Generate all metadata files

5. **Updated: src/embeddings/axis_extraction.py** (~100 lines changes)
   - Replace binary pole assumption with n-ary detection
   - New export methods for metadata files

### New Scripts

1. **scripts/cluster_adjectives.py** (~250 lines)
   - Phase 1.5 executable
   - Load singletons, cluster adjectives
   - Output: adjective_axes.json

2. **scripts/merge_and_allocate.py** (~200 lines)
   - Final allocation script
   - Merge antonym + adjective axes
   - Output: dimension_metadata.json, word_axis_assignments.json, dimension_names.json

3. **Updated: scripts/cluster_antonym_axes.py** (~50 lines changes)
   - Add --min-cluster-size parameter
   - Export singletons separately

---

## Next Steps

### After Phase 1 Completes

1. **Phase 2: Bootstrap Training**
   - Train embeddings on discovered axes (100-120 axes, 200-300 dims)
   - Add polarity constraints (antonyms oppose on specific dimensions)
   - Add sparsity targets (40-70% zeros per word)
   - Validate: NOT(good) ≈ bad, big AND round ≈ spherical

2. **Phase 3: Singleton Integration**
   - Parse singleton definitions
   - Extract semantic components
   - Compose embeddings using trained model
   - Add to Memory Graph incrementally

3. **Future: Multilingual Extension**
   - Map discovered axes to Spanish, French, etc.
   - Find corresponding poles in each language
   - Shared embedding space across languages

---

## References

- [ANTONYM_CLUSTERING_IMPLEMENTATION.md](../ANTONYM_CLUSTERING_IMPLEMENTATION.md) - Phase 1 implementation summary
- [docs/ANTONYM_CLUSTERING_GUIDE.md](ANTONYM_CLUSTERING_GUIDE.md) - User guide for clustering system
- [docs/INCREMENTAL_LEARNING_DESIGN.md](INCREMENTAL_LEARNING_DESIGN.md) - 3-graph incremental learning architecture
- [docs/MEMORY_GRAPH_VISION.md](MEMORY_GRAPH_VISION.md) - Long-term vision for reasoning

---

**Last Updated:** 2025-12-01
**Status:** Design complete, implementation in progress
**Next Milestone:** Phase 1 complete → 100-120 axes discovered → Bootstrap training begins
