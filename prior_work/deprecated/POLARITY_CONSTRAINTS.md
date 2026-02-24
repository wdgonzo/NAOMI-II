# Polarity Constraint System

## Overview

This document describes the polarity constraint system implemented to enable compositional semantics in word embeddings through opposite-sign constraints on antonym pairs.

## Key Insight

**Antonyms should be similar in MOST dimensions but have OPPOSITE SIGNS on a FEW key dimensions.**

Example:
- "good" and "bad" are both moral concepts (similar dimensions)
- But "good" is positive morality while "bad" is negative (opposite sign on morality dimension)

This enables compositional semantics:
- **NOT(good) ≈ bad** by flipping signs on polarity dimensions
- **NOT(hot) ≈ cold**
- Logical operators work through vector operations

## Implementation

### 1. Polarity Discovery ([src/embeddings/polarity_discovery.py](src/embeddings/polarity_discovery.py))

Automatically discovers dimensions where antonyms should have opposite signs.

**Algorithm:**
```python
# For each dimension:
1. Calculate signed differences for all antonym pairs
2. Measure discriminative power (mean absolute difference)
3. Measure sign consistency (do all pairs differ in same direction?)
4. Score = discriminative_power × sign_consistency
5. Return top K dimensions
```

**Usage:**
```python
from src.embeddings.polarity_discovery import PolarityDimensionDiscovery

discovery = PolarityDimensionDiscovery(model)
polarity_dims = discovery.discover_polarity_dimensions(
    antonym_pairs=[("good", "bad"), ("hot", "cold"), ...],
    top_k=10,
    min_consistency=0.6
)
```

### 2. Polarity Constraints ([src/embeddings/constraints.py](src/embeddings/constraints.py))

Added `PolarityConstraint` class that enforces opposite-sign relationships.

**Loss Function:**
```python
def compute_loss(self, emb1, emb2):
    loss = 0.0
    for dim_idx in polarity_dims:
        val1 = emb1[dim_idx]
        val2 = emb2[dim_idx]

        sign_product = sign(val1) × sign(val2)

        if sign_product > 0:
            # Same sign - penalty
            loss += (|val1| + |val2|)
        elif sign_product < 0:
            # Opposite signs - reward
            loss -= (|val1| + |val2|)
        else:
            # One is zero - small penalty
            loss += 0.1

    return loss
```

**Usage:**
```python
constraint_loss = ConstraintLoss()
constraint_loss.set_polarity_dimensions([34, 42, 73, ...])
constraint_loss.add_polarity_constraint("good", "bad", weight=2.0)
```

### 3. Logical Operators ([src/embeddings/logical_operators.py](src/embeddings/logical_operators.py))

Implements NOT/AND/OR operations on embeddings.

**NOT Operator:**
```python
def apply_NOT(embedding):
    result = embedding.copy()
    for dim_idx in polarity_dims:
        result[dim_idx] *= -1  # Flip sign
    return result
```

**AND Operator:**
```python
def apply_AND(emb1, emb2, method='average'):
    if method == 'average':
        return (emb1 + emb2) / 2.0
    elif method == 'weighted':
        # Weight by magnitude
        ...
```

**OR Operator:**
```python
def apply_OR(emb1, emb2, method='max'):
    if method == 'max':
        # Take maximum absolute values
        ...
```

**Usage:**
```python
from src.embeddings.logical_operators import LogicalOperators

operators = LogicalOperators(polarity_dims)

# NOT operation
good_emb = model.get_embedding("good")
bad_approx = operators.apply_NOT(good_emb)

# Compare with actual "bad"
bad_emb = model.get_embedding("bad")
similarity = cosine_similarity(bad_approx, bad_emb)
```

### 4. Two-Stage Training ([scripts/train_with_polarity.py](scripts/train_with_polarity.py))

**Stage 1: Pre-train with distance constraints**
- Train with fuzzy distance constraints from WordNet
- Establish basic semantic relationships
- No polarity constraints yet

**Stage 2: Fine-tune with polarity constraints**
- Discover polarity dimensions from pre-trained embeddings
- Add polarity constraints for antonym pairs
- Fine-tune with both distance and polarity losses
- Uses lower learning rate to preserve learned structure

**Usage:**
```bash
python scripts/train_with_polarity.py \
    --knowledge-graph data/knowledge_graph.pkl \
    --stage1-epochs 100 \
    --stage1-lr 0.001 \
    --stage2-epochs 50 \
    --stage2-lr 0.0005 \
    --num-polarity-dims 10 \
    --output-dir checkpoints_polarity \
    --test-logical-ops
```

### 5. Testing ([scripts/test_polarity_constraints.py](scripts/test_polarity_constraints.py))

Comprehensive testing suite that validates:
1. Polarity dimension discovery
2. NOT operation accuracy (NOT(good) ≈ bad)
3. Sign structure analysis
4. Before/after comparison

**Usage:**
```bash
# Test with discovered polarity dimensions
python scripts/test_polarity_constraints.py \
    --checkpoint checkpoints_polarity/stage2_polarity

# Test with saved polarity dimensions
python scripts/test_polarity_constraints.py \
    --checkpoint checkpoints_polarity/stage2_polarity \
    --polarity-dims-file checkpoints_polarity/polarity_dimensions.json

# Compare before and after polarity training
python scripts/test_polarity_constraints.py \
    --checkpoint checkpoints_polarity/stage2_polarity \
    --compare-before checkpoints_polarity/stage1_pretrain \
    --polarity-dims-file checkpoints_polarity/polarity_dimensions.json
```

## Benefits

1. **Compositional Semantics**: Logical operators work through vector operations
2. **Interpretable Dimensions**: Polarity dimensions encode clear semantic axes
3. **Dimension Consolidation**: Fewer, stronger semantic dimensions
4. **Sign-Based Clustering**: Related concepts cluster by sign (positive/negative)

## Expected Results

After polarity training:
- NOT(good) ≈ bad with >0.7 similarity
- NOT(hot) ≈ cold with >0.7 similarity
- Polarity dimensions show clear bipolar structure
- 60-80% of antonym pairs have opposite signs on polarity dimensions

## Integration with Existing System

The polarity constraint system integrates seamlessly:
- Works with existing `ConstraintLoss` class
- Compatible with WordNet-based constraints
- Uses same training infrastructure
- Can be toggled on/off via configuration

## Files Modified/Created

### Created:
- `src/embeddings/polarity_discovery.py` - Automatic polarity dimension discovery
- `src/embeddings/logical_operators.py` - NOT/AND/OR operators
- `scripts/train_with_polarity.py` - Two-stage training script
- `scripts/test_polarity_constraints.py` - Comprehensive testing

### Modified:
- `src/embeddings/constraints.py` - Added PolarityConstraint class and polarity support

## Next Steps

1. **Run two-stage training** on the 1000-sentence corpus:
   ```bash
   python scripts/train_with_polarity.py \
       --knowledge-graph data/knowledge_graph.pkl \
       --stage1-epochs 100 \
       --stage2-epochs 50 \
       --output-dir checkpoints_polarity
   ```

2. **Test polarity constraints**:
   ```bash
   python scripts/test_polarity_constraints.py \
       --checkpoint checkpoints_polarity/stage2_polarity \
       --compare-before checkpoints_polarity/stage1_pretrain \
       --polarity-dims-file checkpoints_polarity/polarity_dimensions.json
   ```

3. **Analyze dimensions**:
   ```bash
   python scripts/analyze_dimensions.py \
       --checkpoint-dir checkpoints_polarity/stage2_polarity
   ```

4. **Test embedding quality**:
   ```bash
   python scripts/test_embeddings.py \
       --checkpoint checkpoints_polarity/stage2_polarity
   ```

## References

- Original insight from user discussion on antonym structure
- Motivated by desire for compositional semantics (NOT operator)
- Based on observation that antonyms are semantically similar except for polarity
