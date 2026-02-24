# Semantic Vector Space: Core Design Goals

## The Vision: Interpretable Dimensional Meaning

### Fundamental Principle
**Each dimension represents ONE consistent semantic axis across ALL words in the vocabulary.**

- **Dimension 0** might be "morality" for EVERY word
- **Dimension 1** might be "gender" for EVERY word
- **Dimension 2** might be "temperature" for EVERY word
- ... as many dimensions as needed to encode all semantic meaning

### Why This Matters
This is fundamentally different from traditional word embeddings (Word2Vec, GloVe, BERT) where:
- Dimensions are opaque/uninterpretable
- Different words use dimensions inconsistently
- You can't do meaningful arithmetic

Our goal: **Create a semantic coordinate system** where dimensions have consistent, interpretable meanings.

---

## Core Design Requirements

### 1. **Dimensional Consistency** (MOST CRITICAL)

The same dimension MUST mean the same thing for EVERY word.

#### ✅ CORRECT:
```python
# Dimension 0 = "Morality" for ALL words
good[0]  = +0.9   # high positive morality
bad[0]   = -0.9   # high negative morality
evil[0]  = -1.0   # extreme negative morality
boy[0]   = 0.0    # morally neutral
chair[0] = 0.0    # morally neutral
```

#### ❌ WRONG (Dimension Mixing):
```python
# NO! Each word using different dimensions for same concept
boy uses dim 0 for gender
girl uses dim 5 for gender  ← WRONG! Same concept, different dims
```

**Why critical:** If dimensions are inconsistent, we can't:
- Interpret what dimensions mean
- Do meaningful vector arithmetic (NOT, AND, OR)
- Cluster words by semantic properties
- Compare words across dimensions

### 2. **Sparsity Through Irrelevance**

Words should be **ZERO on dimensions that don't apply** to their meaning.

#### Example:
```python
chair = [0.0, 0.0, 0.0, 0.4, 0.0, -0.5, 0.0, 0.2, 0.0, ...]
        ^^^  ^^^  ^^^            ^^^       ^^^
      moral gndr temp          many     many
                               zeros    zeros
```

**Intuition:**
- "chair" has no moral value → `chair[morality_dim] = 0`
- "chair" has no gender → `chair[gender_dim] = 0`
- "chair" might have material/shape properties → `chair[material_dim] ≠ 0`

**Expected sparsity:** Unknown! That's part of what we're discovering.
- Might be 30% active, 70% zero
- Might be 60% active, 40% zero
- The network tells us how many axes of meaning exist

**NOT extreme sparsity:** Each word probably needs 20-60+ dimensions to fully capture its meaning. But NOT all dimensions!

### 3. **Selective Polarity for Antonyms**

Antonym pairs should:
- **Share MOST dimensions** (same sign, same value) — they're similar concepts
- **Oppose on 1-5 dimensions** — where they actually differ semantically

#### Example: good vs bad
```python
good = [+0.9, 0.0, 0.0, 0.4, -0.2, 0.5, 0.3, ...]
bad  = [-0.9, 0.0, 0.0, 0.4, -0.2, 0.5, 0.3, ...]
        ^^^^  ^^^  ^^^  ^^^  ^^^^  ^^^  ^^^
      OPPOSITE!  ALL THE SAME (identical!)
       (moral)
```

- **Dimension 0 (morality)**: OPPOSITE — this is what makes them antonyms!
- **All other dimensions**: SAME — they share semantic properties

#### Example: boy vs girl
```python
boy  = [0.0, +0.8, 0.0, 0.3, 0.1, -0.4, 0.2, ...]
girl = [0.0, -0.8, 0.0, 0.3, 0.1, -0.4, 0.2, ...]
        ^^^  ^^^^  ^^^  ^^^  ^^^  ^^^^  ^^^
       SAME OPPOSITE! ALL THE SAME
           (gender)
```

- **Dimension 0 (morality)**: SAME (both morally neutral)
- **Dimension 1 (gender)**: OPPOSITE — this is what makes them antonyms!
- **All other dimensions**: SAME

#### Key Insight:
**Different antonym pairs use different dimensions for their polarity!**
- good/bad: oppose on morality dimension
- boy/girl: oppose on gender dimension
- hot/cold: oppose on temperature dimension
- big/small: oppose on size dimension

### 4. ❌ **Current Problem: Universal Polarity**

Our current implementation forces ALL antonym pairs to oppose on ALL dimensions:

```python
# What current model produces (WRONG):
boy  = [+1.0, +1.0, +1.0, +1.0, +1.0, ...]  ← all positive
girl = [-1.0, -1.0, -1.0, -1.0, -1.0, ...]  ← all negative

good = [+1.0, +1.0, +1.0, +1.0, +1.0, ...]  ← all positive
bad  = [-1.0, -1.0, -1.0, -1.0, -1.0, ...]  ← all negative
```

**Problems:**
1. ✗ boy and good interfere (both max on dim 1, different meanings)
2. ✗ No sparsity (all dimensions active)
3. ✗ No selective polarity (opposite on everything, not just gender/morality)
4. ✗ No dimensional consistency (gender and morality completely mixed)

---

## Compositional Semantics

### The Goal
Enable **logical operations through vector arithmetic**:

#### Negation
```python
NOT(good) = flip_sign_on_morality_dim(good) ≈ bad
NOT(hot) = flip_sign_on_temperature_dim(hot) ≈ cold
NOT(male) = flip_sign_on_gender_dim(male) ≈ female
```

#### Conjunction
```python
AND(good, brave) = combine_on_shared_dims(...) ≈ "good and brave"
```

#### Modification
```python
VERY(hot) = amplify_temperature_dim(hot) ≈ "very hot"
MALE(doctor) = doctor + offset_on_gender_dim ≈ "male doctor"
```

### Why Dimensional Consistency is Required
For `NOT(good) ≈ bad` to work:
- good and bad must use the SAME dimension for morality
- That dimension must be consistent across all words
- We just flip the sign on that one dimension

If dimensions are inconsistent, we can't know which dimension(s) to flip!

---

## Dynamic Dimensionality

### Starting Point
We begin with **128 dimensions** as an initial guess.

### Discovery Process
Through training, we discover:
- **How many dimensions are actually needed** to encode all semantic meaning
- **What each dimension means** (morality, gender, size, color, etc.)
- **Which words use which dimensions**

### Dimension Expansion
As the vocabulary grows or semantic complexity increases:
- We can **add new dimensions** dynamically
- No fixed limit — use as many as needed
- This is a feature, not a bug!

### Scientific Value
**One of the research goals:** Determine the true dimensionality of semantic meaning.
- How many axes are needed to represent all human concepts?
- 50 dimensions? 200? 500?
- This model will tell us!

---

## Training Implications

### What the Loss Function Must Encourage

1. **Dimensional Consistency**
   - Same semantic axes must map to same dimensions
   - How to enforce? This is the hard part!
   - Antonym/synonym constraints help

2. **Sparsity**
   - Words should be zero on irrelevant dimensions
   - Not extreme (1-5 dims), but not dense (all dims)
   - L1 regularization helps, but must be balanced

3. **Selective Polarity**
   - Antonym pairs oppose on 1-5 relevant dimensions
   - Antonym pairs SHARE values on most dimensions
   - Different pairs use different polarity dimensions

### Current Challenge
The existing `PolarityLoss` violates selective polarity — it forces ALL pairs to oppose on ALL dimensions.

**We need:** A loss that lets the network discover which dimension(s) each antonym pair should use for polarity.

---

## Success Criteria

### After Training, We Should See:

#### 1. Dimensional Interpretability
- Can identify: "Dimension 42 encodes morality"
- Can identify: "Dimension 73 encodes gender"
- Each dimension has a clear, consistent semantic meaning

#### 2. Sparsity Statistics
- Most words have 20-60% non-zero dimensions (rough guess)
- Zero values are truly irrelevant, not just small

#### 3. Antonym Structure
- High correlation (0.7-0.9) between antonyms on most dimensions
- Opposite signs on 1-5 dimensions only
- Different antonym pairs use different polarity dimensions

#### 4. Dimensional Clustering
- All "moral" words cluster along the morality dimension
- All "gendered" words cluster along the gender dimension
- Minimal cross-talk between unrelated dimensions

#### 5. Compositionality
- `NOT(good) ≈ bad` (sign flip works)
- `NOT(hot) ≈ cold` (sign flip works)
- Different NOT operations flip different dimensions

---

## Ideal Example

### Discovered Dimension Meanings
```python
# Network discovers these automatically:
DIM_MORALITY = 0
DIM_GENDER = 1
DIM_TEMPERATURE = 2
DIM_SIZE = 3
DIM_AGE = 4
DIM_ANIMACY = 5
# ... continue for all discovered semantic axes
```

### Word Embeddings
```python
embeddings = {
    # Moral concepts (use morality dim)
    'good':  [+0.9, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
    'bad':   [-0.9, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
    'evil':  [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],

    # Gendered concepts (use gender dim + shared dims)
    'boy':   [0.0, +0.8, 0.0, -0.3, +0.6, +0.9, ...],
    'girl':  [0.0, -0.8, 0.0, -0.3, +0.6, +0.9, ...],
    'man':   [0.0, +0.9, 0.0, +0.4, +0.9, +0.9, ...],
           # moral gndr temp size  age  anim

    # Temperature concepts (use temperature dim)
    'hot':   [0.0, 0.0, +0.9, 0.0, 0.0, 0.0, ...],
    'cold':  [0.0, 0.0, -0.9, 0.0, 0.0, 0.0, ...],
    'warm':  [0.0, 0.0, +0.5, 0.0, 0.0, 0.0, ...],

    # Objects (sparse, no morality/gender/temperature)
    'chair': [0.0, 0.0, 0.0, +0.4, 0.0, -0.5, ...],
           # size=medium, inanimate
}
```

### Operations
```python
NOT(good) = good * [−1, 1, 1, 1, ...] = [−0.9, 0.0, ...] ≈ bad ✓
NOT(boy) = boy * [1, −1, 1, 1, ...] = [0.0, −0.8, ...] ≈ girl ✓
NOT(hot) = hot * [1, 1, −1, 1, ...] = [0.0, 0.0, −0.9, ...] ≈ cold ✓
```

---

## Next Steps: Fixing the Loss Function

### Goal
Design a loss function that:
1. ✓ Encourages sparsity (zeros on irrelevant dimensions)
2. ✓ Enforces selective polarity (opposite on relevant dims only)
3. ✓ Maintains dimensional consistency (same axes → same dimensions)
4. ✓ Allows dimension discovery (network chooses which dims for which concepts)

### Challenge
Balancing these objectives is non-trivial. We need the network to:
- Learn which dimensions represent which semantic axes
- Assign antonym pairs to appropriate dimensions
- Keep unrelated dimensions at zero
- Maintain consistency across the vocabulary

This requires careful loss function design + hyperparameter tuning.
