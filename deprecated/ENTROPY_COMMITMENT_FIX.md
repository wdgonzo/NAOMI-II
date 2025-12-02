# Entropy-Based Commitment Fix

## Problem: Soft Count = 10, Hard Count = 0

**Issue**: Model achieved target of 10.0 soft dims/word but hard count still 0.0 at epoch 58.

**Root Cause**: Model "gaming" the soft count target by spreading activation equally across ~21 dimensions at sigmoid ≈ 0.48-0.50 each, instead of committing 10 dimensions to 1.0.

**Math**:
- 21 dims × sigmoid(~0.0) ≈ 21 × 0.5 = 10.5 soft count ✓ (meets target!)
- But hard count (>0.5 threshold) = 0 ❌ (no interpretability!)

## Solution: Entropy Regularization

Added **commitment loss** that penalizes dimensions hovering at 0.5 (uncertain), forcing them to commit to 0.0 (inactive) or 1.0 (active).

### New Loss Function

```python
def relevance_commitment_loss(model):
    """
    Force dimensions to commit to being active (1.0) or inactive (0.0).

    Uses entropy penalty: sigmoid values near 0.5 have HIGH entropy (uncertain).
    We want LOW entropy (committed).

    Entropy of Bernoulli(p) = -p*log(p) - (1-p)*log(1-p)
    - At p=0.5: entropy = 0.69 (maximum uncertainty)
    - At p=0.0 or p=1.0: entropy = 0 (fully committed)
    """
    relevance = torch.sigmoid(model.relevance_logits[:, model.num_anchors:])

    # Compute entropy for each dimension (higher = less committed)
    # Add epsilon to avoid log(0)
    eps = 1e-8
    entropy = -(relevance * torch.log(relevance + eps) +
                (1 - relevance) * torch.log(1 - relevance + eps))

    # Penalize high entropy (force commitment)
    return torch.mean(entropy)
```

### How It Works

**Entropy penalty creates "valleys" at 0 and 1**:

| Sigmoid Value | Entropy | Gradient |
|--------------|---------|----------|
| 0.01 | 0.056 | Small (stable at 0) |
| 0.25 | 0.562 | Large (push to 0 or 1) |
| **0.50** | **0.693** | **Maximum (strong push!)** |
| 0.75 | 0.562 | Large (push to 1 or 0) |
| 0.99 | 0.056 | Small (stable at 1) |

**Result**: Dimensions get "kicked" away from 0.5 and settle at 0 or 1.

### Changes Made

**1. CONFIG (Cell 5)**:
```python
'relevance_commitment_weight': 0.5,  # NEW: Force binary commitment (0 or 1, not 0.5)
```

**2. Loss Functions (Cell 7)**:
- Added `relevance_commitment_loss()` function
- Updated print statement to mention commitment

**3. Training Loop (Cell 11)**:
```python
# Compute commitment loss
commit_loss = relevance_commitment_loss(model)

# Add to total loss
total = (
    d_loss +
    CONFIG['semantic_clustering_weight'] * sc_loss +
    CONFIG['relevance_coherence_weight'] * rc_loss +
    CONFIG['relevance_sparsity_weight'] * rs_loss +
    CONFIG['relevance_commitment_weight'] * commit_loss +  # NEW
    CONFIG['reg_weight'] * r_loss
)
```

Applied to:
- Mixed precision training branch
- Non-mixed precision training branch
- Validation loop

## Expected Behavior

**Before (gaming the system)**:
```
Epoch 58:
  Avg dims/word (soft): 10.0 (target: 10)  ✓ Target met!
  Avg dims/word (hard): 0.0 (>0.5 threshold)  ❌ No commitment!

# What's happening:
# 21 dimensions at sigmoid ≈ 0.48 each
# 21 × 0.48 = 10.08 soft count (meets target!)
# But none > 0.5 threshold (hard count = 0)
```

**After (with entropy commitment)**:
```
Epoch 60:
  Avg dims/word (soft): 10.2 (target: 10)  ✓ Close to target
  Avg dims/word (hard): 3.5 (>0.5 threshold)  ✓ Starting to commit!

Epoch 80:
  Avg dims/word (soft): 10.1 (target: 10)  ✓ At target
  Avg dims/word (hard): 8.2 (>0.5 threshold)  ✓ Most committed!

Epoch 100:
  Avg dims/word (soft): 10.0 (target: 10)  ✓ Perfect!
  Avg dims/word (hard): 9.8 (>0.5 threshold)  ✓ Nearly all committed!

# What's happening:
# 10 dimensions at sigmoid ≈ 0.95 (active)
# 195 dimensions at sigmoid ≈ 0.05 (inactive)
# 10 × 0.95 + 195 × 0.05 = 10.0 soft count ✓
# 10 dimensions > 0.5 threshold (hard count ≈ 10) ✓
```

## Why This Works

The model now faces **two competing pressures**:

1. **Sparsity loss**: "Keep soft count at 10"
2. **Commitment loss**: "Don't hover at 0.5 - pick a side!"

**Only solution**: Commit exactly 10 dimensions to 1.0, rest to 0.0!

- If too many dims at 1.0: Sparsity loss increases (soft count >10)
- If too few dims at 1.0: Sparsity loss increases (soft count <10)
- If dims hover at 0.5: Commitment loss increases (high entropy)

**Equilibrium**: Exactly 10 dimensions committed to ~0.95-0.99 (low entropy), rest at ~0.01-0.05 (low entropy).

## Technical Details

### Entropy of Bernoulli Distribution

For a binary random variable with probability `p`:
```
H(p) = -p*log(p) - (1-p)*log(1-p)
```

**Properties**:
- Maximum at p=0.5: H(0.5) = 0.693
- Minimum at p=0 or p=1: H(0) = H(1) = 0
- Symmetric around 0.5
- Second derivative negative at 0.5 (unstable equilibrium)
- Second derivative positive at 0 and 1 (stable equilibria)

**Gradient behavior**:
```
dH/dp = -log(p) + log(1-p)
```

- At p=0.5: dH/dp = 0 (but unstable!)
- At p<0.5: dH/dp > 0 (push toward 0 or 1)
- At p>0.5: dH/dp < 0 (push toward 0 or 1)

### Weight Tuning

**Current**: `relevance_commitment_weight = 0.5`

**If convergence too slow**:
- Increase to 1.0 or 2.0 (stronger commitment pressure)

**If instability (dims oscillating)**:
- Decrease to 0.2 or 0.3 (gentler commitment)

**If hard count overshoots target**:
- Commitment weight too high relative to sparsity
- Reduce commitment to 0.3 or increase sparsity to 0.1

## Files Modified

**c:\Users\wdgon\Code\NAOMI-II\colab-results\NAOMI_Semantic_Dimensions_Training.ipynb**

**Changes**:
- Cell 5: Added `relevance_commitment_weight: 0.5` to CONFIG
- Cell 7: Added `relevance_commitment_loss()` function
- Cell 11: Integrated `commit_loss` into training and validation

## Success Criteria

✅ Hard count starts increasing from 0
✅ Hard count reaches 8-10 by epoch 100
✅ Soft count stays close to target 10
✅ Semantic axes become interpretable (clear 0/1 relevance)
✅ Dimensional structure emerges naturally

## Troubleshooting

**If hard count still 0 after 20 more epochs**:
- Commitment weight too low
- Increase to 1.0 or 2.0

**If hard count shoots to 50+**:
- Commitment overwhelming sparsity
- Reduce commitment to 0.2
- Or increase sparsity to 0.1

**If loss becomes NaN**:
- Numerical instability in log(0)
- Already handled with `eps = 1e-8`
- If still happens, increase eps to 1e-6

**If dims oscillate (not stable)**:
- Learning rate too high
- Reduce lr to 0.005
- Or reduce commitment weight to 0.3

---

**Date**: 2025-11-29
**Status**: Ready for testing
**Expected Impact**: Hard count should start increasing immediately, reaching target by epoch 100
