# NAOMI-II Polarity Loss Fix

## Problem Summary

The current polarity loss in `NAOMI_100K_Bootstrap_Training.ipynb` Cell 14 always returns 0.0000 because:

```python
# BROKEN VERSION (Current):
polarity_loss = torch.mean(torch.abs(emb1) * torch.abs(emb2) * (sign_product > 0).float())
```

**Why it fails:**
1. Embeddings start near zero (initialized with `randn() * 0.01`)
2. Product `|0.01| × |0.01| = 0.0001` (vanishingly small)
3. No gradient signal → no polarity structure emerges
4. Sparsity loss pushes values toward zero, making problem worse

## Solution: Magnitude-Independent Polarity Loss

Replace Cell 14's `selective_polarity_loss()` method with:

```python
def selective_polarity_loss(self, embeddings, polarity_dims=None):
    """
    ✅ FIXED: Magnitude-independent polarity loss using cosine similarity.

    OLD PROBLEM: torch.abs(emb1) * torch.abs(emb2) → vanishing gradients for small magnitudes
    NEW SOLUTION: F.cosine_similarity() → sign-aware, magnitude-independent
    """
    if len(self.antonym_indices) == 0:
        return torch.tensor(0.0, device=self.device)

    idx1 = self.antonym_indices[:, 0]
    idx2 = self.antonym_indices[:, 1]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    # Phase 1: Basic opposite-sign penalty (before discovery activation)
    if not self.polarity_dims_enabled or polarity_dims is None:
        # Use cosine similarity (sign-aware, magnitude-independent)
        # Goal: antonyms should have negative cosine similarity (opposite direction)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)

        # Penalize positive cosine similarity (same direction)
        # Loss = 0 when cosine = -1 (perfect opposition)
        # Loss = 1 when cosine = +1 (same direction - BAD!)
        polarity_loss = torch.mean(cosine_sim)  # Higher = worse

    # Phase 2: Selective opposition on discovered dims (after discovery activation)
    else:
        polarity_loss = torch.tensor(0.0, device=self.device)

        # Polarity dimensions: should oppose (negative cosine)
        emb1_polar = emb1[:, polarity_dims]
        emb2_polar = emb2[:, polarity_dims]
        cosine_polar = F.cosine_similarity(emb1_polar, emb2_polar, dim=1)
        polarity_loss += torch.mean(cosine_polar)  # Penalize positive cosine

        # Non-polarity dimensions: should be similar (close in value)
        all_dims = set(range(embeddings.shape[1]))
        non_polar_dims = list(all_dims - set(polarity_dims.tolist()))
        if non_polar_dims:
            non_polar_dims_tensor = torch.tensor(non_polar_dims, device=self.device)
            emb1_nonpolar = emb1[:, non_polar_dims_tensor]
            emb2_nonpolar = emb2[:, non_polar_dims_tensor]
            similarity_loss = F.mse_loss(emb1_nonpolar, emb2_nonpolar)
            polarity_loss += 0.5 * similarity_loss

    return polarity_loss
```

## Alternative Solution: Sign-Only Penalty

If cosine similarity still doesn't work, try pure sign-based penalty:

```python
def selective_polarity_loss(self, embeddings, polarity_dims=None):
    """
    ✅ ALTERNATIVE: Pure sign-based penalty (no magnitude dependence).
    """
    if len(self.antonym_indices) == 0:
        return torch.tensor(0.0, device=self.device)

    idx1 = self.antonym_indices[:, 0]
    idx2 = self.antonym_indices[:, 1]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    if not self.polarity_dims_enabled or polarity_dims is None:
        # Sign-only penalty (magnitude-independent)
        sign_product = torch.sign(emb1) * torch.sign(emb2)

        # Penalty: reward opposite signs, penalize same signs
        # +1 when same sign (bad), -1 when opposite (good), 0 when one is zero
        same_sign_penalty = torch.mean((sign_product > 0).float())  # Fraction with same sign
        opposite_sign_reward = torch.mean((sign_product < 0).float())  # Fraction with opposite sign

        polarity_loss = same_sign_penalty - opposite_sign_reward  # Range: -1 to +1

    else:
        # Selective polarity on discovered dims
        polarity_loss = torch.tensor(0.0, device=self.device)

        # Polarity dimensions: penalize same signs
        emb1_polar = emb1[:, polarity_dims]
        emb2_polar = emb2[:, polarity_dims]
        sign_product_polar = torch.sign(emb1_polar) * torch.sign(emb2_polar)
        same_sign_penalty = torch.mean((sign_product_polar > 0).float())
        opposite_sign_reward = torch.mean((sign_product_polar < 0).float())
        polarity_loss += same_sign_penalty - opposite_sign_reward

        # Non-polarity dimensions: should be similar
        all_dims = set(range(embeddings.shape[1]))
        non_polar_dims = list(all_dims - set(polarity_dims.tolist()))
        if non_polar_dims:
            non_polar_dims_tensor = torch.tensor(non_polar_dims, device=self.device)
            emb1_nonpolar = emb1[:, non_polar_dims_tensor]
            emb2_nonpolar = emb2[:, non_polar_dims_tensor]
            similarity_loss = F.mse_loss(emb1_nonpolar, emb2_nonpolar)
            polarity_loss += 0.5 * similarity_loss

    return polarity_loss
```

## Also Fix: Discovery Threshold

In Cell 14, lower the `min_consistency` threshold in `discover_polarity_dimensions()`:

```python
def discover_polarity_dimensions(self, embeddings, top_k=20, min_consistency=0.15):  # ← Changed from 0.3 to 0.15
    """
    Discover which dimensions best capture antonym polarity.

    ✅ FIXED: Lowered threshold from 0.30 to 0.15 (15% consistency)
    Early in training, polarity structure is weak - need lenient threshold.
    """
    # ... rest of method unchanged
```

## Expected Results After Fix

**Before Fix:**
```
Epoch 20:
  Train Loss: 0.1234
  Val Loss: 0.1256
  Polarity Loss: 0.0000  ← BROKEN!
  [Polarity Discovery] No consistent polarity dimensions found yet
```

**After Fix:**
```
Epoch 20:
  Train Loss: 0.1234
  Val Loss: 0.1256
  Polarity Loss: 0.2341  ← NON-ZERO! ✓
  [Polarity Discovery] Found 8 polarity dimensions:
    Dim 42: score=0.1523, consistency=0.18, power=0.845
    Dim 73: score=0.1401, consistency=0.16, power=0.876
    ...
```

## Configuration Changes (Cell 12)

Also update CONFIG to give polarity loss more influence:

```python
CONFIG = {
    # ... (keep other settings)

    # Loss weights - ✅ ADJUSTED FOR FIX
    'polarity_weight': 10.0,       # ← Increased from 3.0 (need strong signal early)
    'sparsity_weight': 0.0001,     # ← Decreased from 0.005 (don't suppress magnitudes)

    # Polarity discovery
    'polarity_min_consistency': 0.15,  # ← Lowered from 0.3 (lenient early discovery)
    'polarity_discovery_start_epoch': 40,
}
```

## Implementation Steps

1. **In Colab Notebook** (`NAOMI_100K_Bootstrap_Training.ipynb`):
   - Edit Cell 14: Replace `selective_polarity_loss()` method with FIXED version
   - Edit Cell 14: Change `min_consistency=0.3` to `min_consistency=0.15`
   - Edit Cell 12: Update CONFIG weights (polarity=10.0, sparsity=0.0001)
   - Save notebook

2. **Test Locally First** (optional):
   - Run `scripts/train_wordnet_bootstrap.py` (already has fixed loss)
   - Verify polarity loss is non-zero
   - Verify dimensions are discovered

3. **Run on Colab**:
   - Upload fixed notebook
   - Run full training (12-15 hours on A100)
   - Monitor polarity loss values (should be non-zero by epoch 10-20)
   - Check dimension discovery at epoch 20, 40, 60

## Why This Fix Works

**Cosine Similarity Approach:**
- `F.cosine_similarity(a, b)` = `dot(a, b) / (||a|| * ||b||)`
- Normalized by magnitude → magnitude-independent
- Range: -1 (opposite) to +1 (same direction)
- When embeddings are small (0.01), cosine still measures direction
- Gradient exists even for tiny magnitudes
- Forces model to organize by direction, not just magnitude

**Sign-Only Approach:**
- Counts fraction of dimensions with same vs opposite signs
- Completely ignores magnitude
- Simpler, more direct
- May be noisier early in training (many zeros → sign=0)

**Recommendation:** Try cosine similarity first (more robust), fall back to sign-only if needed.

## Verification Checklist

After implementing fix:

- [ ] Polarity loss is non-zero by epoch 10-20
- [ ] Polarity loss shows gradual decrease (learning working)
- [ ] Polarity dimensions discovered by epoch 20-40
- [ ] At least 5-10 dimensions with consistency >15%
- [ ] `NOT(good) ≈ bad` works after training
- [ ] Sparsity stays in 40-70% range (not too low)
- [ ] No dimension over-polarization (all dims = polarity)

## Files to Update

1. **`colab-results/NAOMI_100K_Bootstrap_Training.ipynb`** - Cell 14 + Cell 12
2. **`scripts/train_wordnet_bootstrap.py`** - Already has fix ✓
3. **`colab-results/NAOMI_WordNet_Bootstrap.ipynb`** - Create new with fix

---

**Last Updated:** 2025-11-29
**Status:** Fix implemented in `train_wordnet_bootstrap.py`, needs manual application to notebook
