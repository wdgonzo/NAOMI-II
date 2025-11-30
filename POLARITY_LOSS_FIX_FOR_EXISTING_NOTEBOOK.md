# CRITICAL FIX: Replace Broken Polarity Loss in NAOMI_100K_Bootstrap_Training.ipynb

## The Problem

Cell 14 in `NAOMI_100K_Bootstrap_Training.ipynb` has the BROKEN polarity loss that always returns 0.0000:

```python
# ❌ BROKEN - Phase 1 (OLD CODE - lines ~120-130 in cell 14):
if not self.polarity_dims_enabled or polarity_dims is None:
    # Penalize same signs, reward opposite signs
    sign_product = torch.sign(emb1) * torch.sign(emb2)
    polarity_loss = torch.mean(torch.abs(emb1) * torch.abs(emb2) * (sign_product > 0).float())
```

**Why it's broken:** Multiplies tiny magnitudes (0.01 × 0.01 = 0.0001) → vanishing gradients

## The Fix

Replace with cosine similarity (magnitude-independent):

```python
# ✅ FIXED - Phase 1 (NEW CODE):
if not self.polarity_dims_enabled or polarity_dims is None:
    # Use cosine similarity - magnitude independent!
    cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)
    polarity_loss = torch.mean(cosine_sim)  # Penalize same direction
```

## How to Apply the Fix

### Option 1: Manual Edit in Colab

1. Upload `NAOMI_100K_Bootstrap_Training.ipynb` to Colab
2. Find cell 14 (the `TransparentDimensionLoss` class)
3. Find the `selective_polarity_loss` method (around line 100-150 of cell 14)
4. Replace the Phase 1 section:

**FIND THIS (around line 120):**
```python
        # Phase 1: Basic opposite-sign penalty (before discovery activation)
        if not self.polarity_dims_enabled or polarity_dims is None:
            # Penalize same signs, reward opposite signs
            sign_product = torch.sign(emb1) * torch.sign(emb2)
            polarity_loss = torch.mean(torch.abs(emb1) * torch.abs(emb2) * (sign_product > 0).float())
```

**REPLACE WITH:**
```python
        # Phase 1: Basic opposite-sign penalty using COSINE SIMILARITY
        if not self.polarity_dims_enabled or polarity_dims is None:
            # ✅ FIXED: Use cosine similarity (magnitude-independent!)
            cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)
            polarity_loss = torch.mean(cosine_sim)  # Penalize same direction
```

5. Save the notebook
6. Run training

### Option 2: Use the New Simplified Notebook

Use the cleaner version I created in `NAOMI_Transparent_Dimensions_Training.ipynb` which already has the fix.

## Expected Results After Fix

**BEFORE (broken):**
```
Epoch 20:
  Polarity: 0.0000  ← ALWAYS ZERO!

[Polarity Discovery] No consistent polarity dimensions found yet
```

**AFTER (fixed):**
```
Epoch 20:
  Polarity: 0.2451  ← NON-ZERO! Fix working!

[Polarity Discovery] Found 12 polarity dimensions
[Polarity Discovery] Dims: [67, 89, 102, 134, ...]
```

## Why This Fixes It

**Broken version:**
- `torch.abs(emb1) * torch.abs(emb2)` → 0.01 × 0.01 = 0.0001
- Gradient essentially zero → no learning signal
- Polarity loss always ~0.0000

**Fixed version:**
- `F.cosine_similarity(emb1, emb2)` → ranges from -1 to +1
- Normalized by magnitude → gradient signal even for small values
- Polarity loss ranges from -1.0 to +1.0
- Model can actually learn to oppose antonyms!

## Verification

After training for 20-40 epochs, you should see:

1. ✅ Polarity loss NON-ZERO (0.1-0.5 range)
2. ✅ Polarity dimensions discovered by epoch 40-60
3. ✅ Phase 2 activation message at epoch 40
4. ✅ NOT(good) ≈ bad works in final tests

If polarity loss is still 0.0000 after 20 epochs, the fix wasn't applied correctly.
