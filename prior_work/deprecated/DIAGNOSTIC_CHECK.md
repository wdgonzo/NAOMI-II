# Diagnostic: Why No Polarity Dimensions at Epoch 40?

## The Problem

Polarity loss is `-0.99` (very negative = antonyms opposing strongly), but no dimensions discovered with consistency >30%.

## Likely Cause: Uniform Opposition

**Theory:** Antonyms are opposing on ALL dimensions equally instead of specializing to 10-20 specific dimensions.

**Why?**
- Polarity weight = 3.0 (very high)
- No selective pressure - model just flips ALL dimensions for antonyms
- No dimensional specialization

**Example:**
```
good:  [0.01, 0.01, 0.01, 0.01, ...]  (all small positive)
bad:   [-0.01, -0.01, -0.01, -0.01, ...]  (all small negative)
```

Result: Every dimension has ~50% consistency (random), none reach 30% threshold.

## Add Diagnostic Cell

Add this cell after epoch 40 to check:

```python
# DIAGNOSTIC: Check dimensional structure
if epoch == 40:
    print("\\n" + "="*70)
    print("DIAGNOSTIC: Checking dimensional opposition pattern")
    print("="*70)

    embeddings_np = model.embeddings.detach().cpu().numpy()
    idx1 = antonym_tensor[:, 0].cpu().numpy()
    idx2 = antonym_tensor[:, 1].cpu().numpy()

    emb1 = embeddings_np[idx1]
    emb2 = embeddings_np[idx2]

    # Check per-dimension statistics
    diffs = emb1 - emb2

    print(f"\\nPer-dimension analysis (first 20 dims):")
    print(f"{'Dim':<5} {'Mean Diff':<12} {'Abs Diff':<12} {'Consistency':<12} {'Pass?'}")
    print("-" * 70)

    for dim in range(min(20, embeddings_np.shape[1])):
        dim_diffs = diffs[:, dim]
        mean_diff = np.mean(dim_diffs)
        abs_diff = np.mean(np.abs(dim_diffs))
        consistency = np.abs(np.mean(np.sign(dim_diffs)))
        passed = "✓" if consistency >= 0.3 else "✗"

        print(f"{dim:<5} {mean_diff:+.6f}    {abs_diff:.6f}      {consistency:.6f}      {passed}")

    # Overall statistics
    all_consistencies = []
    for dim in range(embeddings_np.shape[1]):
        dim_diffs = diffs[:, dim]
        consistency = np.abs(np.mean(np.sign(dim_diffs)))
        all_consistencies.append(consistency)

    all_consistencies = np.array(all_consistencies)

    print(f"\\nOverall dimensional consistency:")
    print(f"  Mean: {np.mean(all_consistencies):.3f}")
    print(f"  Median: {np.median(all_consistencies):.3f}")
    print(f"  Max: {np.max(all_consistencies):.3f}")
    print(f"  Dims >30%: {np.sum(all_consistencies >= 0.3)}")
    print(f"  Dims >20%: {np.sum(all_consistencies >= 0.2)}")
    print(f"  Dims >10%: {np.sum(all_consistencies >= 0.1)}")

    # Check if opposition is uniform
    if np.std(all_consistencies) < 0.05:
        print(f"\\n⚠️  WARNING: Uniform opposition detected!")
        print(f"  All dimensions have similar consistency (~{np.mean(all_consistencies):.2f})")
        print(f"  No dimensional specialization!")
        print(f"\\n  SOLUTION: Reduce polarity_weight to encourage specialization")

    print("="*70)
```

## Solutions

### Solution 1: Lower Polarity Weight (RECOMMENDED)

Current: `polarity_weight = 3.0` → Too strong, forces uniform opposition

Try: `polarity_weight = 1.0` → Allows selective opposition

**Rationale:**
- Lower weight lets distance constraints compete with polarity
- Model learns: "oppose antonyms on SOME dimensions, match on others"
- Encourages dimensional specialization

### Solution 2: Lower Consistency Threshold

Current: `polarity_min_consistency = 0.3` (30%)

Try: `polarity_min_consistency = 0.2` (20%)

**Only if diagnostics show consistencies in 20-30% range**

### Solution 3: Add Selective Polarity Loss (BEST LONG-TERM)

Replace uniform polarity loss with selective version that:
1. Discovers candidate polarity dims (>20% consistency)
2. Applies strong opposition ONLY on those dims
3. Encourages SIMILARITY on non-polarity dims

This is what the old v2.3 "Phase 2" was trying to do.

## Quick Fix to Try Now

If you're at epoch 40 with no dimensions:

1. **Stop training** (it won't improve)
2. **Restart with `polarity_weight = 1.0`** instead of 3.0
3. Train to epoch 60
4. Should see 10-20 dimensions discovered

OR

Continue current run but lower threshold:
- Change `polarity_min_consistency` to `0.15` in the notebook
- Rerun epoch 40 discovery (or wait for epoch 60)

## Expected Diagnostic Output

**If uniform opposition:**
```
Dim   Mean Diff    Abs Diff     Consistency  Pass?
----------------------------------------------------------
0     +0.000123    0.012456      0.145        ✗
1     -0.000089    0.011234      0.132        ✗
2     +0.000201    0.013567      0.156        ✗
...
Overall: Mean: 0.14, Max: 0.19, Dims >30%: 0

⚠️  WARNING: Uniform opposition detected!
```

**If selective opposition (good):**
```
Dim   Mean Diff    Abs Diff     Consistency  Pass?
----------------------------------------------------------
0     +0.000123    0.012456      0.045        ✗
1     -0.000089    0.011234      0.823        ✓  ← Polarity dim!
2     +0.000201    0.013567      0.032        ✗
...
67    +0.045678    0.089234      0.745        ✓  ← Polarity dim!
...
Overall: Mean: 0.08, Max: 0.87, Dims >30%: 12

✓ Selective opposition working!
```
