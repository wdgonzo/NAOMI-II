# NAOMI-II Transparent Dimension Training - Implementation Outline

**Date:** 2025-11-29
**Prepared for:** Uploading to Google Colab for testing
**Estimated Time:** 2-3 hours setup + 12-24 hours training

---

## 📋 Executive Summary

You have **TWO OPTIONS** for training transparent semantic dimensions:

### **Option A: WordNet Bootstrap (RECOMMENDED - Start Here)**
- **What:** Train on pure WordNet relations to discover 10-20 interpretable dimensions
- **Why:** Clean semantic structure, no Wikipedia noise, cheaper to test
- **Data:** ~150K-200K sense-tagged words, 524 antonym pairs
- **Time:** 15 minutes on A100 ($0.50)
- **Notebook:** `NAOMI_WordNet_Bootstrap.ipynb` (NEW - creating now)
- **Success Rate:** High confidence (fixed the root cause)

### **Option B: Wikipedia Full Training (Original Approach)**
- **What:** Train on 100K Wikipedia + WordNet together
- **Why:** Richer context, larger vocabulary (197K)
- **Data:** Already prepared and uploaded to your Google Drive
- **Time:** 10-12 hours on A100 ($7-8)
- **Notebook:** `NAOMI_100K_Bootstrap_Training.ipynb` (NEEDS MANUAL FIX)
- **Success Rate:** Lower (still has broken polarity loss in Cell 14)

**RECOMMENDATION:** Start with Option A to validate the fix works, then scale to Option B.

---

## 🚀 Quick Start (Option A - Recommended)

### Prerequisites

1. **Google Colab Pro+** ($50/month for A100 access)
2. **Google Drive** (free 15GB sufficient)
3. **NAOMI-II repository** (will clone in notebook)

### Step-by-Step Workflow

#### **Phase 1: Setup (5 minutes)**

1. Upload `NAOMI_WordNet_Bootstrap.ipynb` to Google Drive
2. Open in Colab: Right-click → Open with → Google Colaboratory
3. Change runtime: Runtime → Change runtime type → Hardware accelerator → GPU → A100
4. Run Cell 1: Verify A100 GPU detected

#### **Phase 2: Build WordNet Graph (10 minutes on Colab)**

**Option 2A: Build in Colab (automatic)**
- Run Cell 2: Clone NAOMI-II repo
- Run Cell 3: Mount Google Drive
- Run Cell 4: Build WordNet graph directly in Colab
  - Extracts 150K-200K sense-tagged words
  - Builds 500K-800K semantic relations
  - Saves to `/content/data/wordnet_only_graph/`

**Option 2B: Build locally and upload (if you prefer)**
```bash
# On your local machine:
python scripts/extract_wordnet_only_graph.py \
  --output-dir data/wordnet_only_graph

# Upload to Google Drive:
# My Drive/NAOMI-II-data/wordnet_only_graph/
```

#### **Phase 3: Train Phase 1 Bootstrap (15 minutes on A100)**

- Run Cell 5: Load WordNet graph
- Run Cell 6: Configure training (polarity weight 10.0, 200 epochs)
- Run Cell 7: Initialize model with anchor dimensions
- Run Cell 8: Start training

**Monitor for:**
```
Epoch 20/200:
  Train Loss: 0.2341
  Val Loss: 0.2356
  Polarity Loss: 0.2341  ← NON-ZERO! ✓
  [Polarity Discovery] Found 8 polarity dimensions ← SUCCESS!
```

**If polarity loss is still 0, see Troubleshooting section below.**

#### **Phase 4: Validate & Visualize (5 minutes)**

- Run Cell 9: Test polarity structure (5 automated tests)
- Run Cell 10: Generate visualizations
- Run Cell 11: Save results to Google Drive

**Expected Output:**
```
TEST 1: POLARITY DIMENSION DISCOVERY
✓ Found 15 polarity dimensions

TEST 2: COMPOSITIONAL SEMANTICS (NOT OPERATION)
  NOT(good):
    1. bad ← TARGET!
  ✓ PASS: 'bad' found at rank 1

Tests passed: 5/5
🎉 ALL TESTS PASSED!
```

#### **Phase 5: Download Results**

Results automatically saved to: `My Drive/NAOMI-II-results/wordnet_bootstrap/`

Files:
- `embeddings_best.npy` - Trained embeddings
- `polarity_dimensions.json` - Discovered dimensions
- `training_history.json` - Training curves
- `semantic_axes_report.txt` - Interpretability analysis

---

## 📝 Option B: Wikipedia Full Training (Manual Fix Required)

### Why Manual Fix Needed

The existing notebook `NAOMI_100K_Bootstrap_Training.ipynb` has the broken polarity loss in Cell 14. You need to manually update it before running.

### Manual Fix Instructions

**File:** `colab-results/NAOMI_100K_Bootstrap_Training.ipynb`
**Cell:** Cell 14 (search for `def selective_polarity_loss`)

**Find this code (BROKEN):**
```python
def selective_polarity_loss(self, embeddings, polarity_dims=None):
    # ...
    # Phase 1: Basic opposite-sign penalty
    if not self.polarity_dims_enabled or polarity_dims is None:
        sign_product = torch.sign(emb1) * torch.sign(emb2)
        polarity_loss = torch.mean(torch.abs(emb1) * torch.abs(emb2) * (sign_product > 0).float())
        # ↑ PROBLEM: Multiplies tiny magnitudes → vanishing gradients
```

**Replace with (FIXED):**
```python
def selective_polarity_loss(self, embeddings, polarity_dims=None):
    """
    ✅ FIXED: Magnitude-independent polarity loss using cosine similarity.
    """
    if len(self.antonym_indices) == 0:
        return torch.tensor(0.0, device=self.device)

    idx1 = self.antonym_indices[:, 0]
    idx2 = self.antonym_indices[:, 1]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    # Phase 1: Use cosine similarity (magnitude-independent)
    if not self.polarity_dims_enabled or polarity_dims is None:
        # Cosine similarity: -1 (opposite) to +1 (same direction)
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)

        # Penalize positive cosine (same direction), reward negative (opposite)
        polarity_loss = torch.mean(cosine_sim)

    # Phase 2: Selective opposition on discovered dims
    else:
        polarity_loss = torch.tensor(0.0, device=self.device)

        # Polarity dimensions: should oppose
        emb1_polar = emb1[:, polarity_dims]
        emb2_polar = emb2[:, polarity_dims]
        cosine_polar = F.cosine_similarity(emb1_polar, emb2_polar, dim=1)
        polarity_loss += torch.mean(cosine_polar)

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

**Also update Cell 12 CONFIG:**
```python
CONFIG = {
    # ... (keep other settings)

    'polarity_weight': 10.0,       # ← CHANGE from 3.0 to 10.0
    'sparsity_weight': 0.0001,     # ← CHANGE from 0.005 to 0.0001
    'polarity_min_consistency': 0.15,  # ← CHANGE from 0.3 to 0.15
}
```

### After Manual Fix

1. Upload fixed notebook to Google Drive
2. Open in Colab
3. Follow same workflow as Option A (Phase 1-5)
4. Training time: 10-12 hours (vs 15 minutes for WordNet-only)

---

## 🔧 Detailed Configuration

### Training Hyperparameters

| Parameter | WordNet Bootstrap | Wikipedia Full | Explanation |
|-----------|-------------------|----------------|-------------|
| **polarity_weight** | 10.0 | 10.0 | High to force discovery |
| **sparsity_weight** | 0.0001 | 0.0001 | Low to allow magnitude growth |
| **learning_rate** | 0.001 | 0.01 | Conservative for WordNet |
| **epochs** | 200 | 150 | More epochs for discovery |
| **batch_size** | 1024 | 262144 | Smaller for WordNet (less data) |
| **embedding_dim** | 128 | 256 | Start conservative |
| **polarity_min_consistency** | 0.15 | 0.15 | Lenient early discovery (15%) |
| **preserve_anchors** | True | True | Never train first 51 dims |

### Loss Function Weights

**WordNet Bootstrap:**
```python
total_loss = (
    distance_loss +                    # Semantic relations (weight=1.0 implicit)
    0.0001 * sparsity_loss +          # Very low - allow magnitude growth
    10.0 * polarity_loss +            # Very high - force discovery!
    0.05 * regularization_loss
)
```

**Wikipedia Full:**
```python
total_loss = (
    0.3 * parse_loss +                 # Parse-derived relations
    0.7 * wordnet_loss +              # WordNet relations (emphasized)
    0.0001 * sparsity_loss +          # Very low early on
    10.0 * polarity_loss +            # Force discovery
    0.05 * regularization_loss
)
```

---

## 📊 Expected Training Progress

### Successful Training (What You Want to See)

**Epoch 1-20: Initial Exploration**
```
Epoch 10:
  Polarity Loss: 0.3421  ← NON-ZERO! ✓
  Sparsity: 15.2%
  No polarity dims discovered yet (normal - needs more training)
```

**Epoch 20-50: Discovery Phase**
```
Epoch 20:
  Polarity Loss: 0.2841
  Sparsity: 22.1%
  [Polarity Discovery] Found 5 polarity dimensions ← BREAKTHROUGH!

Epoch 40:
  Polarity Loss: 0.1923
  Sparsity: 28.5%
  [Polarity Discovery] Found 12 polarity dimensions ← GROWING!
  [Phase 2 Activated] Switching to selective polarity loss
```

**Epoch 50-100: Reinforcement Phase**
```
Epoch 60:
  Polarity Loss: 0.1234
  Sparsity: 35.2%
  [Polarity Discovery] Found 15 polarity dimensions ← STABLE!

Epoch 100:
  Polarity Loss: 0.0821
  Sparsity: 42.1% ← In target range (40-70%)!
  [Polarity Discovery] Found 18 polarity dimensions
```

**Epoch 100-200: Convergence**
```
Epoch 150:
  Polarity Loss: 0.0612
  Sparsity: 48.7%
  [Polarity Discovery] Found 18 polarity dimensions
  ✓ New best model!

Epoch 180:
  No improvement for 20 epochs
  [Early Stopping] Training complete
```

### Failed Training (What to Avoid)

**Red Flags:**

```
Epoch 20:
  Polarity Loss: 0.0000  ← STILL ZERO! ❌ BROKEN!
  [Polarity Discovery] No polarity dimensions found

Epoch 40:
  Polarity Loss: 0.0000  ← STILL ZERO!
  Sparsity: 65.3% ← Too high too early

→ DIAGNOSIS: Polarity loss not working
→ FIX: Check if cosine similarity fix was applied
```

```
Epoch 100:
  Polarity Loss: 0.0234  ← Non-zero but low
  Sparsity: 8.2% ← Way too low (should be 40-70%)
  [Polarity Discovery] Found 3 polarity dimensions ← Not enough

→ DIAGNOSIS: Sparsity too low, not enough polarity dims
→ FIX: Increase sparsity weight to 0.001, train longer
```

---

## 🧪 Validation Tests

### Test 1: Polarity Discovery (Automated)

**Pass Criteria:**
- ✓ 5-20 polarity dimensions discovered
- ✓ Sign consistency >15% per dimension
- ✓ Variance >0.01 (discriminative power)

**Example Output:**
```
TEST 1: POLARITY DIMENSION DISCOVERY
✓ Found 15 polarity dimensions: [42, 73, 18, 91, ...]
  Dim 42: variance=0.2841, activation=34.2%
  Dim 73: variance=0.2105, activation=28.7%
✓ PASS
```

### Test 2: Compositional Semantics (Automated)

**Pass Criteria:**
- ✓ NOT(good) finds 'bad' in top-5 neighbors
- ✓ At least 60% success rate across test pairs

**Example Output:**
```
TEST 2: COMPOSITIONAL SEMANTICS (NOT OPERATION)
  NOT(good):
    1. bad (dist: 0.123) ← TARGET!
    2. evil (dist: 0.245)
    3. terrible (dist: 0.287)
  ✓ PASS: 'bad' found at rank 1

Overall success rate: 80% (4/5)
✓ COMPOSITIONAL SEMANTICS WORKING!
```

### Test 3: Selective Polarity (Automated)

**Pass Criteria:**
- ✓ Different antonym types use different dimensions
- ✓ Overlap ratio <50%

**Example Output:**
```
TEST 3: SELECTIVE POLARITY
  Morality antonyms (good/bad):
    Dim 42: score=1.523
  Temperature antonyms (hot/cold):
    Dim 73: score=1.401

  Dimension overlap ratio: 28.3%
  ✓ EXCELLENT: Different antonym types use different dimensions!
```

### Test 4: Sparsity (Automated)

**Pass Criteria:**
- ✓ Overall sparsity 40-70%
- ✓ Most dimensions show selective usage

**Example Output:**
```
TEST 4: SPARSITY
  Overall sparsity: 48.7%
  ✓ PASS: Sparsity in target range (40-70%)

  Dimension categories:
    Unused (<10%):    12 dims
    Sparse (10-30%):  45 dims
    Moderate (30-60%): 58 dims
    Saturated (>60%):  13 dims
  ✓ GOOD: Most dimensions show selective usage!
```

### Test 5: Dimensional Consistency (Manual)

**Pass Criteria:**
- Can identify what each dimension represents
- Positive/negative poles have consistent semantic meaning

**Example Output:**
```
TEST 5: DIMENSIONAL CONSISTENCY
  Dimension 42:
    Positive pole: good, excellent, wonderful, great
    Negative pole: bad, terrible, awful, horrible
    → Interpretation: MORALITY axis ✓

  Dimension 73:
    Positive pole: hot, warm, burning, heat
    Negative pole: cold, cool, freezing, ice
    → Interpretation: TEMPERATURE axis ✓
```

---

## 🐛 Troubleshooting Guide

### Problem 1: Polarity Loss Still Returns 0

**Symptoms:**
```
Epoch 20:
  Polarity Loss: 0.0000
  [Polarity Pairs] Matched 524/971 antonym pairs
```

**Diagnosis Steps:**

1. **Check if fix was applied:**
   ```python
   # In Cell 14, search for "cosine_similarity"
   # Should see: cosine_sim = F.cosine_similarity(emb1, emb2, dim=1)
   ```

2. **Check antonym pair matching:**
   ```
   [Polarity Pairs] Matched 0/971 antonym pairs
   # ↑ If matched = 0, antonyms not in vocabulary!
   ```

3. **Check embedding magnitudes:**
   ```python
   # Add to training loop:
   print(f"Mean |emb|: {torch.mean(torch.abs(model.embeddings)).item():.4f}")
   # Should be >0.01 by epoch 10
   ```

**Fixes:**

- **If fix not applied:** Apply cosine similarity fix to Cell 14
- **If matched = 0:** Check WordNet graph build (antonyms missing)
- **If magnitudes too small:** Decrease sparsity weight to 0.00001

### Problem 2: No Polarity Dimensions Discovered

**Symptoms:**
```
Epoch 40:
  Polarity Loss: 0.2341  ← Non-zero (good!)
  [Polarity Discovery] No consistent polarity dimensions found
```

**Diagnosis:**

1. Check discovery threshold:
   ```python
   # In Cell 14, discover_polarity_dimensions()
   # min_consistency should be 0.15 (not 0.3)
   ```

2. Check polarity scores manually:
   ```python
   # Add debug print to discover_polarity_dimensions():
   print(f"Top 5 polarity scores: {polarity_scores[:5]}")
   # Should see scores >0.01
   ```

**Fixes:**

- Lower `min_consistency` to 0.10 or 0.15
- Increase `polarity_weight` to 20.0 or 50.0
- Train longer (300-500 epochs)

### Problem 3: Sparsity Out of Range

**Too Low (<40%):**
```
Epoch 100:
  Sparsity: 25.3%  ← Too dense!
```
**Fix:** Increase sparsity weight from 0.0001 to 0.001

**Too High (>70%):**
```
Epoch 100:
  Sparsity: 82.1%  ← Too sparse!
```
**Fix:** Decrease sparsity weight from 0.0001 to 0.00001

### Problem 4: Colab Disconnects

**Prevention:**
- Click in browser every 30 minutes (prevents idle timeout)
- Use Colab Pro+ (longer sessions)
- Enable "Keep-Alive" browser extension

**Recovery:**
- Run Cell 3: Mount Google Drive
- Load latest checkpoint:
  ```python
  checkpoint = torch.load('checkpoints/best_model.pt')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch'] + 1
  ```

---

## 💾 File Organization

### Google Drive Structure

```
My Drive/
├── NAOMI-II-data/
│   ├── wordnet_only_graph/         # Phase 1 input data
│   │   ├── vocabulary.json
│   │   ├── triples.pkl
│   │   ├── training_examples.pkl
│   │   └── graph_stats.json
│   │
│   └── wikipedia_100k_graph/       # Phase 2 input data (if using)
│       ├── vocabulary.json
│       ├── triples.pkl
│       ├── training_examples.pkl
│       └── graph_stats.json
│
└── NAOMI-II-results/
    ├── wordnet_bootstrap/          # Phase 1 results
    │   ├── embeddings_best.npy
    │   ├── polarity_dimensions.json
    │   ├── training_history.json
    │   ├── best_model.pt
    │   └── semantic_axes_report.txt
    │
    └── wikipedia_100k/             # Phase 2 results (if using)
        ├── embeddings_best.npy
        ├── polarity_dimensions.json
        ├── training_history.json
        └── best_model.pt
```

### Local Repository Structure

```
NAOMI-II/
├── scripts/
│   ├── extract_wordnet_only_graph.py    ✅ NEW
│   ├── initialize_anchor_dimensions.py  ✅ NEW
│   ├── train_wordnet_bootstrap.py       ✅ NEW
│   ├── test_polarity_structure.py       ✅ NEW
│   └── visualize_discovered_dimensions.py ✅ NEW
│
├── colab-results/
│   ├── NAOMI_WordNet_Bootstrap.ipynb    ✅ NEW (creating)
│   └── NAOMI_100K_Bootstrap_Training.ipynb (needs manual fix)
│
├── POLARITY_LOSS_FIX.md                  ✅ Fix documentation
├── REFACTORING_SUMMARY.md                ✅ Overview
└── IMPLEMENTATION_OUTLINE.md             ✅ This file
```

---

## 📈 Success Metrics

### Minimum Viable Product (MVP)

**Must Have:**
- [ ] Polarity loss non-zero by epoch 10-20
- [ ] 5-20 polarity dimensions discovered by epoch 50-100
- [ ] NOT(good) ≈ bad works (60%+ success rate)
- [ ] Sparsity 40-70%

**If ALL 4 criteria met:** ✅ **SUCCESS - transparent dimensions working!**

### Stretch Goals

**Nice to Have:**
- [ ] Selective polarity (different dims for different antonym types)
- [ ] Can identify specific semantic axes (morality, temperature, etc.)
- [ ] Compositional operations work (NOT, AND, VERY)
- [ ] Balanced polarity dimensions (30%+ balance +/-)

---

## ⏱️ Timeline & Cost

### Option A: WordNet Bootstrap

| Phase | Time | Cost | Checkpoint |
|-------|------|------|------------|
| Setup notebook | 5 min | $0 | Upload to Colab |
| Build WordNet graph | 10 min | $0 | Graph saved |
| Train Phase 1 | 15 min | $0.50 | Embeddings trained |
| Validate & visualize | 5 min | $0 | Tests passed |
| **TOTAL** | **35 min** | **$0.50** | **MVP complete** |

### Option B: Wikipedia Full

| Phase | Time | Cost | Checkpoint |
|-------|------|------|------------|
| Setup notebook | 5 min | $0 | Upload to Colab |
| Load pre-built graph | 5 min | $0 | Data loaded |
| Train full model | 10-12 hrs | $7-8 | Embeddings trained |
| Validate & visualize | 5 min | $0 | Tests passed |
| **TOTAL** | **10-12 hrs** | **$7-8** | **Full system** |

**RECOMMENDATION:** Start with Option A ($0.50, 35 min) to validate fix, then scale to Option B if needed.

---

## 🎯 Next Actions

### Immediate (Today - 1 hour)

1. **Review this document** - Understand the two options
2. **Choose Option A or B** - WordNet bootstrap vs Wikipedia full
3. **Open Colab notebook** - NAOMI_WordNet_Bootstrap.ipynb (creating next)
4. **Run Phase 1** - Setup + build graph (15 minutes)

### Short-term (This Week)

5. **Train Phase 1 bootstrap** - 15 minutes on A100 ($0.50)
6. **Run validation tests** - Check if polarity dims discovered
7. **Generate visualizations** - Understand discovered dimensions
8. **Review results** - Determine if ready for Phase 2

### Medium-term (Next Week)

9. **Scale to Wikipedia** (optional) - If Phase 1 successful
10. **Benchmark evaluation** - SimLex-999, WordSim-353
11. **Translation tests** - Use for English → Spanish
12. **Production integration** - Add to NAOMI-II pipeline

---

## 📚 Additional Resources

### Documentation

- **[POLARITY_LOSS_FIX.md](POLARITY_LOSS_FIX.md)** - Technical details of the fix
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Complete workflow guide
- **[SEMANTIC_VECTOR_SPACE_GOALS.md](SEMANTIC_VECTOR_SPACE_GOALS.md)** - Design philosophy
- **[docs/INCREMENTAL_LEARNING_DESIGN.md](docs/INCREMENTAL_LEARNING_DESIGN.md)** - 3-graph architecture

### Scripts Reference

All scripts have comprehensive docstrings and usage examples:

```bash
# See help for any script:
python scripts/extract_wordnet_only_graph.py --help
python scripts/train_wordnet_bootstrap.py --help
python scripts/test_polarity_structure.py --help
python scripts/visualize_discovered_dimensions.py --help
```

### Support

**If you encounter issues:**

1. Check [POLARITY_LOSS_FIX.md](POLARITY_LOSS_FIX.md) troubleshooting section
2. Review training logs for error messages
3. Verify configuration matches recommendations
4. Test locally first (cheaper to debug)

---

## ✅ Checklist

Before starting training:

- [ ] Google Colab Pro+ account active
- [ ] A100 GPU runtime selected
- [ ] Google Drive mounted
- [ ] Repository cloned
- [ ] Dependencies installed (torch, numpy, nltk, etc.)
- [ ] WordNet downloaded (nltk.download('wordnet'))
- [ ] Configuration reviewed (polarity weight, sparsity weight)

After training:

- [ ] Polarity loss is non-zero
- [ ] Polarity dimensions discovered (5-20)
- [ ] Validation tests passed (minimum 3/5)
- [ ] Results saved to Google Drive
- [ ] Visualizations generated
- [ ] Ready to scale or deploy

---

**Last Updated:** 2025-11-29
**Status:** Ready for implementation
**Next Step:** Create `NAOMI_WordNet_Bootstrap.ipynb` notebook

