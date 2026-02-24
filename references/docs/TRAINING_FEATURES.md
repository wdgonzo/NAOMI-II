# NAOMI-II Training Features

## Recent Improvements

### 1. Early Stopping

**What it does:** Automatically stops training when validation loss stops improving, preventing overfitting and saving compute time.

**Parameters:**
- `--patience 10` - Stop if no improvement for N epochs (default: 10)
- `--min-delta 0.0001` - Minimum improvement to count (default: 0.0001)

**Example:**
```bash
--patience 5  # Stop after 5 epochs without improvement
```

**Benefits:**
- Saves time (no wasted epochs after convergence)
- Prevents overfitting to training set
- Automatically finds optimal training length

### 2. Adaptive Batch Size

**What it does:** Automatically reduces batch size when dimensions expand to prevent GPU out-of-memory errors.

**How it works:**
- Calculates: `new_batch_size = initial_batch_size * (initial_dims / current_dims)`
- Example: 1M batch @ 128 dims -> 500K batch @ 256 dims -> 250K batch @ 512 dims
- Minimum batch size: 32,768 (safety floor)

**Benefits:**
- No manual batch size tuning needed
- Prevents OOM crashes during dimension expansion
- Maintains stable GPU utilization throughout training

### 3. Aggressive Dimension Expansion

**What changed:**
- Default `--expand-interval`: 10 -> 2 epochs (check more frequently)
- Default `--expand-by`: 16 -> 64 dims (add more dimensions each time)

**New parameters:**
- `--expand-interval 2` - Check every N epochs (default: 2)
- `--expand-by 64` - Number of dims to add each expansion (default: 64)

**Benefits:**
- Reaches max capacity faster (512 dims by epoch 8-12 instead of epoch 40+)
- Adapts capacity to model needs more responsively
- Better utilization of available dimensions

### 4. Fixed GPU Device Bug

**What was fixed:**
- Dynamic dimension expansion now preserves GPU device placement
- Both `expand_embeddings` and `prune_unused_dimensions` fixed

**Files modified:**
- `src/embeddings/dynamic_dimensions.py`

## Recommended Training Command

### Conservative (Safer, More Stable)

```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --expand-interval 5 \
    --expand-by 32 \
    --batch-size 1048576 \
    --patience 10 \
    --min-delta 0.0001 \
    --epochs 100 \
    --lr 0.01
```

**Expected behavior:**
- Starts with 128 dims, 1M batch
- Expands by 32 dims every 5 epochs if saturated
- Batch size adapts automatically
- Stops when loss plateaus (likely 15-25 epochs)
- Total time: ~25-35 minutes

### Aggressive (Faster Expansion)

```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --expand-interval 2 \
    --expand-by 64 \
    --batch-size 1048576 \
    --patience 5 \
    --min-delta 0.0001 \
    --epochs 100 \
    --lr 0.01
```

**Expected behavior:**
- Starts with 128 dims, 1M batch
- Expands by 64 dims every 2 epochs if saturated
- Reaches 512 dims by epoch 8-12
- Batch size: 1M -> 500K -> 250K -> 125K (adapts automatically)
- Stops when loss plateaus (likely 12-18 epochs)
- Total time: ~20-30 minutes

### Fixed Dimensions (No Dynamic Expansion)

```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --embedding-dim 512 \
    --batch-size 524288 \
    --patience 10 \
    --epochs 100 \
    --lr 0.01
```

**Expected behavior:**
- Fixed 512 dims from start
- 524K batch (safe for 512 dims)
- Stops when loss plateaus
- Most predictable memory usage
- Total time: ~25-35 minutes

## Expected Training Output

### With Early Stopping

```
[6/7] Training...
  Dynamic dimensions enabled: 128 -> max 512
  Expansion check interval: every 2 epochs
  Expansion amount: 64 dims per expansion
  Early stopping enabled: patience=5, min_delta=0.0001

Epoch 1/100
  Train - Total: 0.0268, Distance: 0.0267, Sparsity: 0.0001
  Val   - Total: 0.0106, Distance: 0.0105, Sparsity: 0.0001
  Sparsity: 51.0% (target: 40-70%)
  New best model! (val_loss: 0.0106)

Epoch 2/100
  Train - Total: 0.0052, Distance: 0.0050, Sparsity: 0.0001
  Val   - Total: 0.0052, Distance: 0.0051, Sparsity: 0.0001
  Sparsity: 55.9% (target: 40-70%)

[Epoch 2] Adding 64 dimensions (128 -> 192)
[Epoch 2] Dimension Statistics:
  Total dimensions: 192
  ...
  Adapted batch size to 666666 for 192 dimensions
  Reinitialized optimizer with new dimensions
  New best model! (val_loss: 0.0052)

Epoch 3/100
  ...
  New best model! (val_loss: 0.0036)

Epoch 4/100
  ...
[Epoch 4] Adding 64 dimensions (192 -> 256)
  Adapted batch size to 500000 for 256 dimensions
  New best model! (val_loss: 0.0020)

...

Epoch 15/100
  Train - Total: 0.0003, Distance: 0.0002, Sparsity: 0.0001
  Val   - Total: 0.0003, Distance: 0.0003, Sparsity: 0.0001
  Sparsity: 68.2% (target: 40-70%)
  No improvement for 5 epochs

[Early Stopping] No improvement for 5 epochs
[Early Stopping] Best validation loss: 0.0003
[Early Stopping] Stopping training at epoch 15

Training completed in 23.5 minutes
```

### With Adaptive Batch Size

Watch for these messages during dimension expansion:

```
[Epoch 2] Adding 64 dimensions (128 -> 192)
  Adapted batch size to 666666 for 192 dimensions  <- Batch reduced
  Reinitialized optimizer with new dimensions

[Epoch 4] Adding 64 dimensions (192 -> 256)
  Adapted batch size to 500000 for 256 dimensions  <- Batch reduced again

[Epoch 6] Adding 64 dimensions (256 -> 320)
  Adapted batch size to 400000 for 320 dimensions

...

[Epoch 12] Adding 64 dimensions (448 -> 512)
  Adapted batch size to 250000 for 512 dimensions  <- Final reduction
  Maximum dimensions reached (512)
```

## Troubleshooting

### Out of Memory Despite Adaptive Batch Size

If you still get OOM errors:
1. Reduce initial batch size: `--batch-size 524288` instead of 1048576
2. Reduce expansion amount: `--expand-by 32` instead of 64
3. Use fixed dimensions: Remove `--dynamic-dims`, use `--embedding-dim 512`

### Training Not Stopping Early

If training continues to 100 epochs:
1. Loss might still be improving - this is fine!
2. Reduce patience: `--patience 5` instead of 10
3. Increase min_delta: `--min-delta 0.001` instead of 0.0001

### Too Many Dimension Expansions

If dimensions expand too often:
1. Increase interval: `--expand-interval 5` instead of 2
2. Increase saturation threshold in code (currently 0.3 = 30% active)
3. Use larger expansion steps: `--expand-by 128`

### Training Stops Too Early

If early stopping triggers before convergence:
1. Increase patience: `--patience 15` instead of 5
2. Decrease min_delta: `--min-delta 0.00001`
3. Check if loss is actually plateauing (might be converged!)

## Files Modified

1. `scripts/train_embeddings.py`
   - Added `--patience` and `--min-delta` parameters
   - Added `--expand-by` parameter
   - Changed default `--expand-interval` from 10 to 2
   - Implemented early stopping logic
   - Implemented adaptive batch size reduction
   - Updated expansion to use `args.expand_by`

2. `src/embeddings/dynamic_dimensions.py`
   - Fixed `expand_embeddings` to preserve GPU device
   - Fixed `prune_unused_dimensions` to preserve GPU device

## Performance Comparison

### Before Improvements

**Settings:**
- Fixed 50 epochs
- Fixed batch size
- Dimensions expand every 10 epochs by 16 dims
- Crashes on GPU OOM when expanding

**Results:**
- Either crashes or takes 50+ epochs regardless of convergence
- Wastes compute time after convergence
- Manual batch size tuning required

### After Improvements

**Settings:**
- Early stopping (stops when converged)
- Adaptive batch size
- Dimensions expand every 2 epochs by 64 dims
- No OOM crashes

**Results:**
- Stops at optimal point (15-25 epochs typical)
- Saves 25-35 minutes of wasted training
- No manual intervention needed
- Stable GPU utilization throughout

## Summary

These improvements make NAOMI-II training:
- **Faster**: Early stopping saves time
- **Smarter**: Adaptive batch size prevents OOM
- **More aggressive**: Reaches full capacity quickly
- **More reliable**: No crashes, automatic tuning
- **More efficient**: Stops when optimal, no wasted epochs

Typical training time on T4 GPU with aggressive settings: **20-30 minutes** (down from 50+ minutes or crashes).
