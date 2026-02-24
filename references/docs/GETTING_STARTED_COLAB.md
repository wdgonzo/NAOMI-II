# Getting Started: Training NAOMI-II on Google Colab GPU

This guide walks through the complete process of training NAOMI-II embeddings using Google Colab with GPU acceleration.

## Overview

**What we're doing:**
- Training NAOMI-II embeddings on 157K WordNet vocabulary
- Using Google Colab Pro (free for students) with T4 GPU
- Unsupervised learning with dynamic dimension expansion (128→512 dims)

**Timeline:**
- Setup: 30-60 minutes (one time)
- Training: 12-24 hours on T4 GPU
- Analysis: 1-2 hours after training

**Requirements:**
- Google account with student Colab Pro access
- VSCode with Colab extension installed ✓
- Pre-generated training data on your Surface (already complete ✓)
- Google Drive with ~3GB free space

---

## Step 1: Upload Training Data to Google Drive

### Option A: Using Google Drive Desktop (Recommended)

1. **Install Google Drive Desktop** (if not already installed)
   - Download from: https://www.google.com/drive/download/
   - Install and sign in with your student Google account
   - Choose a local sync folder

2. **Create data folder in Drive**
   - Open Google Drive in browser
   - Create new folder: `NAOMI-II-data`
   - Inside it, create two subfolders:
     - `full_wordnet`
     - `wordnet_training`

3. **Copy data from Surface to Drive**

   Open PowerShell or Command Prompt on Surface:
   ```powershell
   # Navigate to your NAOMI-II directory
   cd C:\Users\wdgon\Code\NAOMI-II

   # Find your Google Drive mount (usually G:\ or similar)
   # Adjust drive letter based on your system

   # Copy full_wordnet folder
   xcopy "data\full_wordnet" "G:\My Drive\NAOMI-II-data\full_wordnet" /E /I /Y

   # Copy wordnet_training folder
   xcopy "data\wordnet_training" "G:\My Drive\NAOMI-II-data\wordnet_training" /E /I /Y
   ```

4. **Verify upload**
   - Check Google Drive web interface
   - Should see ~2-3GB of data in NAOMI-II-data folder
   - Verify both folders have files:
     - `full_wordnet/synsets.json` (~50MB)
     - `wordnet_training/edges.pkl` (~200MB)
     - `wordnet_training/vocabulary.json` (~20MB)

### Option B: Using Web Upload (Alternative)

1. Go to https://drive.google.com
2. Create folder `NAOMI-II-data`
3. Drag and drop these folders from File Explorer:
   - `C:\Users\wdgon\Code\NAOMI-II\data\full_wordnet`
   - `C:\Users\wdgon\Code\NAOMI-II\data\wordnet_training`
4. Wait for upload to complete (may take 30-60 minutes)

### Option C: Compress First (Fastest)

1. **Compress data on Surface**:
   ```powershell
   cd C:\Users\wdgon\Code\NAOMI-II

   # Install 7-Zip if needed, or use built-in tar
   tar -czf naomi_training_data.tar.gz data\full_wordnet data\wordnet_training
   ```

2. **Upload the .tar.gz file to Drive** (single file, faster)

3. **Extract in Colab later** (we'll add this to the notebook)

---

## Step 2: Open Notebook in VSCode with Colab

1. **Open VSCode**

2. **Open the NAOMI-II project folder**
   - File → Open Folder
   - Navigate to `C:\Users\wdgon\Code\NAOMI-II`
   - Click "Select Folder"

3. **Open the Colab notebook**
   - In Explorer panel, navigate to: `notebooks/NAOMI_Training_Colab.ipynb`
   - Click to open

4. **Select Colab Runtime**
   - Click "Select Kernel" button in top-right
   - Choose "Colab Runtime"
   - Sign in with your Google student account
   - Grant necessary permissions

5. **Verify connection**
   - You should see "Colab Runtime" in the kernel selector
   - A small cloud icon indicates connection to Colab

---

## Step 3: Configure GPU Runtime

### In VSCode (with Colab extension):

1. **The extension should auto-configure GPU**, but verify:
   - Run the first cell (GPU verification)
   - Check output shows:
     ```
     CUDA available: True
     GPU: Tesla T4
     VRAM: ~15 GB
     ```

### If GPU not detected:

1. **Open notebook in Colab web interface**:
   - Go to https://colab.research.google.com
   - File → Upload notebook
   - Select `notebooks/NAOMI_Training_Colab.ipynb`

2. **Change runtime type**:
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (or Premium/A100 if available)
   - Click **Save**

3. **Verify GPU**:
   - Run first cell
   - Should show Tesla T4 with ~15GB VRAM

4. **Continue in web interface** or return to VSCode

---

## Step 4: Run Training Setup Cells

Execute these cells in order (click play button or Shift+Enter):

### Cell 1: Verify GPU ✓
```python
# Should show:
# CUDA available: True
# GPU: Tesla T4
# VRAM: 15.XX GB
```

### Cell 2: Clone Repository
```python
# Clones NAOMI-II from GitHub
# Sets working directory
```

**Note:** Update the GitHub URL in this cell with your actual repository URL:
```python
!git clone https://github.com/YOUR_USERNAME/NAOMI-II.git
```

If you haven't pushed to GitHub yet:
```python
# Alternative: Skip cloning and manually upload necessary files
# We only need the scripts/ folder for training
```

### Cell 3: Install Dependencies
```python
# Installs: nltk, numpy, torch, tqdm
# Downloads WordNet
```

### Cell 4: Mount Google Drive
```python
# Mounts your Google Drive
# You'll need to authorize access
```

**Important:** Update the Drive path in the next cell to match your folder structure:
```python
DRIVE_DATA_PATH = "/content/drive/MyDrive/NAOMI-II-data"
```

### Cell 5: Copy Data to Colab Storage
```python
# Copies data from Drive to local Colab storage
# This makes training faster (local SSD vs Drive)
# Takes 5-10 minutes
```

**Verify data copied:**
```
✓ Data copied successfully:
data/full_wordnet/synsets.json         50M
data/full_wordnet/antonyms.json        1M
data/wordnet_training/edges.pkl        200M
data/wordnet_training/vocabulary.json  20M
```

---

## Step 5: Start Training on GPU

### Run the training cell:

```python
!python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 128  # Larger batch for GPU!
```

### Expected output:

```
======================================================================
NAOMI-II EMBEDDING TRAINING
======================================================================

[1/7] Initializing device...
[GPU] Using CUDA device: Tesla T4
[GPU] VRAM: 15.00 GB available

[2/7] Loading training data...
  Vocabulary size: 157306
  Training edges: 15671914

[3/7] Creating dataset...
  Dataset size: 15671914 samples
  Train: 14104723, Validation: 1567191

[4/7] Initializing embedding model...
  Embedding dim: 128 (51 anchor + 77 learned)
  Model parameters: 20,135,168
  Model loaded to GPU

[5/7] Initializing optimizer and loss functions...
  [UNSUPERVISED MODE] Skipping manual dimension assignments
  Training with distance + sparsity constraints only
  Optimizer: Adam (lr=0.001)

[6/7] Training...
  Dynamic dimensions enabled: 128 -> max 512
  Expansion check interval: every 10 epochs

Epoch 1:   0%|          | 0/110193 [00:00<?, ?it/s]
Epoch 1:   1%|▏         | 1000/110193 [00:05<5:32:15, 100.21it/s, loss=0.0284]
Epoch 1:   2%|▏         | 2000/110193 [00:10<5:28:12, 120.45it/s, loss=0.0231]
...
```

### Training Parameters Explained:

- `--training-data data/wordnet_training` - Where training edges are stored
- `--unsupervised` - Use distance + sparsity only (no manual labels)
- `--dynamic-dims` - Enable automatic dimension expansion
- `--embedding-dim 128` - Start with 128 dimensions
- `--max-dims 512` - Can expand up to 512 dimensions
- `--epochs 50` - Train for 50 complete passes through data
- `--lr 0.001` - Learning rate (how fast model updates)
- `--batch-size 128` - Process 128 samples at once (GPU can handle larger batches)

### Performance Expectations:

**On T4 GPU:**
- Speed: ~100-200 iterations/second
- Batches per epoch: ~110,000 (with batch_size=128)
- Time per epoch: ~10-20 minutes
- **Total time: 8-17 hours for 50 epochs**

Compare to Surface 11 CPU: ~12.5 days!

---

## Step 6: Monitor Training Progress

### What to watch:

1. **Loss Trends** (should decrease over time):
   ```
   Epoch 1: loss=0.0284
   Epoch 5: loss=0.0198
   Epoch 10: loss=0.0156
   Epoch 20: loss=0.0112
   Epoch 50: loss=0.0087
   ```

2. **Dimension Statistics** (printed every 10 epochs):
   ```
   [Epoch 10] Dimension Statistics:
     Total dimensions: 128
     Mean sparsity: 45.2%
     Dimension usage:
       Unused (<10%): 12
       Sparse (10-30%): 28
       Moderate (30-70%): 64
       Saturated (>70%): 24

   [Decision] 18.8% of dimensions saturated (threshold: 80%)
   [Decision] No expansion needed yet
   ```

3. **Dimension Expansion Events**:
   ```
   [Epoch 20] Dimension Statistics:
     Total dimensions: 128
     Saturated: 85.2%

   [EXPANSION] Adding 16 new dimensions (128 -> 144)
   [EXPANSION] Reinitializing optimizer
   ```

4. **GPU Utilization**:
   - In a new notebook cell, run:
     ```python
     !watch -n 5 nvidia-smi
     ```
   - Should show ~80-95% GPU utilization
   - Memory usage: ~4-8 GB

### If Training is Slow (<50 it/s):

1. **Verify GPU is being used**:
   ```python
   import torch
   print(f"Using device: {next(model.parameters()).device}")
   # Should show: cuda:0
   ```

2. **Check GPU utilization**:
   ```python
   !nvidia-smi
   # GPU should show high % utilization
   ```

3. **Increase batch size** (if GPU has free memory):
   ```bash
   # Try batch_size=256 or even 512
   --batch-size 256
   ```

---

## Step 7: Checkpoints and Saving

### Automatic Checkpointing

The training script automatically saves:

- **Every 5 epochs**: `data/checkpoints/model_epoch_5.pkl`
- **Best model**: `data/checkpoints/best_model.pkl` (lowest validation loss)
- **Final embeddings**: `data/checkpoints/embeddings_final.npy`

### Save to Google Drive (Recommended)

Add this cell and run periodically to backup checkpoints:

```python
# Copy checkpoints to Drive for safekeeping
!mkdir -p /content/drive/MyDrive/NAOMI-II-results/checkpoints
!cp -r data/checkpoints/* /content/drive/MyDrive/NAOMI-II-results/checkpoints/

# Show what was saved
!ls -lh /content/drive/MyDrive/NAOMI-II-results/checkpoints/
```

Run this:
- Every few hours during training
- Definitely when training completes
- Before closing Colab session

### If Session Disconnects

Colab Pro sessions last 24 hours, but can disconnect. If this happens:

1. **Reconnect to runtime** (session may still be alive)
2. **Check if training is still running**:
   ```python
   !ps aux | grep train_embeddings
   ```

3. **If training stopped, resume from checkpoint**:
   ```bash
   # Find latest checkpoint
   !ls -lt data/checkpoints/ | head

   # Resume training
   !python scripts/train_embeddings.py \
       --training-data data/wordnet_training \
       --resume data/checkpoints/model_epoch_30.pkl \
       --epochs 50 \
       --unsupervised \
       --dynamic-dims \
       --batch-size 128
   ```

---

## Step 8: Download Trained Model

When training completes:

### Option A: Download via Colab

```python
from google.colab import files

# Download key files
files.download('data/checkpoints/best_model.pkl')
files.download('data/checkpoints/embeddings_final.npy')
files.download('data/checkpoints/training_log.json')
```

### Option B: Copy to Google Drive (then download later)

```python
# Already saved to Drive in Step 7
# Just download from drive.google.com on your Surface
```

### Option C: Use Google Drive Desktop

1. Checkpoints saved to `My Drive/NAOMI-II-results/checkpoints/`
2. Google Drive Desktop will auto-sync to Surface
3. Find them in: `G:\My Drive\NAOMI-II-results\checkpoints\`

---

## Step 9: Verify Trained Embeddings

Add this analysis cell to verify quality:

```python
import numpy as np
import json

# Load trained embeddings
embeddings = np.load('data/checkpoints/embeddings_final.npy')
with open('data/wordnet_training/vocabulary.json', 'r') as f:
    vocab = json.load(f)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Vocabulary size: {len(vocab['word_to_id'])}")
print(f"\nStatistics:")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std: {embeddings.std():.4f}")
print(f"  Min: {embeddings.min():.4f}")
print(f"  Max: {embeddings.max():.4f}")

# Compute sparsity
sparsity = np.mean(np.abs(embeddings) < 0.001)
print(f"\nSparsity: {sparsity:.1%}")

# Check specific words
word_to_id = vocab['word_to_id']
test_words = ['dog_wn.01_n', 'run_wn.01_v', 'happy_wn.01_a']

print("\nSample embeddings:")
for word in test_words:
    if word in word_to_id:
        idx = word_to_id[word]
        emb = embeddings[idx]
        print(f"\n{word}:")
        print(f"  Dimensions: {len(emb)}")
        print(f"  Sparsity: {np.mean(np.abs(emb) < 0.001):.1%}")
        print(f"  Active dims: {np.sum(np.abs(emb) > 0.001)}")
```

Expected output:
```
Embeddings shape: (157306, 192)  # May have expanded to 192+ dims
Vocabulary size: 157306

Statistics:
  Mean: 0.0012
  Std: 0.1234
  Min: -0.8765
  Max: 0.9123

Sparsity: 45.3%

Sample embeddings:
dog_wn.01_n:
  Dimensions: 192
  Sparsity: 43.2%
  Active dims: 109
```

---

## Step 10: Next Steps (Back on Surface)

Once you have the trained embeddings:

1. **Copy to Surface**
   ```powershell
   # Create results directory
   cd C:\Users\wdgon\Code\NAOMI-II
   mkdir results\trained_embeddings

   # Copy from Google Drive
   copy "G:\My Drive\NAOMI-II-results\checkpoints\*" "results\trained_embeddings\"
   ```

2. **Run Dimension Discovery Analysis**
   ```bash
   python scripts/analyze_dimensions.py \
       --embeddings results/trained_embeddings/embeddings_final.npy \
       --vocab data/wordnet_training/vocabulary.json \
       --output results/dimension_analysis/
   ```

3. **Test Semantic Queries**
   ```bash
   python scripts/test_embeddings.py \
       --embeddings results/trained_embeddings/embeddings_final.npy \
       --vocab data/wordnet_training/vocabulary.json
   ```

4. **Visualize Results**
   ```bash
   python scripts/visualize_embeddings.py \
       --embeddings results/trained_embeddings/embeddings_final.npy \
       --vocab data/wordnet_training/vocabulary.json \
       --output results/visualizations/
   ```

---

## Troubleshooting

### "No GPU available"
- Runtime → Change runtime type → GPU → T4
- Restart runtime and try again
- If Colab Pro, make sure subscription is active

### "Out of Memory" (GPU)
- Reduce batch size: `--batch-size 64` or `--batch-size 32`
- Reduce max dimensions: `--max-dims 256`
- Restart runtime to clear GPU memory

### "Session disconnected"
- Colab Pro sessions last 24 hours
- Reconnect - session may still be alive
- Use Keep Alive script (see VSCODE_COLAB_SETUP.md)

### "Drive mounting failed"
- Re-run the drive.mount() cell
- Sign in again if needed
- Check internet connection

### "Training is slow (<50 it/s)"
- Verify GPU is enabled (check Cell 1 output)
- Check nvidia-smi shows GPU usage
- Make sure model is on GPU (not CPU fallback)

### "Data not found"
- Verify Drive path is correct
- Check data uploaded to correct folder
- Re-run data copy cell (Cell 5)

---

## Summary Checklist

- [ ] **Step 1**: Upload data to Google Drive (2-3 GB)
- [ ] **Step 2**: Open notebook in VSCode with Colab extension
- [ ] **Step 3**: Configure T4 GPU runtime
- [ ] **Step 4**: Run setup cells (1-5)
- [ ] **Step 5**: Start training (12-24 hours on GPU)
- [ ] **Step 6**: Monitor progress periodically
- [ ] **Step 7**: Save checkpoints to Drive
- [ ] **Step 8**: Download trained model
- [ ] **Step 9**: Verify embeddings quality
- [ ] **Step 10**: Run analysis on Surface

---

## Key Files Reference

**On Surface (before upload):**
- `data/full_wordnet/synsets.json` - 117K synsets extracted
- `data/wordnet_training/edges.pkl` - 15.67M training edges
- `data/wordnet_training/vocabulary.json` - 157K word vocabulary

**On Google Drive:**
- `My Drive/NAOMI-II-data/full_wordnet/` - Uploaded data
- `My Drive/NAOMI-II-data/wordnet_training/` - Uploaded training data
- `My Drive/NAOMI-II-results/checkpoints/` - Saved checkpoints

**On Colab (during training):**
- `/content/NAOMI-II/` - Cloned repository
- `/content/data/` - Copied training data (local SSD)
- `/content/data/checkpoints/` - Training checkpoints

**On Surface (after training):**
- `results/trained_embeddings/` - Downloaded checkpoints
- `results/dimension_analysis/` - Analysis results
- `results/visualizations/` - Embedding visualizations

---

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Data upload to Drive | 30-60 min | One-time, depends on internet |
| VSCode + Colab setup | 5 min | Already have extension installed |
| Run setup cells | 10 min | Installing deps, copying data |
| **GPU Training** | **12-24 hrs** | **Main time investment** |
| Download checkpoints | 10 min | Via Drive sync |
| Analysis on Surface | 1-2 hrs | Dimension discovery, viz |

**Total: ~1.5 days from start to analyzed results**

Much better than 12.5 days on Surface CPU! 🚀

---

## Cost

- **Google Colab Pro**: FREE (student account)
- **Google Drive storage**: FREE (3GB well within student quota)
- **Total cost**: $0

---

Good luck with training! You're ready to go! 🎉
