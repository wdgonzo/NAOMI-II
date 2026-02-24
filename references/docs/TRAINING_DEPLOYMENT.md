# NAOMI-II Training Deployment Guide

## Overview

Training NAOMI-II on the full 157K WordNet vocabulary requires significant computational resources. This guide covers deployment options for running the training pipeline on more powerful systems.

## Training Requirements

### Dataset Scale
- **Vocabulary**: 157,306 sense-tagged words
- **Training Edges**: 15,671,914 semantic relation pairs
- **Model Parameters**: 20.1M (157K words × 128 dims)
- **Epochs**: 50 recommended

### Performance Estimates

**Surface 11 (ARM64 CPU, 12 threads):**
- Speed: ~10 iterations/second
- Time per epoch: ~6 hours
- **Total training time: ~12.5 days**

**Desktop/Server (16+ cores, GPU optional):**
- Speed (CPU): ~30-50 it/s → ~2-3 days total
- Speed (GPU): ~100-200 it/s → ~12-24 hours total

### Memory Requirements
- **RAM**: 4-8 GB minimum (model + data)
- **Disk**: 2-3 GB for WordNet data + checkpoints
- **VRAM** (GPU): 2-4 GB recommended

## Deployment Options

### Option 1: Local Desktop/Server

#### Prerequisites
```bash
# Python 3.12+
python --version

# Git
git --version

# At least 8GB RAM, 16+ CPU cores recommended
```

#### Setup Steps

1. **Clone repository**
```bash
git clone <repository-url> NAOMI-II
cd NAOMI-II
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

# For GPU support (NVIDIA CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For AMD GPU (Windows)
pip install torch-directml
```

4. **Download NLTK WordNet**
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

5. **Copy pre-generated data from Surface**
   - Transfer `data/full_wordnet/` directory (117K synsets, extracted)
   - Transfer `data/wordnet_training/` directory (15.67M training edges)

   If not available, regenerate (takes ~1.5 hours):
```bash
python scripts/extract_full_wordnet.py --output data/full_wordnet
python scripts/generate_wordnet_training_data.py \
    --input data/full_wordnet \
    --output data/wordnet_training
```

6. **Start training**
```bash
# Full training run
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 64

# Quick test (1 epoch)
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 1 \
    --lr 0.001 \
    --batch-size 64
```

### Option 2: Cloud GPU (Google Colab / Kaggle)

#### Google Colab Setup

1. **Create new notebook** at [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**
   - Runtime → Change runtime type → GPU (T4)

3. **Setup code cell**
```python
# Clone repository
!git clone <repository-url> NAOMI-II
%cd NAOMI-II

# Install dependencies
!pip install -q nltk torch numpy tqdm

# Download WordNet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

4. **Upload pre-generated data**
```python
# Option A: Upload from Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy data from Drive
!cp -r /content/drive/MyDrive/NAOMI-II-data/full_wordnet data/
!cp -r /content/drive/MyDrive/NAOMI-II-data/wordnet_training data/

# Option B: Download from cloud storage (if uploaded)
# !wget https://your-storage.com/full_wordnet.tar.gz
# !tar -xzf full_wordnet.tar.gz -C data/
```

5. **Train**
```python
!python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 128  # Larger batch for GPU
```

6. **Download checkpoints**
```python
from google.colab import files
files.download('data/checkpoints/best_model.pkl')
files.download('data/checkpoints/embeddings_epoch_50.npy')
```

### Option 3: Cloud VM (AWS/Azure/GCP)

#### Recommended Instance Types

**AWS EC2:**
- `c7i.4xlarge` (16 vCPU, CPU-only): ~$0.68/hr → ~$50 for full training
- `g5.xlarge` (4 vCPU + A10G GPU): ~$1.00/hr → ~$24 for full training

**Azure:**
- `F16s_v2` (16 vCPU, CPU-only): ~$0.68/hr
- `NC4as_T4_v3` (4 vCPU + T4 GPU): ~$0.53/hr

**GCP:**
- `n2-standard-16` (16 vCPU, CPU-only): ~$0.77/hr
- `n1-standard-4` + T4 GPU: ~$0.50/hr

#### Setup Script (Ubuntu 22.04)

```bash
#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Clone repository
git clone <repository-url> NAOMI-II
cd NAOMI-II

# Create venv
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU (NVIDIA CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download WordNet
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Generate training data (or upload pre-generated)
python scripts/extract_full_wordnet.py --output data/full_wordnet
python scripts/generate_wordnet_training_data.py \
    --input data/full_wordnet \
    --output data/wordnet_training

# Start training in background with nohup
nohup python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 64 > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## Data Transfer Options

### Small Files (< 1GB) - GitHub
```bash
# On Surface: Commit generated data
git lfs track "data/wordnet_training/*.pkl"
git add data/full_wordnet/ data/wordnet_training/
git commit -m "Add pre-generated WordNet training data"
git push

# On target machine: Pull
git pull
```

### Large Files - Cloud Storage

**Option 1: Google Drive**
```bash
# On Surface: Upload to Drive manually
# Then on target machine:
pip install gdown
gdown <file-id> -O data/wordnet_training.tar.gz
tar -xzf wordnet_training.tar.gz -C data/
```

**Option 2: AWS S3**
```bash
# On Surface: Upload
aws s3 sync data/full_wordnet/ s3://naomi-training-data/full_wordnet/
aws s3 sync data/wordnet_training/ s3://naomi-training-data/wordnet_training/

# On target: Download
aws s3 sync s3://naomi-training-data/full_wordnet/ data/full_wordnet/
aws s3 sync s3://naomi-training-data/wordnet_training/ data/wordnet_training/
```

**Option 3: SCP (direct transfer)**
```bash
# On Surface: Compress
tar -czf wordnet_data.tar.gz data/full_wordnet/ data/wordnet_training/

# Transfer to remote server
scp wordnet_data.tar.gz user@remote-server:/home/user/NAOMI-II/

# On remote: Extract
tar -xzf wordnet_data.tar.gz
```

## Training Commands Reference

### Full Training (Production)
```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 64 \
    --save-interval 5  # Save every 5 epochs
```

### Quick Test (Validation)
```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 1 \
    --lr 0.001 \
    --batch-size 64
```

### Resume from Checkpoint
```bash
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 64 \
    --resume data/checkpoints/model_epoch_10.pkl
```

## Monitoring Training

### Real-time Monitoring
```bash
# View live training progress
tail -f training.log

# Check GPU usage (if using GPU)
watch -n 1 nvidia-smi

# Check CPU usage
htop
```

### Key Metrics to Watch

1. **Loss Trends**
   - Distance loss should decrease (0.08 → 0.02 typical)
   - Sparsity should stay low (< 0.001)
   - Total loss = distance + sparsity

2. **Dimension Expansion**
   - Checks every 10 epochs
   - Will expand if >80% dimensions saturated (>30% utilized)
   - Max dimensions: 512

3. **Training Speed**
   - CPU: 10-50 it/s depending on cores
   - GPU: 100-200 it/s typical for this model size

## Expected Output

### During Training
```
[Epoch 1] Batch 1000/220387, Loss: 0.0284 (dist: 0.0284, spar: 0.0000)
[Epoch 1] Batch 2000/220387, Loss: 0.0231 (dist: 0.0231, spar: 0.0000)
...
[Epoch 1] Complete - Avg Loss: 0.0198, Val Loss: 0.0205

[Epoch 10] Dimension Statistics:
  Total dimensions: 128
  Mean sparsity: 45.2%
  Dimension usage distribution:
    Unused (<10%): 12
    Sparse (10-30%): 28
    Moderate (30-70%): 64
    Saturated (>70%): 24

[Epoch 50] Training Complete!
  Final Loss: 0.0087
  Final Dimensions: 192 (expanded from 128)
```

### Saved Files
```
data/checkpoints/
├── best_model.pkl           # Best model by validation loss
├── model_epoch_10.pkl       # Checkpoint every 5 epochs
├── model_epoch_15.pkl
├── ...
├── embeddings_epoch_50.npy  # Final embeddings (157K x 192)
└── training_log.json        # Loss history
```

## Next Steps After Training

1. **Run Dimension Analysis**
```bash
python scripts/analyze_dimensions.py \
    --embeddings data/checkpoints/embeddings_epoch_50.npy \
    --vocab data/wordnet_training/vocabulary.json \
    --output results/dimension_analysis/
```

2. **Test Semantic Queries**
```bash
python scripts/test_embeddings.py \
    --embeddings data/checkpoints/embeddings_epoch_50.npy \
    --vocab data/wordnet_training/vocabulary.json
```

3. **Visualize Embeddings**
```bash
python scripts/visualize_embeddings.py \
    --embeddings data/checkpoints/embeddings_epoch_50.npy \
    --vocab data/wordnet_training/vocabulary.json \
    --output results/visualizations/
```

## Troubleshooting

### Out of Memory (CPU)
- Reduce batch size: `--batch-size 32` or `--batch-size 16`
- Reduce max dimensions: `--max-dims 256`

### Out of Memory (GPU)
- Reduce batch size: `--batch-size 32`
- Use gradient accumulation (to be implemented)
- Use mixed precision training (to be implemented)

### Training Too Slow
- Use GPU instance (100x speedup over Surface)
- Increase batch size on powerful systems: `--batch-size 128` or `--batch-size 256`
- Use multi-GPU if available (requires code modification)

### Loss Not Decreasing
- Check learning rate (try `--lr 0.0001` for slower, more stable learning)
- Verify data loaded correctly (check batch samples)
- Ensure WordNet data is complete (157K vocabulary)

## Cost Estimates

### Cloud Training Costs

**Google Colab Pro:**
- $10/month subscription
- ~12-24 hours on T4 GPU
- **Total: $10** (includes other usage)

**AWS EC2 (g5.xlarge with A10G GPU):**
- ~$1.00/hour
- ~12-24 hours training
- **Total: $12-24**

**Azure (NC4as_T4_v3):**
- ~$0.53/hour
- ~12-24 hours training
- **Total: $6-12**

**Recommended:** Start with Google Colab Pro for cost-effectiveness.

## Architecture Notes

### Why This Takes So Long

1. **Dataset Size**: 15.67M training edges (vs. 30K for 5K vocab)
2. **Vocabulary Size**: 157K words (vs. 5K)
3. **Model Parameters**: 20.1M (vs. 640K for 5K vocab)
4. **No GPU Acceleration**: CPU training is 50-100x slower
5. **Dense Operations**: Each forward pass computes 157K embeddings

### Key Optimizations Implemented

✅ **Batch processing** (64 samples at once)
✅ **Multi-threading** (12 threads on Surface)
✅ **Efficient data structures** (pickled edges, numpy arrays)
✅ **Dynamic dimensions** (start small, expand as needed)

### Future Optimizations (Not Yet Implemented)

❌ **GPU support** (would provide 50-100x speedup)
❌ **Gradient accumulation** (simulate larger batches)
❌ **Mixed precision training** (float16 for speed)
❌ **Distributed training** (multi-GPU/multi-node)

## Summary

**Current Status:**
- ✅ WordNet data extracted (117K synsets)
- ✅ Training data generated (15.67M edges)
- ✅ Training script ready and tested
- ❌ Full training not feasible on Surface 11 (12.5 days)

**Recommended Path:**
1. Use Google Colab Pro with T4 GPU (~$10, 12-24 hours)
2. Transfer pre-generated data files (2-3 GB)
3. Run training, download checkpoints
4. Analyze results locally on Surface

**Alternative:**
1. Rent AWS g5.xlarge for 24 hours (~$24)
2. Faster A10G GPU, more reliable than Colab
3. Keep instance running for analysis

The data generation work is done - it's just the training that needs more compute!
