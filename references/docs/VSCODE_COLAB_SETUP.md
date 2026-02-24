# Using Google Colab with VSCode

There are two ways to use Google Colab with VSCode:

## Option 1: Colab VSCode Extension (Recommended)

### Setup

1. **Install the extension**
   - Open VSCode
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Colab"
   - Install **"Colab Notebooks"** by Google

2. **Open the notebook**
   - In VSCode, open `notebooks/NAOMI_Training_Colab.ipynb`
   - Click "Select Kernel" → "Colab Runtime"
   - Sign in with your Google account (student account for free Pro)

3. **Run cells**
   - Click the play button next to each cell
   - Or use Shift+Enter to run current cell

### Benefits
- Native VSCode interface
- Full Colab features (GPU, Drive access)
- Better editing experience than web interface

## Option 2: SSH to Colab Runtime (Advanced)

This gives you full terminal access to Colab VMs.

### Setup Colab for SSH

1. **Create a new Colab notebook** with this code:

```python
# Install colab_ssh
!pip install colab_ssh --upgrade

# Setup SSH
from colab_ssh import launch_ssh_cloudflared, init_git_cloudflared
launch_ssh_cloudflared(password="YOUR_PASSWORD")
```

2. **Run the cell** - it will output an SSH command like:
```bash
ssh root@randomly-generated-url.trycloudflare.com
```

### Connect from VSCode

1. **Install Remote-SSH extension**
   - Extensions → Search "Remote - SSH"
   - Install by Microsoft

2. **Add SSH host**
   - Press F1 → "Remote-SSH: Add New SSH Host"
   - Paste the SSH command from Colab
   - Choose config file location

3. **Connect**
   - Press F1 → "Remote-SSH: Connect to Host"
   - Select the Colab host
   - Enter password when prompted

4. **Open NAOMI-II folder**
   - File → Open Folder
   - Navigate to `/content/NAOMI-II`

### Clone and Setup via SSH

Once connected to Colab via SSH:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/NAOMI-II.git
cd NAOMI-II

# Install dependencies
pip install nltk numpy torch tqdm

# Download WordNet
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy data
cp -r /content/drive/MyDrive/NAOMI-II-data/full_wordnet data/
cp -r /content/drive/MyDrive/NAOMI-II-data/wordnet_training data/

# Start training
python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --unsupervised \
    --dynamic-dims \
    --embedding-dim 128 \
    --max-dims 512 \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 128
```

### Monitor Training

With SSH access, you can:
- Use VSCode's integrated terminal
- Edit files directly on Colab VM
- Run commands in real-time
- Use VSCode debugger

## Option 3: Traditional Colab Web Interface

If extensions don't work, just use the web interface:

1. **Upload notebook**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook
   - Select `notebooks/NAOMI_Training_Colab.ipynb`

2. **Enable GPU**
   - Runtime → Change runtime type → GPU → T4

3. **Run cells sequentially**
   - Click play button on each cell
   - Or Runtime → Run all

## Data Transfer to Google Drive

### Method 1: Web Upload (Small files < 100MB)

1. Go to [drive.google.com](https://drive.google.com)
2. Create folder: `NAOMI-II-data`
3. Upload `full_wordnet/` folder
4. Upload `wordnet_training/` folder

### Method 2: Google Drive Desktop (Recommended for large files)

1. **Install Google Drive Desktop**
   - Download from [google.com/drive/download](https://www.google.com/drive/download/)
   - Sign in with student account

2. **Copy data to Drive**
   ```bash
   # On Windows (Surface)
   xcopy "C:\Users\wdgon\Code\NAOMI-II\data\full_wordnet" "G:\My Drive\NAOMI-II-data\full_wordnet" /E /I
   xcopy "C:\Users\wdgon\Code\NAOMI-II\data\wordnet_training" "G:\My Drive\NAOMI-II-data\wordnet_training" /E /I
   ```

3. **Wait for sync** - Will show in Drive when complete

### Method 3: Compress and Upload (Fastest)

1. **Compress data** (on Surface):
   ```bash
   # Create archive
   tar -czf naomi_training_data.tar.gz data/full_wordnet data/wordnet_training
   ```

2. **Upload to Drive** (drag & drop the .tar.gz file)

3. **Extract in Colab**:
   ```python
   # In Colab notebook
   !tar -xzf /content/drive/MyDrive/naomi_training_data.tar.gz -C .
   ```

## Pro Tips

### Keep Session Alive

Colab Pro sessions timeout if idle. To prevent:

1. **Use this JavaScript snippet** in browser console (F12):
```javascript
function KeepAlive() {
  console.log("Keeping session alive");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(KeepAlive, 60000); // Click every minute
```

2. **Or use `colab_keep_alive` package**:
```python
!pip install colabkeepalive
from colabkeepalive import keep_alive
keep_alive()
```

### Monitor GPU Usage

Add this cell to monitor GPU:

```python
# Monitor GPU usage in real-time
!watch -n 1 nvidia-smi
```

Or programmatically:

```python
import torch
import time

while True:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    time.sleep(10)
```

### Save Checkpoints Periodically

Add this to your training loop:

```python
# In Colab, save to Drive every N epochs
import shutil

def save_checkpoint_to_drive(epoch):
    src = f"data/checkpoints/model_epoch_{epoch}.pkl"
    dst = f"/content/drive/MyDrive/NAOMI-II-results/checkpoints/"
    shutil.copy(src, dst)
    print(f"Saved checkpoint {epoch} to Drive")
```

### Resume Training After Disconnect

If session disconnects, resume from last checkpoint:

```python
# Find latest checkpoint
!ls -lt data/checkpoints/ | head

# Resume training
!python scripts/train_embeddings.py \
    --training-data data/wordnet_training \
    --resume data/checkpoints/model_epoch_10.pkl \
    --epochs 50 \
    --unsupervised \
    --dynamic-dims \
    --batch-size 128
```

## Troubleshooting

### "No GPU available"
- Runtime → Change runtime type → GPU → T4
- Restart runtime
- If still no GPU, wait 30 mins (quota limit)

### "Out of Memory"
- Reduce batch size: `--batch-size 64` or `--batch-size 32`
- Restart runtime to clear memory
- Use gradient checkpointing (if implemented)

### "Session disconnected"
- Reconnect to same runtime (will preserve state)
- If state lost, resume from checkpoint
- Use Keep Alive script (see above)

### "Drive mounting failed"
- Re-run the mount cell
- Check internet connection
- Clear browser cache and retry

### "Data not found"
- Verify Drive path: `!ls /content/drive/MyDrive/`
- Check folder names match exactly
- Ensure data finished syncing to Drive

## Quick Start Checklist

- [ ] Install Colab extension in VSCode (or use web interface)
- [ ] Upload training data to Google Drive
- [ ] Open `notebooks/NAOMI_Training_Colab.ipynb`
- [ ] Enable T4 GPU runtime
- [ ] Run setup cells (1-4)
- [ ] Start training (cell 5)
- [ ] Monitor progress
- [ ] Download checkpoints when complete

**Estimated total time:** 15 minutes setup + 12-24 hours training

Good luck! 🚀
