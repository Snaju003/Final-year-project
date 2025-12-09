# Google Colab Integration Guide

## Quick Start (5 Minutes)

### ⚠️ IMPORTANT: If GitHub Clone Fails

If you see this error:
```
fatal: could not read Username for 'https://github.com': No such device or address
```

**Follow Solution 2 below instead** (Google Drive method - more reliable)

---

### Option 1: Using Pre-built Notebook (If Repo is PUBLIC)
1. Open Google Colab: https://colab.research.google.com/
2. Go to **File → Open notebook → GitHub**
3. Enter your repo: `Snaju003/Final-year-project`
4. Select `Deepfake_Detection_Colab.ipynb`
5. Runtime → Change runtime type → Select **GPU**
6. Run cells in order from top to bottom

### Option 2: Upload Project to Google Drive (RECOMMENDED - Always Works)

**This method works even if repo is private or network is blocked.**

1. **On your local machine**, zip your project:
   ```bash
   # Windows PowerShell
   Compress-Archive -Path "E:\Final-year-project" -DestinationPath "Final-year-project.zip"
   ```

2. **Upload to Google Drive:**
   - Open https://drive.google.com/
   - Upload `Final-year-project.zip` to `MyDrive/`

3. **In Colab, run this cell:**
   ```python
   from google.colab import drive
   import zipfile
   
   drive.mount('/content/drive', force_remount=True)
   
   with zipfile.ZipFile('/content/drive/MyDrive/Final-year-project.zip', 'r') as z:
       z.extractall('/content')
   
   %cd /content/Final-year-project
   !pip install -q -r requirements.txt
   ```

4. Continue with remaining cells

### Option 3: Manual Setup (For Troubleshooting)
1. Create a new notebook in Google Colab
2. Copy-paste from `COLAB_QUICK_START.md` (in repo)
3. This auto-detects GitHub/Drive and sets up accordingly

---

## Step-by-Step Integration

### Step 1: Prepare Your GitHub Repository
```bash
# Make sure your repo is public
git remote -v  # Check remote is set correctly
git push origin main  # Push all changes
```

### Step 2: Check File Structure
Your project should have:
```
Final-year-project/
├── requirements.txt          # ✅ Updated with all dependencies
├── src/
│   ├── inference.py          # Model inference code
│   ├── train.py              # Training code
│   └── models.py             # Model definitions
├── models/
│   ├── best_model.pth        # Pre-trained weights
│   └── ensemble/
│       ├── ensemble_final.pth
│       └── ensemble_best.pth
└── colab_setup.py            # ✅ Setup script
```

### Step 3: Essential Code Updates

Make sure your code handles both local and Colab environments:

```python
import os
import sys

# Handle path issues in Colab
if 'google.colab' in sys.modules:
    # Running in Colab
    PROJECT_ROOT = '/content/Final-year-project'
else:
    # Running locally
    PROJECT_ROOT = os.getcwd()

sys.path.insert(0, PROJECT_ROOT)
```

### Step 4: Upload Data to Google Drive (Optional)

If you need to process your own data:

```python
from google.colab import drive
drive.mount('/content/drive')

# Your data should be in: /content/drive/MyDrive/your_data
```

### Step 5: Download Results

```python
from google.colab import files
import json

# Save results
results = {'image1.jpg': 0.95, 'image2.jpg': 0.12}
with open('results.json', 'w') as f:
    json.dump(results, f)

# Download to local machine
files.download('results.json')
```

---

## Common Issues & Solutions

### Issue 1: "fatal: could not read Username for 'https://github.com'"
**Root Cause:** GitHub is unreachable (network blocked or repo is private)

**Solution:**
- Make repo PUBLIC: GitHub → Settings → Visibility → Public
- **OR** Use Google Drive method (most reliable):
  ```python
  from google.colab import drive
  import zipfile
  drive.mount('/content/drive')
  with zipfile.ZipFile('/content/drive/MyDrive/Final-year-project.zip', 'r') as z:
      z.extractall('/content')
  %cd /content/Final-year-project
  ```

### Issue 2: "[Errno 2] No such file or directory: 'Final-year-project'"
**Root Cause:** Clone or extraction failed

**Solution:**
1. Check if file exists: `import os; print(os.listdir('.'))`
2. If using Drive: verify ZIP is uploaded to `MyDrive/`
3. If using GitHub: check if repo is public
4. Try Drive method instead (more reliable)

### Issue 3: ModuleNotFoundError
```python
import sys
sys.path.insert(0, '/content/Final-year-project')
```

### Issue 2: CUDA/GPU Not Available
- Go to **Runtime → Change runtime type → Hardware accelerator → GPU**
- Select T4 GPU (free tier) or higher

### Issue 3: Out of Memory
- Reduce batch size in your code
- Use `torch.cuda.empty_cache()` to clear GPU memory
- Process images in smaller batches

### Issue 4: File Not Found
- Always use absolute paths in Colab
- Check path exists with: `import os; print(os.path.exists(path))`

### Issue 5: Package Version Conflicts
```python
# Install specific versions
!pip install torch==2.0.1 torchvision==0.15.2 -q --upgrade
```

---

## Performance Tips

1. **GPU Selection**: Use T4 GPU (free, sufficient for inference)
2. **Batch Processing**: Process images in batches of 32-64 for better speed
3. **Model Size**: Use `ensemble_final.pth` (optimized) instead of other checkpoints
4. **Cache**: Enable image caching to avoid re-downloading during repeated runs

---

## Example Workflow

```python
# 1. Setup
!git clone https://github.com/Snaju003/Final-year-project.git
%cd Final-year-project
!pip install -q -r requirements.txt

# 2. Import
import torch
from src.inference import EnsembleModel

# 3. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnsembleModel().to(device)
model.load_state_dict(torch.load('models/ensemble/ensemble_final.pth', map_location=device))
model.eval()

# 4. Run Inference
# ... your inference code here

# 5. Download Results
from google.colab import files
files.download('results.json')
```

---

## Additional Resources

- **Colab Free Tier Limits**: 12 hours per session, 12 GB RAM, T4 GPU
- **Colab Pro**: More compute, longer sessions
- **GitHub Integration**: Direct notebook loading from GitHub repos
- **Google Drive Integration**: Store large datasets and models

---

## Next Steps

1. Test the provided notebook with sample images
2. Adapt the batch processing code for your specific use case
3. Upload your training data to Google Drive (optional)
4. Create custom analysis cells for your needs
5. Share the notebook link with collaborators

---

Happy Colab-ing! 🚀
