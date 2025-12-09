# Google Colab Setup Checklist

## Before Using Colab

- [ ] Project pushed to GitHub (or ready to upload to Drive)
- [ ] `requirements.txt` is up to date
- [ ] Model weights (`models/ensemble/ensemble_final.pth`) available
- [ ] Have a Google account with Google Drive access

---

## Option A: GitHub Method (For Public Repos)

- [ ] Repository is PUBLIC (Settings → Visibility → Public)
- [ ] Latest changes pushed: `git push origin main`
- [ ] Open https://colab.research.google.com/
- [ ] File → Open → GitHub → `Snaju003/Final-year-project`
- [ ] Select `Deepfake_Detection_Colab.ipynb`
- [ ] Runtime → GPU
- [ ] Run Cell 1 (Environment Setup)
- [ ] Run Cell 2 (GPU Check)
- [ ] Continue with remaining cells

---

## Option B: Google Drive Method (Most Reliable)

### Step 1: Prepare Project ZIP
- [ ] On local machine, zip the entire project:
  ```bash
  Compress-Archive -Path "E:\Final-year-project" -DestinationPath "Final-year-project.zip"
  ```

### Step 2: Upload to Google Drive
- [ ] Open https://drive.google.com/
- [ ] Create folder: `MyDrive/Colab_Projects/` (optional)
- [ ] Upload `Final-year-project.zip` to `MyDrive/`
- [ ] Verify file shows in Drive

### Step 3: Setup in Colab
- [ ] Open https://colab.research.google.com/
- [ ] Create new notebook or open provided `Deepfake_Detection_Colab.ipynb`
- [ ] Runtime → GPU
- [ ] Run this cell:
  ```python
  from google.colab import drive
  import zipfile
  import os
  
  drive.mount('/content/drive', force_remount=True)
  
  # Extract ZIP
  zip_path = '/content/drive/MyDrive/Final-year-project.zip'
  with zipfile.ZipFile(zip_path, 'r') as z:
      z.extractall('/content')
  
  %cd /content/Final-year-project
  print("✅ Setup complete")
  ```
- [ ] Run Cell 2 (Install Dependencies)
- [ ] Run Cell 3 (GPU Check)
- [ ] Continue with remaining cells

---

## After Setup - Verification

- [ ] `import torch; print(torch.__version__)` → Should show 2.0.1
- [ ] `torch.cuda.is_available()` → Should return `True`
- [ ] `os.listdir('src')` → Should show `.py` files
- [ ] `os.path.exists('models/ensemble/ensemble_final.pth')` → Should return `True`
- [ ] Model loads: `from src.inference import EnsembleModel`
- [ ] Can create test image and predict

---

## Common Fixes

### If GitHub Clone Fails:
- [ ] Check if repo is public (Settings → Visibility)
- [ ] Try Drive method instead
- [ ] Check internet connection: `!ping github.com`

### If Drive Mount Fails:
- [ ] Make sure you're signed into correct Google account
- [ ] Try remounting: `from google.colab import drive; drive.mount('/content/drive', force_remount=True)`

### If imports fail:
- [ ] Verify working directory: `print(os.getcwd())`
- [ ] Should end with `/Final-year-project`
- [ ] Run: `%cd /content/Final-year-project`

### If GPU not available:
- [ ] Runtime → Change Runtime Type → Hardware Accelerator → GPU
- [ ] Select T4 or higher
- [ ] Restart runtime: Runtime → Restart Runtime

### If out of memory:
- [ ] Reduce batch size: change `batch_size=32` to `batch_size=16`
- [ ] Clear GPU: `torch.cuda.empty_cache()`
- [ ] Use smaller images: change `(128, 128)` to `(96, 96)`

---

## Running Inference

### Single Image:
```python
from src.colab_inference import ColabEnsembleInference

inference = ColabEnsembleInference()
result = inference.predict_single('path/to/image.jpg')
print(result)
```

### Batch Processing:
```python
results = inference.predict_batch('path/to/images/', batch_size=32)
```

### Download Results:
```python
from google.colab import files
import json

with open('results.json', 'w') as f:
    json.dump(results, f)

files.download('results.json')
```

---

## Resource Limits (Colab Free Tier)

- ✅ Session duration: 12 hours
- ✅ GPU: T4 (15 GB VRAM)
- ✅ RAM: 12 GB
- ✅ Disk: ~100 GB
- ❌ No background execution
- ❌ Idle timeout: ~30 minutes

---

## Performance Tips

| Task | Recommended Setting |
|------|---------------------|
| Inference | `batch_size=64`, T4 GPU |
| Fine-tuning | `batch_size=32`, T4 GPU |
| Training from scratch | `batch_size=16`, A100 GPU (Pro) |
| Dataset: Small (<100 images) | T4 sufficient |
| Dataset: Large (>1000 images) | Consider Colab Pro or local |

---

## Troubleshooting Checklist

- [ ] Is the Colab GPU enabled? (Runtime → Change Runtime Type)
- [ ] Is the project directory correct? (`os.getcwd()` ends with `Final-year-project`)
- [ ] Are all dependencies installed? (`pip list | grep torch`)
- [ ] Is the model file present? (`os.path.exists('models/...')`)
- [ ] Is enough GPU memory available? (`nvidia-smi`)
- [ ] Are imports working? (`from src.inference import EnsembleModel`)

---

## Quick Copy-Paste Setup

Create a **new Colab notebook** and run this cell:

```python
# Auto-setup with fallback
!git clone https://github.com/Snaju003/Final-year-project.git 2>/dev/null || \
(from google.colab import drive; \
 drive.mount('/content/drive', force_remount=True); \
 import zipfile; \
 zipfile.ZipFile('/content/drive/MyDrive/Final-year-project.zip').extractall('/content'))

%cd /content/Final-year-project
!pip install -q -r requirements.txt

print("✅ Setup complete!")
```

---

## Still Having Issues?

1. Check `COLAB_QUICK_START.md` for quick-start code
2. Check `COLAB_FILE_ACCESS.md` for file access methods
3. Check `COLAB_INTEGRATION_GUIDE.md` for detailed guide
4. Verify project structure with: `!find . -type f -name "*.py" | head -20`

---

**Good luck with Colab! 🚀**
