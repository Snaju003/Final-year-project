# Google Colab - Quick Reference Guide

## Error You Got

```
fatal: could not read Username for 'https://github.com'
[Errno 2] No such file or directory
```

## Cause → Solution Flow

```
Error occurred?
    ↓
Is your repo PUBLIC on GitHub?
    ├─ YES → GitHub clone should work (retry cell)
    └─ NO → Use Google Drive method ⭐
```

---

## The 2 Main Methods

### Method 1: GitHub (Only if repo is PUBLIC)
```bash
# In Colab cell:
!git clone https://github.com/Snaju003/Final-year-project.git
%cd Final-year-project
!pip install -q -r requirements.txt
```
⏱️ **Time:** ~2 minutes  
✅ **Best for:** Public repos, team collaboration  
❌ **Problem:** Won't work if repo is private

---

### Method 2: Google Drive ⭐ (RECOMMENDED)
**Step 1:** On your local machine
```bash
Compress-Archive -Path "E:\Final-year-project" -DestinationPath "fp.zip"
# Upload fp.zip to Google Drive
```

**Step 2:** In Colab
```python
from google.colab import drive
import zipfile

drive.mount('/content/drive', force_remount=True)
with zipfile.ZipFile('/content/drive/MyDrive/fp.zip', 'r') as z:
    z.extractall('/content')

%cd /content/Final-year-project
!pip install -q -r requirements.txt
```

⏱️ **Time:** ~3 minutes  
✅ **Best for:** Private repos, reliable setup  
✅ **Works:** Always (no network dependency)

---

## After Setup - Verification

```python
# Should all return ✅

import os
assert os.path.exists('src'), "❌ src not found"
assert os.path.exists('models'), "❌ models not found"

import torch
assert torch.cuda.is_available(), "❌ GPU not available"

from src.inference import EnsembleModel
model = EnsembleModel()

print("✅ ALL CHECKS PASSED - Ready to use!")
```

---

## Run Inference

### Single Image
```python
from src.colab_inference import ColabEnsembleInference
inference = ColabEnsembleInference()
result = inference.predict_single('image.jpg')
print(result['prediction'])  # "DEEPFAKE" or "REAL"
```

### Multiple Images
```python
results = inference.predict_batch('path/to/images')
for r in results:
    print(f"{r['image']}: {r['prediction']} ({r['confidence']:.1%})")
```

### Download Results
```python
from google.colab import files
import json

with open('results.json', 'w') as f:
    json.dump(results, f)

files.download('results.json')
```

---

## Common Errors & Quick Fixes

### ❌ "Cannot find module src"
```python
import sys
sys.path.insert(0, '/content/Final-year-project')
from src.inference import EnsembleModel  # Now works
```

### ❌ "GPU not available"
→ Runtime → Change Runtime Type → Hardware Accelerator → GPU

### ❌ "File not found"
```python
import os
print(os.getcwd())  # Should end with Final-year-project
print(os.listdir('src'))  # Should list Python files
```

### ❌ "Out of Memory"
```python
# Reduce batch size
results = inference.predict_batch('images/', batch_size=16)  # was 32

# Clear GPU
torch.cuda.empty_cache()
```

### ❌ "Cannot mount Google Drive"
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)  # Add force_remount=True
```

---

## Setup Decision Tree

```
Start here
    ↓
Is repo on GitHub?
    ├─ YES
    │  ↓
    │  Is it PUBLIC?
    │  ├─ YES → Use GitHub clone (quick!)
    │  └─ NO  → Use Google Drive (safer)
    │
    └─ NO → Use Google Drive
```

---

## Checklists

### Before Colab
- [ ] Have project files ready
- [ ] Have GPU access (Colab Free account)
- [ ] Pick your method (GitHub or Drive)

### GitHub Method
- [ ] Repo is PUBLIC (Settings → Visibility)
- [ ] Latest changes pushed
- [ ] Open Colab
- [ ] Run clone cell

### Drive Method
- [ ] Project zipped
- [ ] ZIP uploaded to Drive
- [ ] Open Colab
- [ ] Run Drive cell

### After Setup
- [ ] Check GPU: `torch.cuda.is_available()` → True
- [ ] Check directory: `os.getcwd()` → ends with `Final-year-project`
- [ ] Check imports: `from src.inference import EnsembleModel` → works
- [ ] Test inference: model produces output

---

## File Locations in Colab

| File | Location |
|------|----------|
| Project root | `/content/Final-year-project` |
| Source code | `/content/Final-year-project/src/` |
| Models | `/content/Final-year-project/models/` |
| Google Drive | `/content/drive/MyDrive/` |
| Uploaded images | `/content/uploads/` (if using file upload) |

---

## Performance Tips

**GPU Usage:**
```bash
# Monitor during inference
!nvidia-smi
```

**Faster Batch Processing:**
```python
# Optimal batch size for T4 GPU
results = inference.predict_batch('images/', batch_size=64)
```

**Memory Management:**
```python
import torch
torch.cuda.empty_cache()  # Clear GPU memory
```

---

## Documentation Files Reference

| File | Purpose | When to Read |
|------|---------|--------------|
| **COLAB_QUICK_START.md** | Get started NOW | First time users |
| **COLAB_CHECKLIST.md** | Step-by-step setup | Doing full setup |
| **COLAB_FILE_ACCESS.md** | All setup methods | Troubleshooting |
| **COLAB_INTEGRATION_GUIDE.md** | Detailed guide | Advanced usage |
| **COLAB_SOLUTION.md** | Big picture | Understanding |
| **This file** | Quick reference | Quick lookup |

---

## Success Indicators ✅

When you see these, your setup is working:

```
✅ GPU Available: Tesla T4
✅ Project ready on /content/Final-year-project
✅ Can import EnsembleModel
✅ Model inference returns value between 0-1
✅ Batch processing shows progress bar
✅ Can download results.json
```

---

## Getting Help

1. Check corresponding `.md` file for detailed help
2. Look at error message - usually tells what's wrong
3. Check your working directory: `print(os.getcwd())`
4. Verify GPU is enabled: Runtime → GPU type
5. Check file exists: `os.path.exists(path)`

---

## One-Liner Setup (Copy & Paste)

```python
# For public repos (make sure repo is PUBLIC first!)
!git clone https://github.com/Snaju003/Final-year-project.git && cd Final-year-project && pip install -q -r requirements.txt && echo "✅ Done"
```

```python
# For private/offline (need zip in Drive)
from google.colab import drive; import zipfile; drive.mount('/content/drive'); zipfile.ZipFile('/content/drive/MyDrive/fp.zip').extractall('/content'); import os; os.chdir('/content/Final-year-project'); exec(open('requirements.txt').read()) if False else __import__('subprocess').run(__import__('sys').executable + ' -m pip install -q -r requirements.txt', shell=True); print("✅ Done")
```

---

## TL;DR (Too Long; Didn't Read)

1. **Repo is public?** → Use GitHub clone
2. **Repo is private?** → Zip & upload to Drive
3. **Run setup cells** → Wait 3 minutes
4. **Test GPU** → Check `torch.cuda.is_available()`
5. **Run inference** → `inference.predict_single('image.jpg')`
6. **Download results** → `files.download('results.json')`

That's it! 🚀

---

**Last Updated:** December 9, 2025  
**Status:** ✅ Ready for Production
