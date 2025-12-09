# COLAB QUICK START - Copy this into a Colab cell to get started

## Cell 1: Auto-Setup (Copy-paste this entire cell)

```python
# Auto-setup script for Google Colab
import os
import subprocess
import sys
from pathlib import Path

print("🚀 STARTING COLAB SETUP\n")

# Step 1: Try GitHub clone
print("Step 1: Attempting GitHub clone...")
try:
    !git clone https://github.com/Snaju003/Final-year-project.git
    %cd Final-year-project
    print("✅ GitHub clone successful\n")
except:
    print("⚠️  GitHub clone failed\n")
    
    # Step 2: Fall back to Google Drive
    print("Step 2: Setting up from Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        
        import shutil
        import zipfile
        from pathlib import Path
        
        drive_path = '/content/drive/MyDrive'
        
        # Look for ZIP file
        zip_files = list(Path(drive_path).glob('*.zip'))
        found = False
        
        for zf in zip_files:
            if 'Final-year-project' in zf.name or 'fyp' in zf.name.lower():
                print(f"Found: {zf.name}, extracting...")
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall('/content')
                found = True
                break
        
        if not found:
            # Look for folder
            project_folder = Path(drive_path) / 'Final-year-project'
            if project_folder.exists():
                print(f"Found folder, copying...")
                shutil.copytree(project_folder, '/content/Final-year-project')
                found = True
        
        if found:
            %cd /content/Final-year-project
            print("✅ Setup from Google Drive successful\n")
        else:
            print("❌ Project not found in Drive")
            print("📋 Please upload 'Final-year-project.zip' to Google Drive/MyDrive/\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")

# Step 3: Install dependencies
print("Step 3: Installing dependencies...")
if os.path.exists('requirements.txt'):
    !pip install -q -r requirements.txt
    print("✅ Dependencies installed\n")
else:
    print("⚠️  requirements.txt not found, installing essentials...")
    packages = [
        'torch==2.0.1', 'torchvision==0.15.2', 'timm==0.9.12',
        'facenet-pytorch==2.5.3', 'opencv-python==4.8.1.78',
        'tqdm==4.65.0', 'pytorch-gradcam==0.2.1',
        'scikit-image==0.21.0', 'scikit-learn==1.3.2',
        'Pillow==10.0.1'
    ]
    for pkg in packages:
        !pip install -q {pkg}
    print("✅ Essential packages installed\n")

# Step 4: Verify
print("Step 4: Verifying setup...")
checks = {
    "src/": os.path.exists('src'),
    "models/": os.path.exists('models'),
    "requirements.txt": os.path.exists('requirements.txt'),
}

all_ok = all(checks.values())
for check, result in checks.items():
    print(f"  {'✅' if result else '❌'} {check}")

if all_ok:
    print("\n🎉 SETUP COMPLETE!\n")
else:
    print("\n⚠️  Some components missing - you may need to upload files\n")

print("=" * 60)
print("Next: Go to Cell 2 to test the setup")
print("=" * 60)
```

---

## Cell 2: Check GPU & Test

```python
import torch

print("GPU Status:")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  Using CPU")

# Verify project
import os
print("\nProject Structure:")
print(f"Current: {os.getcwd()}")
print(f"✅ src/: {os.path.exists('src')}")
print(f"✅ models/: {os.path.exists('models')}")
```

---

## Cell 3: Load & Test Model

```python
import sys
import torch
from src.inference import EnsembleModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnsembleModel().to(device)

# Try to load weights
model_path = 'models/ensemble/ensemble_final.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model loaded from {model_path}")
else:
    print(f"⚠️  Weights not found - using random init")

model.eval()
print(f"✅ Model ready on {device}")

# Test inference
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create dummy image
dummy_img = torch.randn(1, 3, 128, 128).to(device)
with torch.no_grad():
    output = model(dummy_img)

print(f"✅ Inference works! Output: {output.item():.4f}")
```

---

## Cell 4: Predict on Images

```python
from pathlib import Path
from tqdm import tqdm

def predict_batch(image_dir, batch_size=32):
    """Predict on all images in a directory"""
    
    images = list(Path(image_dir).glob('*.jpg')) + \
             list(Path(image_dir).glob('*.png'))
    
    results = []
    
    for img_path in tqdm(images):
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = model(img_t).item()
            
            results.append({
                'image': img_path.name,
                'probability': prob,
                'label': 'DEEPFAKE' if prob > 0.5 else 'REAL'
            })
        except Exception as e:
            print(f"Error: {img_path} - {e}")
    
    return results

# Example: Upload images, then run
# results = predict_batch('/content/my_images')
# for r in results:
#     print(f"{r['image']}: {r['label']} ({r['probability']:.4f})")
```

---

## Cell 5: Download Results

```python
from google.colab import files
import json

# Save results
with open('predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

# Download
files.download('predictions.json')
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `fatal: could not read Username` | Repo is private. Upload to Google Drive instead |
| `ModuleNotFoundError: src` | Wrong working directory. Run: `%cd /content/Final-year-project` |
| `No such file or directory: 'Final-year-project'` | Repo not cloned. Check network or upload to Drive |
| `CUDA out of memory` | Reduce batch_size or image resolution |
| `Model not found` | Upload `models/ensemble/ensemble_final.pth` to Drive |

---

**Need help?** Check `COLAB_FILE_ACCESS.md` in the repo for detailed setup options.
