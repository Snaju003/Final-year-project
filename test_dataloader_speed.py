"""
DataLoader Speed Diagnostic Script
Tests actual image loading speed to identify bottlenecks
"""

import os
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Fix Unicode encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============ SETTINGS ============
DATA_ROOT = Path(r"E:\Final-year-project-data\data\dataset 2")
IMG_SIZE = 224
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test different num_workers configurations
NUM_WORKERS_CONFIGS = [0, 2, 4, 6, 8, 12]

# ============ DATASET ============
class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection with real/fake labels."""

    def __init__(self, real_files, fake_files, transform=None):
        self.files = []
        self.labels = []

        # Add real files (label 0)
        for f in real_files:
            self.files.append(str(f))
            self.labels.append(0)

        # Add fake files (label 1)
        for f in fake_files:
            self.files.append(str(f))
            self.labels.append(1)

        self.transform = transform
        print(f"📊 Dataset: {len(real_files)} real + {len(fake_files)} fake = {len(self.files)} total")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


# ============ TRANSFORMS ============
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============ LOAD DATA ============
def load_data(data_root, max_samples=None):
    """Load a subset of data for testing."""
    print("\n📂 Loading data...")

    train_dir = data_root / "train"
    train_real = list((train_dir / "Real").glob("*"))
    train_fake = list((train_dir / "Fake").glob("*"))

    if max_samples:
        train_real = train_real[:max_samples]
        train_fake = train_fake[:max_samples]

    print(f"✓ Loaded {len(train_real)} real + {len(train_fake)} fake images")
    return train_real, train_fake


# ============ BENCHMARK ============
def benchmark_loader(loader, num_batches=20):
    """Benchmark DataLoader speed."""
    times = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(loader, total=num_batches, desc="Loading")):
        batch_time = time.time()
        if batch_idx == 0:
            start_time = batch_time
            continue
        times.append(batch_time - start_time)
        start_time = batch_time
        
        if batch_idx >= num_batches:
            break

    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        images_per_sec = BATCH_SIZE / avg_time
        return avg_time * 1000, std_time * 1000, images_per_sec
    return 0, 0, 0


# ============ MAIN ============
def main():
    print(f"\n{'='*80}")
    print(f"📊 DataLoader Speed Diagnostic")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'='*80}\n")

    # Load data (use subset for faster testing)
    train_real, train_fake = load_data(DATA_ROOT, max_samples=500)

    # Create dataset
    dataset = DeepfakeDataset(train_real, train_fake, transform=transform)

    # Test different configurations
    results = []

    for num_workers in NUM_WORKERS_CONFIGS:
        print(f"\n🧪 Testing num_workers={num_workers}...")
        
        try:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(DEVICE.type == "cuda"),
                persistent_workers=False,
                prefetch_factor=2,
            )

            avg_time, std_time, img_per_sec = benchmark_loader(loader, num_batches=20)
            
            result = {
                'num_workers': num_workers,
                'avg_batch_time_ms': avg_time,
                'std_batch_time_ms': std_time,
                'images_per_sec': img_per_sec,
            }
            results.append(result)
            
            print(f"   ✓ Batch Time: {avg_time:.1f}±{std_time:.1f}ms")
            print(f"   ✓ Speed: {img_per_sec:.0f} img/s")
            
            # Cleanup
            del loader
            
        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY - Ranked by Speed")
    print(f"{'='*80}")
    
    sorted_results = sorted(results, key=lambda x: x['images_per_sec'], reverse=True)
    
    print(f"\n{'Workers':<10} {'Batch Time (ms)':<20} {'Speed (img/s)':<15}")
    print(f"{'-'*45}")
    
    for r in sorted_results:
        workers = r['num_workers']
        batch_time = r['avg_batch_time_ms']
        speed = r['images_per_sec']
        marker = "🟢 BEST" if r == sorted_results[0] else ""
        print(f"{workers:<10} {batch_time:<20.1f} {speed:<15.0f} {marker}")

    best = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"✨ RECOMMENDATION:")
    print(f"   Use num_workers={best['num_workers']} for fastest loading")
    print(f"   Expected epoch time at 20 epochs: ~{(best['avg_batch_time_ms'] * 1000 / 128 * len(dataset) / BATCH_SIZE / 60):.1f} minutes")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
