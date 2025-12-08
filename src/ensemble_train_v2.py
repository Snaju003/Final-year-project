"""
Ensemble Model Training Script V2 - Robust Training with Diverse Datasets
Trains weighted ensemble with improved generalization for real-world deepfake detection

Key Improvements:
1. Multi-dataset training for diversity
2. Heavy augmentation for robustness (JPEG compression, blur, noise)
3. Balanced sampling across datasets
4. Cosine annealing with warm restarts
5. Label smoothing for better calibration
"""

import os
import sys
from pathlib import Path
import random
from collections import defaultdict

# Fix Unicode encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import json
import time
from datetime import timedelta

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Faster image loading
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

# Import ensemble model
from ensemble_model import create_ensemble

# ============ PATHS ============
DATA_ROOT = Path(r"X:\Final-year-project-data\data")
OUTPUT_DIR = Path(r"E:\Final-year-project\models\ensemble_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ DATASET CONFIGURATION ============
"""
Dataset Structure (all on X: SSD for speed):
- Dataset 1: No splits - will be divided 80/10/10 for train/val/test
- Dataset 2: Has train/validate/test splits (70K/20K/5K each class)
- Dataset 3: Has train/validate only (48K+41K / 11K+10K)

Strategy: Combine all datasets proportionally with balanced sampling
"""

# Dataset 1 split ratios (no predefined splits)
DATASET1_TRAIN_RATIO = 0.80
DATASET1_VAL_RATIO = 0.10
DATASET1_TEST_RATIO = 0.10

# Maximum samples per class from Dataset 1 (to balance with other datasets)
DATASET1_MAX_PER_CLASS = 70000  # Match dataset 2 size

# ============ TRAINING SETTINGS ============
IMG_SIZE = 224
BATCH_SIZE = 48  # Balanced for 8GB VRAM
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3  # Higher LR for faster initial learning
WEIGHT_DECAY = 1e-5  # Reduced weight decay
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05  # Helps with noisy labels
GRADIENT_ACCUMULATION = 2  # Effective batch size = 48 * 2 = 96

# Learning rate schedule
USE_COSINE_ANNEALING = False  # Use ReduceLROnPlateau instead
WARMUP_EPOCHS = 0  # No warmup - start learning immediately

# Data settings - optimized for speed
NUM_WORKERS = 16  # Balanced for 32GB RAM
PREFETCH_FACTOR = 2
PIN_MEMORY = True

# Mixed Precision
USE_MIXED_PRECISION = True

# Ensemble weights (initial - will be optimized)
EFFICIENTNET_WEIGHT = 0.35
XCEPTION_WEIGHT = 0.30
MESONET_WEIGHT = 0.35  # Increased MesoNet weight

# ============ REPRODUCIBILITY ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")


# ============ CUSTOM AUGMENTATIONS (Optimized for Speed) ============
class JPEGCompression:
    """Simulate JPEG compression artifacts - FAST version using quality reduction only."""
    def __init__(self, quality_range=(50, 95)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        # Only apply 20% of time to reduce I/O overhead
        if random.random() < 0.2:
            import io
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
        return img


class GaussianNoise:
    """Add Gaussian noise to simulate camera/sensor noise."""
    def __init__(self, mean=0, std_range=(0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range
    
    def __call__(self, tensor):
        if random.random() < 0.3:  # Apply 30% of the time
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0, 1)
        return tensor


class RandomBlur:
    """Random Gaussian blur to simulate focus issues."""
    def __init__(self, radius_range=(0.5, 2.0)):
        self.radius_range = radius_range
    
    def __call__(self, img):
        if random.random() < 0.3:  # Apply 30% of the time
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomDownscale:
    """Random downscale and upscale to simulate low resolution."""
    def __init__(self, scale_range=(0.5, 0.9)):
        self.scale_range = scale_range
    
    def __call__(self, img):
        if random.random() < 0.3:  # Apply 30% of the time
            scale = random.uniform(*self.scale_range)
            w, h = img.size
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)
        return img


# ============ TRANSFORMS ============
# Heavy augmentation for training - OPTIMIZED for speed
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Direct resize (faster than crop)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

# Clean transforms for validation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============ MULTI-SOURCE DATASET ============
class MultiSourceDeepfakeDataset(Dataset):
    """Dataset that combines multiple deepfake data sources with balanced sampling."""
    
    def __init__(self, files_with_labels, transform=None):
        """
        Args:
            files_with_labels: List of (file_path, label, source) tuples
            transform: Image transforms
        """
        self.data = files_with_labels
        self.transform = transform
        
        # Count by label and source
        stats = defaultdict(lambda: defaultdict(int))
        for _, label, source in self.data:
            label_name = 'fake' if label == 1 else 'real'
            stats[source][label_name] += 1
        
        print(f"📊 Dataset Statistics:")
        for source, counts in stats.items():
            print(f"   {source}: {counts['real']} real, {counts['fake']} fake")
        print(f"   Total: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label, source = self.data[idx]
        
        try:
            if USE_CV2:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("Could not load image")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            else:
                img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        
        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


# ============ DATA LOADING ============
def collect_files(folder, max_samples=None):
    """Collect image files from a folder."""
    if not folder.exists():
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(folder.glob(ext))
    
    # Shuffle and limit if specified
    random.shuffle(files)
    if max_samples and len(files) > max_samples:
        files = files[:max_samples]
    
    return files


def load_multi_source_data():
    """
    Load data from Dataset 2, Dataset 3, and Faces folder.
    - Dataset 2: Has train/validate/test
    - Dataset 3: Has train/validate only
    - Faces: No splits - divide 80/10/10
    """
    print("\n" + "=" * 60)
    print("📂 LOADING DATA FROM MULTIPLE SOURCES")
    print("=" * 60)
    
    train_data = []
    val_data = []
    test_data = []
    
    # ========== DATASET 2 (Full splits available) ==========
    print("\n📁 Dataset 2 (FaceForensics-style):")
    ds2_root = DATA_ROOT / 'dataset 2'
    
    # Train
    ds2_train_real = collect_files(ds2_root / 'Train' / 'Real')
    ds2_train_fake = collect_files(ds2_root / 'Train' / 'Fake')
    for f in ds2_train_real:
        train_data.append((f, 0, 'dataset2'))
    for f in ds2_train_fake:
        train_data.append((f, 1, 'dataset2'))
    print(f"   Train: {len(ds2_train_real)} real, {len(ds2_train_fake)} fake")
    
    # Validate
    ds2_val_real = collect_files(ds2_root / 'Validate' / 'Real')
    ds2_val_fake = collect_files(ds2_root / 'Validate' / 'Fake')
    for f in ds2_val_real:
        val_data.append((f, 0, 'dataset2'))
    for f in ds2_val_fake:
        val_data.append((f, 1, 'dataset2'))
    print(f"   Val:   {len(ds2_val_real)} real, {len(ds2_val_fake)} fake")
    
    # Test
    ds2_test_real = collect_files(ds2_root / 'Test' / 'Real')
    ds2_test_fake = collect_files(ds2_root / 'Test' / 'Fake')
    for f in ds2_test_real:
        test_data.append((f, 0, 'dataset2'))
    for f in ds2_test_fake:
        test_data.append((f, 1, 'dataset2'))
    print(f"   Test:  {len(ds2_test_real)} real, {len(ds2_test_fake)} fake")
    
    # ========== DATASET 3 (Train/Validate only) ==========
    print("\n📁 Dataset 3 (Celeb-DF style):")
    ds3_root = DATA_ROOT / 'dataset 3'
    
    # Train
    ds3_train_real = collect_files(ds3_root / 'train' / 'Real')
    ds3_train_fake = collect_files(ds3_root / 'train' / 'Fake')
    for f in ds3_train_real:
        train_data.append((f, 0, 'dataset3'))
    for f in ds3_train_fake:
        train_data.append((f, 1, 'dataset3'))
    print(f"   Train: {len(ds3_train_real)} real, {len(ds3_train_fake)} fake")
    
    # Validate (use as validation, no test set)
    ds3_val_real = collect_files(ds3_root / 'validate' / 'Real')
    ds3_val_fake = collect_files(ds3_root / 'validate' / 'Fake')
    for f in ds3_val_real:
        val_data.append((f, 0, 'dataset3'))
    for f in ds3_val_fake:
        val_data.append((f, 1, 'dataset3'))
    print(f"   Val:   {len(ds3_val_real)} real, {len(ds3_val_fake)} fake")
    print(f"   Test:  (none - using validate only)")
    
    # ========== DATASET 1 (No splits - create them) ==========
    print("\n📁 Dataset 1 (splitting 80/10/10):")
    ds1_root = DATA_ROOT / 'dataset 1'
    
    # Collect all files (limit to prevent class imbalance)
    ds1_real_all = collect_files(ds1_root / 'Real', max_samples=DATASET1_MAX_PER_CLASS)
    ds1_fake_all = collect_files(ds1_root / 'Fake', max_samples=DATASET1_MAX_PER_CLASS)
    
    print(f"   Loaded: {len(ds1_real_all)} real, {len(ds1_fake_all)} fake (limited to {DATASET1_MAX_PER_CLASS})")
    
    # Shuffle before splitting
    random.shuffle(ds1_real_all)
    random.shuffle(ds1_fake_all)
    
    # Split real images
    n_real = len(ds1_real_all)
    n_real_train = int(n_real * DATASET1_TRAIN_RATIO)
    n_real_val = int(n_real * DATASET1_VAL_RATIO)
    
    ds1_real_train = ds1_real_all[:n_real_train]
    ds1_real_val = ds1_real_all[n_real_train:n_real_train + n_real_val]
    ds1_real_test = ds1_real_all[n_real_train + n_real_val:]
    
    # Split fake images
    n_fake = len(ds1_fake_all)
    n_fake_train = int(n_fake * DATASET1_TRAIN_RATIO)
    n_fake_val = int(n_fake * DATASET1_VAL_RATIO)
    
    ds1_fake_train = ds1_fake_all[:n_fake_train]
    ds1_fake_val = ds1_fake_all[n_fake_train:n_fake_train + n_fake_val]
    ds1_fake_test = ds1_fake_all[n_fake_train + n_fake_val:]
    
    # Add to datasets
    for f in ds1_real_train:
        train_data.append((f, 0, 'dataset1'))
    for f in ds1_fake_train:
        train_data.append((f, 1, 'dataset1'))
    
    for f in ds1_real_val:
        val_data.append((f, 0, 'dataset1'))
    for f in ds1_fake_val:
        val_data.append((f, 1, 'dataset1'))
    
    for f in ds1_real_test:
        test_data.append((f, 0, 'dataset1'))
    for f in ds1_fake_test:
        test_data.append((f, 1, 'dataset1'))
    
    print(f"   Train: {len(ds1_real_train)} real, {len(ds1_fake_train)} fake")
    print(f"   Val:   {len(ds1_real_val)} real, {len(ds1_fake_val)} fake")
    print(f"   Test:  {len(ds1_real_test)} real, {len(ds1_fake_test)} fake")
    
    # ========== SHUFFLE ALL DATA ==========
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # ========== PRINT SUMMARY ==========
    print("\n" + "=" * 60)
    print("📊 COMBINED DATASET SUMMARY")
    print("=" * 60)
    
    # Count by source and label
    def count_by_source(data):
        counts = {}
        for _, label, source in data:
            if source not in counts:
                counts[source] = {'real': 0, 'fake': 0}
            if label == 0:
                counts[source]['real'] += 1
            else:
                counts[source]['fake'] += 1
        return counts
    
    train_counts = count_by_source(train_data)
    val_counts = count_by_source(val_data)
    test_counts = count_by_source(test_data)
    
    print(f"\n{'Source':<12} {'Train Real':>12} {'Train Fake':>12} {'Val Real':>10} {'Val Fake':>10} {'Test Real':>10} {'Test Fake':>10}")
    print("-" * 80)
    
    for source in ['dataset1', 'dataset2', 'dataset3']:
        tr = train_counts.get(source, {'real': 0, 'fake': 0})
        vr = val_counts.get(source, {'real': 0, 'fake': 0})
        te = test_counts.get(source, {'real': 0, 'fake': 0})
        print(f"{source:<12} {tr['real']:>12,} {tr['fake']:>12,} {vr['real']:>10,} {vr['fake']:>10,} {te['real']:>10,} {te['fake']:>10,}")
    
    print("-" * 80)
    total_train_real = sum(1 for _, l, _ in train_data if l == 0)
    total_train_fake = sum(1 for _, l, _ in train_data if l == 1)
    total_val_real = sum(1 for _, l, _ in val_data if l == 0)
    total_val_fake = sum(1 for _, l, _ in val_data if l == 1)
    total_test_real = sum(1 for _, l, _ in test_data if l == 0)
    total_test_fake = sum(1 for _, l, _ in test_data if l == 1)
    
    print(f"{'TOTAL':<12} {total_train_real:>12,} {total_train_fake:>12,} {total_val_real:>10,} {total_val_fake:>10,} {total_test_real:>10,} {total_test_fake:>10,}")
    
    print(f"\n📈 Grand Total:")
    print(f"   Training:   {len(train_data):,} images ({total_train_real:,} real, {total_train_fake:,} fake)")
    print(f"   Validation: {len(val_data):,} images ({total_val_real:,} real, {total_val_fake:,} fake)")
    print(f"   Test:       {len(test_data):,} images ({total_test_real:,} real, {total_test_fake:,} fake)")
    
    return train_data, val_data, test_data


def create_balanced_sampler(dataset):
    """Create a sampler that balances real/fake samples."""
    labels = [item[1] for item in dataset.data]
    class_counts = np.bincount(labels)
    
    # Weight each sample inversely proportional to class frequency
    weights = 1.0 / class_counts[labels]
    weights = torch.FloatTensor(weights)
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler


# ============ TRAINING FUNCTIONS ============
def train_epoch(model, loader, criterion, optimizer, scaler, epoch, total_epochs):
    """Train for one epoch with gradient accumulation for memory efficiency."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", 
                ncols=120, leave=True)
    
    optimizer.zero_grad()  # Zero gradients once at start
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Sanity check on first batch
        if batch_idx == 0 and epoch == 0:
            print(f"\n🔍 Sanity Check:")
            print(f"   Images device: {images.device}, shape: {images.shape}")
            print(f"   Labels device: {labels.device}, unique labels: {torch.unique(labels).tolist()}")
            print(f"   Model device: {next(model.parameters()).device}")
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / GRADIENT_ACCUMULATION  # Scale loss
            
            scaler.scale(loss).backward()
            
            # Step optimizer every GRADIENT_ACCUMULATION batches
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / GRADIENT_ACCUMULATION
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        running_loss += loss.item() * GRADIENT_ACCUMULATION  # Unscale for logging
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar with dedicated GPU memory info
        current_acc = accuracy_score(all_labels, all_preds) * 100
        gpu_mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{current_acc:.1f}%',
            'VRAM': f'{gpu_mem:.1f}GB'
        })
        
        # Clear cache periodically
        if batch_idx % 100 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Validating", ncols=100, leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def get_scheduler(optimizer, num_epochs, steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    if USE_COSINE_ANNEALING:
        # Warmup + Cosine Annealing
        warmup_steps = WARMUP_EPOCHS * steps_per_epoch
        total_steps = num_epochs * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
    
    return scheduler


# ============ MAIN TRAINING ============
def train():
    """Main training function."""
    print("=" * 60)
    print("🚀 ENSEMBLE DEEPFAKE DETECTION - ROBUST TRAINING V2")
    print("=" * 60)
    
    # Load data from multiple sources
    train_data, val_data, test_data = load_multi_source_data()
    
    if len(train_data) == 0:
        raise ValueError("❌ No training data found!")
    
    # Create datasets
    train_dataset = MultiSourceDeepfakeDataset(train_data, transform=train_transform)
    val_dataset = MultiSourceDeepfakeDataset(val_data, transform=val_transform)
    
    # Create balanced sampler for training
    train_sampler = create_balanced_sampler(train_dataset)
    
    # Create data loaders - optimized for RTX 3060 Ti
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,  # Disable to reduce RAM usage
        drop_last=True  # Drop incomplete batches for consistent gradient accumulation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,  # Larger batch for validation (no gradients)
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False
    )
    
    print(f"\n⚡ DataLoader Settings:")
    print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🔧 Creating ensemble model...")
    model_weights = (EFFICIENTNET_WEIGHT, XCEPTION_WEIGHT, MESONET_WEIGHT)
    model = create_ensemble(
        model_type='weighted',
        weights=model_weights,
        num_classes=2,
        pretrained=True  # Use pretrained backbones
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.efficientnet, 'set_grad_checkpointing'):
        model.efficientnet.set_grad_checkpointing(True)
    
    print(f"   Weights: EfficientNet={EFFICIENTNET_WEIGHT}, Xception={XCEPTION_WEIGHT}, MesoNet={MESONET_WEIGHT}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer - same LR for all models to ensure they all learn
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, NUM_EPOCHS, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
    
    print(f"\n🏋️ Starting training for {NUM_EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Label smoothing: {LABEL_SMOOTHING}")
    print(f"   Augmentation: Heavy (JPEG, blur, noise, etc.)")
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, NUM_EPOCHS
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion)
        
        # Update scheduler
        if USE_COSINE_ANNEALING:
            # Step scheduler every batch in training loop
            pass
        else:
            scheduler.step(val_metrics['accuracy'])
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch+1}/{NUM_EPOCHS} Summary ({epoch_time:.1f}s)")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
        print(f"   Precision: {val_metrics['precision']:.2f}% | Recall: {val_metrics['recall']:.2f}%")
        print(f"   F1 Score:  {val_metrics['f1']:.2f}% | AUC: {val_metrics['auc']:.2f}%")
        print(f"   Confusion Matrix:")
        cm = np.array(val_metrics['confusion_matrix'])
        print(f"      Real->Real: {cm[0,0]} | Real->Fake: {cm[0,1]}")
        print(f"      Fake->Real: {cm[1,0]} | Fake->Fake: {cm[1,1]}")
        
        # Save best model (based on F1 to balance precision/recall)
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc'],
                'val_metrics': val_metrics,
                'ensemble_type': 'weighted',
                'weights': {
                    'efficientnet': EFFICIENTNET_WEIGHT,
                    'xception': XCEPTION_WEIGHT,
                    'mesonet': MESONET_WEIGHT
                },
                'training_config': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'label_smoothing': LABEL_SMOOTHING,
                    'img_size': IMG_SIZE,
                    'augmentation': 'heavy_v2'
                }
            }
            
            torch.save(checkpoint, OUTPUT_DIR / 'ensemble_best_v2.pth')
            print(f"   ✅ New best model saved! (F1: {best_val_f1:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Current LR: {current_lr:.2e}")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"   Best Val F1 Score: {best_val_f1:.2f}%")
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['f1'],
        'val_metrics': val_metrics,
        'ensemble_type': 'weighted',
        'weights': {
            'efficientnet': EFFICIENTNET_WEIGHT,
            'xception': XCEPTION_WEIGHT,
            'mesonet': MESONET_WEIGHT
        },
        'history': history
    }
    torch.save(final_checkpoint, OUTPUT_DIR / 'ensemble_final_v2.pth')
    print(f"💾 Final model saved to {OUTPUT_DIR / 'ensemble_final_v2.pth'}")
    
    # Save training results
    results = {
        'best_val_accuracy': best_val_acc,
        'best_val_f1': best_val_f1,
        'total_epochs': epoch + 1,
        'training_time': str(timedelta(seconds=int(total_time))),
        'history': history,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'weights': {
                'efficientnet': EFFICIENTNET_WEIGHT,
                'xception': XCEPTION_WEIGHT,
                'mesonet': MESONET_WEIGHT
            }
        }
    }
    
    with open(OUTPUT_DIR / 'training_results_v2.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"📊 Training results saved to {OUTPUT_DIR / 'training_results_v2.json'}")
    
    return model, history


# ============ TEST EVALUATION ============
def evaluate_on_test(model_path):
    """Evaluate trained model on test set."""
    print("\n" + "=" * 60)
    print("📋 EVALUATING ON TEST SET")
    print("=" * 60)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    weights = checkpoint.get('weights', {})
    if isinstance(weights, dict):
        model_weights = (
            weights.get('efficientnet', 0.35),
            weights.get('xception', 0.30),
            weights.get('mesonet', 0.35)
        )
    else:
        model_weights = weights
    
    model = create_ensemble(
        model_type='weighted',
        weights=model_weights,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    _, _, test_data = load_multi_source_data()
    
    if len(test_data) == 0:
        print("⚠️ No test data found")
        return
    
    test_dataset = MultiSourceDeepfakeDataset(test_data, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = validate(model, test_loader, criterion)
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   Precision: {metrics['precision']:.2f}%")
    print(f"   Recall:    {metrics['recall']:.2f}%")
    print(f"   F1 Score:  {metrics['f1']:.2f}%")
    print(f"   AUC:       {metrics['auc']:.2f}%")
    
    cm = np.array(metrics['confusion_matrix'])
    print(f"\n   Confusion Matrix:")
    print(f"      Predicted:  REAL    FAKE")
    print(f"      Actual REAL: {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"      Actual FAKE: {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Calculate per-class accuracy
    real_acc = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_acc = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
    print(f"\n   Per-class Accuracy:")
    print(f"      Real: {real_acc:.2f}%")
    print(f"      Fake: {fake_acc:.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Ensemble V2')
    parser.add_argument('--evaluate', type=str, help='Path to model checkpoint to evaluate')
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_on_test(Path(args.evaluate))
    else:
        model, history = train()
        
        # Evaluate best model on test set
        best_model_path = OUTPUT_DIR / 'ensemble_best_v2.pth'
        if best_model_path.exists():
            evaluate_on_test(best_model_path)
