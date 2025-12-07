"""
Ensemble Model Training Script - Optimized with Enhanced Statistics Display
Trains weighted ensemble of EfficientNet, XceptionNet, and MesoNet for deepfake detection

Weights:
- EfficientNet: 40%
- XceptionNet: 35%
- MesoNet: 25%
"""

import os
import sys
from pathlib import Path

# Fix Unicode encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
import random
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
DATA_ROOT = Path(r"E:\Final-year-project-data\data\dataset 2")
OUTPUT_DIR = Path(r"E:\Final-year-project\models\ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ TRAINING SETTINGS ============
IMG_SIZE = 224

# Default batch size (may be overridden by auto-tuner)
DEFAULT_BATCH_SIZE = 128  # Good starting guess for RTX 3060 Ti (8GB VRAM)

NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5

# ✅ AUTO TUNING CONFIG
AUTO_TUNE_BATCH_SIZE = False  # Set to False for fixed batch size
BATCH_SIZE = 64  # Balanced batch size
BATCH_SIZE_CANDIDATES = [256, 224, 192, 160, 128, 96, 64, 48, 32]

# ✅ OPTIMIZED: Use CPU cores efficiently (Ryzen 7 5700X → 16 threads)
CPU_CORES = os.cpu_count() or 16
# More workers with faster image loading (OpenCV is 2-3x faster than PIL)
NUM_WORKERS = 12

# Mixed Precision Training
USE_MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = batch_size * 2

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory
EMPTY_CACHE_INTERVAL = 50  # Clear GPU cache every N batches

# Ensemble weights
EFFICIENTNET_WEIGHT = 0.40
XCEPTION_WEIGHT = 0.35
MESONET_WEIGHT = 0.25

# ============ REPRODUCIBILITY ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ✅ OPTIMIZED: Maximize CPU thread usage for parallel processing
torch.set_num_threads(CPU_CORES)  # Use all available cores

# Enable TF32 and CUDA optimizations
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"📊 CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(
        f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
    )


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
        print(
            f"📊 Dataset: {len(real_files)} real + {len(fake_files)} fake = {len(self.files)} total"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]

        try:
            # Use faster image loading - CV2 is faster than PIL
            if USE_CV2:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Could not load image with CV2")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            else:
                img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"⚠️  Failed to load {img_path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


# ============ TRANSFORMS ============
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============ OPTIMIZED COLLATE FUNCTION ============
def fast_collate_fn(batch):
    """Fast collate function using CPU threading for parallel stacking."""
    imgs, labels = zip(*batch)
    # Stack on CPU first (parallel), then move to device
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


# ============ DATA LOADING ============
def load_data(data_root):
    """Load data from predefined train/validate/test splits."""
    print("\n📂 Loading data from predefined splits...")

    train_dir = data_root / "train"
    validate_dir = data_root / "validate"
    test_dir = data_root / "test"

    train_real = (
        list((train_dir / "Real").glob("*"))
        if (train_dir / "Real").exists()
        else []
    )
    train_fake = (
        list((train_dir / "Fake").glob("*"))
        if (train_dir / "Fake").exists()
        else []
    )

    val_real = (
        list((validate_dir / "Real").glob("*"))
        if (validate_dir / "Real").exists()
        else []
    )
    val_fake = (
        list((validate_dir / "Fake").glob("*"))
        if (validate_dir / "Fake").exists()
        else []
    )

    test_real = (
        list((test_dir / "Real").glob("*"))
        if (test_dir / "Real").exists()
        else []
    )
    test_fake = (
        list((test_dir / "Fake").glob("*"))
        if (test_dir / "Fake").exists()
        else []
    )

    print(f"Train data: {len(train_real)} real, {len(train_fake)} fake")
    print(f"Val data:   {len(val_real)} real, {len(val_fake)} fake")
    if test_real or test_fake:
        print(f"Test data:  {len(test_real)} real, {len(test_fake)} fake")

    if len(train_real) == 0 or len(train_fake) == 0:
        raise ValueError("❌ No training data found!")

    if len(val_real) == 0 or len(val_fake) == 0:
        raise ValueError("❌ No validation data found!")

    return (train_real, train_fake), (val_real, val_fake), (test_real, test_fake)


# ============ AUTO BATCH SIZE TUNER ============
def auto_tune_batch_size(
    model,
    img_size,
    device,
    candidates,
    use_mixed_precision=True,
    accumulation_steps=1,
):
    """
    Tries different batch sizes (largest → smallest) and returns the biggest one
    that can run a forward + backward pass without CUDA OOM.
    """
    if device.type != "cuda":
        print("⚠️ Auto batch size tuning is only useful on CUDA. Skipping.")
        return DEFAULT_BATCH_SIZE

    print("\n🧪 Auto-tuning batch size for your GPU...")
    print(f"   Candidates (largest → smallest): {candidates}")

    model.train()

    for bs in candidates:
        try:
            print(f"   ▶ Testing batch size {bs}...", end="", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            dummy_inputs = torch.randn(
                bs, 3, img_size, img_size, device=device
            )
            dummy_labels = torch.zeros(
                bs, dtype=torch.long, device=device
            )

            test_optimizer = optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=0.0
            )
            test_scaler = GradScaler(
                enabled=(use_mixed_precision and device.type == "cuda")
            )

            test_optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision and device.type == "cuda":
                with autocast():
                    outputs = model(dummy_inputs)
                    loss = nn.functional.cross_entropy(
                        outputs, dummy_labels
                    ) / accumulation_steps
                test_scaler.scale(loss).backward()
                test_scaler.step(test_optimizer)
                test_scaler.update()
            else:
                outputs = model(dummy_inputs)
                loss = nn.functional.cross_entropy(
                    outputs, dummy_labels
                ) / accumulation_steps
                loss.backward()
                test_optimizer.step()

            del dummy_inputs, dummy_labels, outputs, loss, test_optimizer, test_scaler
            torch.cuda.empty_cache()

            used_mem = (
                torch.cuda.max_memory_allocated(device) / 1e9
            )
            print(f" ✅ OK (approx peak mem: {used_mem:.2f} GB)")
            return bs

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(" ❌ OOM, trying smaller...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"\n❌ Unexpected error at batch size {bs}: {e}")
                raise e

    print(
        "⚠️ Could not find a safe batch size from candidates, "
        "falling back to 32"
    )
    return 32


# ============ ENHANCED TRAINING WITH STATS ============
def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
    scaler=None,
    accumulation_steps=1,
):
    """Train for one epoch with enhanced statistics display and async GPU ops."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_times = []
    
    # Track per-class accuracy
    class_correct = [0, 0]
    class_total = [0, 0]

    epoch_start_time = time.time()
    data_load_times = []
    forward_times = []
    backward_times = []
    
    pbar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
        colour='blue'
    )
    
    optimizer.zero_grad(set_to_none=True)
    
    # Create CUDA streams for asynchronous operations
    if device.type == "cuda":
        compute_stream = torch.cuda.default_stream(device)
        prefetch_stream = torch.cuda.Stream(device)

    for batch_idx, (images, labels) in enumerate(pbar):
        batch_start = time.time()
        
        # Asynchronous GPU transfer on separate stream
        if device.type == "cuda":
            with torch.cuda.stream(prefetch_stream):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            torch.cuda.current_stream(device).wait_stream(prefetch_stream)
        else:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        forward_start = time.time()
        data_load_times.append(forward_start - batch_start)
        
        backward_start = time.time()
        forward_times.append(backward_start - forward_start)
        
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        backward_times.append(time.time() - backward_start)

        # Calculate metrics
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Per-class accuracy
        for i in range(labels.size(0)):
            label_val = labels[i].item()
            class_total[label_val] += 1
            if predicted[i] == labels[i]:
                class_correct[label_val] += 1
        
        batch_times.append(time.time() - batch_start)
        
        # Periodic GPU cache clearing to prevent memory fragmentation
        if (batch_idx + 1) % EMPTY_CACHE_INTERVAL == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Calculate statistics
        current_acc = 100.0 * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        avg_batch_time = np.mean(batch_times[-100:])  # Last 100 batches
        samples_per_sec = labels.size(0) / avg_batch_time
        
        # GPU memory stats
        if device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated(device) / 1e9
            gpu_mem_cached = torch.cuda.memory_reserved(device) / 1e9
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Real': f'{100.0 * class_correct[0] / max(class_total[0], 1):.1f}%',
                'Fake': f'{100.0 * class_correct[1] / max(class_total[1], 1):.1f}%',
                'GPU': f'{gpu_mem:.1f}GB',
                'Speed': f'{samples_per_sec:.1f} img/s'
            })
        else:
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Real': f'{100.0 * class_correct[0] / max(class_total[0], 1):.1f}%',
                'Fake': f'{100.0 * class_correct[1] / max(class_total[1], 1):.1f}%',
                'Speed': f'{samples_per_sec:.1f} img/s'
            })

    epoch_time = time.time() - epoch_start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    # Per-class accuracies
    real_acc = 100.0 * class_correct[0] / max(class_total[0], 1)
    fake_acc = 100.0 * class_correct[1] / max(class_total[1], 1)
    
    # Profiling stats
    profiling_stats = {
        'data_loading_ms': np.mean(data_load_times) * 1000 if data_load_times else 0,
        'forward_pass_ms': np.mean(forward_times) * 1000 if forward_times else 0,
        'backward_pass_ms': np.mean(backward_times) * 1000 if backward_times else 0,
    }
    
    return avg_loss, accuracy, real_acc, fake_acc, epoch_time, profiling_stats


def validate(model, val_loader, criterion, device, epoch, num_epochs, use_mixed_precision=True):
    """Validate the model with enhanced statistics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Per-class tracking
    class_correct = [0, 0]
    class_total = [0, 0]

    val_start_time = time.time()

    with torch.inference_mode():
        pbar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch}/{num_epochs} [Val]  ",
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
            colour='green'
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_mixed_precision and device.type == "cuda":
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label_val = labels[i].item()
                class_total[label_val] += 1
                if predicted[i] == labels[i]:
                    class_correct[label_val] += 1
            
            # Update progress bar
            current_acc = 100.0 * sum(class_correct) / sum(class_total)
            avg_loss = running_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Real': f'{100.0 * class_correct[0] / max(class_total[0], 1):.1f}%',
                'Fake': f'{100.0 * class_correct[1] / max(class_total[1], 1):.1f}%'
            })

    val_time = time.time() - val_start_time
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracies
    real_acc = 100.0 * class_correct[0] / max(class_total[0], 1)
    fake_acc = 100.0 * class_correct[1] / max(class_total[1], 1)

    return (epoch_loss, accuracy, precision * 100, recall * 100, f1 * 100, 
            auc, cm, real_acc, fake_acc, val_time)


def print_epoch_summary(epoch, num_epochs, train_stats, val_stats, lr, improved=False, profiling=None):
    """Print a comprehensive epoch summary."""
    train_loss, train_acc, train_real_acc, train_fake_acc, train_time = train_stats
    (val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, 
     cm, val_real_acc, val_fake_acc, val_time) = val_stats
    
    print(f"\n{'='*80}")
    print(f"📊 EPOCH {epoch}/{num_epochs} SUMMARY")
    print(f"{'='*80}")
    
    # Training Stats
    print(f"\n🔵 TRAINING:")
    print(f"   Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    print(f"   Real Accuracy: {train_real_acc:.2f}% | Fake Accuracy: {train_fake_acc:.2f}%")
    print(f"   Time: {train_time:.1f}s")
    
    # Validation Stats
    print(f"\n🟢 VALIDATION:")
    print(f"   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
    print(f"   Precision: {val_prec:.2f}% | Recall: {val_rec:.2f}% | F1: {val_f1:.2f}%")
    print(f"   AUC-ROC: {val_auc:.4f}")
    print(f"   Real Accuracy: {val_real_acc:.2f}% | Fake Accuracy: {val_fake_acc:.2f}%")
    print(f"   Time: {val_time:.1f}s")
    
    # Confusion Matrix
    print(f"\n📈 CONFUSION MATRIX:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"   Real    [{cm[0,0]:6d}  {cm[0,1]:6d}]")
    print(f"   Fake    [{cm[1,0]:6d}  {cm[1,1]:6d}]")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   True Negatives:  {tn:6d} | False Positives: {fp:6d}")
    print(f"   False Negatives: {fn:6d} | True Positives:  {tp:6d}")
    
    # Learning Rate
    print(f"\n⚙️  Learning Rate: {lr:.2e}")
    
    # Profiling stats
    if profiling:
        print(f"\n⏱️  PERFORMANCE PROFILING:")
        data_load = profiling.get('data_loading_ms', 0) or 0
        forward = profiling.get('forward_pass_ms', 0) or 0
        backward = profiling.get('backward_pass_ms', 0) or 0
        
        print(f"   Data Loading: {data_load:.1f}ms")
        print(f"   Forward Pass: {forward:.1f}ms")
        print(f"   Backward Pass: {backward:.1f}ms")
        
        # Find bottleneck safely
        timings = {'data_loading_ms': data_load, 'forward_pass_ms': forward, 'backward_pass_ms': backward}
        non_zero_timings = {k: v for k, v in timings.items() if v > 0}
        if non_zero_timings:
            bottleneck = max(non_zero_timings.values())
            bottleneck_name = [k for k, v in non_zero_timings.items() if v == bottleneck][0]
            bottleneck_name_clean = bottleneck_name.replace('_ms', '').replace('_', ' ').title()
            print(f"   🔴 Bottleneck: {bottleneck_name_clean} ({bottleneck:.1f}ms)")
    
    # Improvement indicator
    if improved:
        print(f"\n✨ NEW BEST MODEL! (Accuracy: {val_acc:.2f}%, F1: {val_f1:.2f}%)")
    
    print(f"{'='*80}\n")


# ============ MAIN TRAINING ============
def main():
    print(f"\n{'=' * 80}")
    print(f"🎯 ENSEMBLE MODEL TRAINING - Deepfake Detection")
    print(f"{'=' * 80}")
    print(f"📊 Architecture:")
    print(f"   - EfficientNet (40%)")
    print(f"   - XceptionNet  (35%)")
    print(f"   - MesoNet      (25%)")
    print(f"📸 Image Loading: {'OpenCV (fast)' if USE_CV2 else 'PIL (standard)'}")
    print(f"{'=' * 80}\n")

    # Load data paths
    (train_real, train_fake), (val_real, val_fake), (test_real, test_fake) = load_data(DATA_ROOT)

    # Create ensemble model
    print(f"\n🔨 Creating ensemble model...")
    model = create_ensemble(
        model_type="weighted",
        weights=(EFFICIENTNET_WEIGHT, XCEPTION_WEIGHT, MESONET_WEIGHT),
        num_classes=2,
        pretrained=True,
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    if USE_GRADIENT_CHECKPOINTING:
        print(f"   Enabling gradient checkpointing (trade compute for memory)...")
        # Enable checkpointing for EfficientNet backbone
        try:
            if hasattr(model.efficientnet.backbone, 'set_grad_checkpointing'):
                model.efficientnet.backbone.set_grad_checkpointing(True)
                print(f"      ✓ EfficientNet checkpointing enabled")
        except (AssertionError, AttributeError) as e:
            print(f"      ⚠️  EfficientNet checkpointing not supported")
        
        # XceptionNet doesn't support gradient checkpointing
        try:
            if hasattr(model.xception.backbone, 'set_grad_checkpointing'):
                model.xception.backbone.set_grad_checkpointing(True)
                print(f"      ✓ XceptionNet checkpointing enabled")
        except (AssertionError, AttributeError) as e:
            print(f"      ⚠️  XceptionNet checkpointing not supported")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # === BATCH SIZE CONFIGURATION ===
    batch_size = BATCH_SIZE  # Use fixed batch size
    print(f"\n📦 Batch Size: {batch_size} (fixed)")

    # Build datasets
    train_dataset = DeepfakeDataset(
        train_real, train_fake, transform=train_transform
    )
    val_dataset = DeepfakeDataset(
        val_real, val_fake, transform=val_transform
    )

    # ✅ OPTIMIZED DATALOADERS - Aggressive prefetching for CPU-GPU overlap
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=8,  # Balanced: enough for GPU feeding without memory overflow
        collate_fn=fast_collate_fn,  # Parallel CPU stacking
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        prefetch_factor=8,  # Balanced: enough for GPU feeding without memory overflow
        collate_fn=fast_collate_fn,  # Parallel CPU stacking
    )

    print(f"\n📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"📊 DataLoader workers: {NUM_WORKERS}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Mixed precision training setup
    scaler = GradScaler() if USE_MIXED_PRECISION and device.type == "cuda" else None

    print(f"\n{'=' * 80}")
    print(f"🚀 Training Configuration:")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - Batch Size (per step): {batch_size}")
    print(
        f"   - Effective Batch Size: {batch_size * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"   - Weight Decay: {WEIGHT_DECAY}")
    print(f"   - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(
        f"   - Mixed Precision: {'Enabled (FP16)' if USE_MIXED_PRECISION else 'Disabled'}"
    )
    print(f"   - Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   - Data Loading Workers: {NUM_WORKERS} (optimized CPU utilization)")
    print(f"   - CPU Threads: {CPU_CORES} (all cores for compute)")
    print(f"   - Prefetch Factor: 8x (balanced CPU-GPU overlap)")
    print(f"   - Gradient Checkpointing: {'Enabled' if USE_GRADIENT_CHECKPOINTING else 'Disabled'} (memory optimization)")
    print(f"   - GPU Cache Clear Interval: {EMPTY_CACHE_INTERVAL} batches")
    print(f"{'=' * 80}\n")

    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    start_time = time.time()

    training_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            NUM_EPOCHS,
            scaler=scaler,
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )
        train_stats = train_result[:5]
        profiling = train_result[5] if len(train_result) > 5 else {}
        
        # Validate
        val_stats = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            NUM_EPOCHS,
            use_mixed_precision=USE_MIXED_PRECISION,
        )
        
        # Update learning rate
        scheduler.step(val_stats[0])  # val_loss
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if improved
        val_acc = val_stats[1]
        val_f1 = val_stats[4]
        improved = (val_acc > best_val_acc) or (val_f1 > best_val_f1)
        
        # Print comprehensive summary
        print_epoch_summary(epoch, NUM_EPOCHS, train_stats, val_stats, current_lr, improved, profiling)

        # Store history
        train_loss, train_acc, train_real_acc, train_fake_acc, train_time = train_stats
        (val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, 
         cm, val_real_acc, val_fake_acc, val_time) = val_stats
        
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_real_acc": float(train_real_acc),
            "train_fake_acc": float(train_fake_acc),
            "train_time": float(train_time),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_real_acc": float(val_real_acc),
            "val_fake_acc": float(val_fake_acc),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
            "val_time": float(val_time),
            "learning_rate": float(current_lr),
            "confusion_matrix": cm.tolist(),
        }
        training_history.append(epoch_record)

        # Save best model
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_f1 = max(best_val_f1, val_f1)
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "weights": {
                    "efficientnet": EFFICIENTNET_WEIGHT,
                    "xception": XCEPTION_WEIGHT,
                    "mesonet": MESONET_WEIGHT,
                },
                "batch_size": batch_size,
            }

            torch.save(checkpoint, OUTPUT_DIR / "ensemble_best.pth")
            print(f"💾 Saved best model checkpoint\n")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n{'='*80}")
                print(f"⚠️  EARLY STOPPING")
                print(f"{'='*80}")
                print(f"No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
                print(f"Best Accuracy: {best_val_acc:.2f}% | Best F1: {best_val_f1:.2f}%")
                print(f"{'='*80}\n")
                break

    elapsed = time.time() - start_time

    # ============ TEST EVALUATION ============
    if test_real or test_fake:
        print(f"\n{'=' * 80}")
        print(f"🧪 EVALUATING ON TEST SET")
        print(f"{'=' * 80}\n")
        
        # Load best model
        best_checkpoint = torch.load(OUTPUT_DIR / "ensemble_best.pth", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        
        # Create test dataset and loader
        test_dataset = DeepfakeDataset(
            test_real, test_fake, transform=val_transform
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
            collate_fn=fast_collate_fn,
        )
        
        # Evaluate on test set
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, test_cm, test_real_acc, test_fake_acc, test_time = validate(
            model,
            test_loader,
            criterion,
            device,
            0,
            1,
            use_mixed_precision=USE_MIXED_PRECISION,
        )
        
        print(f"\n{'='*80}")
        print(f"📊 TEST SET RESULTS")
        print(f"{'='*80}")
        print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
        print(f"Precision: {test_prec:.2f}% | Recall: {test_rec:.2f}% | F1: {test_f1:.2f}%")
        print(f"AUC-ROC: {test_auc:.4f}")
        print(f"Real Accuracy: {test_real_acc:.2f}% | Fake Accuracy: {test_fake_acc:.2f}%")
        print(f"Time: {test_time:.1f}s")
        print(f"{'='*80}\n")
    else:
        elapsed = time.time() - start_time

    # Save final checkpoint
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": best_val_acc,
        "val_f1": best_val_f1,
        "weights": {
            "efficientnet": EFFICIENTNET_WEIGHT,
            "xception": XCEPTION_WEIGHT,
            "mesonet": MESONET_WEIGHT,
        },
        "batch_size": batch_size,
    }
    torch.save(final_checkpoint, OUTPUT_DIR / "ensemble_final.pth")

    # Save training history to JSON
    results = {
        "summary": {
            "best_accuracy": float(best_val_acc),
            "best_f1": float(best_val_f1),
            "total_epochs_trained": len(training_history),
            "total_epochs": NUM_EPOCHS,
            "training_time_minutes": float(elapsed / 60),
            "training_time_formatted": str(timedelta(seconds=int(elapsed))),
        },
        "ensemble_weights": {
            "efficientnet": EFFICIENTNET_WEIGHT,
            "xception": XCEPTION_WEIGHT,
            "mesonet": MESONET_WEIGHT,
        },
        "hyperparameters": {
            "learning_rate": float(LEARNING_RATE),
            "batch_size": int(batch_size),
            "weight_decay": float(WEIGHT_DECAY),
            "img_size": int(IMG_SIZE),
            "num_workers": int(NUM_WORKERS),
            "mixed_precision": bool(USE_MIXED_PRECISION),
            "gradient_accumulation_steps": int(GRADIENT_ACCUMULATION_STEPS),
        },
        "training_history": training_history,
    }

    with open(OUTPUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"🎉 TRAINING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"⏱️  Total Time: {elapsed / 60:.1f} minutes ({str(timedelta(seconds=int(elapsed)))})")
    print(f"📊 Best Accuracy: {best_val_acc:.2f}%")
    print(f"📊 Best F1 Score: {best_val_f1:.2f}%")
    print(f"📊 Total Epochs Trained: {len(training_history)}/{NUM_EPOCHS}")
    print(f"\n💾 Models saved:")
    print(f"   - Best:  {OUTPUT_DIR / 'ensemble_best.pth'}")
    print(f"   - Final: {OUTPUT_DIR / 'ensemble_final.pth'}")
    print(f"   - Results: {OUTPUT_DIR / 'training_results.json'}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    if os.name == "nt":
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    main()