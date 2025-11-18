"""
Optimized Deepfake Detection Training - Maximum Hardware Utilization
- Faster data loading with prefetching
- Mixed precision training (FP16)
- Optimized batch processing
- Reduced bottlenecks
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
import time

# ============ PATHS ============
DATA_ROOT = Path(r"E:\Final-year-project-data\data\faces")
REAL_DIR = DATA_ROOT / "real"
FAKE_DIR = DATA_ROOT / "fake"
MODEL_DIR = Path(r"E:\Final-year-project\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============ OPTIMIZED HYPERPARAMETERS ============
IMG_SIZE = 224
BATCH_SIZE = 96              # Balanced for memory
NUM_EPOCHS = 15              # Reduced from 20 (diminishing returns after 15)
LEARNING_RATE = 2e-4         # Slightly higher for faster convergence
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 6              # Reduced to prevent memory overflow
PREFETCH_FACTOR = 2          # Reduced prefetch
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Enable all optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")


# ============ OPTIMIZED DATASET ============
class FastDeepfakeDataset(Dataset):
    """Optimized dataset with caching."""
    
    def __init__(self, real_files, fake_files, transform=None):
        self.files = []
        self.labels = []
        
        # Real faces (label 0)
        for f in real_files:
            self.files.append(str(f))  # Store as string for faster pickling
            self.labels.append(0)
        
        # Fake faces (label 1)
        for f in fake_files:
            self.files.append(str(f))
            self.labels.append(1)
        
        self.transform = transform
        print(f"📊 {len(real_files)} real + {len(fake_files)} fake = {len(self.files)} total")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            # Fast image loading
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        
        except Exception as e:
            # Return black image on error (rare)
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


# ============ LIGHTER DATA AUGMENTATION ============
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============ SIMPLIFIED MODEL ============
class FastDeepfakeDetector(nn.Module):
    """Lighter model for faster training."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Use EfficientNet-B0 instead of B4 (4x faster, similar accuracy)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, global_pool='')
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Simpler classifier with proper pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        # Extract features (4D: batch, channels, height, width)
        features = self.backbone(x)
        
        # Pool to (batch, channels, 1, 1)
        pooled = self.pool(features)
        
        # Classify
        output = self.classifier(pooled)
        return output


# ============ OPTIMIZED TRAINING ============
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Optimized training with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Fast validation."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return epoch_loss, accuracy, precision * 100, recall * 100, f1 * 100, auc


# ============ DATA PREPARATION ============
def prepare_data():
    """Load and split data efficiently."""
    print("\n📂 Loading data...")
    
    real_files = list(REAL_DIR.glob("*.jpg"))
    fake_files = list(FAKE_DIR.glob("*.jpg"))
    
    print(f"Found {len(real_files)} real, {len(fake_files)} fake")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("❌ No faces found!")
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(real_files)
    np.random.shuffle(fake_files)
    
    # Balance dataset (use equal amounts)
    min_samples = min(len(real_files), len(fake_files))
    real_files = real_files[:min_samples]
    fake_files = fake_files[:min_samples]
    
    print(f"Using {min_samples} samples per class (balanced)")
    
    # Split
    n = min_samples
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    
    train_real, train_fake = real_files[:train_end], fake_files[:train_end]
    val_real, val_fake = real_files[train_end:val_end], fake_files[train_end:val_end]
    test_real, test_fake = real_files[val_end:], fake_files[val_end:]
    
    return (train_real, train_fake), (val_real, val_fake), (test_real, test_fake)


# ============ MAIN ============
def main():
    print(f"\n{'='*60}")
    print(f"🤖 OPTIMIZED DEEPFAKE DETECTION TRAINING")
    print(f"{'='*60}")
    
    # Data
    (train_real, train_fake), (val_real, val_fake), (test_real, test_fake) = prepare_data()
    
    train_dataset = FastDeepfakeDataset(train_real, train_fake, transform=train_transform)
    val_dataset = FastDeepfakeDataset(val_real, val_fake, transform=val_transform)
    test_dataset = FastDeepfakeDataset(test_real, test_fake, transform=val_transform)
    
    # Optimized dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False  # Don't keep workers alive to save memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,  # Same batch size for validation
        shuffle=False,
        num_workers=4,  # Fewer workers for validation
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Model
    print(f"\n🏗️  Building model...")
    model = FastDeepfakeDetector(num_classes=2, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
    best_val_acc = 0.0
    
    print(f"\n{'='*60}")
    print(f"🚀 Training for {NUM_EPOCHS} epochs (Optimized for speed)")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        print(f"\n📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.2f}%, AUC={val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc
            }, MODEL_DIR / "best_model.pth")
            print(f"✅ Saved best model (Acc: {val_acc:.2f}%)")
    
    elapsed = time.time() - start_time
    
    # Test
    print(f"\n{'='*60}")
    print(f"🧪 FINAL TEST")
    print(f"{'='*60}")
    
    checkpoint = torch.load(MODEL_DIR / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate(model, test_loader, criterion, device)
    
    print(f"\n📊 Test: Acc={test_acc:.2f}%, Precision={test_prec:.2f}%, Recall={test_rec:.2f}%, F1={test_f1:.2f}%, AUC={test_auc:.4f}")
    
    results = {
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc),
        'training_time_minutes': elapsed / 60
    }
    
    with open(MODEL_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE! Time: {elapsed/60:.1f} min")
    print(f"💾 Model: {MODEL_DIR / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()