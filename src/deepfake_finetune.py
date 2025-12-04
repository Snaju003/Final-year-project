"""
Fine-Tuning Script for Deepfake Detection Model

Use cases:
1. Add new deepfake techniques to existing model
2. Improve performance on specific video types
3. Adapt to different datasets
4. Continue training for better accuracy
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
MODEL_PATH = Path(r"E:\Final-year-project\models\best_model.pth")
NEW_DATA_ROOT = Path(r"E:\Final-year-project-data\data\dataset 3")  # Your new data with train/validate/test splits
OUTPUT_DIR = Path(r"E:\Final-year-project\models\finetuned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ FINE-TUNING SETTINGS ============
IMG_SIZE = 224
BATCH_SIZE = 96
NUM_EPOCHS = 5               # Fewer epochs for fine-tuning
LEARNING_RATE = 1e-5         # Much lower LR (10x smaller)
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 6

# Fine-tuning strategy
FREEZE_BACKBONE = False      # Set to True to only train classifier
USE_PREDEFINED_SPLITS = True # Use train/validate splits from dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")


# ============ MODEL ARCHITECTURE (same as training) ============
class FastDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features)
        output = self.classifier(pooled)
        return output


# ============ DATASET ============
class DeepfakeDataset(Dataset):
    def __init__(self, real_files, fake_files, transform=None):
        self.files = []
        self.labels = []
        
        for f in real_files:
            self.files.append(str(f))
            self.labels.append(0)
        
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
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label


# ============ TRANSFORMS ============
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============ LOAD PRETRAINED MODEL ============
def load_pretrained_model(model_path):
    """Load existing trained model."""
    print(f"📂 Loading model from {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = FastDeepfakeDetector(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Loaded model (Original Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
    return model, checkpoint


# ============ FREEZE LAYERS ============
def freeze_backbone(model):
    """Freeze backbone, only train classifier."""
    print("❄️  Freezing backbone (only training classifier)")
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")


# ============ DATA LOADING ============
def load_data(data_root):
    """Load data from predefined train/validate/test splits."""
    print("\n📂 Loading data from predefined splits...")
    
    train_dir = data_root / "train"
    validate_dir = data_root / "validate"
    
    # Load training data
    train_real = list((train_dir / "Real").glob("*")) if (train_dir / "Real").exists() else []
    train_fake = list((train_dir / "Fake").glob("*")) if (train_dir / "Fake").exists() else []
    
    # Load validation data
    val_real = list((validate_dir / "Real").glob("*")) if (validate_dir / "Real").exists() else []
    val_fake = list((validate_dir / "Fake").glob("*")) if (validate_dir / "Fake").exists() else []
    
    print(f"Train data: {len(train_real)} real, {len(train_fake)} fake")
    print(f"Val data:   {len(val_real)} real, {len(val_fake)} fake")
    
    if len(train_real) == 0 or len(train_fake) == 0:
        raise ValueError("❌ No training data found!")
    
    if len(val_real) == 0 or len(val_fake) == 0:
        raise ValueError("❌ No validation data found!")
    
    return (train_real, train_fake), (val_real, val_fake)


# ============ TRAINING ============
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
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


# ============ MAIN FINE-TUNING ============
def main():
    print(f"\n{'='*60}")
    print(f"🔧 FINE-TUNING DEEPFAKE DETECTION MODEL")
    print(f"{'='*60}")
    
    # Load pretrained model
    model, original_checkpoint = load_pretrained_model(MODEL_PATH)
    
    # Freeze backbone if needed
    if FREEZE_BACKBONE:
        freeze_backbone(model)
    else:
        print("🔥 Training all layers (full fine-tuning)")
    
    # Load data
    (train_real, train_fake), (val_real, val_fake) = load_data(NEW_DATA_ROOT)
    
    train_dataset = DeepfakeDataset(train_real, train_fake, transform=train_transform)
    val_dataset = DeepfakeDataset(val_real, val_fake, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"\n📊 Splits: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Optimizer with lower learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    print(f"\n{'='*60}")
    print(f"🚀 Fine-tuning for {NUM_EPOCHS} epochs")
    print(f"Learning Rate: {LEARNING_RATE} (10x lower than training)")
    print(f"{'='*60}\n")
    
    best_val_acc = original_checkpoint.get('val_acc', 0)
    best_epoch_metrics = None
    best_epoch_num = 0
    all_epoch_metrics = []
    improved = False
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"\n📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.2f}%, AUC={val_auc:.4f}")
        
        # Store all epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'val_f1': float(val_f1),
            'val_auc': float(val_auc)
        }
        all_epoch_metrics.append(epoch_metrics)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improved = True
            best_epoch_num = epoch + 1
            best_epoch_metrics = epoch_metrics
            print(f"✅ New best model found (Val Acc: {val_acc:.2f}%)")
        elif best_epoch_metrics is None:
            best_epoch_metrics = epoch_metrics
            best_epoch_num = epoch + 1
    
    elapsed = time.time() - start_time
    
    # Save the best model checkpoint
    final_checkpoint = {
        'epoch': best_epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': best_epoch_metrics['val_acc'],
        'val_f1': best_epoch_metrics['val_f1'],
        'val_auc': best_epoch_metrics['val_auc'],
        'original_acc': original_checkpoint.get('val_acc', 0),
        'finetuned': True
    }
    
    torch.save(final_checkpoint, OUTPUT_DIR / "finetuned_model.pth")
    
    # Save results to JSON
    original_acc = original_checkpoint.get('val_acc', 0)
    improvement = best_val_acc - original_acc
    
    results = {
        'fine_tuning_summary': {
            'original_accuracy': float(original_acc),
            'final_accuracy': float(best_val_acc),
            'improvement': float(improvement),
            'improved': bool(improved),
            'best_epoch': int(best_epoch_num),
            'total_epochs': int(NUM_EPOCHS),
            'training_time_minutes': float(elapsed / 60)
        },
        'best_epoch_metrics': best_epoch_metrics,
        'all_epoch_metrics': all_epoch_metrics,
        'hyperparameters': {
            'learning_rate': float(LEARNING_RATE),
            'batch_size': int(BATCH_SIZE),
            'weight_decay': float(WEIGHT_DECAY),
            'freeze_backbone': bool(FREEZE_BACKBONE),
            'img_size': int(IMG_SIZE)
        }
    }
    
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"🎉 FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"📊 Original Accuracy: {original_acc:.2f}%")
    print(f"📊 Final Accuracy:    {best_val_acc:.2f}%")
    print(f"📊 Improvement:       {improvement:+.2f}%")
    print(f"\n💾 Model saved:   {OUTPUT_DIR / 'finetuned_model.pth'}")
    print(f"📄 Results saved: {OUTPUT_DIR / 'results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()