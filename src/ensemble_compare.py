"""
Ensemble Model Comparison Script
Compare performance of Ensemble V1 vs V2 on test datasets

This script:
1. Loads both ensemble models (v1 and v2)
2. Evaluates them on the same test data
3. Generates detailed comparison metrics
4. Creates visualizations and comparison reports
"""

import os
import sys
from pathlib import Path
import json
import time
from datetime import timedelta

# Fix Unicode encoding for Windows PowerShell
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
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
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import ensemble model
from ensemble_model import create_ensemble

# Try to import CV2 for faster image loading
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    print("⚠️ OpenCV not available, using PIL (slower)")

# ============ PATHS ============
DATA_ROOT = Path(r"X:\Final-year-project-data\data")
MODEL_V1_PATH = Path(r"E:\Final-year-project\models\ensemble\ensemble_best.pth")
MODEL_V2_PATH = Path(r"E:\Final-year-project\models\ensemble_v2\ensemble_best_v2.pth")
OUTPUT_DIR = Path(r"E:\Final-year-project\results\comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ SETTINGS ============
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 12
PIN_MEMORY = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

if device.type == "cuda":
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")


# ============ TRANSFORMS ============
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============ DATASET ============
class MultiSourceDeepfakeDataset(Dataset):
    """Dataset for loading test images from multiple sources."""
    
    def __init__(self, files_with_labels, transform=None):
        self.data = files_with_labels
        self.transform = transform
    
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
            
            return img, label, source
        
        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label, source


# ============ DATA LOADING ============
def collect_files(folder, max_samples=None):
    """Collect image files from a folder."""
    if not folder.exists():
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(folder.glob(ext))
    
    if max_samples and len(files) > max_samples:
        import random
        random.shuffle(files)
        files = files[:max_samples]
    
    return files


def load_test_data():
    """Load test data from all available datasets."""
    print("\n" + "=" * 60)
    print("📂 LOADING TEST DATA")
    print("=" * 60)
    
    test_data = []
    
    # Dataset 1 - use last 10% as test
    print("\n📁 Dataset 1:")
    ds1_root = DATA_ROOT / 'dataset 1'
    if ds1_root.exists():
        ds1_real = collect_files(ds1_root / 'Real', max_samples=70000)
        ds1_fake = collect_files(ds1_root / 'Fake', max_samples=70000)
        
        # Take last 10% for test
        import random
        random.seed(42)
        random.shuffle(ds1_real)
        random.shuffle(ds1_fake)
        
        n_test_real = int(len(ds1_real) * 0.10)
        n_test_fake = int(len(ds1_fake) * 0.10)
        
        ds1_test_real = ds1_real[-n_test_real:]
        ds1_test_fake = ds1_fake[-n_test_fake:]
        
        for f in ds1_test_real:
            test_data.append((f, 0, 'dataset1'))
        for f in ds1_test_fake:
            test_data.append((f, 1, 'dataset1'))
        
        print(f"   Test: {len(ds1_test_real)} real, {len(ds1_test_fake)} fake")
    
    # Dataset 2 - has explicit test split
    print("\n📁 Dataset 2:")
    ds2_root = DATA_ROOT / 'dataset 2'
    if ds2_root.exists():
        ds2_test_real = collect_files(ds2_root / 'Test' / 'Real')
        ds2_test_fake = collect_files(ds2_root / 'Test' / 'Fake')
        
        for f in ds2_test_real:
            test_data.append((f, 0, 'dataset2'))
        for f in ds2_test_fake:
            test_data.append((f, 1, 'dataset2'))
        
        print(f"   Test: {len(ds2_test_real)} real, {len(ds2_test_fake)} fake")
    
    # Dataset 3 - no test split, skip
    print("\n📁 Dataset 3: (no test split, using validation only)")
    
    import random
    random.shuffle(test_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST DATA SUMMARY")
    print("=" * 60)
    
    from collections import defaultdict
    stats = defaultdict(lambda: {'real': 0, 'fake': 0})
    for _, label, source in test_data:
        if label == 0:
            stats[source]['real'] += 1
        else:
            stats[source]['fake'] += 1
    
    total_real = sum(s['real'] for s in stats.values())
    total_fake = sum(s['fake'] for s in stats.values())
    
    print(f"\n{'Source':<12} {'Real':>10} {'Fake':>10} {'Total':>10}")
    print("-" * 45)
    for source in sorted(stats.keys()):
        s = stats[source]
        total = s['real'] + s['fake']
        print(f"{source:<12} {s['real']:>10,} {s['fake']:>10,} {total:>10,}")
    print("-" * 45)
    print(f"{'TOTAL':<12} {total_real:>10,} {total_fake:>10,} {len(test_data):>10,}")
    
    return test_data


# ============ MODEL LOADING ============
def load_model(checkpoint_path, model_name):
    """Load a trained ensemble model."""
    print(f"\n📦 Loading {model_name}...")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract weights
    weights = checkpoint.get('weights', {})
    if isinstance(weights, dict):
        model_weights = (
            weights.get('efficientnet', 0.35),
            weights.get('xception', 0.30),
            weights.get('mesonet', 0.35)
        )
    else:
        model_weights = weights
    
    # Create model
    model = create_ensemble(
        model_type='weighted',
        weights=model_weights,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Print model info
    print(f"   Weights: EfficientNet={model_weights[0]:.2f}, "
          f"Xception={model_weights[1]:.2f}, MesoNet={model_weights[2]:.2f}")
    
    if 'epoch' in checkpoint:
        print(f"   Trained for {checkpoint['epoch']} epochs")
    if 'val_acc' in checkpoint:
        print(f"   Best Val Acc: {checkpoint['val_acc']:.2f}%")
    if 'val_f1' in checkpoint:
        print(f"   Best Val F1: {checkpoint['val_f1']:.2f}%")
    
    return model, checkpoint


# ============ EVALUATION ============
@torch.no_grad()
def evaluate_model(model, loader, model_name):
    """Evaluate a model on test data."""
    print(f"\n🔍 Evaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_sources = []
    
    # Per-source tracking
    from collections import defaultdict
    source_preds = defaultdict(list)
    source_labels = defaultdict(list)
    source_probs = defaultdict(list)
    
    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Testing {model_name}", ncols=100)
    
    for images, labels, sources in pbar:
        images = images.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Track by source
        for pred, label, prob, source in zip(preds, labels.numpy(), probs[:, 1].cpu().numpy(), sources):
            source_preds[source].append(pred)
            source_labels[source].append(label)
            source_probs[source].append(prob)
    
    eval_time = time.time() - start_time
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    real_acc = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100 if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fake_acc = cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100 if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    # Per-source metrics
    source_metrics = {}
    for source in source_preds.keys():
        s_acc = accuracy_score(source_labels[source], source_preds[source]) * 100
        s_prec, s_rec, s_f1, _ = precision_recall_fscore_support(
            source_labels[source], source_preds[source], average='binary', zero_division=0
        )
        try:
            s_auc = roc_auc_score(source_labels[source], source_probs[source]) * 100
        except:
            s_auc = 0.0
        
        source_metrics[source] = {
            'accuracy': s_acc,
            'precision': s_prec * 100,
            'recall': s_rec * 100,
            'f1': s_f1 * 100,
            'auc': s_auc,
            'n_samples': len(source_labels[source])
        }
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'confusion_matrix': cm.tolist(),
        'eval_time': eval_time,
        'source_metrics': source_metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


# ============ COMPARISON & VISUALIZATION ============
def print_comparison(metrics_v1, metrics_v2):
    """Print side-by-side comparison of both models."""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Ensemble V1':>20} {'Ensemble V2':>20} {'Difference':>15}")
    print("-" * 80)
    
    # Overall metrics
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1'),
        ('AUC', 'auc'),
        ('Real Accuracy', 'real_accuracy'),
        ('Fake Accuracy', 'fake_accuracy'),
    ]
    
    for display_name, key in metrics_to_compare:
        v1_val = metrics_v1[key]
        v2_val = metrics_v2[key]
        diff = v2_val - v1_val
        
        # Color code the difference
        diff_str = f"{diff:+.2f}%"
        if diff > 0:
            diff_str = f"✅ {diff_str}"
        elif diff < 0:
            diff_str = f"❌ {diff_str}"
        else:
            diff_str = f"   {diff_str}"
        
        print(f"{display_name:<25} {v1_val:>19.2f}% {v2_val:>19.2f}% {diff_str:>15}")
    
    print(f"\n{'Evaluation Time':<25} {metrics_v1['eval_time']:>18.2f}s {metrics_v2['eval_time']:>18.2f}s {metrics_v2['eval_time']-metrics_v1['eval_time']:>+14.2f}s")
    
    # Confusion matrices
    print("\n" + "=" * 80)
    print("CONFUSION MATRICES")
    print("=" * 80)
    
    cm_v1 = np.array(metrics_v1['confusion_matrix'])
    cm_v2 = np.array(metrics_v2['confusion_matrix'])
    
    print(f"\nEnsemble V1:")
    print(f"   Predicted:  REAL    FAKE")
    print(f"   Actual REAL: {cm_v1[0,0]:5d}  {cm_v1[0,1]:5d}")
    print(f"   Actual FAKE: {cm_v1[1,0]:5d}  {cm_v1[1,1]:5d}")
    
    print(f"\nEnsemble V2:")
    print(f"   Predicted:  REAL    FAKE")
    print(f"   Actual REAL: {cm_v2[0,0]:5d}  {cm_v2[0,1]:5d}")
    print(f"   Actual FAKE: {cm_v2[1,0]:5d}  {cm_v2[1,1]:5d}")
    
    # Per-source comparison
    print("\n" + "=" * 80)
    print("PER-SOURCE COMPARISON")
    print("=" * 80)
    
    sources = set(metrics_v1['source_metrics'].keys()) | set(metrics_v2['source_metrics'].keys())
    
    for source in sorted(sources):
        print(f"\n📁 {source.upper()}:")
        
        if source in metrics_v1['source_metrics'] and source in metrics_v2['source_metrics']:
            s1 = metrics_v1['source_metrics'][source]
            s2 = metrics_v2['source_metrics'][source]
            
            print(f"   {'Metric':<15} {'V1':>10} {'V2':>10} {'Diff':>10}")
            print(f"   {'-'*45}")
            print(f"   {'Accuracy':<15} {s1['accuracy']:>9.2f}% {s2['accuracy']:>9.2f}% {s2['accuracy']-s1['accuracy']:>+9.2f}%")
            print(f"   {'F1 Score':<15} {s1['f1']:>9.2f}% {s2['f1']:>9.2f}% {s2['f1']-s1['f1']:>+9.2f}%")
            print(f"   {'AUC':<15} {s1['auc']:>9.2f}% {s2['auc']:>9.2f}% {s2['auc']-s1['auc']:>+9.2f}%")
            print(f"   {'Samples':<15} {s1['n_samples']:>10,}")


def create_visualizations(metrics_v1, metrics_v2):
    """Create comparison visualizations."""
    print("\n📊 Creating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    v1_values = [metrics_v1['accuracy'], metrics_v1['precision'], 
                 metrics_v1['recall'], metrics_v1['f1'], metrics_v1['auc']]
    v2_values = [metrics_v2['accuracy'], metrics_v2['precision'], 
                 metrics_v2['recall'], metrics_v2['f1'], metrics_v2['auc']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, v1_values, width, label='Ensemble V1', alpha=0.8)
    bars2 = ax.bar(x + width/2, v2_values, width, label='Ensemble V2', alpha=0.8)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Ensemble V1 vs V2 - Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([80, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: metrics_comparison.png")
    plt.close()
    
    # 2. Confusion Matrix Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_v1 = np.array(metrics_v1['confusion_matrix'])
    cm_v2 = np.array(metrics_v2['confusion_matrix'])
    
    # Normalize for better visualization
    cm_v1_norm = cm_v1.astype('float') / cm_v1.sum(axis=1)[:, np.newaxis]
    cm_v2_norm = cm_v2.astype('float') / cm_v2.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_v1_norm, annot=cm_v1, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Normalized Count'})
    ax1.set_title('Ensemble V1 - Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    sns.heatmap(cm_v2_norm, annot=cm_v2, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Normalized Count'})
    ax2.set_title('Ensemble V2 - Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: confusion_matrices.png")
    plt.close()
    
    # 3. Per-Source Performance Comparison
    sources = list(metrics_v1['source_metrics'].keys())
    if sources:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        v1_accs = [metrics_v1['source_metrics'][s]['accuracy'] for s in sources]
        v2_accs = [metrics_v2['source_metrics'][s]['accuracy'] for s in sources]
        
        x = np.arange(len(sources))
        width = 0.35
        
        ax.bar(x - width/2, v1_accs, width, label='Ensemble V1', alpha=0.8)
        ax.bar(x + width/2, v2_accs, width, label='Ensemble V2', alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Source Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in sources])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'source_comparison.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: source_comparison.png")
        plt.close()
    
    # 4. Class-wise Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['Real', 'Fake']
    v1_class_accs = [metrics_v1['real_accuracy'], metrics_v1['fake_accuracy']]
    v2_class_accs = [metrics_v2['real_accuracy'], metrics_v2['fake_accuracy']]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, v1_class_accs, width, label='Ensemble V1', alpha=0.8)
    bars2 = ax.bar(x + width/2, v2_class_accs, width, label='Ensemble V2', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Class-wise Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([80, 100])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: class_accuracy.png")
    plt.close()


def save_comparison_report(metrics_v1, metrics_v2):
    """Save detailed comparison report as JSON."""
    print("\n💾 Saving comparison report...")
    
    # Prepare report
    report = {
        'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': {
            'ensemble_v1': {
                'accuracy': metrics_v1['accuracy'],
                'precision': metrics_v1['precision'],
                'recall': metrics_v1['recall'],
                'f1': metrics_v1['f1'],
                'auc': metrics_v1['auc'],
                'real_accuracy': metrics_v1['real_accuracy'],
                'fake_accuracy': metrics_v1['fake_accuracy'],
                'confusion_matrix': metrics_v1['confusion_matrix'],
                'eval_time': metrics_v1['eval_time'],
                'source_metrics': metrics_v1['source_metrics']
            },
            'ensemble_v2': {
                'accuracy': metrics_v2['accuracy'],
                'precision': metrics_v2['precision'],
                'recall': metrics_v2['recall'],
                'f1': metrics_v2['f1'],
                'auc': metrics_v2['auc'],
                'real_accuracy': metrics_v2['real_accuracy'],
                'fake_accuracy': metrics_v2['fake_accuracy'],
                'confusion_matrix': metrics_v2['confusion_matrix'],
                'eval_time': metrics_v2['eval_time'],
                'source_metrics': metrics_v2['source_metrics']
            }
        },
        'improvements': {
            'accuracy': metrics_v2['accuracy'] - metrics_v1['accuracy'],
            'precision': metrics_v2['precision'] - metrics_v1['precision'],
            'recall': metrics_v2['recall'] - metrics_v1['recall'],
            'f1': metrics_v2['f1'] - metrics_v1['f1'],
            'auc': metrics_v2['auc'] - metrics_v1['auc'],
            'real_accuracy': metrics_v2['real_accuracy'] - metrics_v1['real_accuracy'],
            'fake_accuracy': metrics_v2['fake_accuracy'] - metrics_v1['fake_accuracy']
        },
        'summary': {
            'v2_better_metrics': sum(1 for k in ['accuracy', 'precision', 'recall', 'f1', 'auc'] 
                                     if metrics_v2[k] > metrics_v1[k]),
            'total_metrics': 5,
            'overall_winner': 'V2' if metrics_v2['f1'] > metrics_v1['f1'] else 'V1'
        }
    }
    
    # Save JSON report
    with open(OUTPUT_DIR / 'comparison_results.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"   ✅ Saved: comparison_results.json")
    
    # Create markdown report
    md_report = f"""# Ensemble Model Comparison Report

**Generated:** {report['comparison_date']}

## Summary

- **Overall Winner:** Ensemble {report['summary']['overall_winner']}
- **Better Metrics:** {report['summary']['v2_better_metrics']}/{report['summary']['total_metrics']}

## Performance Metrics

| Metric | Ensemble V1 | Ensemble V2 | Improvement |
|--------|-------------|-------------|-------------|
| Accuracy | {metrics_v1['accuracy']:.2f}% | {metrics_v2['accuracy']:.2f}% | {report['improvements']['accuracy']:+.2f}% |
| Precision | {metrics_v1['precision']:.2f}% | {metrics_v2['precision']:.2f}% | {report['improvements']['precision']:+.2f}% |
| Recall | {metrics_v1['recall']:.2f}% | {metrics_v2['recall']:.2f}% | {report['improvements']['recall']:+.2f}% |
| F1 Score | {metrics_v1['f1']:.2f}% | {metrics_v2['f1']:.2f}% | {report['improvements']['f1']:+.2f}% |
| AUC | {metrics_v1['auc']:.2f}% | {metrics_v2['auc']:.2f}% | {report['improvements']['auc']:+.2f}% |

## Class-wise Performance

| Class | V1 Accuracy | V2 Accuracy | Improvement |
|-------|-------------|-------------|-------------|
| Real | {metrics_v1['real_accuracy']:.2f}% | {metrics_v2['real_accuracy']:.2f}% | {report['improvements']['real_accuracy']:+.2f}% |
| Fake | {metrics_v1['fake_accuracy']:.2f}% | {metrics_v2['fake_accuracy']:.2f}% | {report['improvements']['fake_accuracy']:+.2f}% |

## Confusion Matrices

### Ensemble V1
```
             Predicted
             Real    Fake
Actual Real  {metrics_v1['confusion_matrix'][0][0]:5d}  {metrics_v1['confusion_matrix'][0][1]:5d}
Actual Fake  {metrics_v1['confusion_matrix'][1][0]:5d}  {metrics_v1['confusion_matrix'][1][1]:5d}
```

### Ensemble V2
```
             Predicted
             Real    Fake
Actual Real  {metrics_v2['confusion_matrix'][0][0]:5d}  {metrics_v2['confusion_matrix'][0][1]:5d}
Actual Fake  {metrics_v2['confusion_matrix'][1][0]:5d}  {metrics_v2['confusion_matrix'][1][1]:5d}
```

## Evaluation Time

- **Ensemble V1:** {metrics_v1['eval_time']:.2f}s
- **Ensemble V2:** {metrics_v2['eval_time']:.2f}s

## Visualizations

Generated visualizations:
- `metrics_comparison.png` - Overall metrics comparison
- `confusion_matrices.png` - Confusion matrices side-by-side
- `source_comparison.png` - Per-source performance
- `class_accuracy.png` - Class-wise accuracy comparison
"""
    
    with open(OUTPUT_DIR / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"   ✅ Saved: comparison_report.md")


# ============ MAIN ============
def main():
    """Main comparison function."""
    print("\n" + "=" * 80)
    print("🔬 ENSEMBLE MODEL COMPARISON - V1 vs V2")
    print("=" * 80)
    
    # Check if models exist
    if not MODEL_V1_PATH.exists():
        print(f"\n❌ Ensemble V1 not found: {MODEL_V1_PATH}")
        return
    
    if not MODEL_V2_PATH.exists():
        print(f"\n❌ Ensemble V2 not found: {MODEL_V2_PATH}")
        return
    
    # Load test data
    test_data = load_test_data()
    
    if len(test_data) == 0:
        print("\n❌ No test data found!")
        return
    
    # Create dataset and loader
    test_dataset = MultiSourceDeepfakeDataset(test_data, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"\n📊 Test Dataset: {len(test_dataset)} images")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Batches: {len(test_loader)}")
    
    # Load models
    model_v1, checkpoint_v1 = load_model(MODEL_V1_PATH, "Ensemble V1")
    model_v2, checkpoint_v2 = load_model(MODEL_V2_PATH, "Ensemble V2")
    
    # Evaluate both models
    print("\n" + "=" * 80)
    print("🚀 STARTING EVALUATION")
    print("=" * 80)
    
    metrics_v1 = evaluate_model(model_v1, test_loader, "Ensemble V1")
    metrics_v2 = evaluate_model(model_v2, test_loader, "Ensemble V2")
    
    # Print comparison
    print_comparison(metrics_v1, metrics_v2)
    
    # Create visualizations
    create_visualizations(metrics_v1, metrics_v2)
    
    # Save report
    save_comparison_report(metrics_v1, metrics_v2)
    
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\n📂 Results saved to: {OUTPUT_DIR}")
    print(f"   - comparison_results.json (detailed metrics)")
    print(f"   - comparison_report.md (human-readable report)")
    print(f"   - metrics_comparison.png (bar chart)")
    print(f"   - confusion_matrices.png (heatmaps)")
    print(f"   - source_comparison.png (per-source performance)")
    print(f"   - class_accuracy.png (class-wise accuracy)")
    
    # Print summary
    print(f"\n🏆 WINNER: Ensemble {'V2' if metrics_v2['f1'] > metrics_v1['f1'] else 'V1'}")
    print(f"   Based on F1 Score: V1={metrics_v1['f1']:.2f}%, V2={metrics_v2['f1']:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
