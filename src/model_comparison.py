"""
Model Comparison Script
Compare performance of single model vs ensemble on test dataset
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ensemble_model import create_ensemble, EfficientNetDetector
from ensemble_train import DeepfakeDataset

# ============ PATHS ============
SINGLE_MODEL_PATH = Path(r"E:\Final-year-project\models\finetuned\finetuned_model.pth")
ENSEMBLE_MODEL_PATH = Path(r"E:\Final-year-project\models\ensemble\ensemble_final.pth")
TEST_DATA_ROOT = Path(r"E:\Final-year-project-data\data\dataset 2\test")
OUTPUT_DIR = Path(r"E:\Final-year-project\results\comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ SETTINGS ============
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


# ============ LOAD MODELS ============
def load_single_model(model_path):
    """Load single EfficientNet model."""
    print(f"📂 Loading single model from {model_path}")
    
    model = EfficientNetDetector(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Single model loaded")
    return model


def load_ensemble_model(model_path):
    """Load ensemble model."""
    print(f"📂 Loading ensemble model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    ensemble_type = checkpoint.get('ensemble_type', 'weighted')
    model_weights = checkpoint.get('model_weights', (0.4, 0.35, 0.25))
    
    model = create_ensemble(
        model_type=ensemble_type,
        weights=model_weights,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Ensemble model loaded")
    return model


# ============ EVALUATION ============
def evaluate_model(model, test_loader, model_name):
    """Evaluate model on test set."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"⚠️  Could not compute AUC-ROC: {e}")
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision * 100),
        'recall': float(recall * 100),
        'f1_score': float(f1 * 100),
        'auc_roc': float(auc),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'per_class': {
            'real': {
                'precision': float(per_class_precision[0] * 100),
                'recall': float(per_class_recall[0] * 100),
                'f1': float(per_class_f1[0] * 100)
            },
            'fake': {
                'precision': float(per_class_precision[1] * 100),
                'recall': float(per_class_recall[1] * 100),
                'f1': float(per_class_f1[1] * 100)
            }
        },
        'predictions': {
            'all_preds': [int(x) for x in all_preds],
            'all_labels': [int(x) for x in all_labels],
            'all_probs': [float(x) for x in all_probs]
        }
    }
    
    # Print results
    print(f"\n📊 Results for {model_name}:")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TN: {tn:4d} | FP: {fp:4d}")
    print(f"FN: {fn:4d} | TP: {tp:4d}")
    
    return results


# ============ VISUALIZATION ============
def plot_comparison(single_results, ensemble_results, save_dir):
    """Create comparison visualizations."""
    
    # 1. Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        values = [single_results[metric], ensemble_results[metric]]
        bars = ax.bar(['Single Model', 'Ensemble'], values, color=['#3498db', '#e74c3c'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel(f'{name} (%)', fontsize=11)
        ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight improvement
        improvement = ensemble_results[metric] - single_results[metric]
        color = 'green' if improvement > 0 else 'red'
        ax.text(0.5, 95, f'{improvement:+.2f}%', 
               ha='center', fontsize=10, color=color, fontweight='bold',
               transform=ax.transData)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: metrics_comparison.png")
    
    # 2. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (results, title) in enumerate([
        (single_results, 'Single Model'),
        (ensemble_results, 'Ensemble Model')
    ]):
        cm = np.array([[
            results['confusion_matrix']['true_negatives'],
            results['confusion_matrix']['false_positives']
        ], [
            results['confusion_matrix']['false_negatives'],
            results['confusion_matrix']['true_positives']
        ]])
        
        if sns is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                       cbar_kws={'label': 'Count'})
        else:
            # Fallback without seaborn
            im = axes[idx].imshow(cm, cmap='Blues')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Real', 'Fake'])
            axes[idx].set_yticklabels(['Real', 'Fake'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', color='white', fontweight='bold')
        
        axes[idx].set_title(f'{title}\nAccuracy: {results["accuracy"]:.2f}%', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=11)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: confusion_matrices.png")
    
    # 3. ROC Curve Comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for results, label, color in [
        (single_results, 'Single Model', '#3498db'),
        (ensemble_results, 'Ensemble', '#e74c3c')
    ]:
        fpr, tpr, _ = roc_curve(results['predictions']['all_labels'], 
                                results['predictions']['all_probs'])
        
        ax.plot(fpr, tpr, label=f"{label} (AUC={results['auc_roc']:.4f})", 
               color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: roc_comparison.png")
    
    # 4. Per-Class Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = ['Real', 'Fake']
    metrics_pc = ['precision', 'recall', 'f1']
    
    for class_idx, class_name in enumerate(['real', 'fake']):
        ax = axes[class_idx]
        
        x = np.arange(len(metrics_pc))
        width = 0.35
        
        single_vals = [single_results['per_class'][class_name][m] for m in metrics_pc]
        ensemble_vals = [ensemble_results['per_class'][class_name][m] for m in metrics_pc]
        
        bars1 = ax.bar(x - width/2, single_vals, width, label='Single Model', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title(f'{classes[class_idx]} Class Performance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Precision', 'Recall', 'F1'], fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: per_class_comparison.png")


# ============ MAIN ============
def main():
    print(f"\n{'='*60}")
    print(f"📊 MODEL COMPARISON EVALUATION")
    print(f"{'='*60}")
    
    # Check if models exist
    if not SINGLE_MODEL_PATH.exists():
        print(f"❌ Single model not found: {SINGLE_MODEL_PATH}")
        return
    
    if not ENSEMBLE_MODEL_PATH.exists():
        print(f"❌ Ensemble model not found: {ENSEMBLE_MODEL_PATH}")
        return
    
    # Load test data
    print(f"\n📂 Loading test data from {TEST_DATA_ROOT}")
    
    if not TEST_DATA_ROOT.exists():
        print(f"❌ Test data directory not found: {TEST_DATA_ROOT}")
        print(f"Please ensure your test data is in the correct location")
        return
    
    test_real = list((TEST_DATA_ROOT / "Real").glob("*")) if (TEST_DATA_ROOT / "Real").exists() else []
    test_fake = list((TEST_DATA_ROOT / "Fake").glob("*")) if (TEST_DATA_ROOT / "Fake").exists() else []
    
    print(f"Test data: {len(test_real)} real, {len(test_fake)} fake")
    
    if not test_real or not test_fake:
        print("❌ No test data found!")
        print(f"Expected directories:")
        print(f"  - {TEST_DATA_ROOT / 'Real'}")
        print(f"  - {TEST_DATA_ROOT / 'Fake'}")
        return
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DeepfakeDataset(test_real, test_fake, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    # Load models
    single_model = load_single_model(SINGLE_MODEL_PATH)
    ensemble_model = load_ensemble_model(ENSEMBLE_MODEL_PATH)
    
    # Evaluate both models
    single_results = evaluate_model(single_model, test_loader, "Single EfficientNet")
    ensemble_results = evaluate_model(ensemble_model, test_loader, "Ensemble (EfficientNet + Xception + MesoNet)")
    
    # Calculate improvements
    print(f"\n{'='*60}")
    print(f"📈 IMPROVEMENT ANALYSIS")
    print(f"{'='*60}")
    
    improvements = {
        'accuracy': ensemble_results['accuracy'] - single_results['accuracy'],
        'precision': ensemble_results['precision'] - single_results['precision'],
        'recall': ensemble_results['recall'] - single_results['recall'],
        'f1_score': ensemble_results['f1_score'] - single_results['f1_score'],
        'auc_roc': ensemble_results['auc_roc'] - single_results['auc_roc']
    }
    
    for metric, improvement in improvements.items():
        symbol = '📈' if improvement > 0 else '📉'
        color_text = 'IMPROVED' if improvement > 0 else 'DECREASED'
        print(f"{symbol} {metric.replace('_', ' ').title()}: {improvement:+.2f}% - {color_text}")
    
    # Create visualizations
    print(f"\n{'='*60}")
    print(f"📊 GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_comparison(single_results, ensemble_results, OUTPUT_DIR)
    
    # Save detailed comparison
    comparison_data = {
        'single_model': single_results,
        'ensemble_model': ensemble_results,
        'improvements': improvements,
        'test_dataset': {
            'total_samples': len(test_dataset),
            'real_samples': len(test_real),
            'fake_samples': len(test_fake)
        }
    }
    
    # Remove predictions from saved JSON (too large)
    comparison_data['single_model'].pop('predictions', None)
    comparison_data['ensemble_model'].pop('predictions', None)
    
    with open(OUTPUT_DIR / 'comparison_results.json', 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    print(f"💾 Saved: comparison_results.json")
    
    print(f"\n{'='*60}")
    print(f"✅ COMPARISON COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nVisualization files:")
    print(f"  - metrics_comparison.png")
    print(f"  - confusion_matrices.png")
    print(f"  - roc_comparison.png")
    print(f"  - per_class_comparison.png")
    print(f"  - comparison_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()