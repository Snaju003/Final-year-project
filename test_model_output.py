"""
Quick test to check what the model is actually predicting
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from ensemble_model import create_ensemble
from torchvision import transforms
from PIL import Image

# Load model
MODEL_PATH = Path(r"E:\Final-year-project\models\ensemble_v2\ensemble_best_v2.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Check if it's an ensemble or single model
if 'weights' in checkpoint:
    # Ensemble model
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
    print(f"Ensemble model loaded!")
    print(f"Weights: {model_weights}")
else:
    # Single EfficientNet model
    from ensemble_model import EfficientNetDetector
    
    model = EfficientNetDetector(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Single EfficientNet model loaded!")

model.eval()
print(f"Model loaded successfully!")

# Test with dummy input
print("\n" + "="*60)
print("Testing with random noise (should give ~50/50 predictions)")
print("="*60)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for i in range(5):
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        
        print(f"\nTest {i+1}:")
        print(f"  Raw output: {output[0].cpu().numpy()}")
        print(f"  Probabilities [Real, Fake]: [{probs[0][0].item():.4f}, {probs[0][1].item():.4f}]")
        print(f"  Prediction: {'FAKE' if pred_class == 1 else 'REAL'}")

# Test with actual images from dataset 5 (parquet)
print("\n" + "="*60)
print("Testing with Dataset 5 (Parquet file)")
print("="*60)

try:
    import pandas as pd
    import io
    
    parquet_path = Path(r"X:\Final-year-project-data\data\dataset 5\test-00000-of-00001.parquet")
    
    if parquet_path.exists():
        print(f"Loading {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        print(f"Dataset has {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            print(f"Label distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")
        
        # Test on first 10 samples (5 real, 5 fake if available)
        num_samples = min(10, len(df))
        correct = 0
        total = 0
        
        for idx in range(num_samples):
            row = df.iloc[idx]
            
            # Get image bytes
            if 'image' in row:
                img_bytes = row['image']['bytes']
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Get label (assuming 0=real, 1=fake or 'real'/'fake' strings)
                true_label = row.get('label', None)
                if isinstance(true_label, str):
                    true_label = 1 if true_label.lower() == 'fake' else 0
                
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred_class = output.argmax(1).item()
                    
                    label_str = 'FAKE' if true_label == 1 else 'REAL'
                    pred_str = 'FAKE' if pred_class == 1 else 'REAL'
                    is_correct = (pred_class == true_label)
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    print(f"\nSample {idx+1} (True: {label_str}):")
                    print(f"  Raw output: {output[0].cpu().numpy()}")
                    print(f"  Probabilities [Real, Fake]: [{probs[0][0].item():.4f}, {probs[0][1].item():.4f}]")
                    print(f"  Prediction: {pred_str}")
                    print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        print(f"\n📊 Accuracy on {total} samples: {correct}/{total} ({correct/total*100:.1f}%)")
    else:
        print(f"❌ Parquet file not found: {parquet_path}")

except ImportError:
    print("❌ pandas not installed. Install with: pip install pandas pyarrow")
except Exception as e:
    print(f"❌ Error loading parquet: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("If all predictions are REAL (class 0), the model is biased.")
print("If predictions vary randomly, the model is working but may need retraining.")
print("If predictions match the labels, the model is working correctly!")
