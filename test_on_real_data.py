"""Test the model on actual training data"""
import torch
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, 'src')
from ensemble_model import create_ensemble

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ckpt = torch.load('models/ensemble/ensemble_final.pth', map_location=device, weights_only=False)
weights = ckpt.get('weights', {})
model_weights = (weights.get('efficientnet', 0.4), weights.get('xception', 0.35), weights.get('mesonet', 0.25))

model = create_ensemble(model_type='weighted', weights=model_weights, num_classes=2, pretrained=False).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on actual training data
import os

test_images = [
    (r'E:\Final-year-project-data\data\dataset 2\test\Real\real_0.jpg', 'REAL'),
    (r'E:\Final-year-project-data\data\dataset 2\test\Real\real_1.jpg', 'REAL'),
    (r'E:\Final-year-project-data\data\dataset 2\test\Real\real_10.jpg', 'REAL'),
    (r'E:\Final-year-project-data\data\dataset 2\test\Fake\fake_0.jpg', 'FAKE'),
    (r'E:\Final-year-project-data\data\dataset 2\test\Fake\fake_1.jpg', 'FAKE'),
    (r'E:\Final-year-project-data\data\dataset 2\test\Fake\fake_10.jpg', 'FAKE'),
]

print("\n--- Testing on actual test set images ---")
correct = 0
total = 0

for path, label in test_images:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    
    try:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = 'FAKE' if probs[0][1] > 0.5 else 'REAL'
            is_correct = pred == label
            correct += int(is_correct)
            total += 1
            status = "✅" if is_correct else "❌"
            print(f'{status} {label} -> Predicted: {pred} (Real: {probs[0][0].item()*100:.1f}%, Fake: {probs[0][1].item()*100:.1f}%)')
    except Exception as e:
        print(f'Error loading {label}: {e}')

print(f"\nAccuracy on test samples: {correct}/{total} = {correct/total*100:.1f}%")
