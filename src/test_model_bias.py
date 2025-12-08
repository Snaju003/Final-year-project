"""Quick diagnostic test for the ensemble model"""
import torch
from ensemble_model import create_ensemble

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ckpt = torch.load('models/ensemble/ensemble_final.pth', map_location=device, weights_only=False)

# Get weights
weights = ckpt.get('weights', {})
model_weights = (weights.get('efficientnet', 0.4), weights.get('xception', 0.35), weights.get('mesonet', 0.25))
print(f"Weights: {model_weights}")

model = create_ensemble(model_type='weighted', weights=model_weights, num_classes=2, pretrained=False).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f"\nVal Acc from checkpoint: {ckpt.get('val_acc', 'N/A')}")
print(f"Val F1 from checkpoint: {ckpt.get('val_f1', 'N/A')}")

# Test with random inputs (multiple times)
print("\n--- Testing with random noise (should be ~50/50) ---")
for i in range(5):
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)
        probs = torch.softmax(output, dim=1)
        print(f"Test {i+1}: Real={probs[0][0].item()*100:.1f}%, Fake={probs[0][1].item()*100:.1f}%")

# Test with black image
print("\n--- Testing with black image ---")
black_input = torch.zeros(1, 3, 224, 224).to(device)
# Normalize it like we do in inference
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
black_normalized = (black_input - mean) / std
with torch.no_grad():
    output = model(black_normalized)
    probs = torch.softmax(output, dim=1)
    print(f"Black image: Real={probs[0][0].item()*100:.1f}%, Fake={probs[0][1].item()*100:.1f}%")

# Test with white image
print("\n--- Testing with white image ---")
white_input = torch.ones(1, 3, 224, 224).to(device)
white_normalized = (white_input - mean) / std
with torch.no_grad():
    output = model(white_normalized)
    probs = torch.softmax(output, dim=1)
    print(f"White image: Real={probs[0][0].item()*100:.1f}%, Fake={probs[0][1].item()*100:.1f}%")

# Check individual models on random input
print("\n--- Individual model responses to random input ---")
test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    ind = model.get_individual_predictions(test_input)
    print(f"EfficientNet - Real: {ind['efficientnet'][0][0].item()*100:.1f}%, Fake: {ind['efficientnet'][0][1].item()*100:.1f}%")
    print(f"Xception     - Real: {ind['xception'][0][0].item()*100:.1f}%, Fake: {ind['xception'][0][1].item()*100:.1f}%")
    print(f"MesoNet      - Real: {ind['mesonet'][0][0].item()*100:.1f}%, Fake: {ind['mesonet'][0][1].item()*100:.1f}%")
