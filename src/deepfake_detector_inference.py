"""
Deepfake Detection Inference Script
Test trained model on new images or videos
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import timm
from tqdm import tqdm
import json

# ============ PATHS ============
MODEL_PATH = Path(r"X:\Final-year-project\models\best_model.pth")
OUTPUT_DIR = Path(r"X:\Final-year-project\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ SETTINGS ============
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5  # Threshold for fake detection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


# ============ MODEL ARCHITECTURE (MUST match training) ============
class AttentionModule(nn.Module):
    """Spatial attention mechanism."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class DeepfakeDetector(nn.Module):
    """Original EfficientNet-B4 with attention (for old models)."""
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        
        # EfficientNet-B4 backbone
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Attention module
        self.attention = AttentionModule(self.feature_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Classify
        output = self.classifier(features)
        return output


class FastDeepfakeDetector(nn.Module):
    """New EfficientNet-B0 without attention (for new models)."""
    def __init__(self, num_classes=2, pretrained=False):
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


# ============ LOAD MODEL ============
def load_model(model_path):
    """Load trained model."""
    print(f"📂 Loading model from {model_path}")
    
    # Try loading with DeepfakeDetector (B4 with attention) first
    try:
        model = DeepfakeDetector(num_classes=2, pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded (DeepfakeDetector - EfficientNet-B4)")
        return model
    except RuntimeError as e:
        if "Missing key" in str(e) or "size mismatch" in str(e):
            print(f"⚠️  DeepfakeDetector failed, trying FastDeepfakeDetector...")
            try:
                model = FastDeepfakeDetector(num_classes=2, pretrained=False).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print(f"✅ Model loaded (FastDeepfakeDetector - EfficientNet-B0)")
                return model
            except Exception as e2:
                print(f"❌ Both model architectures failed: {e2}")
                raise
        else:
            raise


# ============ IMAGE PREPROCESSING ============
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============ PREDICT SINGLE IMAGE ============
def predict_image(model, image_path):
    """Predict if a single image is real or fake."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = probs[0][pred_class].item()
        
        label = "FAKE" if pred_class == 1 else "REAL"
        fake_prob = probs[0][1].item()
        
        return {
            'prediction': label,
            'confidence': confidence * 100,
            'fake_probability': fake_prob * 100,
            'real_probability': probs[0][0].item() * 100
        }
    
    except Exception as e:
        print(f"❌ Error predicting {image_path}: {e}")
        return None


# ============ PREDICT VIDEO ============
def predict_video(model, video_path, frame_step=30, output_video=None):
    """
    Predict deepfake in video by analyzing faces.
    
    Args:
        model: Trained model
        video_path: Path to video file
        frame_step: Analyze every Nth frame
        output_video: Path to save annotated video (optional)
    """
    print(f"\n🎬 Analyzing video: {video_path.name}")
    
    # Initialize face detector
    mtcnn = MTCNN(keep_all=True, device=device)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer (optional)
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (int(width), int(height)))
        if not writer.isOpened():
            print(f"⚠️  Warning: Could not create video writer, skipping video output")
            writer = None
    
    frame_predictions = []
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                boxes, _ = mtcnn.detect([rgb_frame])
            except:
                boxes = None
            
            if boxes is not None and len(boxes) > 0 and boxes[0] is not None:
                for box in boxes[0]:
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Extract face
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    
                    # Predict
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(face_tensor)
                        probs = torch.softmax(output, dim=1)
                        fake_prob = probs[0][1].item()
                    
                    frame_predictions.append(fake_prob)
                    
                    # Annotate frame
                    color = (0, 0, 255) if fake_prob > CONFIDENCE_THRESHOLD else (0, 255, 0)
                    label = f"{'FAKE' if fake_prob > CONFIDENCE_THRESHOLD else 'REAL'}: {fake_prob*100:.1f}%"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if writer and writer.isOpened():
            writer.write(frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if writer and writer.isOpened():
        writer.release()
    
    # Aggregate predictions
    if frame_predictions and len(frame_predictions) > 0:
        frame_predictions = np.array(frame_predictions)
        avg_fake_prob = np.mean(frame_predictions)
        max_fake_prob = np.max(frame_predictions)
        verdict = "FAKE" if avg_fake_prob > CONFIDENCE_THRESHOLD else "REAL"
        
        result = {
            'video': video_path.name,
            'verdict': verdict,
            'avg_fake_probability': float(avg_fake_prob * 100),
            'max_fake_probability': float(max_fake_prob * 100),
            'frames_analyzed': len(frame_predictions),
            'confidence': float(max(avg_fake_prob, 1 - avg_fake_prob) * 100)
        }
        
        print(f"\n📊 Results:")
        print(f"Verdict: {verdict}")
        print(f"Avg Fake Probability: {avg_fake_prob*100:.2f}%")
        print(f"Max Fake Probability: {max_fake_prob*100:.2f}%")
        print(f"Frames Analyzed: {len(frame_predictions)}")
        
        return result
    else:
        print("❌ No faces detected in video")
        return None


# ============ BATCH PREDICT IMAGES ============
def predict_folder(model, folder_path, save_results=True):
    """Predict all images in a folder."""
    folder_path = Path(folder_path)
    image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n📂 Analyzing {len(image_files)} images from {folder_path.name}")
    
    results = []
    real_count = 0
    fake_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        result = predict_image(model, img_path)
        if result:
            result['filename'] = img_path.name
            results.append(result)
            
            if result['prediction'] == 'FAKE':
                fake_count += 1
            else:
                real_count += 1
    
    print(f"\n📊 Summary:")
    print(f"Total: {len(results)} images")
    print(f"Real:  {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"Fake:  {fake_count} ({fake_count/len(results)*100:.1f}%)")
    
    if save_results:
        output_file = OUTPUT_DIR / f"{folder_path.name}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"💾 Results saved: {output_file}")
    
    return results


# ============ MAIN ============
def main():
    """Main inference pipeline."""
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please train the model first using the training script.")
        return
    
    # Load model
    model = load_model(MODEL_PATH)
    
    print(f"\n{'='*60}")
    print(f"🔍 DEEPFAKE DETECTION - INFERENCE MODE")
    print(f"{'='*60}")
    print(f"\nOptions:")
    print(f"1. Predict single image")
    print(f"2. Predict video")
    print(f"3. Predict folder of images")
    print(f"4. Exit")
    
    while True:
        choice = input(f"\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if Path(img_path).exists():
                result = predict_image(model, img_path)
                if result:
                    print(f"\n📊 Result:")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}%")
                    print(f"Fake Probability: {result['fake_probability']:.2f}%")
            else:
                print("❌ File not found")
        
        elif choice == '2':
            vid_path = input("Enter video path: ").strip()
            save_output = input("Save annotated video? (y/n): ").strip().lower() == 'y'
            
            if Path(vid_path).exists():
                output_path = None
                if save_output:
                    output_path = OUTPUT_DIR / f"annotated_{Path(vid_path).name}"
                
                result = predict_video(model, Path(vid_path), frame_step=30, output_video=output_path)
                
                if result and save_output:
                    print(f"💾 Annotated video saved: {output_path}")
            else:
                print("❌ File not found")
        
        elif choice == '3':
            folder = input("Enter folder path: ").strip()
            if Path(folder).exists():
                predict_folder(model, folder, save_results=True)
            else:
                print("❌ Folder not found")
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()