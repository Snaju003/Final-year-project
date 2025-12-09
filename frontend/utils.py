"""
Utility functions for the Streamlit deepfake detection frontend
Adapted from deepfake_detector_inference.py
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import timm
import tempfile

# Add the src directory to Python path to import models
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

# Model paths and settings
MODEL_PATH = Path(__file__).parent.parent / "models" / "finetuned" / "finetuned_model.pth"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ MODEL ARCHITECTURES ============
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

# ============ IMAGE PREPROCESSING ============
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ MODEL LOADING ============
def load_deepfake_model():
    """Load the trained deepfake detection model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    print(f"Loading model from {MODEL_PATH}")
    
    # Try loading with DeepfakeDetector (B4 with attention) first
    try:
        model = DeepfakeDetector(num_classes=2, pretrained=False).to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully (DeepfakeDetector - EfficientNet-B4)")
        return model
    except RuntimeError as e:
        if "Missing key" in str(e) or "size mismatch" in str(e):
            print("DeepfakeDetector failed, trying FastDeepfakeDetector...")
            try:
                model = FastDeepfakeDetector(num_classes=2, pretrained=False).to(device)
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print("Model loaded successfully (FastDeepfakeDetector - EfficientNet-B0)")
                return model
            except Exception as e2:
                print(f"Both model architectures failed: {e2}")
                raise
        else:
            raise

# ============ VIDEO PREDICTION ============
def predict_video_frames(model, video_path, frame_step=30, progress_callback=None):
    """
    Predict deepfake in video by analyzing faces in frames.
    
    Args:
        model: Trained model
        video_path: Path to video file (can be temporary file)
        frame_step: Analyze every Nth frame
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Dictionary with prediction results or None if no faces detected
    """
    print(f"Analyzing video: {video_path}")
    
    # Initialize face detector
    mtcnn = MTCNN(keep_all=True, device=device)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_predictions = []
    frame_idx = 0
    frames_processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        if progress_callback:
            progress = (frame_idx / total_frames) * 100
            progress_callback(progress)
        
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
                    try:
                        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        face_tensor = transform(face_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = model(face_tensor)
                            probs = torch.softmax(output, dim=1)
                            fake_prob = probs[0][1].item()
                        
                        frame_predictions.append(fake_prob)
                        frames_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
        
        frame_idx += 1
    
    cap.release()
    
    # Final progress update
    if progress_callback:
        progress_callback(100)
    
    # Aggregate predictions
    if frame_predictions and len(frame_predictions) > 0:
        frame_predictions = np.array(frame_predictions)
        avg_fake_prob = np.mean(frame_predictions)
        max_fake_prob = np.max(frame_predictions)
        verdict = "FAKE" if avg_fake_prob > CONFIDENCE_THRESHOLD else "REAL"
        
        result = {
            'verdict': verdict,
            'avg_fake_probability': float(avg_fake_prob * 100),
            'max_fake_probability': float(max_fake_prob * 100),
            'frames_analyzed': len(frame_predictions),
            'confidence': float(max(avg_fake_prob, 1 - avg_fake_prob) * 100),
            'frames_processed': frames_processed,
            'total_frames': total_frames
        }
        
        print(f"Analysis complete:")
        print(f"Verdict: {verdict}")
        print(f"Avg Fake Probability: {avg_fake_prob*100:.2f}%")
        print(f"Max Fake Probability: {max_fake_prob*100:.2f}%")
        print(f"Frames Analyzed: {len(frame_predictions)}")
        
        return result
    else:
        print("No faces detected in video")
        return None

# ============ UTILITY FUNCTIONS ============
def cleanup_temp_files(file_path):
    """Clean up temporary files."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"Warning: Could not clean up {file_path}: {e}")

def get_model_info():
    """Get information about the loaded model."""
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        return {
            'path': str(MODEL_PATH),
            'size_mb': round(size_mb, 2),
            'device': str(device),
            'exists': True
        }
    else:
        return {
            'path': str(MODEL_PATH),
            'exists': False,
            'device': str(device)
        }

def predict_single_image(model, image_path):
    """Predict if a single image is real or fake (utility function)."""
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
        print(f"Error predicting image {image_path}: {e}")
        return None