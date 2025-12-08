"""
Ensemble Model Inference Script
Test ensemble model on new images and videos with detailed predictions
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
from tqdm import tqdm
import json
from ensemble_model import create_ensemble

# ============ PATHS ============
MODEL_PATH = Path(r"E:\Final-year-project\models\ensemble_v2\ensemble_best_v2.pth")
OUTPUT_DIR = Path(r"E:\Final-year-project\results\ensemble_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ SETTINGS ============
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


# ============ LOAD MODEL ============
def load_ensemble_model(model_path):
    """Load trained ensemble model."""
    print(f"📂 Loading ensemble model from {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get ensemble configuration
    ensemble_type = checkpoint.get('ensemble_type', 'weighted')
    
    # Handle weights - can be dict or tuple
    weights_data = checkpoint.get('weights', checkpoint.get('model_weights', (0.4, 0.35, 0.25)))
    if isinstance(weights_data, dict):
        model_weights = (
            weights_data.get('efficientnet', 0.4),
            weights_data.get('xception', 0.35),
            weights_data.get('mesonet', 0.25)
        )
    else:
        model_weights = weights_data
    
    print(f"Ensemble type: {ensemble_type}")
    print(f"Model weights: EfficientNet={model_weights[0]}, Xception={model_weights[1]}, MesoNet={model_weights[2]}")
    
    # Create model
    model = create_ensemble(
        model_type=ensemble_type,
        weights=model_weights,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test model with dummy input to verify it works
    print("\n🧪 Testing model with dummy input...")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        test_output = model(dummy_input)
        test_probs = torch.softmax(test_output, dim=1)
        print(f"   Dummy output: {test_output[0].cpu().numpy()}")
        print(f"   Dummy probs [Real, Fake]: [{test_probs[0][0].item():.4f}, {test_probs[0][1].item():.4f}]")
    
    # Get validation accuracy - handle different checkpoint formats
    val_acc = checkpoint.get('val_acc', checkpoint.get('val_metrics', {}).get('accuracy', 'N/A'))
    acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, (int, float)) else str(val_acc)
    print(f"✅ Model loaded (Val Acc: {acc_str})")
    
    return model


# ============ IMAGE PREPROCESSING ============
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============ PREDICT SINGLE IMAGE ============
def predict_image(model, image_path, show_individual=False):
    """Predict if a single image is real or fake."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Ensemble prediction
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = probs[0][pred_class].item()
            
            # DEBUG: Print raw output and probabilities
            print(f"\n🔍 DEBUG - Raw output: {output[0].cpu().numpy()}")
            print(f"🔍 DEBUG - Probabilities [Real, Fake]: [{probs[0][0].item():.4f}, {probs[0][1].item():.4f}]")
            print(f"🔍 DEBUG - Predicted class: {pred_class} ({'FAKE' if pred_class == 1 else 'REAL'})")
            
            # Individual model predictions
            individual_preds = None
            if show_individual:
                individual_preds = model.get_individual_predictions(img_tensor)
                individual_preds = {
                    'efficientnet': {
                        'fake_prob': individual_preds['efficientnet'][0][1].item() * 100,
                        'real_prob': individual_preds['efficientnet'][0][0].item() * 100
                    },
                    'xception': {
                        'fake_prob': individual_preds['xception'][0][1].item() * 100,
                        'real_prob': individual_preds['xception'][0][0].item() * 100
                    },
                    'mesonet': {
                        'fake_prob': individual_preds['mesonet'][0][1].item() * 100,
                        'real_prob': individual_preds['mesonet'][0][0].item() * 100
                    }
                }
        
        label = "FAKE" if pred_class == 1 else "REAL"
        fake_prob = probs[0][1].item()
        
        result = {
            'prediction': label,
            'confidence': confidence * 100,
            'fake_probability': fake_prob * 100,
            'real_probability': probs[0][0].item() * 100
        }
        
        if individual_preds:
            result['individual_models'] = individual_preds
        
        return result
    
    except Exception as e:
        print(f"❌ Error predicting {image_path}: {e}")
        return None


# ============ PREDICT VIDEO ============
def predict_video(model, video_path, frame_step=30, output_video=None, show_individual=False):
    """
    Predict deepfake in video by analyzing faces.
    
    Args:
        model: Trained ensemble model
        video_path: Path to video file
        frame_step: Analyze every Nth frame
        output_video: Path to save annotated video (optional)
        show_individual: Show individual model contributions
    """
    print(f"\n🎬 Analyzing video: {video_path.name}")
    
    mtcnn = MTCNN(keep_all=True, device=device)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    frame_predictions = []
    individual_predictions = {'efficientnet': [], 'xception': [], 'mesonet': []}
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
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
                        real_prob = probs[0][0].item()
                        
                        # DEBUG: Print first few predictions
                        if len(frame_predictions) < 3:
                            print(f"\n🔍 Frame {frame_idx} - Output: {output[0].cpu().numpy()}")
                            print(f"🔍 Probs [Real, Fake]: [{real_prob:.4f}, {fake_prob:.4f}]")
                        
                        if show_individual:
                            ind_preds = model.get_individual_predictions(face_tensor)
                            individual_predictions['efficientnet'].append(ind_preds['efficientnet'][0][1].item())
                            individual_predictions['xception'].append(ind_preds['xception'][0][1].item())
                            individual_predictions['mesonet'].append(ind_preds['mesonet'][0][1].item())
                    
                    frame_predictions.append(fake_prob)
                    
                    # Annotate frame
                    color = (0, 0, 255) if fake_prob > CONFIDENCE_THRESHOLD else (0, 255, 0)
                    label = f"{'FAKE' if fake_prob > CONFIDENCE_THRESHOLD else 'REAL'}: {fake_prob*100:.1f}%"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if writer:
            writer.write(frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if writer:
        writer.release()
    
    # Aggregate predictions
    if frame_predictions:
        frame_predictions = np.array(frame_predictions)
        avg_fake_prob = np.mean(frame_predictions)
        max_fake_prob = np.max(frame_predictions)
        verdict = "FAKE" if avg_fake_prob > CONFIDENCE_THRESHOLD else "REAL"
        
        result = {
            'video': video_path.name,
            'verdict': verdict,
            'avg_fake_probability': float(avg_fake_prob * 100),
            'max_fake_probability': float(max_fake_prob * 100),
            'min_fake_probability': float(np.min(frame_predictions) * 100),
            'std_fake_probability': float(np.std(frame_predictions) * 100),
            'frames_analyzed': len(frame_predictions),
            'confidence': float(max(avg_fake_prob, 1 - avg_fake_prob) * 100)
        }
        
        if show_individual:
            result['individual_model_contributions'] = {
                'efficientnet_avg': float(np.mean(individual_predictions['efficientnet']) * 100),
                'xception_avg': float(np.mean(individual_predictions['xception']) * 100),
                'mesonet_avg': float(np.mean(individual_predictions['mesonet']) * 100)
            }
        
        print(f"\n📊 Results:")
        print(f"Verdict: {verdict}")
        print(f"Avg Fake Probability: {avg_fake_prob*100:.2f}%")
        print(f"Max Fake Probability: {max_fake_prob*100:.2f}%")
        print(f"Frames Analyzed: {len(frame_predictions)}")
        
        if show_individual:
            print(f"\n🔍 Individual Model Contributions:")
            print(f"EfficientNet: {result['individual_model_contributions']['efficientnet_avg']:.2f}%")
            print(f"XceptionNet:  {result['individual_model_contributions']['xception_avg']:.2f}%")
            print(f"MesoNet:      {result['individual_model_contributions']['mesonet_avg']:.2f}%")
        
        return result
    else:
        print("❌ No faces detected in video")
        return None


# ============ BATCH PREDICT IMAGES ============
def predict_folder(model, folder_path, save_results=True, show_individual=False):
    """Predict all images in a folder."""
    folder_path = Path(folder_path)
    image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n📂 Analyzing {len(image_files)} images from {folder_path.name}")
    
    results = []
    real_count = 0
    fake_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        result = predict_image(model, img_path, show_individual=show_individual)
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
        print("Please train the ensemble model first.")
        return
    
    # Load model
    model = load_ensemble_model(MODEL_PATH)
    
    print(f"\n{'='*60}")
    print(f"🔍 ENSEMBLE DEEPFAKE DETECTION - INFERENCE MODE")
    print(f"{'='*60}")
    print(f"\nOptions:")
    print(f"1. Predict single image")
    print(f"2. Predict single image (with individual model details)")
    print(f"3. Predict video")
    print(f"4. Predict video (with individual model details)")
    print(f"5. Predict folder of images")
    print(f"6. Exit")
    
    while True:
        choice = input(f"\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if Path(img_path).exists():
                result = predict_image(model, img_path, show_individual=False)
                if result:
                    print(f"\n📊 Result:")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}%")
                    print(f"Fake Probability: {result['fake_probability']:.2f}%")
            else:
                print("❌ File not found")
        
        elif choice == '2':
            img_path = input("Enter image path: ").strip()
            if Path(img_path).exists():
                result = predict_image(model, img_path, show_individual=True)
                if result:
                    print(f"\n📊 Ensemble Result:")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}%")
                    print(f"Fake Probability: {result['fake_probability']:.2f}%")
                    print(f"\n🔍 Individual Models:")
                    for model_name, preds in result['individual_models'].items():
                        print(f"{model_name.capitalize()}: Fake={preds['fake_prob']:.2f}%, Real={preds['real_prob']:.2f}%")
            else:
                print("❌ File not found")
        
        elif choice == '3':
            vid_path = input("Enter video path: ").strip()
            save_output = input("Save annotated video? (y/n): ").strip().lower() == 'y'
            
            if Path(vid_path).exists():
                output_path = None
                if save_output:
                    output_path = OUTPUT_DIR / f"annotated_{Path(vid_path).name}"
                
                result = predict_video(model, Path(vid_path), frame_step=30, 
                                     output_video=output_path, show_individual=False)
                
                if result and save_output:
                    print(f"💾 Annotated video saved: {output_path}")
            else:
                print("❌ File not found")
        
        elif choice == '4':
            vid_path = input("Enter video path: ").strip()
            save_output = input("Save annotated video? (y/n): ").strip().lower() == 'y'
            
            if Path(vid_path).exists():
                output_path = None
                if save_output:
                    output_path = OUTPUT_DIR / f"annotated_{Path(vid_path).name}"
                
                result = predict_video(model, Path(vid_path), frame_step=30, 
                                     output_video=output_path, show_individual=True)
                
                if result and save_output:
                    print(f"💾 Annotated video saved: {output_path}")
            else:
                print("❌ File not found")
        
        elif choice == '5':
            folder = input("Enter folder path: ").strip()
            show_ind = input("Show individual model details? (y/n): ").strip().lower() == 'y'
            
            if Path(folder).exists():
                predict_folder(model, folder, save_results=True, show_individual=show_ind)
            else:
                print("❌ Folder not found")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
    