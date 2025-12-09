"""
Colab-friendly inference wrapper for the deepfake detection ensemble model.
This module handles both local and Colab environments seamlessly.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, List, Tuple
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Handle Colab environment
IN_COLAB = 'google.colab' in sys.modules
PROJECT_ROOT = '/content/Final-year-project' if IN_COLAB else os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class ColabEnsembleInference:
    """Ensemble model inference wrapper optimized for Google Colab"""
    
    def __init__(self, model_path: str = 'models/ensemble/ensemble_final.pth', 
                 device: str = None):
        """
        Initialize the inference wrapper
        
        Args:
            model_path: Path to the ensemble model weights
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✅ Inference ready on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load the ensemble model"""
        from src.inference import EnsembleModel
        
        model = EnsembleModel().to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model weights not found at {model_path}. Using random init.")
        
        model.eval()
        return model
    
    def predict_single(self, image_path: Union[str, Path]) -> Dict[str, float]:
        """
        Predict on a single image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary with deepfake probability
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
            
            prob = output.item()
            return {
                'image': str(image_path),
                'deepfake_probability': prob,
                'prediction': 'DEEPFAKE' if prob > 0.5 else 'REAL',
                'confidence': max(prob, 1 - prob)
            }
        except Exception as e:
            return {
                'image': str(image_path),
                'error': str(e)
            }
    
    def predict_batch(self, image_dir: Union[str, Path], 
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict on multiple images in a directory
        
        Args:
            image_dir: Directory containing images
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        image_dir = Path(image_dir)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(ext))
            image_paths.extend(image_dir.glob(ext.upper()))
        
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), 
                     desc="Processing images"):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                images = []
                valid_paths = []
                
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(self.transform(img))
                        valid_paths.append(path)
                    except:
                        results.append({
                            'image': str(path),
                            'error': 'Failed to load image'
                        })
                
                if images:
                    batch_tensor = torch.stack(images).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(batch_tensor)
                    
                    for path, prob in zip(valid_paths, outputs):
                        prob_val = prob.item()
                        results.append({
                            'image': str(path.name),
                            'deepfake_probability': prob_val,
                            'prediction': 'DEEPFAKE' if prob_val > 0.5 else 'REAL',
                            'confidence': max(prob_val, 1 - prob_val)
                        })
            
            except Exception as e:
                for path in batch_paths:
                    results.append({
                        'image': str(path),
                        'error': str(e)
                    })
        
        return results
    
    def predict_video_frames(self, video_path: str, 
                            sample_rate: int = 5) -> Dict[str, list]:
        """
        Extract frames from video and predict
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every nth frame
        
        Returns:
            Dictionary with frame predictions
        """
        import cv2
        
        results = {
            'video': video_path,
            'frames': [],
            'summary': {}
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            predictions = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(image_tensor)
                    
                    prob = output.item()
                    predictions.append(prob)
                    results['frames'].append({
                        'frame': frame_count,
                        'deepfake_probability': prob,
                        'prediction': 'DEEPFAKE' if prob > 0.5 else 'REAL'
                    })
                
                frame_count += 1
            
            cap.release()
            
            # Summary statistics
            if predictions:
                results['summary'] = {
                    'total_frames': frame_count,
                    'sampled_frames': len(predictions),
                    'avg_deepfake_prob': float(np.mean(predictions)),
                    'max_deepfake_prob': float(np.max(predictions)),
                    'min_deepfake_prob': float(np.min(predictions)),
                    'deepfake_frames': sum(1 for p in predictions if p > 0.5),
                    'real_frames': sum(1 for p in predictions if p <= 0.5)
                }
        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("✅ GPU cache cleared")


# Convenience functions for Colab notebooks
def setup_inference():
    """Quick setup for Colab notebooks"""
    return ColabEnsembleInference()


def predict_image(image_path: str) -> Dict:
    """Quick single image prediction"""
    inference = ColabEnsembleInference()
    return inference.predict_single(image_path)


def predict_images_batch(image_dir: str, batch_size: int = 32) -> List[Dict]:
    """Quick batch prediction"""
    inference = ColabEnsembleInference()
    return inference.predict_batch(image_dir, batch_size)


if __name__ == "__main__":
    # Example usage
    inference = ColabEnsembleInference()
    print("Inference module ready!")
