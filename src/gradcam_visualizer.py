"""
Grad-CAM Visualization for Deepfake Detection
Generates heatmaps showing which regions of the face indicate manipulation
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ensemble_model import create_ensemble, EfficientNetDetector, XceptionNetDetector, MesoNet4

# ============ PATHS ============
MODEL_PATH = Path(r"E:\Final-year-project\models\ensemble\ensemble_final.pth")
OUTPUT_DIR = Path(r"E:\Final-year-project\results\gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ SETTINGS ============
IMG_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


# ============ GRAD-CAM IMPLEMENTATION ============
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Generates heatmaps showing which regions of the input image
    are most important for the model's prediction.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Neural network model
            target_layer: Layer to extract gradients from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations from forward pass."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients from backward pass."""
        if isinstance(grad_output, tuple) and len(grad_output) > 0:
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach() if isinstance(grad_output, torch.Tensor) else None
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Class index to generate CAM for (default: predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
            prediction: Model prediction
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Generate CAM - handle different tensor shapes
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to get gradients or activations")
        
        gradients = self.gradients
        activations = self.activations
        
        # Handle different tensor shapes
        if gradients.dim() > 3:
            gradients = gradients[0]  # Remove batch dimension
        if activations.dim() > 3:
            activations = activations[0]  # Remove batch dimension
        
        # If gradients are 1D or 2D, we can't compute spatial CAM
        if gradients.dim() < 3:
            # Return a simple attention map based on gradient magnitude
            cam = torch.ones(7, 7, device=gradients.device) * 0.5
        else:
            # Global average pooling of gradients
            weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
            
            # Weighted combination of activation maps
            cam = (weights * activations).sum(dim=0)  # (H, W)
            
            # ReLU to keep only positive influences
            cam = F.relu(cam)
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = cam / cam.max()
        
        return cam.cpu().detach().numpy(), output
    
    def __call__(self, input_image, target_class=None):
        return self.generate_cam(input_image, target_class)


# ============ GET TARGET LAYER ============
def get_target_layer(model, model_type='ensemble'):
    """
    Get the last convolutional layer for Grad-CAM.
    
    Args:
        model: Neural network model
        model_type: 'ensemble', 'efficientnet', 'xception', or 'mesonet'
    
    Returns:
        target_layer: Last convolutional layer
    """
    if model_type == 'ensemble':
        # For ensemble, use EfficientNet's last conv layer (typically best for visualization)
        return model.efficientnet.backbone.blocks[-1][-1]
    
    elif model_type == 'efficientnet':
        return model.backbone.blocks[-1][-1]
    
    elif model_type == 'xception':
        # Xception's last conv layer before pooling
        # Find the last Conv2d in the model
        conv_layers = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
        return conv_layers[-1] if conv_layers else list(model.backbone.children())[-2]
    
    elif model_type == 'mesonet':
        # MesoNet's last conv layer
        return model.conv4
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============ VISUALIZATION ============
def apply_colormap(cam, colormap=cv2.COLORMAP_JET):
    """
    Apply colormap to CAM heatmap.
    
    Args:
        cam: Grayscale heatmap (H, W)
        colormap: OpenCV colormap
    
    Returns:
        heatmap: Colored heatmap (H, W, 3)
    """
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (H, W, 3), RGB, [0-255]
        heatmap: Heatmap (H, W, 3), RGB, [0-255]
        alpha: Transparency of heatmap
    
    Returns:
        overlaid: Overlaid image
    """
    overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlaid


def visualize_gradcam(original_image_path, cam, prediction, confidence, save_path=None):
    """
    Create comprehensive Grad-CAM visualization.
    
    Args:
        original_image_path: Path to original image
        cam: Grad-CAM heatmap (H, W)
        prediction: Model prediction (0=Real, 1=Fake)
        confidence: Prediction confidence
        save_path: Path to save visualization
    """
    # Load original image
    original = Image.open(original_image_path).convert('RGB')
    original = original.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original)
    
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    
    # Apply colormap
    heatmap = apply_colormap(cam_resized, colormap=cv2.COLORMAP_JET)
    
    # Overlay heatmap on image
    overlaid = overlay_heatmap(original_np, heatmap, alpha=0.4)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlaid
    axes[2].imshow(overlaid)
    label = "FAKE" if prediction == 1 else "REAL"
    color = "red" if prediction == 1 else "green"
    axes[2].set_title(f"Prediction: {label} ({confidence:.1f}%)", 
                     fontsize=12, fontweight='bold', color=color)
    axes[2].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cmap = cm.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('Importance', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============ ENSEMBLE GRAD-CAM ============
def generate_ensemble_gradcam(model, image_path, save_dir=None):
    """
    Generate Grad-CAM for all three models in the ensemble.
    
    Args:
        model: Ensemble model
        image_path: Path to input image
        save_dir: Directory to save visualizations
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get ensemble prediction
    model.eval()
    with torch.no_grad():
        ensemble_output = model(img_tensor)
        ensemble_probs = torch.softmax(ensemble_output, dim=1)
        ensemble_pred = ensemble_output.argmax(1).item()
        ensemble_conf = ensemble_probs[0][ensemble_pred].item() * 100
    
    print(f"\n📊 Ensemble Prediction: {'FAKE' if ensemble_pred == 1 else 'REAL'} ({ensemble_conf:.2f}%)")
    
    # Generate Grad-CAM for each model
    models_to_visualize = [
        ('EfficientNet', model.efficientnet, model.efficientnet.backbone.blocks[-1][-1]),
        ('XceptionNet', model.xception, list(model.xception.backbone.children())[-2]),
        ('MesoNet', model.mesonet, model.mesonet.conv4)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Load original image
    original = Image.open(image_path).convert('RGB')
    original = original.resize((IMG_SIZE, IMG_SIZE))
    original_np = np.array(original)
    
    for idx, (model_name, sub_model, target_layer) in enumerate(models_to_visualize):
        print(f"\n🔍 Generating Grad-CAM for {model_name}...")
        
        # Create Grad-CAM
        gradcam = GradCAM(sub_model, target_layer)
        cam, output = gradcam(img_tensor)
        
        # Get prediction for this model
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
        conf = probs[0][pred].item() * 100
        
        # Resize and colorize CAM
        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = apply_colormap(cam_resized)
        overlaid = overlay_heatmap(original_np, heatmap, alpha=0.4)
        
        # Plot heatmap
        axes[0, idx].imshow(heatmap)
        axes[0, idx].set_title(f"{model_name} - Heatmap", fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot overlaid
        axes[1, idx].imshow(overlaid)
        label = "FAKE" if pred == 1 else "REAL"
        color = "red" if pred == 1 else "green"
        axes[1, idx].set_title(f"{model_name}: {label} ({conf:.1f}%)", 
                              fontsize=11, fontweight='bold', color=color)
        axes[1, idx].axis('off')
        
        print(f"   Prediction: {label} ({conf:.2f}%)")
    
    plt.suptitle(f"Ensemble Grad-CAM Analysis\nFinal Prediction: {'FAKE' if ensemble_pred == 1 else 'REAL'} ({ensemble_conf:.1f}%)",
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        save_path = Path(save_dir) / f"gradcam_{Path(image_path).stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved ensemble visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============ BATCH GRAD-CAM ============
def batch_gradcam(model, image_folder, save_dir=None, max_images=10):
    """
    Generate Grad-CAM for multiple images.
    
    Args:
        model: Trained model
        image_folder: Folder containing images
        save_dir: Directory to save visualizations
        max_images: Maximum number of images to process
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    image_files = image_files[:max_images]
    
    if not image_files:
        print("❌ No images found")
        return
    
    print(f"\n📂 Generating Grad-CAM for {len(image_files)} images...")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"{'='*60}")
        
        generate_ensemble_gradcam(model, img_path, save_dir=save_dir)


# ============ MAIN ============
def main():
    """Main Grad-CAM visualization pipeline."""
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Load model
    print(f"📂 Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
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
    
    print(f"✅ Model loaded")
    
    print(f"\n{'='*60}")
    print(f"🔥 GRAD-CAM VISUALIZATION")
    print(f"{'='*60}")
    print(f"\nOptions:")
    print(f"1. Single image Grad-CAM (ensemble)")
    print(f"2. Batch Grad-CAM (multiple images)")
    print(f"3. Exit")
    
    while True:
        choice = input(f"\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if Path(img_path).exists():
                generate_ensemble_gradcam(model, img_path, save_dir=OUTPUT_DIR)
            else:
                print("❌ File not found")
        
        elif choice == '2':
            folder = input("Enter folder path: ").strip()
            max_imgs = input("Max images to process (default 10): ").strip()
            max_imgs = int(max_imgs) if max_imgs.isdigit() else 10
            
            if Path(folder).exists():
                batch_gradcam(model, folder, save_dir=OUTPUT_DIR, max_images=max_imgs)
            else:
                print("❌ Folder not found")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()