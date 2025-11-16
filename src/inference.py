import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path

# ===========================
# 1. Define Ensemble Model
# ===========================

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights


class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet-B0 (pretend DFDC pretrain)
        self.eff = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.eff.classifier[1] = nn.Linear(self.eff.classifier[1].in_features, 1)

        # ResNet50 used as Xception substitute
        self.xcp = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.xcp.fc = nn.Linear(self.xcp.fc.in_features, 1)

        # ------- FIXED MESONET BLOCK --------
        # For input 128x128 → 64x64 → 32x32
        self.meso = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 128 -> 64
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 64 -> 32
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 1)       # 16384
        )
        # ------------------------------------

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o1 = self.sigmoid(self.eff(x))
        o2 = self.sigmoid(self.xcp(x))
        o3 = self.sigmoid(self.meso(x))

        # average the predictions
        return (o1 + o2 + o3) / 3


# ===========================
# 2. Load model weights
# ===========================

def load_model():
    model = EnsembleModel()
    WEIGHTS = "weights/ensemble.pth"

    if not os.path.exists(WEIGHTS):
        print("\n❌ No weights found. Using RANDOM untrained model.\n")
        return model.eval()

    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt)
    print("\n✅ Loaded ensemble weights.\n")
    return model.eval()


# ===========================
# 3. Video face prediction
# ===========================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def predict_video(video_faces_folder):
    model = load_model()
    video_faces_folder = Path(video_faces_folder)

    # List all extracted JPG face images
    face_files = sorted([f for f in video_faces_folder.glob("*.jpg")])
    if len(face_files) == 0:
        print("❌ No faces found in", video_faces_folder)
        return

    preds = []
    for face_path in face_files:
        img = Image.open(face_path).convert("RGB")
        t = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(t)
            preds.append(out.item())

    score = float(np.mean(preds))

    print("\n=======================")
    print(f"Fake Probability: {score:.4f}")
    print("=======================\n")

    if score > 0.5:
        print("FINAL RESULT: **FAKE VIDEO**")
    else:
        print("FINAL RESULT: **REAL VIDEO**")


# ===========================
# 4. Command-line interface
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", required=True, help="Folder containing extracted faces")
    args = parser.parse_args()

    predict_video(args.faces)
