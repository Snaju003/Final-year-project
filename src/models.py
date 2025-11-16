# src/models.py
"""
Contains:
- Pretrained Xception
- Pretrained EfficientNet-B0
- MesoNet (lightweight)
- EnsembleFusion model
- TemporalConsistencyScorer (your novelty)

All models return: real probability, fake probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================
# 1) MESONET (lightweight CNN)
# =============================
class Meso4(nn.Module):
    """
    Simple lightweight MesoNet — good for mobile & fast inference.
    """
    def __init__(self):
        super(Meso4, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ====================================
# 2) XCEPTION Model (pretrained)
# ====================================
def load_xception():
    """
    Loads Xception via torchvision. We replace the final FC layer for 2 classes.
    """

    model = models.xception(weights="Xception_Weights.IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


# ====================================
# 3) EfficientNet-B0 Model (pretrained)
# ====================================
def load_efficientnet():
    model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(1280, 2)
    return model


# ====================================
# 4) Ensemble Fusion Layer (Novel part)
# ====================================
class EnsembleFusion(nn.Module):
    """
    Takes outputs from:
    - Xception (2)
    - EfficientNet-B0 (2)
    - MesoNet (2)
    Concatenates → Linear Fusion → Output (2)
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 2)   # 3 models × 2 output = 6

    def forward(self, out_xcep, out_eff, out_meso):
        x = torch.cat([out_xcep, out_eff, out_meso], dim=1)
        return self.fc(x)


# ======================================================
# 5) TEMPORAL CONSISTENCY MODULE (Your Novel Contribution)
# ======================================================
class TemporalConsistencyScorer:
    """
    Lightweight temporal score.
    Measures frame-to-frame probability volatility:
        high jumps → suspicious → likely deepfake
    """

    def __init__(self):
        self.prev_score = None

    def update(self, fake_prob):
        """
        fake_prob: float (0–1)
        returns: consistency anomaly score
        """

        if self.prev_score is None:
            self.prev_score = fake_prob
            return 0.0

        diff = abs(fake_prob - self.prev_score)
        self.prev_score = fake_prob

        # Simple normalization:
        return float(diff)


# ======================================================
# 6) Helper: build full model set
# ======================================================
def build_all_models(device="cpu"):
    xcep = load_xception().to(device)
    eff  = load_efficientnet().to(device)
    meso = Meso4().to(device)
    fusion = EnsembleFusion().to(device)

    return {
        "xception": xcep,
        "efficientnet": eff,
        "mesonet": meso,
        "fusion": fusion,
        "temporal": TemporalConsistencyScorer()
    }
