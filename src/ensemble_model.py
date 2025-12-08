"""
Ensemble Model Architectures
Implements EfficientNet, MesoNet4, and XceptionNet for deepfake detection
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


# ============ EFFICIENTNET (Already in your code, included for completeness) ============
class EfficientNetDetector(nn.Module):
    """EfficientNet-B0 for deepfake detection."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, global_pool='')
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
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
    
    def get_features(self, x):
        """Extract features for ensemble."""
        features = self.backbone(x)
        pooled = self.pool(features)
        return pooled.flatten(1)


# ============ MESONET4 ============
class MesoNet4(nn.Module):
    """
    MesoNet4: Lightweight CNN designed specifically for deepfake detection.
    Focuses on mesoscopic properties of images.
    
    Reference: "MesoNet: a Compact Facial Video Forgery Detection Network" (2018)
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Classifier
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 14 * 14, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
    
    def forward(self, x):
        # Input: 224x224
        x = self.conv1(x)      # 224x224x8
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 112x112x8
        
        x = self.conv2(x)      # 112x112x8
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 56x56x8
        
        x = self.conv3(x)      # 56x56x16
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 28x28x16
        
        x = self.conv4(x)      # 28x28x16
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 14x14x16
        
        x = self.flatten(x)    # 16*14*14
        x = self.dropout1(x)
        x = self.fc1(x)        # 16
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)        # num_classes
        
        return x
    
    def get_features(self, x):
        """Extract features before final classification."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        
        return x


# ============ XCEPTIONNET ============
class XceptionNetDetector(nn.Module):
    """
    Modified Xception architecture for deepfake detection.
    Uses depthwise separable convolutions to capture manipulation artifacts.
    
    Based on: "Xception: Deep Learning with Depthwise Separable Convolutions" (2017)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained Xception backbone
        if pretrained:
            # Use timm's xception model
            self.backbone = timm.create_model('xception', pretrained=True, num_classes=0)
        else:
            self.backbone = timm.create_model('xception', pretrained=False, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        """Extract features for ensemble."""
        return self.backbone(x)


# ============ WEIGHTED ENSEMBLE ============
class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble of EfficientNet, MesoNet, and XceptionNet.
    
    Weights are learned during training or can be set manually based on
    individual model performance.
    """
    
    def __init__(self, efficientnet_weight=0.4, xception_weight=0.35, mesonet_weight=0.25, 
                 num_classes=2, pretrained=True, learnable_weights=False):
        super().__init__()
        
        # Initialize models
        self.efficientnet = EfficientNetDetector(num_classes=num_classes, pretrained=pretrained)
        self.xception = XceptionNetDetector(num_classes=num_classes, pretrained=pretrained)
        self.mesonet = MesoNet4(num_classes=num_classes)
        
        # Ensemble weights
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable weights (will be optimized during training)
            self.weights = nn.Parameter(torch.tensor([efficientnet_weight, xception_weight, mesonet_weight]))
        else:
            # Fixed weights
            self.register_buffer('weights', torch.tensor([efficientnet_weight, xception_weight, mesonet_weight]))
    
    def forward(self, x):
        # Get predictions from each model
        eff_out = self.efficientnet(x)
        xce_out = self.xception(x)
        mes_out = self.mesonet(x)
        
        # Normalize weights
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # Weighted average of logits
        ensemble_logits = (
            normalized_weights[0] * eff_out +
            normalized_weights[1] * xce_out +
            normalized_weights[2] * mes_out
        )
        
        return ensemble_logits
    
    def get_individual_predictions(self, x):
        """Get predictions from each model separately (useful for analysis)."""
        with torch.no_grad():
            eff_out = self.efficientnet(x)
            xce_out = self.xception(x)
            mes_out = self.mesonet(x)
            
            eff_prob = torch.softmax(eff_out, dim=1)
            xce_prob = torch.softmax(xce_out, dim=1)
            mes_prob = torch.softmax(mes_out, dim=1)
        
        return {
            'efficientnet': eff_prob,
            'xception': xce_prob,
            'mesonet': mes_prob
        }


# ============ HELPER FUNCTION ============
def create_ensemble(model_type='weighted', weights=(0.4, 0.35, 0.25), num_classes=2, pretrained=True):
    """
    Factory function to create ensemble model.
    
    Args:
        model_type: 'weighted' (weighted average) or 'learnable' (learnable weights)
        weights: tuple of (efficientnet_weight, xception_weight, mesonet_weight)
        num_classes: number of output classes (default: 2 for binary classification)
        pretrained: whether to use pretrained backbones
    
    Returns:
        Ensemble model
    """
    learnable = (model_type == 'learnable')
    
    model = WeightedEnsemble(
        efficientnet_weight=weights[0],
        xception_weight=weights[1],
        mesonet_weight=weights[2],
        num_classes=num_classes,
        pretrained=pretrained,
        learnable_weights=learnable
    )
    
    return model


# ============ TEST CODE ============
if __name__ == "__main__":
    print("Testing ensemble models...")
    
    # Test input
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test EfficientNet
    print("\n1. Testing EfficientNet...")
    eff_model = EfficientNetDetector(num_classes=2, pretrained=False)
    eff_out = eff_model(dummy_input)
    print(f"   Output shape: {eff_out.shape}")
    
    # Test MesoNet
    print("\n2. Testing MesoNet4...")
    meso_model = MesoNet4(num_classes=2)
    meso_out = meso_model(dummy_input)
    print(f"   Output shape: {meso_out.shape}")
    
    # Test XceptionNet
    print("\n3. Testing XceptionNet...")
    xce_model = XceptionNetDetector(num_classes=2, pretrained=False)
    xce_out = xce_model(dummy_input)
    print(f"   Output shape: {xce_out.shape}")
    
    # Test Ensemble
    print("\n4. Testing Weighted Ensemble...")
    ensemble = create_ensemble(model_type='weighted', weights=(0.4, 0.35, 0.25), pretrained=False)
    ensemble_out = ensemble(dummy_input)
    print(f"   Output shape: {ensemble_out.shape}")
    
    # Test individual predictions
    print("\n5. Testing individual predictions...")
    individual = ensemble.get_individual_predictions(dummy_input)
    for name, pred in individual.items():
        print(f"   {name}: {pred.shape}")
    
    print("\n✅ All models working correctly!")