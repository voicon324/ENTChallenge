"""
DinoV2 Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DinoV2Backbone(nn.Module):
    """DinoV2 backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'dinov2_vitb14'):
        super().__init__()
        
        # Load pre-trained DinoV2
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            print(f"✅ Loaded {model_name} from torch.hub")
        except Exception as e:
            print(f"⚠️ Could not load {model_name}: {e}")
            # Fallback to ViT
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=True)
            self.backbone.heads = nn.Identity()
            print("💡 Using fallback ViT model")
        
        # Get output dimension
        self.feature_dim = self._get_feature_dim()
        
    def _get_feature_dim(self) -> int:
        """Get feature dimension"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        return features.shape[-1]
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)

class DinoV2Head(nn.Module):
    """Classification/Feature head for DinoV2"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        features = self.feature_projection(x)
        
        if return_features:
            return features
        
        logits = self.classifier(features)
        return logits

class DinoV2Model(nn.Module):
    """Complete DinoV2 model for image retrieval"""
    
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Initialize backbone
        self.backbone = DinoV2Backbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize head
        self.head = DinoV2Head(
            input_dim=self.backbone.feature_dim,
            hidden_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Store config
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Pass through head
        output = self.head(backbone_features, return_features=return_features)
        
        return output
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)
    
    def get_backbone_features(self, x):
        """Get raw backbone features"""
        return self.backbone(x)
