"""
EntVit Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class EntVitBackbone(nn.Module):
    """EntVit backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'google/vit-base-patch16-224'):
        super().__init__()
        
        # Load pre-trained ViT
        try:
            from transformers import ViTModel, ViTConfig
            self.config = ViTConfig.from_pretrained(model_name)
            self.backbone = ViTModel.from_pretrained(model_name)
            self.feature_dim = self.config.hidden_size
            print(f"✅ Loaded {model_name} from transformers")
        except Exception as e:
            print(f"⚠️ Could not load {model_name}: {e}")
            # Fallback to torchvision ViT
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=True)
            self.backbone.heads = nn.Identity()
            self.feature_dim = 768
            print("💡 Using fallback ViT model")
        
    def forward(self, x):
        """Forward pass"""
        try:
            # For transformers ViT
            outputs = self.backbone(x)
            return outputs.last_hidden_state[:, 0]  # CLS token
        except:
            # For torchvision ViT
            return self.backbone(x)

class EntVitHead(nn.Module):
    """Classification/Feature head for EntVit"""
    
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

class EntVitModel(nn.Module):
    """Complete EntVit model for image retrieval"""
    
    def __init__(self, 
                 model_name: str = 'google/vit-base-patch16-224',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Initialize backbone
        self.backbone = EntVitBackbone(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Initialize head
        self.head = EntVitHead(
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
