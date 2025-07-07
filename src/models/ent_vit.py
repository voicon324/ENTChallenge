"""
EntVit Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

class EntVitBackbone(nn.Module):
    """EntVit backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'egeozsoy/EndoViT'):
        super().__init__()
        
        # First try to load EndoViT from HuggingFace
        try:
            print("📥 Loading pretrained EndoViT from HuggingFace...")
            from huggingface_hub import snapshot_download
            from timm.models.vision_transformer import VisionTransformer
            from functools import partial
            
            # Download model files
            model_path = snapshot_download(repo_id=model_name, revision="main")
            model_weights_path = Path(model_path) / "pytorch_model.bin"
            
            if model_weights_path.exists():
                # Define the EndoViT model architecture (based on ViT-B/16)
                self.backbone = VisionTransformer(
                    patch_size=16, 
                    embed_dim=768, 
                    depth=12, 
                    num_heads=12, 
                    mlp_ratio=4, 
                    qkv_bias=True, 
                    norm_layer=partial(nn.LayerNorm, eps=1e-6)
                ).eval()
                
                # Load the pretrained weights
                model_weights = torch.load(model_weights_path, map_location='cpu', weights_only=False)
                if 'model' in model_weights:
                    model_weights = model_weights['model']
                    
                loading_info = self.backbone.load_state_dict(model_weights, strict=False)
                self.feature_dim = 768
                print(f"✅ Successfully loaded pretrained EndoViT: {loading_info}")
            else:
                raise FileNotFoundError("EndoViT weights not found")
                
        except Exception as e:
            print(f"⚠️ Could not load EndoViT from HuggingFace: {e}")
            print("💡 Falling back to standard ViT with ImageNet pretrained weights")
            # Fallback to standard pretrained ViT
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Identity()
            self.feature_dim = 768
            print("✅ Using fallback ViT model with pretrained weights")
        
    def forward(self, x):
        """Forward pass"""
        try:
            # For EndoViT (timm ViT) - use forward_features method
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                # Take the CLS token (first token) from the sequence
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    return features[:, 0]  # CLS token
                else:
                    return features
            # For torchvision ViT
            else:
                return self.backbone(x)
        except Exception as e:
            print(f"⚠️ Error in EntVit forward pass: {e}")
            # Fallback to regular forward
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
                 model_name: str = 'egeozsoy/EndoViT',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Initialize backbone with EndoViT
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
