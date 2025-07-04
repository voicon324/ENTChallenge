"""
"Nhà máy" tạo ra model (DinoV2, EntVit)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path

class DinoV2Model(nn.Module):
    """DinoV2 Model for Image Retrieval"""
    
    def __init__(self, 
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained DinoV2
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            backbone_dim = 768  # DinoV2 ViT-B/14 output dimension
        except Exception as e:
            print(f"⚠️ Could not load DinoV2 from torch.hub: {e}")
            print("💡 Using fallback ViT model")
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=True)
            # Remove classifier head
            self.backbone.heads = nn.Identity()
            backbone_dim = 768
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Classification head (for fine-tuning)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store dimensions
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project features
        projected_features = self.feature_projection(features)
        
        if return_features:
            return projected_features
        
        # Classification
        logits = self.classifier(projected_features)
        
        return logits
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)

class EntVitModel(nn.Module):
    """EndoViT Model for Image Retrieval"""
    
    def __init__(self, 
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Load pre-trained EndoViT from HuggingFace
        try:
            from timm.models.vision_transformer import VisionTransformer
            from functools import partial
            from huggingface_hub import snapshot_download
            
            print("📥 Downloading EndoViT model from HuggingFace...")
            # Download model files
            model_path = snapshot_download(repo_id="egeozsoy/EndoViT", revision="main")
            model_weights_path = Path(model_path) / "pytorch_model.bin"
            
            # Define the EndoViT model architecture
            self.backbone = VisionTransformer(
                patch_size=16, 
                embed_dim=768, 
                depth=12, 
                num_heads=12, 
                mlp_ratio=4, 
                qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ).eval()
            
            # Load the pre-trained weights
            model_weights = torch.load(model_weights_path, map_location='cpu')['model']
            loading_info = self.backbone.load_state_dict(model_weights, strict=False)
            print(f"✅ EndoViT loaded successfully: {loading_info}")
            
            backbone_dim = 768  # EndoViT output dimension
            
        except Exception as e:
            print(f"⚠️ Could not load EndoViT from HuggingFace: {e}")
            print("💡 Using fallback ViT model")
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=True)
            # Remove classifier head
            self.backbone.heads = nn.Identity()
            backbone_dim = 768
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("🔒 Frozen backbone parameters")
        
        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Classification head (for fine-tuning)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store dimensions
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        # Extract features from backbone
        try:
            # For EndoViT - use forward_features method
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                # Take the CLS token (first token) from the sequence
                if len(features.shape) == 3:  # [batch, seq_len, embed_dim]
                    features = features[:, 0]  # CLS token
            else:
                # For fallback ViT
                features = self.backbone(x)
        except Exception as e:
            print(f"⚠️ Error in forward pass: {e}")
            # Fallback to regular forward
            features = self.backbone(x)
        
        # Project features
        projected_features = self.feature_projection(features)
        
        if return_features:
            return projected_features
        
        # Classification
        logits = self.classifier(projected_features)
        
        return logits
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.forward(x, return_features=True)

class ContrastiveModel(nn.Module):
    """Wrapper for contrastive learning"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x1, x2=None):
        """Forward pass for contrastive learning"""
        if x2 is None:
            # Single input - return features
            return self.base_model.get_features(x1)
        else:
            # Pair input - return both features
            f1 = self.base_model.get_features(x1)
            f2 = self.base_model.get_features(x2)
            return f1, f2

def build_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Build model based on configuration
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Built model
    """
    backbone_name = model_config.get('backbone', 'dino_v2')
    feature_dim = model_config.get('feature_dim', 768)
    num_classes = model_config.get('num_classes', 1000)
    dropout = model_config.get('dropout', 0.1)
    freeze_backbone = model_config.get('freeze_backbone', False)
    
    print(f"🏗️ Building model with backbone: {backbone_name}")
    
    # Build base model
    if backbone_name == 'dino_v2':
        model = DinoV2Model(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    elif backbone_name == 'ent_vit':
        model = EntVitModel(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Load checkpoint if provided
    checkpoint_path = model_config.get('pretrained_checkpoint')
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("🔄 Continuing with random initialization")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Number of classes: {num_classes}")
    
    return model

def create_contrastive_model(base_model: nn.Module) -> ContrastiveModel:
    """Create contrastive learning wrapper"""
    return ContrastiveModel(base_model)

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'model_type': type(model).__name__
    }
