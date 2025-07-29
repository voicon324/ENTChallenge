"""
"Factory" to create models (DinoV2, EntVit)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path

# Import models from modules
try:
    from .models.dino_v2 import DinoV2Model as DinoV2Core
    from .models.ent_vit import EntVitModel as EntVitCore
except ImportError:
    # For direct execution
    from models.dino_v2 import DinoV2Model as DinoV2Core
    from models.ent_vit import EntVitModel as EntVitCore

class DinoV2Model(nn.Module):
    """DinoV2 Model for Image Retrieval - Wrapper around core DinoV2Model"""
    
    def __init__(self, 
                 model_name: str = 'dinov2_vitb14',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Use the core DinoV2Model from models module
        self.model = DinoV2Core(
            model_name=model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # Store dimensions for compatibility
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.model_name = model_name
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        return self.model.forward(x, return_features=return_features)
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.model.get_features(x)

class EntVitModel(nn.Module):
    """EndoViT Model for Image Retrieval - Wrapper around core EntVitModel"""
    
    def __init__(self, 
                 model_name = 'egeozsoy/EndoViT',
                 feature_dim: int = 768,
                 num_classes: int = 1000,
                 dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # Use the core EntVitModel from models module
        self.model = EntVitCore(
            model_name='egeozsoy/EndoViT',  # Use actual EndoViT from HuggingFace
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # Store dimensions for compatibility
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """Forward pass"""
        return self.model.forward(x, return_features=return_features)
    
    def get_features(self, x):
        """Get feature embeddings"""
        return self.model.get_features(x)

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
    model_name = model_config.get('model_name', 'dinov2_vitb14')  # DinoV2 variant
    feature_dim = model_config.get('feature_dim', 768)
    num_classes = model_config.get('num_classes', 1000)
    dropout = model_config.get('dropout', 0.1)
    freeze_backbone = model_config.get('freeze_backbone', False)
    
    print(f"🏗️ Building model with backbone: {backbone_name}")
    if backbone_name == 'dino_v2':
        print(f"🔧 Using DinoV2 variant: {model_name}")
    
    # Build base model
    if backbone_name == 'dino_v2':
        model = DinoV2Model(
            model_name=model_name,
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
