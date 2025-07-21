import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from PIL import Image
from torchvision import transforms
from functools import partial

# ===================================================================
# MODEL ARCHITECTURES - SELF-CONTAINED IMPLEMENTATIONS
# ===================================================================

class DinoV2Backbone(nn.Module):
    """DinoV2 backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'dinov2_vitb14'):
        super().__init__()
        
        # Load pre-trained DinoV2 model
        try:
            print(f"📥 Loading pretrained {model_name} from torch.hub...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            print(f"✅ Successfully loaded pretrained {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load pretrained {model_name}: {e}")
        
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
                 model_name: str = 'dinov2_vitb14', #dinoev2_vitb14, dinov2_vits14, dinov2_vitl14
                 feature_dim: int = 768,
                 num_classes: int = 7,
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

class EntVitBackbone(nn.Module):
    """EntVit backbone for feature extraction"""
    
    def __init__(self, model_name: str = 'egeozsoy/EndoViT'):
        super().__init__()
        
        # First try to load EndoViT from HuggingFace
        try:
            print("📥 Loading pretrained EndoViT from HuggingFace...")
            from huggingface_hub import snapshot_download
            from timm.models.vision_transformer import VisionTransformer
            
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
                 num_classes: int = 7,
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

# ===================================================================
# DATA TRANSFORMS - SELF-CONTAINED IMPLEMENTATION
# ===================================================================

def get_transforms(image_size: int = 224, 
                  split: str = 'train', 
                  normalize: bool = True,
                  backbone: str = 'dino_v2') -> transforms.Compose:
    """Get data transforms for different splits"""
    
    if split == 'train':
        transform_list = [
            # Preprocessing - Focus on important regions (endoscopic circle)
            transforms.CenterCrop(size=(450, 450)),
            transforms.Resize((image_size, image_size)),

            # Geometric augmentation (simulate endoscope movement)
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=20,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                )
            ], p=0.7),

            # Color augmentation (simulate lighting and camera variations)
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            
            transforms.RandomAutocontrast(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),

            # Random erasing to simulate occlusion
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    
    if normalize:
        # Use EndoViT-specific normalization if backbone is ent_vit
        if backbone == 'ent_vit':
            # EndoViT-specific normalization parameters
            mean = [0.3464, 0.2280, 0.2228]
            std = [0.2520, 0.2128, 0.2093]
        else:
            # Standard ImageNet normalization for other models
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)

# ===================================================================
# SIMPLE MODEL CONFIGURATION AND FUNCTIONS
# ===================================================================

# Model configurations - PHẢI KHỚP VỚI CONFIG ĐÃ TRAIN
MODEL_CONFIGS = {
    'dinov2_vits14': {
        'backbone': 'dino_v2',
        'model_name': 'dinov2_vits14',
        'feature_dim': 768,  # ✅ Khớp với configs/dinov2_vits14.yaml
        'checkpoint_path': 'pretrained/dinov2_vits14.pth',  # ✅ Đã lưu lại
        'description': 'DinoV2 ViT-S/14'
    },
    'dinov2_vitb14': {
        'backbone': 'dino_v2',
        'model_name': 'dinov2_vitb14',
        'feature_dim': 768,  # ✅ Khớp với configs/dinov2_vitb14.yaml
        'checkpoint_path': 'pretrained/dinov2_vitb14.pth',  # ✅ Đã lưu lại
        'description': 'DinoV2 ViT-B/14'
    },
    'dinov2_vitl14': {
        'backbone': 'dino_v2',
        'model_name': 'dinov2_vitl14',
        'feature_dim': 768,  # ✅ Khớp với configs/dinov2_vitl14.yaml
        'checkpoint_path': 'pretrained/dinov2_vitl14.pth',  # ✅ Đã lưu lại
        'description': 'DinoV2 ViT-L/14'
    },
    'ent_vit': {
        'backbone': 'ent_vit',
        'model_name': 'ent_vit',
        'feature_dim': 768,  # ✅ Khớp với configs/ent-vit.yaml
        'checkpoint_path': 'pretrained/ent_vit.pth',  # ✅ Đã lưu lại
        'description': 'EndoViT'
    }
}

# ENT Classes (7 classes theo checkpoint)
ENT_CLASSES = ['ear', 'nose', 'throat', 'vc', 'class_4', 'class_5', 'class_6']

# Global device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_best_checkpoint(model_name: str, workspace_root: Path = Path(".")) -> Optional[str]:
    """
    Get checkpoint path for a model from config
    
    Args:
        model_name: Model name
        workspace_root: Root workspace directory
        
    Returns:
        Path to checkpoint or None
    """
    if model_name not in MODEL_CONFIGS:
        return None
    
    config = MODEL_CONFIGS[model_name]
    checkpoint_path = workspace_root / config['checkpoint_path']
    
    if checkpoint_path.exists():
        print(f"✅ Found checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return None

def load_model(model_name: str, 
               checkpoint_path: Optional[str] = None,
               num_classes: int = 7,  # ✅ Khớp với config đã train
               dropout: float = 0.1,
               freeze_backbone: bool = False) -> nn.Module:
    """
    Load a model with checkpoint
    
    Args:
        model_name: Model name (e.g., 'dinov2_vitb14')
        checkpoint_path: Full path to checkpoint file (e.g., '/path/to/model.pth')
                        If None, will auto-discover best checkpoint
        num_classes: Number of classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone
        
    Returns:
        Loaded model
    """
    print(f"🔄 Loading model: {model_name}")
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Auto-discover checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint(model_name)
        if checkpoint_path:
            print(f"🔍 Auto-discovered checkpoint: {checkpoint_path}")
        else:
            print(f"🔍 No checkpoint found for {model_name}")
    
    # Validate checkpoint path if provided
    checkpoint_path_str = None
    if checkpoint_path:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            checkpoint_path_str = str(checkpoint_file)
            print(f"✅ Found checkpoint: {checkpoint_path_str}")
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"🔄 Loading pretrained weights only")
    else:
        print(f"🔄 No checkpoint provided, loading pretrained weights only")
    
    # Build model directly with parameters
    print(f"🏗️ Building model with backbone: {config['backbone']}")
    print(f"🔧 Feature dimension: {config['feature_dim']}")
    
    if config['backbone'] == 'dino_v2':
        model = DinoV2Model(
            model_name=config['model_name'],
            feature_dim=config['feature_dim'],
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    elif config['backbone'] == 'ent_vit':
        model = EntVitModel(
            feature_dim=config['feature_dim'],
            num_classes=num_classes,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")
    
    # Load checkpoint if available
    if checkpoint_path_str:
        print(f"📥 Loading checkpoint: {checkpoint_path_str}")
        try:
            checkpoint = torch.load(checkpoint_path_str, map_location='cpu')
            
            # # Handle different checkpoint formats
            # if 'model_state_dict' in checkpoint:
            #     state_dict = checkpoint['model_state_dict']
            # else:
            #     state_dict = checkpoint
            state_dict = checkpoint
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("🔄 Continuing with pretrained weights only")
    
    model = model.to(DEVICE)
    model.eval()

    import os
    # save model_state_dict only
    model_state_dict = model.state_dict()
    model_save_path = os.path.join("pretrained", f"{model_name}.pth")
    torch.save(model_state_dict, model_save_path)
    
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Feature dimension: {config['feature_dim']}")
    print(f"   Number of classes: {num_classes}")
    print(f"✅ Model loaded successfully")
    
    return model



# ===================================================================
# DEMO AND MAIN FUNCTION
# ===================================================================

def main():
    """Demo usage with simple loop - Load models only"""
    
    print(f"🔧 ENT Model Loader Demo")
    print(f"📁 Workspace root: .")
    print(f"🖥️  Device: {DEVICE}")
    
    # List available models
    print("\n🏗️  Available Models:")
    print("=" * 50)
    
    workspace_root = Path(".")
    for model_name, config in MODEL_CONFIGS.items():
        checkpoint_path = workspace_root / config['checkpoint_path']
        
        if checkpoint_path.exists():
            status_icon = "✅"
            print(f"{status_icon} {model_name:15} - {config['description']}")
            print(f"   💾 Checkpoint: {checkpoint_path.name}")
        else:
            status_icon = "❌"
            print(f"{status_icon} {model_name:15} - {config['description']}")
            print(f"   💾 Checkpoint: Not found")
        print()
    
    # Test loading all 4 models
    print("\n" + "="*60)
    print("🎯 DEMO: Loading All 4 Models")
    print("="*60)
    
    models = {}
    model_names = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'ent_vit']
    
    for model_name in model_names:
        try:
            print(f"\n🔄 Loading {model_name}...")
            # Auto-discover checkpoint path instead of hardcoding
            model = load_model(model_name)
            models[model_name] = model
            print(f"✅ {model_name} loaded successfully!")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
    
    print(f"\n🎉 Demo completed! Loaded {len(models)}/{len(model_names)} models successfully.")
    print(f"💡 Models are ready for inference. Use model(image_tensor) for predictions.")

if __name__ == "__main__":
    main()
           