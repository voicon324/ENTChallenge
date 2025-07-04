#!/usr/bin/env python3
"""
Test script to verify EndoViT model implementation
"""

import torch
import yaml
from pathlib import Path
from src.model_factory import build_model
from src.utils import process_entvit_image
import glob

def test_entvit_model():
    """Test EndoViT model implementation"""
    
    print("🧪 Testing EndoViT Model Implementation")
    print("=" * 50)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure we're using ent_vit backbone
    config['model']['backbone'] = 'ent_vit'
    
    print(f"📋 Model Config:")
    print(f"   Backbone: {config['model']['backbone']}")
    print(f"   Feature dim: {config['model']['feature_dim']}")
    print(f"   Num classes: {config['model']['num_classes']}")
    
    # Test model creation
    print("\n🏗️ Building model...")
    try:
        model = build_model(config['model'])
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False
    
    # Test model forward pass
    print("\n🔄 Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    try:
        with torch.no_grad():
            # Test feature extraction
            features = model.get_features(dummy_input)
            print(f"✅ Feature extraction successful: {features.shape}")
            
            # Test full forward pass
            logits = model(dummy_input)
            print(f"✅ Full forward pass successful: {logits.shape}")
            
            # Verify output dimensions
            expected_feature_dim = config['model']['feature_dim']
            expected_num_classes = config['model']['num_classes']
            
            assert features.shape == (batch_size, expected_feature_dim), \
                f"Expected features shape {(batch_size, expected_feature_dim)}, got {features.shape}"
            
            assert logits.shape == (batch_size, expected_num_classes), \
                f"Expected logits shape {(batch_size, expected_num_classes)}, got {logits.shape}"
            
            print("✅ All output dimensions are correct")
            
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return False
    
    # Test image processing function
    print("\n🖼️ Testing image processing...")
    image_dir = Path("Dataset/images")
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png"))[:3]  # Test with first 3 images
        
        if image_files:
            try:
                processed_images = []
                for img_path in image_files:
                    processed_img = process_entvit_image(img_path)
                    processed_images.append(processed_img)
                    print(f"✅ Processed {img_path.name}: {processed_img.shape}")
                
                # Test batch processing
                batch_tensor = torch.stack(processed_images).to(device)
                with torch.no_grad():
                    batch_features = model.get_features(batch_tensor)
                    print(f"✅ Batch processing successful: {batch_features.shape}")
                    
            except Exception as e:
                print(f"❌ Error processing images: {e}")
                return False
        else:
            print("⚠️ No images found in Dataset/images directory")
    else:
        print("⚠️ Dataset/images directory not found")
    
    print("\n🎉 All tests passed! EndoViT model is working correctly.")
    return True

if __name__ == "__main__":
    success = test_entvit_model()
    exit(0 if success else 1)
