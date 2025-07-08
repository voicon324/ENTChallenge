#!/usr/bin/env python3
"""
Script để test NTXentPairDataset
"""

import yaml
import torch
from src.data_loader import create_ntxent_dataloaders, ImageRetrievalDataset, NTXentPairDataset, get_transforms
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def test_ntxent_dataset():
    """Test NTXentPairDataset functionality"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 Testing NTXentPairDataset...")
    
    # Create dataloaders
    train_loader, val_loader = create_ntxent_dataloaders(config['data'], config['model']['backbone'])
    
    print(f"📊 Train batches: {len(train_loader)}")
    print(f"📊 Val batches: {len(val_loader)}")
    
    # Test one batch
    for batch_idx, (view1, view2, labels) in enumerate(train_loader):
        print(f"✅ Batch {batch_idx + 1}:")
        print(f"   View1 shape: {view1.shape}")
        print(f"   View2 shape: {view2.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels: {labels[:5].tolist()}")
        
        # Verify that view1 and view2 are different (due to different augmentations)
        diff = torch.abs(view1 - view2).mean()
        print(f"   Mean difference between views: {diff:.4f}")
        
        if batch_idx >= 2:  # Test only first 3 batches
            break
    
    print("✅ NTXentPairDataset test completed successfully!")

def visualize_ntxent_pairs():
    """Visualize NT-Xent pairs"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create transform
    transform = get_transforms(
        config['data']['image_size'], 
        'train', 
        False,  # No normalization for visualization
        config['model']['backbone']
    )
    
    # Create base dataset
    base_dataset = ImageRetrievalDataset(
        config['data']['path'], 
        split='train',
        transform=None,
        image_size=config['data']['image_size']
    )
    
    # Create NT-Xent dataset
    ntxent_dataset = NTXentPairDataset(base_dataset, transform=transform)
    
    # Get a few samples
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle('NT-Xent Augmented Pairs (Top: View1, Bottom: View2)')
    
    to_pil = ToPILImage()
    
    for i in range(6):
        view1, view2, label = ntxent_dataset[i]
        
        # Convert to PIL for visualization
        img1 = to_pil(view1)
        img2 = to_pil(view2)
        
        # Plot
        axes[0, i].imshow(img1)
        axes[0, i].set_title(f'View1 (Label: {label})')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(img2)
        axes[1, i].set_title(f'View2 (Label: {label})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ntxent_pairs_visualization.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'ntxent_pairs_visualization.png'")

if __name__ == '__main__':
    test_ntxent_dataset()
    try:
        visualize_ntxent_pairs()
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
