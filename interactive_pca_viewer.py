#!/usr/bin/env python3
"""
Interactive PCA Viewer with real-time threshold adjustment
"""

import sys
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from pathlib import Path
import argparse

# Add src to path
sys.path.append('src')

from src.visualize import PCAVisualizer
from src.model_factory import DinoV2Model, EntVitModel

def load_model_from_checkpoint(config_path: str, checkpoint_path: str):
    """Load model from checkpoint"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    if model_config['backbone'] == 'dino_v2':
        model = DinoV2Model(
            model_name=model_config['model_name'],
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            freeze_backbone=model_config['freeze_backbone']
        )
    elif model_config['backbone'] == 'ent_vit':
        model = EntVitModel(
            model_name=model_config['model_name'],
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            freeze_backbone=model_config['freeze_backbone']
        )
    else:
        raise ValueError(f"Unsupported backbone: {model_config['backbone']}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def create_interactive_pca_viewer(image_path: str, models_to_load: list = None):
    """Create interactive PCA viewer with threshold slider"""
    
    print("🎨 Creating Interactive PCA Viewer")
    print("=" * 40)
    
    # Load image
    image_pil = Image.open(image_path).convert('RGB')
    print(f"📸 Loaded image: {os.path.basename(image_path)}")
    
    # Discover and load models
    models = {}
    visualizers = {}
    
    model_dir = Path("outputs")
    config_dir = Path("configs")
    
    for model_path in model_dir.iterdir():
        if not model_path.is_dir():
            continue
            
        model_name = model_path.name
        if models_to_load and model_name not in models_to_load:
            continue
        
        best_model_path = model_path / 'best_model.pth'
        if not best_model_path.exists():
            continue
        
        # Find corresponding config
        config_path = None
        for config_file in config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                if config.get('wandb', {}).get('run_name') == model_name:
                    config_path = config_file
                    break
            except:
                continue
        
        if not config_path:
            continue
        
        try:
            print(f"🔧 Loading model: {model_name}")
            model, config = load_model_from_checkpoint(str(config_path), str(best_model_path))
            models[model_name] = model
            visualizers[model_name] = PCAVisualizer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {e}")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    # Generate PCA data
    print("📊 Generating PCA data...")
    pca_data = {}
    
    for model_name, visualizer in visualizers.items():
        print(f"🔍 Processing model: {model_name}")
        features_list = visualizer.extract_patch_features([image_pil], patch_size=14)
        pca_images, pca = visualizer.compute_pca_visualization(features_list, n_components=3)
        
        pca_data[model_name] = {
            'pca_features': pca_images[0],
            'visualizer': visualizer
        }
    
    # Create interactive plot
    num_models = len(pca_data)
    fig, axes = plt.subplots(2, num_models + 1, figsize=(4 * (num_models + 1), 8))
    
    if num_models == 1:
        axes = axes.reshape(2, 2)
    
    # Original image
    axes[0, 0].imshow(image_pil)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Model names
    for i, model_name in enumerate(pca_data.keys()):
        col_idx = i + 1
        axes[0, col_idx].text(0.5, 0.5, model_name, 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[0, col_idx].transAxes,
                             fontsize=12, fontweight='bold')
        axes[0, col_idx].axis('off')
    
    # PCA axes storage
    pca_axes = {}
    for i, model_name in enumerate(pca_data.keys()):
        pca_axes[model_name] = axes[1, i + 1]
    
    # Slider
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Threshold %', 0, 100, valinit=25, valfmt='%.1f%%')
    
    # Reset button
    ax_reset = plt.axes([0.75, 0.05, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    # Update function
    def update_pca(threshold_val):
        for model_name, data in pca_data.items():
            ax = pca_axes[model_name]
            ax.clear()
            
            pca_features = data['pca_features']
            visualizer = data['visualizer']
            
            # Reshape to spatial dimensions
            pca_spatial = visualizer.reshape_to_spatial(pca_features, patch_size=14)
            
            # Apply threshold
            first_component = pca_spatial[:, :, 0]
            threshold = np.percentile(first_component, threshold_val)
            mask = first_component > threshold
            
            pca_masked = pca_spatial.copy()
            pca_masked[~mask] = 0
            
            # Display
            ax.imshow(pca_masked, cmap='viridis')
            ax.set_title(f'PCA - {model_name}\\nThreshold: {threshold_val:.1f}%')
            ax.axis('off')
        
        fig.canvas.draw_idle()
    
    def reset_slider(event):
        slider.reset()
    
    # Connect events
    slider.on_changed(update_pca)
    button_reset.on_clicked(reset_slider)
    
    # Initial update
    update_pca(25)
    
    # Set title
    fig.suptitle(f'Interactive PCA Viewer - {os.path.basename(image_path)}', 
                fontsize=14, fontweight='bold')
    
    print("🎨 Interactive display created!")
    print("📝 Use the slider to adjust threshold")
    print("🔄 Click Reset to return to default")
    print("❌ Close the window to exit")
    
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive PCA Viewer')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--models', nargs='*', help='Specific models to load')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print("🎯 Starting Interactive PCA Viewer...")
    print("🖥️  Backend: matplotlib")
    print(f"📸 Image: {args.image}")
    if args.models:
        print(f"🎯 Models: {', '.join(args.models)}")
    
    create_interactive_pca_viewer(args.image, args.models)

if __name__ == '__main__':
    main()
