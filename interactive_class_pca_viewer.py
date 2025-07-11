#!/usr/bin/env python3
"""
Interactive PCA Viewer for Entire Class
View PCA visualizations for all images in a class at once
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import cv2 as cv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

def crop_image(img_path):
    # Read image in color (3 channels)
    img_color = cv.imread(img_path, cv.IMREAD_COLOR)
    # Read image in grayscale for contour detection
    img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    # finding contour and extremepoint using grayscale image
    blurred_img = cv.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv.threshold(blurred_img, 45, 255, cv.THRESH_BINARY)
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)
    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv.contourArea)

    extLeft = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    extRight = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    extTop = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    extBot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

    # crop the color image (3 channels)
    cropped_img = img_color[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return cropped_img
# def load_class_images(data_dir: str, class_name: str, max_images: int = 20, image_size: int = 224) -> List[Image.Image]:
#     """
#     Load the first image from a class and create 4 augmented versions
    
#     Args:
#         data_dir: Directory containing class folders
#         class_name: Name of the class
#         max_images: Not used, kept for compatibility
#         image_size: Size to resize images to (same as training)
        
#     Returns:
#         List of 4 augmented PIL images
#     """
#     class_dir = os.path.join(data_dir, class_name)
#     if not os.path.exists(class_dir):
#         raise ValueError(f"Class directory not found: {class_dir}")
    
#     # Get all image files
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
#     image_files = []
    
#     for file in os.listdir(class_dir):
#         if any(file.lower().endswith(ext) for ext in image_extensions):
#             image_files.append(os.path.join(class_dir, file))
    
#     if not image_files:
#         raise ValueError(f"No image files found in {class_dir}")
    
#     # Sort and get first image
#     image_files.sort()
#     first_image_path = image_files[0]
    
#     # Load and crop the first image
#     try:
#         cropped_img = crop_image(first_image_path)
#         # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
#         img_rgb = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)
#         base_image = Image.fromarray(img_rgb)
#     except Exception as e:
#         raise ValueError(f"Error loading first image {first_image_path}: {e}")
    
#     # Create diverse augmentation transforms
#     crop_percent = 0.9
#     augment_transforms = [
#         # Original (resized only)
#         transforms.Compose([
#             transforms.CenterCrop(int(image_size * crop_percent)),
#             transforms.Resize((image_size, image_size)),
#         ]),
        
#         # Rotation + Color Jitter
#         transforms.Compose([
#             transforms.CenterCrop(int(image_size * crop_percent)),
#             transforms.Resize((image_size, image_size)),
#             transforms.RandomRotation(45),
#             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
#         ]),
        
#         # Horizontal Flip + Gaussian Blur
#         transforms.Compose([
#             transforms.CenterCrop(int(image_size * crop_percent)),
#             transforms.Resize((image_size, image_size)),
#             transforms.RandomHorizontalFlip(p=1.0),
#             transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
#         ]),
        
#         # Perspective + Grayscale
#         transforms.Compose([
#             transforms.CenterCrop(int(image_size * crop_percent)),
#             transforms.Resize((image_size, image_size)),
#             transforms.RandomPerspective(distortion_scale=0.3, p=1.0),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.ColorJitter(brightness=0.3),
#         ]),
#     ]
    
#     # Generate 4 augmented versions
#     images = []
#     for i, transform in enumerate(augment_transforms):
#         try:
#             augmented_img = transform(base_image)
#             images.append(augmented_img)
#         except Exception as e:
#             print(f"⚠️ Error applying augmentation {i+1}: {e}")
#             # Fallback to original image if augmentation fails
#             images.append(transforms.Resize((image_size, image_size))(base_image))
    
#     print(f"📸 Created 4 augmented versions from first image in class '{class_name}' (resized to {image_size}x{image_size})")
#     return images

def load_class_images(data_dir: str, class_name: str, max_images: int = 20, image_size: int = 224) -> List[Image.Image]:
    """
    Load images from a specific class and resize them to training size
    
    Args:
        data_dir: Directory containing class folders
        class_name: Name of the class
        max_images: Maximum number of images to load
        image_size: Size to resize images to (same as training)
        
    Returns:
        List of PIL images resized to training size
    """
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.exists(class_dir):
        raise ValueError(f"Class directory not found: {class_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for file in os.listdir(class_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(class_dir, file))
    
    # Sort and limit
    image_files.sort()
    image_files = image_files[:max_images]
    
    # Create transform to resize images to training size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(int(image_size * 0.8)),  # Crop to 90% of size
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # Load and resize images
    images = []
    for img_path in image_files:
        try:
            # img = Image.open(img_path).convert('RGB')
            # # Resize to training size
            # img_resized = transform(img)
            # images.append(img_resized)
            img = crop_image(img_path)
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = transform(Image.fromarray(img_rgb))
            images.append(img)
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
    
    print(f"📸 Loaded {len(images)} images from class '{class_name}' (resized to {image_size}x{image_size})")
    return images

def get_available_classes(data_dir: str) -> List[str]:
    """Get list of available classes"""
    if not os.path.exists(data_dir):
        return []
    
    classes = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            classes.append(item)
    
    return sorted(classes)

def create_interactive_class_pca_viewer(class_name: str, 
                                       data_dir: str = "data/processed/test",
                                       models_to_load: List[str] = None,
                                       max_images: int = 20):
    """
    Create interactive PCA viewer for all images in a class
    
    Args:
        class_name: Name of the class to visualize
        data_dir: Directory containing class folders
        models_to_load: List of specific models to load
        max_images: Maximum number of images to display
    """
    print(f"🎯 Creating Interactive Class PCA Viewer for: {class_name}")
    print(f"📁 Data directory: {data_dir}")
    
    # Load class images
    try:
        # Get image size from first available model config
        image_size = 224  # Default
        outputs_dir = Path("outputs")
        config_dir = Path("configs")
        
        # Try to get image size from config
        for config_file in config_dir.glob('*.yaml'):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                if 'data' in config and 'image_size' in config['data']:
                    image_size = config['data']['image_size']
                    break
            except:
                continue
        
        print(f"📏 Using image size: {image_size}x{image_size}")
        images = load_class_images(data_dir, class_name, max_images, image_size)
        if not images:
            print(f"❌ No images found in class '{class_name}'")
            return
    except Exception as e:
        print(f"❌ Error loading class images: {e}")
        return
    
    # Get available models
    available_models = []
    outputs_dir = Path("outputs")
    
    for model_dir in outputs_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "best_model.pth").exists():
            available_models.append(model_dir.name)
    
    if not available_models:
        print("❌ No trained models found!")
        return
    
    # Filter models if specified
    if models_to_load:
        available_models = [m for m in available_models if m in models_to_load]
    
    print(f"🤖 Found models: {', '.join(available_models)}")
    
    # Load models and compute PCA
    pca_data = {}
    
    # Get available models from outputs directory
    outputs_dir = Path("outputs")
    config_dir = Path("configs")
    
    for model_name in available_models:
        if model_name == 'old_checkpoints':  # Skip this directory
            continue
            
        print(f"\n🔄 Loading model: {model_name}")
        
        model_path = outputs_dir / model_name
        if not model_path.exists():
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
            print(f"⚠️ Config not found for model: {model_name}")
            continue
        
        try:
            model, config = load_model_from_checkpoint(str(config_path), str(best_model_path))
            
            # Create visualizer
            visualizer = PCAVisualizer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Extract features and compute PCA for all images
            print(f"🔍 Extracting features from {len(images)} images...")
            features_list = visualizer.extract_patch_features(images, patch_size=14)
            
            print("📊 Computing PCA...")
            pca_images, pca = visualizer.compute_pca_visualization(features_list, n_components=3)
            
            # Store PCA data
            pca_data[model_name] = {
                'pca_images': pca_images,
                'pca_object': pca,
                'visualizer': visualizer
            }
            
            print(f"✅ Processed {model_name}")
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")
            continue
    
    if not pca_data:
        print("❌ No models could be loaded!")
        return
    
    # Create interactive visualization
    print("\n🎨 Creating interactive visualization...")
    
    num_models = len(pca_data)
    num_images = len(images)
    
    # Create figure with subplots
    # Layout: [Original Images Row] + [Model Rows]
    fig, axes = plt.subplots(num_models + 1, num_images, 
                            figsize=(num_images * 3, (num_models + 1) * 3))
    
    if num_models == 1:
        axes = axes.reshape(2, -1)
    
    # Display original images in first row
    for i, img in enumerate(images):
        if num_images == 1:
            ax = axes[0] if num_models == 1 else axes[0, 0]
        else:
            ax = axes[0, i]
        ax.imshow(img)
        ax.set_title(f'Original {i+1}')
        ax.axis('off')
    
    # PCA axes storage
    pca_axes = {}
    model_names = list(pca_data.keys())
    
    for model_idx, model_name in enumerate(model_names):
        pca_axes[model_name] = []
        for img_idx in range(num_images):
            if num_images == 1:
                if num_models == 1:
                    ax = axes[1]
                else:
                    ax = axes[model_idx + 1, 0]
            else:
                ax = axes[model_idx + 1, img_idx]
            pca_axes[model_name].append(ax)
    
    # Create slider
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Threshold %', 0, 100, valinit=25, valfmt='%.1f%%')
    
    # Create reset button
    ax_reset = plt.axes([0.75, 0.05, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    def update_pca(threshold_val):
        """Update PCA visualization based on threshold"""
        for model_name, data in pca_data.items():
            pca_images = data['pca_images']
            visualizer = data['visualizer']
            axes_list = pca_axes[model_name]
            
            for img_idx, (pca_features, ax) in enumerate(zip(pca_images, axes_list)):
                # Reshape to spatial grid
                pca_spatial = visualizer.reshape_to_spatial(pca_features, patch_size=14)
                
                # Apply threshold on first component
                first_component = pca_spatial[:, :, 0]
                threshold = np.percentile(first_component, threshold_val)
                mask = first_component > threshold
                
                pca_masked = pca_spatial.copy()
                pca_masked[~mask] = 0
                
                # Clear and display - resize PCA image to 224x224
                ax.clear()
                
                # Convert PCA visualization to PIL Image and resize
                if pca_masked.shape[2] == 3:  # RGB
                    # Normalize to 0-255 range
                    pca_img_array = (pca_masked * 255).astype(np.uint8)
                    pca_pil = Image.fromarray(pca_img_array)
                else:
                    # Handle single channel or other formats
                    pca_img_array = (pca_masked[:, :, 0] * 255).astype(np.uint8)
                    pca_pil = Image.fromarray(pca_img_array, mode='L')
                
                # Resize to training size (224x224) using BICUBIC
                pca_resized = pca_pil.resize((224, 224), resample=Image.BICUBIC)
                
                ax.imshow(pca_resized, cmap='viridis')
                ax.set_title(f'{model_name}\nImg {img_idx+1} - {threshold_val:.1f}%')
                ax.axis('off')
        
        fig.canvas.draw_idle()
    
    def reset_slider(event):
        """Reset slider to default value"""
        slider.reset()
    
    # Connect events
    slider.on_changed(update_pca)
    button_reset.on_clicked(reset_slider)
    
    # Initial update
    update_pca(25)
    
    # Set main title
    fig.suptitle(f'Interactive Class PCA Viewer - {class_name} ({len(images)} images)', 
                fontsize=16, fontweight='bold')
    
    print("🎨 Interactive display created!")
    print("📝 Use the slider to adjust threshold for all images")
    print("🔄 Click Reset to return to default")
    print("❌ Close the window to exit")
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Class PCA Viewer')
    parser.add_argument('--class', dest='class_name', 
                       help='Name of the class to visualize')
    parser.add_argument('--data_dir', default='data/processed/train',
                       help='Directory containing class folders')
    parser.add_argument('--models', nargs='*', 
                       help='Specific models to load (default: all available)')
    parser.add_argument('--max_images', type=int, default=20,
                       help='Maximum number of images to display')
    parser.add_argument('--list_classes', action='store_true',
                       help='List available classes and exit')
    
    args = parser.parse_args()
    
    # List classes if requested
    if args.list_classes:
        print("📋 Available classes:")
        classes = get_available_classes(args.data_dir)
        if classes:
            for i, class_name in enumerate(classes, 1):
                print(f"  {i}. {class_name}")
        else:
            print("  No classes found!")
        return
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        return
    
    # Check if class name is provided (unless listing classes)
    if not args.class_name and not args.list_classes:
        print("❌ Class name is required!")
        print("Use --list_classes to see available classes")
        return
    
    # Check if class exists
    if args.class_name:
        available_classes = get_available_classes(args.data_dir)
        if args.class_name not in available_classes:
            print(f"❌ Class '{args.class_name}' not found!")
            print(f"Available classes: {', '.join(available_classes)}")
            return
    
    print("🎯 Starting Interactive Class PCA Viewer...")
    print("🖥️  Backend: matplotlib")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Class: {args.class_name}")
    print(f"🔢 Max images: {args.max_images}")
    
    if args.models:
        print(f"🎯 Models: {', '.join(args.models)}")
    
    create_interactive_class_pca_viewer(
        class_name=args.class_name,
        data_dir=args.data_dir,
        models_to_load=args.models,
        max_images=args.max_images
    )

if __name__ == '__main__':
    main()
