#!/usr/bin/env python3
"""
Batch PCA Class Viewer
Generate PCA visualizations for all images in each class in batches of 8 images
Creates 3-column, 4-row figures: Original | Before Train | After Train
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    """Crop image using contour detection"""
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

def load_all_class_images(class_name: str, image_size: int = 224) -> List[Image.Image]:
    """
    Load all images from a class across train/test/val splits
    
    Args:
        class_name: Name of the class
        image_size: Size to resize images to
        
    Returns:
        List of PIL images
    """
    all_images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Check all splits
    data_splits = ['data/processed/train', 'data/processed/test', 'data/processed/val']
    
    for split_dir in data_splits:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Get all image files in this split
        for file in os.listdir(class_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_images.append(os.path.join(class_dir, file))
    
    # Sort all image paths
    all_images.sort()
    
    # Create transform to resize images to training size
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(int(image_size * 0.8)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # Load and process images
    processed_images = []
    for img_path in all_images:
        try:
            cropped_img = crop_image(img_path)
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            img_rgb = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)
            img = transform(Image.fromarray(img_rgb))
            processed_images.append(img)
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
    
    print(f"📸 Loaded {len(processed_images)} images from class '{class_name}' across all splits")
    return processed_images

def get_available_classes() -> List[str]:
    """Get list of available classes from all splits"""
    classes = set()
    
    data_splits = ['data/processed/train', 'data/processed/test', 'data/processed/val']
    
    for split_dir in data_splits:
        if not os.path.exists(split_dir):
            continue
            
        for item in os.listdir(split_dir):
            item_path = os.path.join(split_dir, item)
            if os.path.isdir(item_path):
                classes.add(item)
    
    return sorted(list(classes))

def resize_pca_visualization(pca_array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize PCA visualization to target size while preserving the data structure
    
    Args:
        pca_array: PCA visualization array (H, W, C)
        target_h: Target height
        target_w: Target width
        
    Returns:
        Resized PCA array
    """
    if pca_array.shape[:2] == (target_h, target_w):
        return pca_array
    
    # Handle different channel configurations
    if len(pca_array.shape) == 3:
        if pca_array.shape[2] == 1:
            # Single channel - resize and keep as single channel
            resized = cv.resize(pca_array[:, :, 0], (target_w, target_h))
            return resized.reshape(target_h, target_w, 1)
        elif pca_array.shape[2] == 3:
            # Three channels - resize all channels
            return cv.resize(pca_array, (target_w, target_h))
        else:
            # Other channel counts - resize first channel only
            resized = cv.resize(pca_array[:, :, 0], (target_w, target_h))
            return resized.reshape(target_h, target_w, 1)
    else:
        # 2D array - add channel dimension
        resized = cv.resize(pca_array, (target_w, target_h))
        return resized.reshape(target_h, target_w, 1)

def apply_threshold_to_pca(pca_spatial: np.ndarray, threshold_val: float) -> np.ndarray:
    """
    Apply threshold to PCA visualization
    
    Args:
        pca_spatial: PCA spatial features (H, W, C)
        threshold_val: Threshold percentile (0-100)
        
    Returns:
        Masked PCA visualization
    """
    # Apply threshold on first component
    first_component = pca_spatial[:, :, 0]
    threshold = np.percentile(first_component, threshold_val)
    mask = first_component > threshold
    
    pca_masked = pca_spatial.copy()
    pca_masked[~mask] = 0
    
    return pca_masked

def create_pca_batch_figure(images_batch: List[Image.Image], 
                           pca_before: List[np.ndarray],
                           pca_after: List[np.ndarray],
                           batch_idx: int,
                           class_name: str,
                           model_name: str,
                           threshold_val: float = 25.0) -> plt.Figure:
    """
    Create a 3-column, 4-row figure for a batch of images by concatenating arrays
    
    Args:
        images_batch: List of original images (up to 8)
        pca_before: List of PCA visualizations before training
        pca_after: List of PCA visualizations after training
        batch_idx: Batch index
        class_name: Class name
        model_name: Model name
        threshold_val: Threshold value for PCA masking
        
    Returns:
        matplotlib Figure
    """
    num_images = len(images_batch)
    rows = min(4, num_images)
    
    # Get target size from first image
    target_h, target_w = np.array(images_batch[0]).shape[:2]
    
    # Convert PIL images to numpy arrays and apply threshold
    original_arrays = []
    before_arrays = []
    after_arrays = []
    
    for i in range(rows):
        if i < num_images:
            # Original image - ensure consistent size
            orig_img = np.array(images_batch[i])
            if orig_img.shape[:2] != (target_h, target_w):
                orig_img = cv.resize(orig_img, (target_w, target_h))
            original_arrays.append(orig_img)
            
            # Before training PCA with threshold
            pca_before_masked = apply_threshold_to_pca(pca_before[i], threshold_val)
            pca_before_resized = resize_pca_visualization(pca_before_masked, target_h, target_w)
            
            # Convert to RGB format for concatenation
            if pca_before_resized.shape[2] == 3:
                pca_before_rgb = (pca_before_resized * 255).astype(np.uint8)
            else:
                pca_before_gray = (pca_before_resized[:, :, 0] * 255).astype(np.uint8)
                pca_before_rgb = np.stack([pca_before_gray, pca_before_gray, pca_before_gray], axis=2)
            before_arrays.append(pca_before_rgb)
            
            # After training PCA with threshold
            pca_after_masked = apply_threshold_to_pca(pca_after[i], threshold_val)
            pca_after_resized = resize_pca_visualization(pca_after_masked, target_h, target_w)
            
            # Convert to RGB format for concatenation
            if pca_after_resized.shape[2] == 3:
                pca_after_rgb = (pca_after_resized * 255).astype(np.uint8)
            else:
                pca_after_gray = (pca_after_resized[:, :, 0] * 255).astype(np.uint8)
                pca_after_rgb = np.stack([pca_after_gray, pca_after_gray, pca_after_gray], axis=2)
            after_arrays.append(pca_after_rgb)
        else:
            # Create empty placeholders for missing images with consistent size
            empty_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            original_arrays.append(empty_img)
            before_arrays.append(empty_img)
            after_arrays.append(empty_img)
    
    # Verify all arrays have the same size before stacking
    print(f"    🔍 Array sizes - Original: {[arr.shape for arr in original_arrays]}")
    print(f"    🔍 Array sizes - Before: {[arr.shape for arr in before_arrays]}")
    print(f"    🔍 Array sizes - After: {[arr.shape for arr in after_arrays]}")
    
    # Stack images vertically for each column
    if rows > 1:
        original_column = np.vstack(original_arrays)
        before_column = np.vstack(before_arrays)
        after_column = np.vstack(after_arrays)
    else:
        original_column = original_arrays[0]
        before_column = before_arrays[0]
        after_column = after_arrays[0]
    
    # Concatenate all columns horizontally with no gap
    final_image = np.hstack([original_column, before_column, after_column])
    
    # Create figure with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(15, rows * 5))
    
    # Display the concatenated image
    ax.imshow(final_image)
    ax.axis('off')
    
    # Remove all margins and padding completely
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    return fig

def process_class_pca_batches(class_name: str, model_name: str, output_dir: str):
    """
    Process all images in a class in batches of 8 and generate PCA visualizations
    for different threshold values
    
    Args:
        class_name: Name of the class to process
        model_name: Name of the model to use
        output_dir: Output directory for saving figures
    """
    print(f"🎯 Processing class: {class_name} with model: {model_name}")
    
    # Create output directory structure
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Define threshold values
    threshold_values = [0, 25, 50, 75]
    
    # Create threshold directories
    threshold_dirs = {}
    for threshold in threshold_values:
        threshold_dir = os.path.join(model_output_dir, f"threshold_{threshold}")
        os.makedirs(threshold_dir, exist_ok=True)
        threshold_dirs[threshold] = threshold_dir
    
    # Get image size from config
    image_size = 224  # Default
    config_dir = Path("configs")
    
    for config_file in config_dir.glob('*.yaml'):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            if 'data' in config and 'image_size' in config['data']:
                image_size = config['data']['image_size']
                break
        except:
            continue
    
    # Load all images from the class
    try:
        all_images = load_all_class_images(class_name, image_size)
        if not all_images:
            print(f"❌ No images found for class '{class_name}'")
            return
    except Exception as e:
        print(f"❌ Error loading images for class '{class_name}': {e}")
        return
    
    # Find model paths
    outputs_dir = Path("outputs")
    model_path = outputs_dir / model_name
    best_model_path = model_path / 'best_model2.pth'
    
    if not best_model_path.exists():
        print(f"❌ Model checkpoint not found: {best_model_path}")
        return
    
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
        print(f"❌ Config not found for model: {model_name}")
        return
    
    try:
        # Load trained model
        print("  🔄 Loading trained model...")
        trained_model, config = load_model_from_checkpoint(str(config_path), str(best_model_path))
        trained_visualizer = PCAVisualizer(trained_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create untrained model (same architecture)
        print("  🔄 Creating untrained model...")
        if config['model']['backbone'] == 'dino_v2':
            untrained_model = DinoV2Model(
                model_name=config['model']['model_name'],
                feature_dim=config['model']['feature_dim'],
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout'],
                freeze_backbone=config['model']['freeze_backbone']
            )
        elif config['model']['backbone'] == 'ent_vit':
            untrained_model = EntVitModel(
                model_name=config['model']['model_name'],
                feature_dim=config['model']['feature_dim'],
                num_classes=config['model']['num_classes'],
                dropout=config['model']['dropout'],
                freeze_backbone=config['model']['freeze_backbone']
            )
        else:
            raise ValueError(f"Unsupported backbone: {config['model']['backbone']}")
        
        untrained_visualizer = PCAVisualizer(untrained_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"💡 This might be due to network issues when downloading pretrained models")
        print(f"💡 Try running again or check your internet connection")
        return
    
    # Process images in batches of 8
    batch_size = 8
    num_batches = (len(all_images) + batch_size - 1) // batch_size
    
    print(f"📊 Processing {len(all_images)} images in {num_batches} batches")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_images))
        
        batch_images = all_images[start_idx:end_idx]
        
        print(f"🔄 Processing batch {batch_idx + 1}/{num_batches} ({len(batch_images)} images)")
        
        try:
            # Extract features and compute PCA for untrained model
            print("  📊 Computing PCA for untrained model...")
            untrained_features = untrained_visualizer.extract_patch_features(batch_images, patch_size=14)
            untrained_pca_images, _ = untrained_visualizer.compute_pca_visualization(untrained_features, n_components=3)
            
            # Extract features and compute PCA for trained model
            print("  📊 Computing PCA for trained model...")
            trained_features = trained_visualizer.extract_patch_features(batch_images, patch_size=14)
            trained_pca_images, _ = trained_visualizer.compute_pca_visualization(trained_features, n_components=3)
            
            # Convert PCA visualizations to proper format
            pca_before = []
            pca_after = []
            
            for i in range(len(batch_images)):
                # Untrained (before)
                pca_spatial_before = untrained_visualizer.reshape_to_spatial(untrained_pca_images[i], patch_size=14)
                pca_before.append(pca_spatial_before)
                
                # Trained (after)
                pca_spatial_after = trained_visualizer.reshape_to_spatial(trained_pca_images[i], patch_size=14)
                pca_after.append(pca_spatial_after)
            
            # Create and save figures for each threshold
            for threshold_val in threshold_values:
                print(f"    📊 Creating figure for threshold {threshold_val}%...")
                
                # Create figure
                fig = create_pca_batch_figure(
                    batch_images, 
                    pca_before, 
                    pca_after,
                    batch_idx + 1,
                    class_name,
                    model_name,
                    threshold_val
                )
                
                # Save figure in corresponding threshold directory with class info in filename
                output_filename = f"{class_name}_batch{batch_idx + 1}_threshold{threshold_val}.png"
                output_path = os.path.join(threshold_dirs[threshold_val], output_filename)
                
                fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                
                print(f"    ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error processing batch {batch_idx + 1}: {e}")
            continue
    
    print(f"✅ Completed processing class '{class_name}' with model '{model_name}'")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch PCA Class Viewer - Generate PCA visualizations with multiple threshold values (0%, 25%, 50%, 75%)')
    parser.add_argument('--class', dest='class_name',
                       help='Name of the class to process (if not specified, process all classes)')
    parser.add_argument('--model', dest='model_name',
                       help='Name of the model to use (if not specified, process all models)')
    parser.add_argument('--output_dir', default='pca',
                       help='Output directory for saving figures (default: pca)')
    parser.add_argument('--list_classes', action='store_true',
                       help='List available classes and exit')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # List classes if requested
    if args.list_classes:
        print("📋 Available classes:")
        classes = get_available_classes()
        if classes:
            for i, class_name in enumerate(classes, 1):
                print(f"  {i}. {class_name}")
        else:
            print("  No classes found!")
        return
    
    # List models if requested
    if args.list_models:
        print("📋 Available models:")
        outputs_dir = Path("outputs")
        available_models = []
        
        for model_dir in outputs_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "best_model2.pth").exists():
                available_models.append(model_dir.name)
        
        if available_models:
            for i, model_name in enumerate(available_models, 1):
                print(f"  {i}. {model_name}")
        else:
            print("  No models found!")
        return
    
    # Get available classes and models
    available_classes = get_available_classes()
    if not available_classes:
        print("❌ No classes found!")
        return
    
    outputs_dir = Path("outputs")
    available_models = []
    for model_dir in outputs_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "best_model2.pth").exists():
            available_models.append(model_dir.name)
    
    if not available_models:
        print("❌ No trained models found!")
        return
    
    # Determine which classes to process
    classes_to_process = [args.class_name] if args.class_name else available_classes
    
    # Determine which models to process
    models_to_process = [args.model_name] if args.model_name else available_models
    
    # Validate inputs
    if args.class_name and args.class_name not in available_classes:
        print(f"❌ Class '{args.class_name}' not found!")
        print(f"Available classes: {', '.join(available_classes)}")
        return
    
    if args.model_name and args.model_name not in available_models:
        print(f"❌ Model '{args.model_name}' not found!")
        print(f"Available models: {', '.join(available_models)}")
        return
    
    print("🎯 Starting Batch PCA Class Viewer...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Classes to process: {len(classes_to_process)}")
    print(f"🤖 Models to process: {len(models_to_process)}")
    print(f"🎚️ Threshold values: 0%, 25%, 50%, 75%")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each combination of class and model
    total_combinations = len(classes_to_process) * len(models_to_process)
    current_combination = 0
    
    for class_name in classes_to_process:
        for model_name in models_to_process:
            current_combination += 1
            print(f"\n🔄 Processing combination {current_combination}/{total_combinations}")
            print(f"   Class: {class_name}")
            print(f"   Model: {model_name}")
            
            try:
                process_class_pca_batches(class_name, model_name, args.output_dir)
            except Exception as e:
                print(f"❌ Error processing {class_name} with {model_name}: {e}")
                continue
    
    print(f"\n✅ Completed! Results saved to: {args.output_dir}")
    print("📁 Directory structure:")
    print("   pca/")
    print("   └── model_name/")
    print("       ├── threshold_0/")
    print("       ├── threshold_25/")
    print("       ├── threshold_50/")
    print("       └── threshold_75/")
    print("           └── classname_batch*_threshold*.png")

if __name__ == '__main__':
    main()
