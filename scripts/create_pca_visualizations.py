#!/usr/bin/env python3
"""
Generate PCA visualizations for trained models
"""

import argparse
import yaml
import torch
import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.visualize import PCAVisualizer
from src.model_factory import DinoV2Model, EntVitModel
from src.data_loader import ImageRetrievalDataset

def load_model_from_checkpoint(config_path: str, checkpoint_path: str):
    """
    Load model from checkpoint
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model based on backbone type
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
            feature_dim=model_config['feature_dim'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            freeze_backbone=model_config['freeze_backbone']
        )
    else:
        raise ValueError(f"Unknown backbone: {model_config['backbone']}")
    
    # Load checkpoint
    print(f"📥 Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config

def load_class_images(data_path: str, split: str = 'test', max_images_per_class: int = 20):
    """
    Load images organized by class
    
    Args:
        data_path: Path to data directory
        split: Data split to use
        max_images_per_class: Maximum number of images per class
        
    Returns:
        Dictionary mapping class names to lists of PIL images
    """
    print(f"📁 Loading images from {data_path}/{split}")
    
    # Create dataset
    dataset = ImageRetrievalDataset(data_path, split=split, transform=None)
    
    # Group images by class
    class_images = defaultdict(list)
    class_paths = defaultdict(list)
    
    for i, (img_path, label_idx) in enumerate(zip(dataset.image_paths, dataset.label_indices)):
        class_name = dataset.idx_to_label[label_idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            class_images[class_name].append(img)
            class_paths[class_name].append(img_path)
            
            # Limit number of images per class
            if len(class_images[class_name]) >= max_images_per_class:
                continue
                
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            continue
    
    # Print statistics
    print(f"📊 Loaded images by class:")
    for class_name, images in class_images.items():
        print(f"  {class_name}: {len(images)} images")
    
    return dict(class_images), dict(class_paths)

def create_pca_visualizations(model_name: str, config_path: str, checkpoint_path: str, 
                             data_path: str, output_dir: str, device: str = 'cuda'):
    """
    Create PCA visualizations for a model with structure: pca_images/model_name/original_name.png
    
    Args:
        model_name: Name of the model (for output directory)
        config_path: Path to config file
        checkpoint_path: Path to checkpoint
        data_path: Path to data directory
        output_dir: Output directory for visualizations
        device: Device to use for inference
    """
    print(f"\n🎯 Creating PCA visualizations for {model_name}")
    print(f"📋 Config: {config_path}")
    print(f"📋 Checkpoint: {checkpoint_path}")
    
    # Create output directory: pca_images/model_name/
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_model_from_checkpoint(config_path, checkpoint_path)
    
    # Create visualizer
    visualizer = PCAVisualizer(model, device=device)
    
    # Load class images
    class_images, class_paths = load_class_images(data_path, split='test', max_images_per_class=20)
    
    # Create class comparison visualization
    print("\n🎨 Creating class comparison PCA...")
    results = visualizer.create_class_comparison_pca(
        class_images=class_images,
        patch_size=14,
        samples_per_class=5,
        threshold_percentile=25,
        save_path=str(model_output_dir / 'class_comparison_pca.png')
    )
    
    # Save individual images with original names
    print("\n🎨 Saving individual PCA images...")
    all_saved_paths = []
    
    for class_name, images in class_images.items():
        print(f"  Processing class: {class_name}")
        
        # Get original image names from paths
        image_names = []
        for img_path in class_paths[class_name]:
            # Extract filename without extension
            img_name = Path(img_path).stem
            image_names.append(f"{class_name}_{img_name}")
        
        # Save individual PCA images
        saved_paths = visualizer.save_individual_pca_images(
            imgs=images,
            image_names=image_names,
            output_dir=str(model_output_dir),
            patch_size=14,
            threshold_percentile=25
        )
        
        all_saved_paths.extend(saved_paths)
        
        # Create group analysis for this class
        sample_images = images[:5]
        pca_images, pca = visualizer.create_pca_visualization(
            imgs=sample_images,
            class_names=[f"{class_name}_group"] * len(sample_images),
            patch_size=14,
            threshold_percentile=25,
            save_path=str(model_output_dir / f'{class_name}_group_analysis.png')
        )
        
        # Analyze PCA components
        analysis = visualizer.analyze_pca_components(
            pca=pca,
            save_path=str(model_output_dir / f'{class_name}_pca_analysis.png')
        )
        
        # Save analysis results
        with open(model_output_dir / f'{class_name}_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
    
    # Create summary report
    print("\n📊 Creating summary report...")
    create_summary_report(model_name, config, class_images, class_paths, all_saved_paths, model_output_dir)
    
    print(f"✅ PCA visualizations completed for {model_name}")
    print(f"📁 Results saved to: {model_output_dir}")
    print(f"📊 Total images processed: {len(all_saved_paths)}")
    print(f"📋 Structure: pca_images/{model_name}/original_name.png")

def create_summary_report(model_name: str, config: dict, class_images: dict, 
                         class_paths: dict, all_saved_paths: list, output_dir: Path):
    """
    Create a summary report of the PCA analysis
    
    Args:
        model_name: Name of the model
        config: Model configuration
        class_images: Dictionary of class images
        class_paths: Dictionary of class image paths
        all_saved_paths: List of all saved PCA image paths
        output_dir: Output directory
    """
    
    report = {
        'model_name': model_name,
        'model_config': config['model'],
        'training_config': config['training'],
        'analysis_timestamp': str(datetime.now()),
        'output_structure': 'pca_images/{model_name}/original_name.png',
        'class_statistics': {
            class_name: len(images) for class_name, images in class_images.items()
        },
        'total_images_processed': len(all_saved_paths),
        'num_classes': len(class_images),
        'pca_settings': {
            'patch_size': 14,
            'n_components': 3,
            'threshold_percentile': 25,
            'visualization_type': 'PCA_only'
        },
        'file_structure': {
            'main_files': [
                'class_comparison_pca.png',
                'pca_summary.json'
            ],
            'individual_pca_files': [
                Path(path).name for path in all_saved_paths
            ],
            'group_analysis_files': [
                f'{class_name}_group_analysis.png' for class_name in class_images.keys()
            ],
            'analysis_files': [
                f'{class_name}_pca_analysis.png' for class_name in class_images.keys()
            ],
            'json_files': [
                f'{class_name}_analysis.json' for class_name in class_images.keys()
            ]
        },
        'original_image_mapping': {
            class_name: [Path(path).name for path in paths]
            for class_name, paths in class_paths.items()
        }
    }
    
    # Save report
    with open(output_dir / 'pca_summary.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Summary report saved to: {output_dir / 'pca_summary.json'}")
    print(f"📁 Created {len(all_saved_paths)} individual PCA files")
    print(f"📋 Structure: pca_images/{model_name}/original_name.png")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate PCA visualizations for trained models')
    parser.add_argument('--model_dir', type=str, default='outputs', 
                       help='Directory containing model checkpoints')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Directory containing config files')
    parser.add_argument('--data_path', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='pca_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--specific_model', type=str, default=None,
                       help='Specific model to analyze (e.g., dinov2_vitl14_ntxent_200epochs)')
    
    args = parser.parse_args()
    
    print("🎯 PCA Visualization Generator")
    print("=" * 50)
    
    model_dir = Path(args.model_dir)
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available models
    if args.specific_model:
        # Analyze specific model
        model_dirs = [model_dir / args.specific_model]
    else:
        # Find all model directories
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    print(f"📁 Found {len(model_dirs)} model directories to analyze")
    
    # Process each model
    for model_path in model_dirs:
        model_name = model_path.name
        
        # Find best model checkpoint
        best_model_path = model_path / 'best_model.pth'
        
        if not best_model_path.exists():
            print(f"⚠️ No best_model.pth found for {model_name}, skipping...")
            continue
        
        # Find corresponding config file
        config_path = None
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if config.get('wandb', {}).get('run_name') == model_name:
                config_path = config_file
                break
        
        if config_path is None:
            print(f"⚠️ No config file found for {model_name}, skipping...")
            continue
        
        try:
            # Create PCA visualizations
            create_pca_visualizations(
                model_name=model_name,
                config_path=str(config_path),
                checkpoint_path=str(best_model_path),
                data_path=args.data_path,
                output_dir=str(output_dir),
                device=args.device
            )
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {e}")
            continue
    
    print("\n🎉 PCA visualization generation completed!")
    print(f"📁 All results saved to: {output_dir}")

if __name__ == '__main__':
    main()
