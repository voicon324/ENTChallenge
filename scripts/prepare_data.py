#!/usr/bin/env python3
"""
Script to prepare and check data
Run: python prepare_data.py
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

def create_sample_data(output_dir: str, num_classes: int = 5, images_per_class: int = 20):
    """Create sample data for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class folders
    class_names = [f"class_{i:02d}" for i in range(num_classes)]
    
    for class_name in class_names:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create dummy images (just placeholder files)
        for i in range(images_per_class):
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            img_path.touch()  # Create empty file
    
    print(f"✅ Created sample data in {output_dir}")
    print(f"   Classes: {num_classes}")
    print(f"   Images per class: {images_per_class}")
    print(f"   Total images: {num_classes * images_per_class}")

def analyze_dataset(data_dir: str) -> Dict:
    """Analyze dataset"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Collect statistics
    stats = {
        'total_images': 0,
        'total_classes': 0,
        'classes': {},
        'splits': {}
    }
    
    # Check for splits
    splits = ['train', 'val', 'test']
    has_splits = any((data_path / split).exists() for split in splits)
    
    if has_splits:
        # Analyze splits
        for split in splits:
            split_dir = data_path / split
            if split_dir.exists():
                split_stats = analyze_split(split_dir)
                stats['splits'][split] = split_stats
                stats['total_images'] += split_stats['total_images']
    else:
        # Analyze root directory
        root_stats = analyze_split(data_path)
        stats.update(root_stats)
    
    return stats

def analyze_split(split_dir: Path) -> Dict:
    """Analyze a split"""
    
    stats = {
        'total_images': 0,
        'total_classes': 0,
        'classes': {}
    }
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Count images in this class
            image_count = 0
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    image_count += 1
            
            if image_count > 0:
                stats['classes'][class_name] = image_count
                stats['total_images'] += image_count
    
    stats['total_classes'] = len(stats['classes'])
    
    return stats

def print_dataset_stats(stats: Dict):
    """Print dataset statistics"""
    
    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    
    if 'splits' in stats and stats['splits']:
        # Print split statistics
        for split, split_stats in stats['splits'].items():
            print(f"\n📁 {split.upper()} Split:")
            print(f"   Total images: {split_stats['total_images']:,}")
            print(f"   Total classes: {split_stats['total_classes']:,}")
            
            if split_stats['classes']:
                # Show class distribution
                class_counts = list(split_stats['classes'].values())
                print(f"   Images per class: {min(class_counts)} - {max(class_counts)}")
                print(f"   Average per class: {sum(class_counts) / len(class_counts):.1f}")
    else:
        # Print overall statistics
        print(f"Total images: {stats['total_images']:,}")
        print(f"Total classes: {stats['total_classes']:,}")
        
        if stats['classes']:
            class_counts = list(stats['classes'].values())
            print(f"Images per class: {min(class_counts)} - {max(class_counts)}")
            print(f"Average per class: {sum(class_counts) / len(class_counts):.1f}")
    
    print("=" * 50)

def create_annotation_files(data_dir: str):
    """Create annotation files for dataset"""
    
    data_path = Path(data_dir)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = data_path / split
        if split_dir.exists():
            annotations = []
            
            # Collect annotations
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                            relative_path = f"{split}/{class_name}/{img_file.name}"
                            annotations.append({
                                'image_path': relative_path,
                                'label': class_name,
                                'class_id': hash(class_name) % 1000  # Simple class ID
                            })
            
            # Save annotations
            if annotations:
                annotation_file = data_path / f"{split}_annotations.json"
                with open(annotation_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                
                print(f"✅ Created {annotation_file} with {len(annotations)} annotations")

def validate_dataset(data_dir: str) -> bool:
    """Validate dataset integrity"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Check for splits
    splits = ['train', 'val', 'test']
    has_splits = any((data_path / split).exists() for split in splits)
    
    if has_splits:
        print("✅ Found split directories")
        
        # Validate each split
        for split in splits:
            split_dir = data_path / split
            if split_dir.exists():
                split_stats = analyze_split(split_dir)
                if split_stats['total_images'] == 0:
                    print(f"⚠️  {split} split has no images")
                else:
                    print(f"✅ {split} split: {split_stats['total_images']} images, {split_stats['total_classes']} classes")
    else:
        print("⚠️  No split directories found, assuming single directory structure")
        root_stats = analyze_split(data_path)
        if root_stats['total_images'] == 0:
            print("❌ No images found in root directory")
            return False
        else:
            print(f"✅ Root directory: {root_stats['total_images']} images, {root_stats['total_classes']} classes")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare and analyze dataset')
    parser.add_argument('--data-dir', '-d', default='data/processed',
                       help='Path to data directory')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample data for testing')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes for sample data')
    parser.add_argument('--images-per-class', type=int, default=20,
                       help='Number of images per class for sample data')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing dataset')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset')
    parser.add_argument('--create-annotations', action='store_true',
                       help='Create annotation files')
    
    args = parser.parse_args()
    
    # Create sample data
    if args.create_sample:
        create_sample_data(
            args.data_dir,
            num_classes=args.num_classes,
            images_per_class=args.images_per_class
        )
    
    # Analyze dataset
    if args.analyze:
        try:
            stats = analyze_dataset(args.data_dir)
            print_dataset_stats(stats)
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
    
    # Validate dataset
    if args.validate:
        is_valid = validate_dataset(args.data_dir)
        if is_valid:
            print("✅ Dataset validation passed")
        else:
            print("❌ Dataset validation failed")
    
    # Create annotations
    if args.create_annotations:
        try:
            create_annotation_files(args.data_dir)
        except Exception as e:
            print(f"❌ Error creating annotations: {e}")
    
    # Default action: analyze
    if not any([args.create_sample, args.analyze, args.validate, args.create_annotations]):
        print("🔍 Running default analysis...")
        try:
            stats = analyze_dataset(args.data_dir)
            print_dataset_stats(stats)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try --create-sample to create sample data")

if __name__ == '__main__':
    main()
