#!/usr/bin/env python3
"""
Script to organize ENT dataset from Dataset/data.json into train/val/test splits
with proper class distribution
"""

import json
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

def load_data(json_path):
    """Load data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_data(data):
    """Analyze data distribution"""
    print(f"📊 Total images: {len(data)}")
    
    # Count by classification
    classifications = [item['Classification'] for item in data]
    class_counts = Counter(classifications)
    
    print(f"📈 Classes distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count} images")
    
    # Count by type
    types = [item['Type'] for item in data]
    type_counts = Counter(types)
    print(f"🔍 Type distribution:")
    for typ, count in type_counts.items():
        print(f"  {typ}: {count} images")
    
    return class_counts, type_counts

def create_balanced_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create balanced train/val/test splits"""
    # Group by classification
    class_data = defaultdict(list)
    for item in data:
        class_data[item['Classification']].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    print(f"📋 Creating balanced splits...")
    
    for cls, items in class_data.items():
        n_items = len(items)
        
        # Calculate split sizes
        n_train = int(n_items * train_ratio)
        n_val = int(n_items * val_ratio)
        n_test = n_items - n_train - n_val
        
        # Shuffle items
        random.shuffle(items)
        
        # Split
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
        
        print(f"  {cls}: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")
    
    # Shuffle all splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

def copy_images_to_structure(data_split, split_name, source_dir, target_dir):
    """Copy images to organized directory structure"""
    print(f"📁 Organizing {split_name} data...")
    
    split_dir = target_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    class_dirs = {}
    for item in data_split:
        cls = item['Classification']
        if cls not in class_dirs:
            class_dir = split_dir / cls
            class_dir.mkdir(exist_ok=True)
            class_dirs[cls] = class_dir
    
    # Copy images
    copied_count = 0
    missing_count = 0
    
    for item in data_split:
        source_path = source_dir / item['Path']
        target_path = class_dirs[item['Classification']] / item['Path']
        
        # Try original path first
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            # Try case variations (image -> Image, Image -> image)
            path_variants = [
                source_dir / item['Path'].replace('image', 'Image'),
                source_dir / item['Path'].replace('Image', 'image')
            ]
            
            copied = False
            for variant_path in path_variants:
                if variant_path.exists():
                    shutil.copy2(variant_path, target_path)
                    copied_count += 1
                    copied = True
                    break
            
            if not copied:
                print(f"⚠️  Missing image: {source_path}")
                missing_count += 1
    
    print(f"✅ {split_name}: Copied {copied_count} images, {missing_count} missing")
    return copied_count, missing_count

def clean_existing_data(target_dir):
    """Clean existing processed data"""
    if target_dir.exists():
        print(f"🧹 Cleaning existing data in {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    dataset_dir = Path('/kaggle/input/entrep/Dataset')
    json_path = dataset_dir / 'data.json'
    images_dir = dataset_dir / 'images'
    target_dir = Path('data/processed')
    
    # Load and analyze data
    print("📖 Loading data...")
    data = load_data(json_path)
    class_counts, type_counts = analyze_data(data)
    
    # Check if images directory exists
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Clean existing data
    clean_existing_data(target_dir)
    
    # Create balanced splits
    train_data, val_data, test_data = create_balanced_splits(data)
    
    print(f"\n📊 Final split sizes:")
    print(f"  Train: {len(train_data)} images")
    print(f"  Val: {len(val_data)} images")
    print(f"  Test: {len(test_data)} images")
    
    # Copy images to organized structure
    print(f"\n📂 Copying images to organized structure...")
    
    train_copied, train_missing = copy_images_to_structure(train_data, 'train', images_dir, target_dir)
    val_copied, val_missing = copy_images_to_structure(val_data, 'val', images_dir, target_dir)
    test_copied, test_missing = copy_images_to_structure(test_data, 'test', images_dir, target_dir)
    
    # Summary
    total_copied = train_copied + val_copied + test_copied
    total_missing = train_missing + val_missing + test_missing
    
    print(f"\n📈 Summary:")
    print(f"  Total images copied: {total_copied}")
    print(f"  Total images missing: {total_missing}")
    print(f"  Success rate: {total_copied / (total_copied + total_missing) * 100:.1f}%")
    
    # Save split information
    splits_info = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    with open(target_dir / 'splits_info.json', 'w', encoding='utf-8') as f:
        json.dump(splits_info, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Split information saved to {target_dir / 'splits_info.json'}")
    print(f"✅ Data organization complete!")

if __name__ == '__main__':
    main()
