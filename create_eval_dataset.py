#!/usr/bin/env python3
"""
Script tạo bộ data evaluation cho image retrieval:
- Query: Các ảnh gốc trong tập test
- Corpus: Toàn bộ ảnh gốc và ảnh đã augment của tập train, val, và test
- Ground Truth: Mapping từ mỗi query đến các phiên bản augment tương ứng trong corpus

Output:
- eval_data/
  ├── query/                    # Ảnh test gốc (queries)
  ├── corpus/                   # Toàn bộ ảnh trong corpus
  └── ground_truth.json         # Mapping từ query đến ground truth trong corpus
"""

import yaml
import torch
import torch.utils.data
import pandas as pd
from pathlib import Path
import argparse
import json
from torchvision import transforms
from PIL import Image
import shutil
import os

from src.data_loader import create_dataloaders
from src.utils import set_seed, setup_logging


def get_augmentation_transform_without_normalize(image_size=224, backbone='dinov2'):
    """
    Tạo transform với augmentation mạnh nhưng KHÔNG normalize.
    Dùng để tạo ảnh augment để lưu trữ.
    """
    return transforms.Compose([
        # Bước 1: Tiền xử lý - Tập trung vào vùng quan trọng (vòng tròn nội soi)
        transforms.Resize((640, 480)),
        transforms.CenterCrop(size=(450, 450)), # Giả sử ảnh gốc ~500x500
        transforms.Resize((image_size, image_size)), # Resize về kích thước chuẩn

        # Bước 2: Augmentation hình học (Mô phỏng chuyển động của ống soi)
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=20,               # Xoay một góc hợp lý
                translate=(0.1, 0.1),     # Dịch chuyển nhẹ
                scale=(0.6, 1.3)          # Zoom vào/ra một chút
            )
        ], p=0.7), # Áp dụng với xác suất 70%

        # Bước 3: Augmentation màu sắc (Mô phỏng điều kiện ánh sáng và camera khác nhau)
        transforms.ColorJitter(
            brightness=0.2,   # Điều chỉnh độ sáng
            contrast=0.2,     # Điều chỉnh độ tương phản
            saturation=0.2,   # Điều chỉnh độ bão hòa
            hue=0.05          # HUE rất nhạy, chỉ nên thay đổi rất ít
        ),
        
        # Các phép biến đổi màu sắc an toàn khác
        transforms.RandomAutocontrast(p=0.2), # Tự động tăng cường độ tương phản

        # Bước 4: Augmentation mô phỏng nhiễu và che khuất
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

        # Chuyển sang Tensor TRƯỚC khi thực hiện RandomErasing
        transforms.ToTensor(),

        # Xóa một vùng nhỏ để mô phỏng bị che khuất (ví dụ: bởi ráy tai)
        transforms.RandomErasing(
            p=0.8, # Áp dụng với xác suất thấp
            scale=(0.02, 0.08), # Xóa một vùng nhỏ
            ratio=(0.3, 3.3),
            value='random' # Điền vào bằng nhiễu ngẫu nhiên thay vì màu đen
        ),
        # KHÔNG có Normalize ở đây!
    ])


def get_standard_transform_without_normalize(image_size=224):
    """
    Tạo transform chuẩn cho ảnh gốc nhưng KHÔNG normalize.
    """
    return transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.CenterCrop(size=(450, 450)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # KHÔNG có Normalize ở đây!
    ])


def get_denormalization_transform(backbone='dinov2'):
    """
    Tạo transform để denormalize ảnh về dạng PIL có thể lưu được.
    """
    if backbone == 'ent_vit':
        mean = [0.3464, 0.2280, 0.2228]
        std = [0.2520, 0.2128, 0.2093]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return transforms.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )


def save_images_from_dataloader(dataloader, output_dir, prefix, transform_func=None, backbone='dinov2'):
    """
    Lưu ảnh từ dataloader vào thư mục output_dir.
    
    Args:
        dataloader: DataLoader chứa ảnh
        output_dir: Thư mục đích
        prefix: Prefix cho tên file (ví dụ: 'train_', 'val_', 'test_')
        transform_func: Hàm transform để áp dụng lên ảnh (nếu có)
        backbone: Loại backbone để xác định denormalization
    
    Returns:
        List các path ảnh đã lưu
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    denormalize = get_denormalization_transform(backbone)
    
    image_counter = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        
        for i in range(batch_size):
            # Lấy ảnh từ batch
            img_tensor = images[i].cpu()
            label = labels[i].item()
            
            # Denormalize ảnh về dạng có thể convert sang PIL
            img_denormalized = denormalize(img_tensor)
            img_pil = transforms.ToPILImage()(img_denormalized)
            
            # Nếu có transform_func, áp dụng nó
            if transform_func is not None:
                img_tensor_processed = transform_func(img_pil)
                img_pil = transforms.ToPILImage()(img_tensor_processed)
            
            # Tạo tên file
            filename = f"{prefix}{image_counter:05d}_class{label}.jpg"
            filepath = output_dir / filename
            
            # Lưu ảnh
            img_pil.save(filepath)
            saved_paths.append(str(filepath))
            
            image_counter += 1
    
    return saved_paths


def create_augmented_images_from_dataloader(dataloader, output_dir, prefix, num_augmentations, backbone='dinov2'):
    """
    Tạo và lưu các phiên bản augment của ảnh từ dataloader.
    
    Args:
        dataloader: DataLoader chứa ảnh
        output_dir: Thư mục đích
        prefix: Prefix cho tên file
        num_augmentations: Số lượng phiên bản augment cho mỗi ảnh
        backbone: Loại backbone
    
    Returns:
        List các path ảnh augment đã lưu (được group theo ảnh gốc)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    augment_transform = get_augmentation_transform_without_normalize(backbone=backbone)
    denormalize = get_denormalization_transform(backbone)
    
    saved_paths = []
    image_counter = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        
        for i in range(batch_size):
            # Lấy ảnh từ batch
            img_tensor = images[i].cpu()
            label = labels[i].item()
            
            # Denormalize ảnh về dạng có thể convert sang PIL
            img_denormalized = denormalize(img_tensor)
            img_pil = transforms.ToPILImage()(img_denormalized)
            
            # Tạo các phiên bản augment
            augmented_paths = []
            for aug_idx in range(num_augmentations):
                # Áp dụng augmentation
                img_tensor_aug = augment_transform(img_pil)
                img_pil_aug = transforms.ToPILImage()(img_tensor_aug)
                
                # Tạo tên file
                filename = f"{prefix}{image_counter:05d}_aug{aug_idx:02d}_class{label}.jpg"
                filepath = output_dir / filename
                
                # Lưu ảnh
                img_pil_aug.save(filepath)
                augmented_paths.append(str(filepath))
            
            saved_paths.append(augmented_paths)
            image_counter += 1
    
    return saved_paths


def create_evaluation_dataset(config_path, output_dir, num_augmentations=5):
    """
    Tạo bộ data evaluation hoàn chỉnh.
    
    Args:
        config_path: Đường dẫn đến file config
        output_dir: Thư mục đích
        num_augmentations: Số lượng phiên bản augment cho mỗi ảnh test
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    backbone = config['model']['backbone']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo các thư mục con
    query_dir = output_dir / 'query'
    corpus_dir = output_dir / 'corpus'
    query_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataloaders
    print("📊 Loading dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config['data'], backbone)
    
    # --- 1. Tạo Query (Test gốc) ---
    print("📊 Creating queries from test set...")
    standard_transform = get_standard_transform_without_normalize()
    query_paths = save_images_from_dataloader(
        test_loader, query_dir, 'query_', standard_transform, backbone
    )
    
    # --- 2. Tạo Corpus ---
    print("📊 Creating corpus...")
    corpus_paths = []
    corpus_info = []
    
    # Corpus Part 1: Ảnh gốc từ train
    print("  - Adding train set (original)...")
    train_paths = save_images_from_dataloader(
        train_loader, corpus_dir, 'train_orig_', standard_transform, backbone
    )
    corpus_paths.extend(train_paths)
    corpus_info.extend([{'type': 'train_original', 'index': i} for i in range(len(train_paths))])
    
    # Corpus Part 2: Ảnh gốc từ val
    print("  - Adding val set (original)...")
    val_paths = save_images_from_dataloader(
        val_loader, corpus_dir, 'val_orig_', standard_transform, backbone
    )
    corpus_paths.extend(val_paths)
    corpus_info.extend([{'type': 'val_original', 'index': i} for i in range(len(val_paths))])
    
    # Corpus Part 3: Ảnh augment từ train (1 augment mỗi ảnh)
    print("  - Adding train set (augmented)...")
    train_aug_paths = create_augmented_images_from_dataloader(
        train_loader, corpus_dir, 'train_aug_', 1, backbone
    )
    # Flatten list of lists
    train_aug_flat = [path for paths in train_aug_paths for path in paths]
    corpus_paths.extend(train_aug_flat)
    corpus_info.extend([{'type': 'train_augmented', 'index': i} for i in range(len(train_aug_flat))])
    
    # Corpus Part 4: Ảnh augment từ val (1 augment mỗi ảnh)
    print("  - Adding val set (augmented)...")
    val_aug_paths = create_augmented_images_from_dataloader(
        val_loader, corpus_dir, 'val_aug_', 1, backbone
    )
    # Flatten list of lists
    val_aug_flat = [path for paths in val_aug_paths for path in paths]
    corpus_paths.extend(val_aug_flat)
    corpus_info.extend([{'type': 'val_augmented', 'index': i} for i in range(len(val_aug_flat))])
    
    # Corpus Part 5: Ảnh augment từ test (ground truth)
    print(f"  - Adding test set (augmented - {num_augmentations} per image)...")
    test_aug_paths = create_augmented_images_from_dataloader(
        test_loader, corpus_dir, 'test_aug_', num_augmentations, backbone
    )
    
    # Tính toán vị trí bắt đầu của test augment trong corpus
    test_aug_start_idx = len(corpus_paths)
    
    # Flatten và thêm vào corpus
    test_aug_flat = [path for paths in test_aug_paths for path in paths]
    corpus_paths.extend(test_aug_flat)
    corpus_info.extend([{'type': 'test_augmented', 'index': i} for i in range(len(test_aug_flat))])
    
    # --- 3. Tạo Ground Truth Mapping ---
    print("📊 Creating ground truth mapping...")
    ground_truth = {}
    
    for query_idx in range(len(query_paths)):
        # Tính toán indices của các phiên bản augment trong corpus
        start_idx = test_aug_start_idx + query_idx * num_augmentations
        end_idx = start_idx + num_augmentations
        
        ground_truth[query_idx] = {
            'query_path': query_paths[query_idx],
            'ground_truth_indices': list(range(start_idx, end_idx)),
            'ground_truth_paths': corpus_paths[start_idx:end_idx]
        }
    
    # --- 4. Lưu metadata ---
    print("📊 Saving metadata...")
    
    # Lưu thông tin corpus
    corpus_metadata = {
        'total_corpus_size': len(corpus_paths),
        'corpus_composition': {
            'train_original': len(train_paths),
            'val_original': len(val_paths),
            'train_augmented': len(train_aug_flat),
            'val_augmented': len(val_aug_flat),
            'test_augmented': len(test_aug_flat)
        },
        'corpus_paths': corpus_paths,
        'corpus_info': corpus_info
    }
    
    # Lưu thông tin query
    query_metadata = {
        'total_queries': len(query_paths),
        'query_paths': query_paths
    }
    
    # Lưu ground truth
    ground_truth_metadata = {
        'num_augmentations_per_query': num_augmentations,
        'test_augmented_start_index': test_aug_start_idx,
        'ground_truth_mapping': ground_truth
    }
    
    # Lưu các file JSON
    with open(output_dir / 'corpus_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(corpus_metadata, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'query_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(query_metadata, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(ground_truth_metadata, f, indent=2, ensure_ascii=False)
    
    # Lưu summary
    summary = {
        'config_used': config_path,
        'backbone': backbone,
        'num_augmentations': num_augmentations,
        'statistics': {
            'total_queries': len(query_paths),
            'total_corpus_size': len(corpus_paths),
            'train_original': len(train_paths),
            'val_original': len(val_paths),
            'train_augmented': len(train_aug_flat),
            'val_augmented': len(val_aug_flat),
            'test_augmented': len(test_aug_flat)
        },
        'directory_structure': {
            'query_dir': str(query_dir),
            'corpus_dir': str(corpus_dir),
            'metadata_files': [
                'corpus_metadata.json',
                'query_metadata.json', 
                'ground_truth.json',
                'summary.json'
            ]
        }
    }
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluation dataset created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Statistics:")
    print(f"   - Total queries: {len(query_paths)}")
    print(f"   - Total corpus size: {len(corpus_paths)}")
    print(f"   - Ground truth augmentations per query: {num_augmentations}")
    print(f"   - Test augmented start index: {test_aug_start_idx}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Create evaluation dataset for image retrieval')
    parser.add_argument('--config', '-c', default='configs/dinov2_vits14.yaml', 
                       help='Path to model config file')
    parser.add_argument('--output', '-o', default='eval_data',
                       help='Output directory for evaluation dataset')
    parser.add_argument('--num_augmentations', '-n', type=int, default=5,
                       help='Number of augmentations per test image for ground truth')
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    
    print("🚀 Starting evaluation dataset creation...")
    print("=" * 60)
    print(f"📁 Config: {args.config}")
    print(f"📁 Output: {args.output}")
    print(f"🔢 Augmentations per query: {args.num_augmentations}")
    print("=" * 60)
    
    try:
        summary = create_evaluation_dataset(
            config_path=args.config,
            output_dir=args.output,
            num_augmentations=args.num_augmentations
        )
        
        print("\n🎉 Dataset creation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
