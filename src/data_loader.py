"""
Class Dataset và hàm tạo DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json
import random
from sklearn.model_selection import train_test_split

class ImageRetrievalDataset(Dataset):
    """Dataset class for image retrieval with 4 ENT classes: ear/nose/throat/vc"""
    
    def __init__(self, 
                 data_path: str, 
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 image_size: int = 224):
        """
        Args:
            data_path: Path to data directory
            split: 'train', 'val', or 'test'
            transform: Data augmentation transforms
            image_size: Size to resize images to
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Define the 4 ENT classes
        self.target_classes = ['ear', 'nose', 'throat', 'vc']
        
        # Load data
        self.image_paths, self.labels = self._load_data()
        
        # Map original labels to target classes
        self.image_paths, self.labels = self._map_to_target_classes()
        
        # Create label to index mapping for the 4 classes
        self.label_to_idx = {label: idx for idx, label in enumerate(self.target_classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert labels to indices
        self.label_indices = [self.label_to_idx[label] for label in self.labels]
        
        print(f"📊 Loaded {len(self.image_paths)} images for {split} split")
        print(f"📋 Number of classes: {len(self.label_to_idx)} (ear/nose/throat/vc)")
        print(f"📈 Class distribution: {self._get_class_distribution()}")
        
    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load image paths and labels"""
        image_paths = []
        labels = []
        
        # Check if annotation file exists
        annotation_file = self.data_path / f"{self.split}_annotations.json"
        if annotation_file.exists():
            # Load from annotation file
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            for item in annotations:
                image_paths.append(str(self.data_path / item['image_path']))
                labels.append(item['label'])
                
        else:
            # Load from directory structure (class_name/image_files)
            split_dir = self.data_path / self.split
            if not split_dir.exists():
                split_dir = self.data_path  # Use root if split dir doesn't exist
            
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.glob("*.jpg"):
                        image_paths.append(str(img_file))
                        labels.append(class_name)
                    for img_file in class_dir.glob("*.png"):
                        image_paths.append(str(img_file))
                        labels.append(class_name)
                    for img_file in class_dir.glob("*.jpeg"):
                        image_paths.append(str(img_file))
                        labels.append(class_name)
        
        return image_paths, labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item at index"""
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        label = self.label_indices[idx]
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return list(self.label_to_idx.keys())
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.label_to_idx)
    
    def _map_to_target_classes(self) -> Tuple[List[str], List[str]]:
        """Map original labels to the 4 target ENT classes"""
        mapped_paths = []
        mapped_labels = []
        
        # Define mapping from original labels to target classes
        class_mapping = {
            # Ear classes
            'ear-left': 'ear',
            'ear-right': 'ear',
            'ear_left': 'ear',
            'ear_right': 'ear',
            'ear': 'ear',
            
            # Nose classes  
            'nose-left': 'nose',
            'nose-right': 'nose',
            'nose_left': 'nose',
            'nose_right': 'nose',
            'nose': 'nose',
            
            # Throat classes
            'throat': 'throat',
            
            # VC (Vocal Cords) classes
            'vc-closed': 'vc',
            'vc-open': 'vc',
            'vc_closed': 'vc',
            'vc_open': 'vc',
            'vc': 'vc',
            'vocal_cord': 'vc',
            'vocal_cords': 'vc'
        }
        
        # Map labels to target classes
        for img_path, label in zip(self.image_paths, self.labels):
            # Normalize label (handle different naming conventions)
            label_normalized = label.lower().replace('_', '-')
            
            if label_normalized in class_mapping:
                mapped_label = class_mapping[label_normalized]
                mapped_paths.append(img_path)
                mapped_labels.append(mapped_label)
            elif label.lower() in class_mapping:
                mapped_label = class_mapping[label.lower()]
                mapped_paths.append(img_path)
                mapped_labels.append(mapped_label)
            else:
                # Try substring matching for cases where label contains target class names
                found_match = False
                for target_class in self.target_classes:
                    if target_class in label.lower():
                        mapped_paths.append(img_path)
                        mapped_labels.append(target_class)
                        found_match = True
                        break
                
                if not found_match:
                    print(f"⚠️ Skipping image with unmapped label: {label}")
        
        original_count = len(self.image_paths)
        mapped_count = len(mapped_paths)
        print(f"🔍 Mapped {mapped_count}/{original_count} images to target classes")
        
        return mapped_paths, mapped_labels
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes"""
        from collections import Counter
        return dict(Counter(self.labels))

class ContrastivePairDataset(Dataset):
    """Dataset for contrastive learning"""
    
    def __init__(self, base_dataset: ImageRetrievalDataset, num_pairs: int = 10000):
        """
        Args:
            base_dataset: Base dataset to create pairs from
            num_pairs: Number of pairs to generate
        """
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        
        # Create pairs
        self.pairs = self._create_pairs()
        
    def _create_pairs(self) -> List[Tuple[int, int, int]]:
        """Create positive and negative pairs"""
        pairs = []
        
        # Group indices by class
        class_to_indices = {}
        for idx, label in enumerate(self.base_dataset.label_indices):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        classes = list(class_to_indices.keys())
        
        for _ in range(self.num_pairs):
            # Create positive pair (same class)
            if random.random() < 0.5:
                class_idx = random.choice(classes)
                if len(class_to_indices[class_idx]) >= 2:
                    idx1, idx2 = random.sample(class_to_indices[class_idx], 2)
                    pairs.append((idx1, idx2, 1))  # 1 for positive
                else:
                    # Fallback to negative pair
                    class1, class2 = random.sample(classes, 2)
                    idx1 = random.choice(class_to_indices[class1])
                    idx2 = random.choice(class_to_indices[class2])
                    pairs.append((idx1, idx2, 0))  # 0 for negative
            else:
                # Create negative pair (different classes)
                class1, class2 = random.sample(classes, 2)
                idx1 = random.choice(class_to_indices[class1])
                idx2 = random.choice(class_to_indices[class2])
                pairs.append((idx1, idx2, 0))  # 0 for negative
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get pair at index"""
        idx1, idx2, label = self.pairs[idx]
        
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        
        return img1, img2, label
    
class NTXentPairDataset(Dataset):
    """Dataset for NT-Xent loss (SimCLR-style): returns two augmented views of each image."""

    def __init__(self, base_dataset: ImageRetrievalDataset, 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            base_dataset: Base dataset to create pairs from
            transform: Augmentation to apply for each view (if None, use base_dataset.transform)
        """
        self.base_dataset = base_dataset
        self.transform = transform if transform is not None else base_dataset.transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            view1: First augmented view of the image
            view2: Second augmented view of the image
            label: Class label index (optional, can be ignored for unsupervised)
        """
        img_path = self.base_dataset.image_paths[idx]
        label = self.base_dataset.label_indices[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading image {img_path}: {e}")
            image = Image.new('RGB', (self.base_dataset.image_size, self.base_dataset.image_size), (0, 0, 0))

        # Apply two different augmentations
        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2, label
        
        

def get_transforms(image_size: int = 224, 
                  split: str = 'train', 
                  normalize: bool = True,
                  backbone: str = 'dino_v2') -> transforms.Compose:
    """Get data transforms for different splits"""
    
    if split == 'train':
        transform_list = [
            # Bước 1: Tiền xử lý - Tập trung vào vùng quan trọng (vòng tròn nội soi)
            # Crop phần trung tâm để loại bỏ phần lớn viền đen, giả sử vòng tròn ở giữa.
            # Điều chỉnh kích thước crop cho phù hợp với ảnh của bạn.
            transforms.CenterCrop(size=(450, 450)), # Giả sử ảnh gốc ~500x500
            transforms.Resize((image_size, image_size)), # Resize về kích thước chuẩn

            # Bước 2: Augmentation hình học (Mô phỏng chuyển động của ống soi)
            # Áp dụng một trong các phép biến đổi hình học một cách ngẫu nhiên
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=20,               # Xoay một góc hợp lý
                    translate=(0.1, 0.1),     # Dịch chuyển nhẹ
                    scale=(0.9, 1.1)          # Zoom vào/ra một chút
                    # Shear (biến dạng trượt) thường không thực tế với ống soi, nên bỏ
                )
            ], p=0.7), # Áp dụng với xác suất 70%

            # transforms.RandomHorizontalFlip(p=0.5), # Rất quan trọng, mô phỏng soi tai trái/phải

            # Bước 3: Augmentation màu sắc (Mô phỏng điều kiện ánh sáng và camera khác nhau)
            # Sử dụng ColorJitter với cường độ vừa phải
            transforms.ColorJitter(
                brightness=0.2,   # Điều chỉnh độ sáng
                contrast=0.2,     # Điều chỉnh độ tương phản
                saturation=0.2,   # Điều chỉnh độ bão hòa
                hue=0.05          # HUE rất nhạy, chỉ nên thay đổi rất ít
            ),
            
            # Các phép biến đổi màu sắc an toàn khác
            transforms.RandomAutocontrast(p=0.2), # Tự động tăng cường độ tương phản

            # Bước 4: Augmentation mô phỏng nhiễu và che khuất
            # Làm mờ nhẹ để mô phỏng ảnh bị out-focus
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

            # Chuyển sang Tensor TRƯỚC khi thực hiện RandomErasing
            transforms.ToTensor(),

            # Xóa một vùng nhỏ để mô phỏng bị che khuất (ví dụ: bởi ráy tai)
            transforms.RandomErasing(
                p=0.2, # Áp dụng với xác suất thấp
                scale=(0.02, 0.08), # Xóa một vùng nhỏ
                ratio=(0.3, 3.3),
                value='random' # Điền vào bằng nhiễu ngẫu nhiên thay vì màu đen
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

def create_splits(data_path: str, 
                 train_split: float = 0.8, 
                 val_split: float = 0.1, 
                 test_split: float = 0.1) -> None:
    """Create train/val/test splits if they don't exist"""
    
    data_path = Path(data_path)
    
    # Check if splits already exist
    if (data_path / 'train').exists():
        print("📁 Splits already exist, skipping creation")
        return
    
    print("📁 Creating train/val/test splits...")
    
    # Collect all images and labels
    all_images = []
    all_labels = []
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_file in class_dir.glob("*.jpg"):
                all_images.append(str(img_file))
                all_labels.append(class_name)
            for img_file in class_dir.glob("*.png"):
                all_images.append(str(img_file))
                all_labels.append(class_name)
            for img_file in class_dir.glob("*.jpeg"):
                all_images.append(str(img_file))
                all_labels.append(class_name)
    
    # Split data
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=(val_split + test_split), 
        stratify=all_labels, random_state=42
    )
    
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=(test_split / (val_split + test_split)), 
        stratify=temp_labels, random_state=42
    )
    
    # Create directories and move files
    splits = {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    }
    
    for split_name, (images, labels) in splits.items():
        split_dir = data_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Create class directories
        for label in set(labels):
            (split_dir / label).mkdir(exist_ok=True)
        
        # Move files
        for img_path, label in zip(images, labels):
            src_path = Path(img_path)
            dst_path = split_dir / label / src_path.name
            src_path.rename(dst_path)
    
    print(f"✅ Created splits: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

def create_dataloaders(data_config: Dict, backbone: str = 'dino_v2') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create splits if they don't exist
    create_splits(
        data_config['path'], 
        data_config.get('train_split', 0.8),
        data_config.get('val_split', 0.1),
        data_config.get('test_split', 0.1)
    )
    
    # Get transforms
    train_transform = get_transforms(
        data_config.get('image_size', 224), 
        'val', 
        data_config.get('normalize', True),
        backbone
    )
    val_transform = get_transforms(
        data_config.get('image_size', 224), 
        'val', 
        data_config.get('normalize', True),
        backbone
    )
    
    # Create datasets
    train_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='train',
        transform=train_transform,
        image_size=data_config.get('image_size', 224)
    )
    
    val_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='val',
        transform=val_transform,
        image_size=data_config.get('image_size', 224)
    )
    
    test_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='test',
        transform=val_transform,
        image_size=data_config.get('image_size', 224)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_contrastive_dataloaders(data_config: Dict, backbone: str = 'dino_v2') -> Tuple[DataLoader, DataLoader]:
    """Create contrastive learning dataloaders"""
    
    # Create base datasets
    train_transform = get_transforms(
        data_config.get('image_size', 224), 
        'train', 
        data_config.get('normalize', True),
        backbone
    )
    
    train_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='train',
        transform=train_transform,
        image_size=data_config.get('image_size', 224)
    )
    
    val_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='val',
        transform=train_transform,
        image_size=data_config.get('image_size', 224)
    )
    
    # Create contrastive datasets
    train_contrastive = ContrastivePairDataset(train_dataset, num_pairs=10000)
    val_contrastive = ContrastivePairDataset(val_dataset, num_pairs=2000)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_contrastive,
        batch_size=data_config.get('batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_contrastive,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_ntxent_dataloaders(data_config: Dict, backbone: str = 'dino_v2') -> Tuple[DataLoader, DataLoader]:
    """Create NT-Xent (SimCLR-style) dataloaders"""
    
    # Create base datasets first
    train_transform = get_transforms(
        data_config.get('image_size', 224), 
        'train', 
        data_config.get('normalize', True),
        backbone
    )
    
    val_transform = get_transforms(
        data_config.get('image_size', 224), 
        'val', 
        data_config.get('normalize', True),
        backbone
    )
    
    train_base_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='train',
        transform=None,  # We'll apply transforms in NTXentPairDataset
        image_size=data_config.get('image_size', 224)
    )
    
    val_base_dataset = ImageRetrievalDataset(
        data_config['path'], 
        split='val',
        transform=None,  # We'll apply transforms in NTXentPairDataset
        image_size=data_config.get('image_size', 224)
    )
    
    # Create NT-Xent datasets
    train_ntxent = NTXentPairDataset(train_base_dataset, transform=train_transform)
    val_ntxent = NTXentPairDataset(val_base_dataset, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ntxent,
        batch_size=data_config.get('batch_size', 32),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ntxent,
        batch_size=data_config.get('batch_size', 32),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader
