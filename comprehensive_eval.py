#!/usr/bin/env python3
"""
Script đánh giá toàn diện cho 4 model với metrics:
- HitRate@1, HitRate@5, HitRate@10
- MRR@1, MRR@5, MRR@10
Corpus bao gồm:
- Train original images
- Val original images  
- Train augmented images (3 versions mỗi ảnh với strong augmentation)
- Val augmented images (3 versions mỗi ảnh với strong augmentation)
- Test augmented images (3 versions mỗi ảnh với strong augmentation - ground truth)
Query: Test original images
Ground Truth: Chính xác các augmented versions của test image đó (không phải chỉ cùng class)
"""

import yaml
import torch
import torch.utils.data
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from torchvision import transforms

from src.data_loader import create_dataloaders
from src.model_factory import build_model
from src.utils import set_seed, setup_logging, calculate_metrics
import torch.nn.functional as F

def calculate_metrics_with_topk(query_embeddings: torch.Tensor, query_labels: torch.Tensor,
                               corpus_embeddings: torch.Tensor, corpus_labels: torch.Tensor, 
                               k_values: list = [1, 5, 10], test_augmented_start_idx: int = None,
                               query_to_augmented_mapping: dict = None) -> dict:
    """
    Tính toán HitRate@k và MRR@k cho cross-split retrieval
    - Query: test set embeddings (original images only)
    - Corpus: train + val + augmented train + augmented val + augmented test images
    - Ground truth: chính xác augmented versions của test images đó (không phải chỉ cùng class)
    """
    results = {}
    
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)
    
    # Tính similarity matrix (query x corpus)
    similarity_matrix = torch.mm(query_embeddings, corpus_embeddings.t())
    
    # Debug info
    n_queries = query_embeddings.size(0)
    n_corpus = corpus_embeddings.size(0)
    unique_query_labels = torch.unique(query_labels)
    unique_corpus_labels = torch.unique(corpus_labels)
    
    print(f"  📊 Debug info:")
    print(f"    - Number of queries (test original): {n_queries}")
    print(f"    - Number of corpus (train+val+augmented): {n_corpus}")
    if test_augmented_start_idx is not None:
        print(f"    - Test augmented start index: {test_augmented_start_idx}")
        print(f"    - Test augmented samples: {n_corpus - test_augmented_start_idx}")
    print(f"    - Query classes: {len(unique_query_labels)}")
    print(f"    - Corpus classes: {len(unique_corpus_labels)}")
    print(f"    - Query samples per class: {[int(torch.sum(query_labels == label).item()) for label in unique_query_labels]}")
    
    for k in k_values:
        # Calculate HitRate@k for cross-split retrieval
        hit_rate = calculate_cross_split_hit_rate_at_k(similarity_matrix, query_labels, corpus_labels, k, test_augmented_start_idx, query_to_augmented_mapping)
        results[f"HitRate@{k}"] = hit_rate
        
        # Calculate MRR@k for cross-split retrieval
        mrr = calculate_cross_split_mrr_at_k(similarity_matrix, query_labels, corpus_labels, k, test_augmented_start_idx, query_to_augmented_mapping)
        results[f"MRR@{k}"] = mrr
    
    return results

def calculate_cross_split_hit_rate_at_k(similarity_matrix: torch.Tensor, query_labels: torch.Tensor, 
                                       corpus_labels: torch.Tensor, k: int, 
                                       test_augmented_start_idx: int = None,
                                       query_to_augmented_mapping: dict = None) -> float:
    """Tính Hit Rate@k cho cross-split retrieval - chỉ tính đúng khi tìm thấy chính xác augmented version của test image đó"""
    n_queries = similarity_matrix.size(0)
    hit_count = 0
    
    # Debug: Show first few queries
    debug_queries = min(3, n_queries)
    
    for i in range(n_queries):
        # Lấy top-k similar items từ corpus
        _, top_k_indices = torch.topk(similarity_matrix[i], k)
        
        # Nếu có mapping và test augmented start index, kiểm tra chính xác augmented versions
        if test_augmented_start_idx is not None and query_to_augmented_mapping is not None:
            # Lấy danh sách các augmented indices của query này
            query_augmented_indices = query_to_augmented_mapping.get(i, [])
            
            # Debug first few queries
            if i < debug_queries:
                print(f"    🔍 Query {i}: Expected augmented indices {query_augmented_indices}, Got top-{k}: {top_k_indices.tolist()}")
            
            # Kiểm tra xem có augmented version chính xác của query này không
            found = False
            for top_idx in top_k_indices:
                if top_idx.item() in query_augmented_indices:
                    hit_count += 1
                    found = True
                    break
            
            if i < debug_queries:
                print(f"    ✅ Query {i}: {'Found' if found else 'Not found'}")
        else:
            # Fallback: kiểm tra cùng class label
            query_label = query_labels[i]
            retrieved_labels = corpus_labels[top_k_indices]
            
            # Nếu có augmented test images trong corpus, ưu tiên check chúng
            if test_augmented_start_idx is not None:
                # Kiểm tra xem có augmented version của cùng class không
                augmented_indices = top_k_indices[top_k_indices >= test_augmented_start_idx]
                if len(augmented_indices) > 0:
                    # Kiểm tra augmented versions (ground truth)
                    augmented_labels = corpus_labels[augmented_indices]
                    if torch.any(augmented_labels == query_label):
                        hit_count += 1
                        continue
            
            # Kiểm tra các ảnh khác cùng class
            if torch.any(retrieved_labels == query_label):
                hit_count += 1
    
    hit_rate = hit_count / n_queries
    print(f"    - HitRate@{k}: {hit_count}/{n_queries} = {hit_rate:.4f}")
    return hit_rate

def calculate_cross_split_mrr_at_k(similarity_matrix: torch.Tensor, query_labels: torch.Tensor, 
                                  corpus_labels: torch.Tensor, k: int,
                                  test_augmented_start_idx: int = None,
                                  query_to_augmented_mapping: dict = None) -> float:
    """Tính Mean Reciprocal Rank@k - chỉ tính đúng khi tìm thấy chính xác augmented version của test image đó"""
    n_queries = similarity_matrix.size(0)
    reciprocal_ranks = []
    
    for i in range(n_queries):
        # Lấy top-k similar items từ corpus
        _, top_k_indices = torch.topk(similarity_matrix[i], k)
        
        best_rank = float('inf')
        
        # Nếu có mapping và test augmented start index, kiểm tra chính xác augmented versions
        if test_augmented_start_idx is not None and query_to_augmented_mapping is not None:
            # Lấy danh sách các augmented indices của query này
            query_augmented_indices = query_to_augmented_mapping.get(i, [])
            
            # Tìm rank tốt nhất của augmented version chính xác
            for rank, top_idx in enumerate(top_k_indices):
                if top_idx.item() in query_augmented_indices:
                    best_rank = min(best_rank, rank + 1)
                    break
        else:
            # Fallback: kiểm tra cùng class label
            query_label = query_labels[i]
            retrieved_labels = corpus_labels[top_k_indices]
            
            # Nếu có augmented test images, ưu tiên tìm chúng trước
            if test_augmented_start_idx is not None:
                for rank, (idx, label) in enumerate(zip(top_k_indices, retrieved_labels)):
                    if label == query_label:
                        if idx >= test_augmented_start_idx:
                            # Đây là augmented version (ground truth)
                            best_rank = min(best_rank, rank + 1)
                            break
                        else:
                            # Đây là ảnh khác cùng class
                            best_rank = min(best_rank, rank + 1)
            else:
                # Tìm rank của item đầu tiên cùng class
                for rank, label in enumerate(retrieved_labels):
                    if label == query_label:
                        best_rank = rank + 1
                        break
        
        if best_rank != float('inf'):
            reciprocal_ranks.append(1.0 / best_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"    - MRR@{k}: {mrr:.4f}")
    return mrr

def extract_features(model, dataloader, device):
    """Extract features from model"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                
                features = model.get_features(images)
                
                all_features.append(features.cpu())
                all_labels.append(targets.cpu())
    
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    return None, None

def get_strong_augmentation_transform(image_size=224):
    """Tạo transform với augmentation mạnh hơn để tăng độ khác biệt"""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_features_with_strong_augmentation(model, dataloader, device, num_augmentations=3):
    """Extract features with strong data augmentation including random crop"""
    model.eval()
    # Force deterministic mode for reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    all_features = []
    all_labels = []
    
    # Get image size from model config
    image_size = 224  # Default size
    strong_transform = get_strong_augmentation_transform(image_size)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(batch) == 2:
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)
                
                # Extract features for original images
                features = model.get_features(images)
                all_features.append(features.cpu())
                all_labels.append(targets.cpu())
                
                # Extract features for augmented versions with stronger augmentation
                for aug_idx in range(num_augmentations):
                    # Set different random seed for each augmentation
                    torch.manual_seed(42 + batch_idx * 1000 + aug_idx)
                    
                    # Create augmented batch
                    augmented_batch = []
                    for i in range(images.size(0)):
                        # Convert tensor back to PIL for augmentation
                        img_tensor = images[i].cpu()
                        # Denormalize
                        img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Convert to PIL
                        img_pil = transforms.ToPILImage()(img_tensor)
                        
                        # Apply strong augmentation
                        augmented_img = strong_transform(img_pil)
                        augmented_batch.append(augmented_img)
                    
                    # Stack augmented images
                    augmented_batch = torch.stack(augmented_batch).to(device)
                    
                    # Extract features
                    augmented_features = model.get_features(augmented_batch)
                    all_features.append(augmented_features.cpu())
                    all_labels.append(targets.cpu())
    
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    return None, None

def evaluate_model(config_path, checkpoint_path=None, model_name="", use_pretrained=True):
    """Đánh giá một model với cross-split retrieval bao gồm augmented train/val/test images"""
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create all dataloaders
    backbone = config['model']['backbone']
    train_loader, val_loader, test_loader = create_dataloaders(config['data'], backbone)
    
    # Build model
    model = build_model(config['model'])
    model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"🚫 {'No checkpoint provided' if checkpoint_path is None else 'Checkpoint not found'}: using pretrained weights only")
    
    # Extract features from test set (queries - original images only)
    print("📊 Extracting features from test set (queries - original images)...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Extract features from train set (corpus - original)
    print("📊 Extracting features from train set (original)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    # Extract features from val set (corpus - original)
    print("📊 Extracting features from val set (original)...")
    val_features, val_labels = extract_features(model, val_loader, device)
    
    # Extract strong augmented features from train set (corpus)
    print("📊 Extracting strong augmented features from train set (5 augmentations)...")
    train_augmented_features, train_augmented_labels = extract_features_with_strong_augmentation(model, train_loader, device, num_augmentations=5)
    
    # Extract strong augmented features from val set (corpus)
    print("📊 Extracting strong augmented features from val set (5 augmentations)...")
    val_augmented_features, val_augmented_labels = extract_features_with_strong_augmentation(model, val_loader, device, num_augmentations=5)
    
    # Extract strong augmented features from test set (corpus - ground truth)
    print("📊 Extracting strong augmented features from test set (5 augmentations)...")
    test_augmented_features, test_augmented_labels = extract_features_with_strong_augmentation(model, test_loader, device, num_augmentations=5)
    
    if (test_features is None or train_features is None or val_features is None or 
        train_augmented_features is None or val_augmented_features is None or test_augmented_features is None):
        print(f"❌ Failed to extract features for {model_name}")
        return None
    
    # Remove original images from augmented features (keep only augmented versions)
    # Format: [original, aug1, aug2, aug3, original, aug1, aug2, aug3, ...]
    # We want: [aug1, aug2, aug3, aug1, aug2, aug3, ...]
    
    def extract_only_augmented(augmented_features, augmented_labels, num_augmentations=5):
        """Extract only augmented versions (skip original)"""
        n_samples = len(augmented_features) // (num_augmentations + 1)
        aug_only_features = []
        aug_only_labels = []
        
        print(f"    🔍 Debug extract_only_augmented:")
        print(f"    - Total features: {len(augmented_features)}")
        print(f"    - Expected samples: {n_samples}")
        print(f"    - Features per sample: {num_augmentations + 1}")
        
        for i in range(n_samples):
            # Skip original (index 0, 6, 12, ...) and take augmented versions
            start_idx = i * (num_augmentations + 1) + 1  # Skip original
            end_idx = start_idx + num_augmentations  # Take augmentations
            
            if i < 3:  # Debug first 3 samples
                print(f"    - Sample {i}: original at {i * (num_augmentations + 1)}, augmented at {start_idx}:{end_idx}")
            
            aug_only_features.append(augmented_features[start_idx:end_idx])
            aug_only_labels.append(augmented_labels[start_idx:end_idx])
        
        return torch.cat(aug_only_features, dim=0), torch.cat(aug_only_labels, dim=0)
    
    # Extract only augmented versions
    train_aug_only_features, train_aug_only_labels = extract_only_augmented(train_augmented_features, train_augmented_labels)
    val_aug_only_features, val_aug_only_labels = extract_only_augmented(val_augmented_features, val_augmented_labels)
    test_aug_only_features, test_aug_only_labels = extract_only_augmented(test_augmented_features, test_augmented_labels)
    
    # Tạo mapping từ query index đến augmented indices trong corpus
    query_to_augmented_mapping = {}
    test_augmented_start_idx = (len(train_features) + len(val_features) + 
                               len(train_aug_only_features) + len(val_aug_only_features))
    
    # Mỗi test image có 5 augmented versions
    num_augmentations = 5
    for query_idx in range(len(test_features)):
        # Augmented versions của query_idx nằm ở vị trí:
        # test_augmented_start_idx + query_idx * num_augmentations đến
        # test_augmented_start_idx + (query_idx + 1) * num_augmentations - 1
        start_aug_idx = test_augmented_start_idx + query_idx * num_augmentations
        end_aug_idx = start_aug_idx + num_augmentations
        query_to_augmented_mapping[query_idx] = list(range(start_aug_idx, end_aug_idx))
    
    print(f"📊 Query to augmented mapping example:")
    for i in range(min(3, len(test_features))):  # Show first 3 mappings
        print(f"  - Query {i} -> Augmented indices: {query_to_augmented_mapping[i]}")
    
    # Debug: Check similarity between query and its augmented versions
    print(f"📊 Debug: Checking similarity between queries and their augmented versions:")
    for i in range(min(3, len(test_features))):
        query_embedding = F.normalize(test_features[i:i+1], p=2, dim=1)
        
        # Get the correct augmented embeddings from test_aug_only_features
        aug_start_in_aug_features = i * 5  # Each query has 5 augmented versions
        aug_embeddings = F.normalize(test_aug_only_features[aug_start_in_aug_features:aug_start_in_aug_features+5], p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mm(query_embedding, aug_embeddings.t())
        print(f"  - Query {i} vs its augmented versions: {similarities.squeeze().tolist()}")
        
        # Also check if the features are exactly the same (they should be since no augmentation)
        print(f"  - Query {i} feature norm: {torch.norm(test_features[i]).item():.6f}")
        print(f"  - Aug features norms: {[torch.norm(test_aug_only_features[aug_start_in_aug_features+j]).item() for j in range(5)]}")
        
        # Debug: Direct comparison with raw features (before normalization)
        raw_similarities = []
        for j in range(5):
            raw_sim = F.cosine_similarity(test_features[i], test_aug_only_features[aug_start_in_aug_features+j], dim=0)
            raw_similarities.append(raw_sim.item())
        print(f"  - Raw cosine similarities: {raw_similarities}")
        
        # Check if they are exactly equal
        for j in range(5):
            is_equal = torch.allclose(test_features[i], test_aug_only_features[aug_start_in_aug_features+j])
            print(f"  - Query {i} == Aug {j}: {is_equal}")
    
    # Combine all corpus: original train + original val + augmented train + augmented val + augmented test
    corpus_features = torch.cat([
        train_features, 
        val_features,
        train_aug_only_features,
        val_aug_only_features,
        test_aug_only_features
    ], dim=0)
    
    corpus_labels = torch.cat([
        train_labels,
        val_labels,
        train_aug_only_labels,
        val_aug_only_labels,
        test_aug_only_labels
    ], dim=0)
    
    # Calculate start index of test augmented samples in corpus
    test_augmented_start_idx = (len(train_features) + len(val_features) + 
                               len(train_aug_only_features) + len(val_aug_only_features))
    
    print(f"📊 Cross-split setup:")
    print(f"  - Test queries (original): {test_features.shape[0]} samples")
    print(f"  - Train corpus (original): {train_features.shape[0]} samples")
    print(f"  - Val corpus (original): {val_features.shape[0]} samples")
    print(f"  - Train augmented corpus: {train_aug_only_features.shape[0]} samples")
    print(f"  - Val augmented corpus: {val_aug_only_features.shape[0]} samples")
    print(f"  - Test augmented corpus: {test_aug_only_features.shape[0]} samples")
    print(f"  - Total corpus: {corpus_features.shape[0]} samples")
    print(f"  - Test augmented start index: {test_augmented_start_idx}")
    
    # Calculate metrics
    metrics = calculate_metrics_with_topk(
        test_features, test_labels, 
        corpus_features, corpus_labels, 
        k_values=[1, 5, 10],
        test_augmented_start_idx=test_augmented_start_idx,
        query_to_augmented_mapping=query_to_augmented_mapping
    )
    
    print(f"📊 Results for {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of all models')
    parser.add_argument('--output', '-o', default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(42)
    
    # Model configurations
    models = [
        {
            'name': 'DINOv2-ViT-S/14',
            'config': 'configs/dinov2_vits14.yaml',
            'checkpoint': 'outputs/dinov2_vits14_ntxent/best_model.pth'
        },
        # {
        #     'name': 'DINOv2-ViT-B/14',
        #     'config': 'configs/dinov2_vitb14.yaml',
        #     'checkpoint': 'outputs/dinov2_vitb14_ntxent/best_model.pth'
        # },
        # {
        #     'name': 'DINOv2-ViT-L/14',
        #     'config': 'configs/dinov2_vitl14.yaml',
        #     'checkpoint': 'outputs/dinov2_vitl14_ntxent/best_model.pth'
        # },
        # {
        #     'name': 'ENT-ViT',
        #     'config': 'configs/ent-vit.yaml',
        #     'checkpoint': 'outputs/ent_vit_ntxent/best_model.pth'
        # }
    ]
    
    results = {}
    
    print("🚀 Starting comprehensive evaluation...")
    print("=" * 80)
    
    for model_config in models:
        model_name = model_config['name']
        config_path = Path(model_config['config'])
        checkpoint_path = Path(model_config['checkpoint'])
        
        print(f"\n🔍 Evaluating {model_name}")
        print("-" * 50)
        
        # Evaluate without fine-tuning (pretrained only)
        print(f"📦 Evaluating {model_name} (Pretrained only)")
        pretrained_results = evaluate_model(
            config_path, 
            checkpoint_path=None, 
            model_name=f"{model_name} (Pretrained)",
            use_pretrained=True
        )
        
        # Evaluate with fine-tuning
        print(f"\n🎯 Evaluating {model_name} (Fine-tuned)")
        finetuned_results = evaluate_model(
            config_path,
            checkpoint_path=checkpoint_path,
            model_name=f"{model_name} (Fine-tuned)",
            use_pretrained=False
        )
        
        # Store results
        results[model_name] = {
            'pretrained': pretrained_results,
            'finetuned': finetuned_results
        }
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Create summary table
    create_summary_table(results)

def create_summary_table(results):
    """Tạo bảng tổng kết kết quả"""
    print("\n📊 SUMMARY TABLE")
    print("=" * 100)
    
    # Create DataFrame for better formatting
    rows = []
    
    for model_name, model_results in results.items():
        if model_results['pretrained']:
            for metric, value in model_results['pretrained'].items():
                rows.append({
                    'Model': model_name,
                    'Training': 'Pretrained Only',
                    'Metric': metric,
                    'Value': f"{value:.4f}"
                })
        
        if model_results['finetuned']:
            for metric, value in model_results['finetuned'].items():
                rows.append({
                    'Model': model_name,
                    'Training': 'Fine-tuned',
                    'Metric': metric,
                    'Value': f"{value:.4f}"
                })
    
    df = pd.DataFrame(rows)
    
    # Print by metric
    metrics = ['HitRate@1', 'HitRate@5', 'HitRate@10', 'MRR@1', 'MRR@5', 'MRR@10']
    
    for metric in metrics:
        print(f"\n📈 {metric}")
        print("-" * 60)
        metric_df = df[df['Metric'] == metric]
        
        if not metric_df.empty:
            # Pivot table for better visualization
            pivot_df = metric_df.pivot(index='Model', columns='Training', values='Value')
            print(pivot_df.to_string())
        
        print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise