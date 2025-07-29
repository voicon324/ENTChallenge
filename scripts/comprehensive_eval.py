#!/usr/bin/env python3
"""
Comprehensive evaluation script for models with metrics:
- HitRate@1, HitRate@5, HitRate@10
- MRR@1, MRR@5, MRR@10
- Recall@1, Recall@5, Recall@10

Evaluation process:
- Query: Original images in the test set.
- Corpus: All original and augmented images from train, val, and test sets.
- Ground Truth: For each query (original test image), the correct results are the 3 corresponding augmented versions of itself in the corpus.
"""

import yaml
import torch
import torch.utils.data
import pandas as pd
from pathlib import Path
import argparse
import json
from torchvision import transforms

from src.data_loader import create_dataloaders
from src.model_factory import build_model
from src.utils import set_seed, setup_logging
import torch.nn.functional as F

# --- Metrics Calculation Functions ---
# The logic of these functions has been streamlined to focus on the main objective:
# accurately finding augmented versions, rather than falling back to class label comparison.

def calculate_metrics_with_topk(query_embeddings: torch.Tensor,
                               corpus_embeddings: torch.Tensor,
                               k_values: list,
                               query_to_augmented_mapping: dict) -> dict:
    """
    Calculate HitRate@k, MRR@k and Recall@k.

    Args:
        query_embeddings: Embeddings of original test images (queries).
        corpus_embeddings: Embeddings of all images in the corpus.
        k_values: List of k values (e.g., [1, 5, 10]).
        query_to_augmented_mapping: Dict mapping from query index to list of
                                   corresponding augmented version indices in corpus.
    """
    results = {}
    
    # Normalize embeddings for cosine similarity calculation
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)
    
    # Calculate similarity matrix (cosine similarity) between queries and corpus
    similarity_matrix = torch.mm(query_embeddings, corpus_embeddings.t())
    
    n_queries = query_embeddings.size(0)
    
    # Get top-k indices for all queries at once for efficiency
    # Get the largest top-k to reuse for smaller k values
    max_k = max(k_values)
    _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)

    for k in k_values:
        # Get top-k for the current k value
        top_k_indices = top_k_indices_all[:, :k]
        
        # --- Calculate HitRate@k ---
        hit_count = 0
        for i in range(n_queries):
            query_augmented_indices = set(query_to_augmented_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i].tolist())
            
            # A non-empty intersection of the two sets means at least 1 ground truth was found
            if not query_augmented_indices.isdisjoint(retrieved_indices):
                hit_count += 1
        
        results[f"HitRate@{k}"] = hit_count / n_queries
        
        # --- Calculate MRR@k ---
        reciprocal_ranks = []
        for i in range(n_queries):
            query_augmented_indices = query_to_augmented_mapping.get(i, [])
            best_rank = float('inf')
            
            # Find the rank (position) of the first found ground truth
            for rank, retrieved_idx in enumerate(top_k_indices[i].tolist()):
                if retrieved_idx in query_augmented_indices:
                    best_rank = rank + 1
                    break
            
            if best_rank != float('inf'):
                reciprocal_ranks.append(1.0 / best_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        results[f"MRR@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        # --- Calculate Recall@k ---
        recall_scores = []
        for i in range(n_queries):
            query_augmented_indices = set(query_to_augmented_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i].tolist())
            
            # Calculate the number of ground truth found
            found_gt = len(query_augmented_indices.intersection(retrieved_indices))
            total_gt = len(query_augmented_indices)
            
            # Recall = number of found ground truths / total number of ground truths
            recall = found_gt / total_gt if total_gt > 0 else 0.0
            recall_scores.append(recall)
        
        results[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    return results

# --- Feature Extraction Functions ---

def extract_features(model, dataloader, device):
    """Extract features for original images (no augmentation)."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            features = model.get_features(images)
            all_features.append(features.cpu())
            all_labels.append(targets.cpu())
    
    if not all_features:
        return None, None
        
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

def get_strong_augmentation_transform(image_size=224, backbone='dinov2'):
    """Create transform with strong augmentation."""
    
    # Determine normalization parameters based on backbone
    if backbone == 'ent_vit':
        # EndoViT-specific normalization parameters
        mean = [0.3464, 0.2280, 0.2228]
        std = [0.2520, 0.2128, 0.2093]
    else:
        # Standard ImageNet normalization for other models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Suggestion: Use torchvision.transforms.v2 to run augmentations on GPU, significantly speeding up the process
    return transforms.Compose([
        # Step 1: Preprocessing - Focus on the important region (endoscopic circle)
        # Crop the central part to remove most of the black border, assuming the circle is in the middle.
        # Adjust the crop size to fit your images.
        # transforms.CenterCrop(size=(450, 450)), # Assuming original image is ~500x500
        transforms.Resize((500, 400)), # Resize to a standard size
        # transforms.CenterCrop(size=(450, 450)), # Assuming original image is ~500x500
        transforms.RandomCrop(size=(image_size, image_size)), # Randomly crop a standard size region
        transforms.Resize((image_size, image_size)), # Resize to a standard size

        # Step 2: Geometric Augmentation (Simulating endoscope movement)
        # Randomly apply one of the geometric transformations
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=20,               # Rotate by a reasonable angle
                translate=(0.1, 0.1),     # Translate slightly
                scale=(0.9, 1.1)          # Zoom in/out a bit
                # Shear is often not realistic for endoscopes, so it's omitted
            )
        ], p=0.7), # Apply with 70% probability

        # transforms.RandomHorizontalFlip(p=0.5), # Very important, simulates looking at left/right ear

        # Step 3: Color Augmentation (Simulating different lighting and camera conditions)
        # Use ColorJitter with moderate intensity
        transforms.ColorJitter(
            brightness=0.2,   # Adjust brightness
            contrast=0.2,     # Adjust contrast
            saturation=0.2,   # Adjust saturation
            hue=0.05          # HUE is very sensitive, should only be changed slightly
        ),
        
        # Other safe color transformations
        transforms.RandomAutocontrast(p=0.2), # Automatically enhance contrast

        # Step 4: Augmentation simulating noise and occlusion
        # Apply slight blur to simulate out-of-focus images
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

        # Convert to Tensor BEFORE performing RandomErasing
        transforms.ToTensor(),

        # Erase a small region to simulate occlusion (e.g., by earwax)
        transforms.RandomErasing(
            p=0.2, # Apply with low probability
            scale=(0.02, 0.08), # Erase a small area
            ratio=(0.3, 3.3),
            value='random' # Fill with random noise instead of black
        ),
        transforms.Normalize(mean=mean, std=std)
    ])

def extract_augmented_features(model, dataloader, device, backbone, num_augmentations=3):
    """
    Extract features for augmented versions of images.
    This function only returns features of the augmented images.
    """
    model.eval()
    all_features = []
    all_labels = []

    if num_augmentations <= 0:
        return torch.tensor
    
    image_size = model.image_size if hasattr(model, 'image_size') else 224
    strong_transform = get_strong_augmentation_transform(image_size, backbone)
    
    # Get standard transform to denormalize image before augmenting
    if backbone == 'ent_vit':
        # EndoViT-specific normalization parameters
        mean = [0.3464, 0.2280, 0.2228]
        std = [0.2520, 0.2128, 0.2093]
    else:
        # Standard ImageNet normalization for other models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    denormalize = transforms.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )

    with torch.no_grad():
        for images, targets in dataloader:
            # `images` is the batch of original images from the dataloader
            batch_size = images.size(0)
            
            # Repeat targets for the augmented versions
            augmented_targets = targets.repeat_interleave(num_augmentations)
            all_labels.append(augmented_targets.cpu())

            # Create and process augmented versions
            batch_augmented_features = []
            for _ in range(num_augmentations):
                augmented_batch_pil = []
                for i in range(batch_size):
                    img_tensor = images[i].cpu()
                    img_denormalized = denormalize(img_tensor)
                    img_pil = transforms.ToPILImage()(img_denormalized)
                    augmented_batch_pil.append(strong_transform(img_pil))

                augmented_batch_tensor = torch.stack(augmented_batch_pil).to(device)
                features = model.get_features(augmented_batch_tensor)
                batch_augmented_features.append(features)
            
            # Concatenate the augmented features in the correct order:
            # [img1_aug1, img2_aug1, ..., img1_aug2, img2_aug2, ...]
            # Needs to be reordered to:
            # [img1_aug1, img1_aug2, ..., img2_aug1, img2_aug2, ...]
            reordered_features = torch.cat(batch_augmented_features, dim=0).reshape(num_augmentations, batch_size, -1).transpose(0, 1).reshape(batch_size * num_augmentations, -1)
            all_features.append(reordered_features.cpu())

    if not all_features:
        return None, None

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def evaluate_model(config_path, checkpoint_path=None, model_name=""):
    """
    Main function to evaluate a model.
    The process has been clarified and the logic simplified.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Load Dataloaders and Model ---
    train_loader, val_loader, test_loader = create_dataloaders(config['data'], config['model']['backbone'])
    model = build_model(config['model'])
    model.to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint loaded. Using pretrained weights from model definition.")
    
    # --- 2. Feature Extraction ---
    print("\n--- Feature Extraction ---")
    # Query: Original Test
    print("📊 Extracting features from test set (Queries)...")
    query_features, query_labels = extract_features(model, test_loader, device)
    
    # Corpus Part 1: Original Images
    print("📊 Extracting features from train set (Corpus - Original)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("📊 Extracting features from val set (Corpus - Original)...")
    val_features, val_labels = extract_features(model, val_loader, device)
    
    # Corpus Part 2: Augmented Images
    num_augmentations = 5
    backbone = config['model']['backbone']
    print(f"📊 Extracting {num_augmentations} augmented features from train set (Corpus - Augmented)...")
    train_aug_features, train_aug_labels = extract_augmented_features(model, train_loader, device, backbone, num_augmentations)
    print(f"📊 Extracting {num_augmentations} augmented features from val set (Corpus - Augmented)...")
    val_aug_features, val_aug_labels = extract_augmented_features(model, val_loader, device, backbone, num_augmentations)
    print(f"📊 Extracting {num_augmentations} augmented features from test set (Corpus - Ground Truth)...")
    test_aug_features, test_aug_labels = extract_augmented_features(model, test_loader, device, backbone, num_augmentations)

    if query_features is None or train_features is None or test_aug_features is None:
        print(f"❌ Failed to extract necessary features for {model_name}. Skipping evaluation.")
        return None

    # --- 3. Build Corpus and Ground Truth Mapping ---
    print("\n--- Building Corpus & Ground Truth ---")
    
    # Concatenate all features to create the complete corpus
    corpus_features = torch.cat([
        train_features, 
        val_features,
        train_aug_features,
        val_aug_features,
        test_aug_features
    ], dim=0)
    
    # (Optional) Concatenate labels if needed for debugging
    corpus_labels = torch.cat([
        train_labels,
        val_labels,
        train_aug_labels,
        val_aug_labels,
        test_aug_labels
    ], dim=0)
    
    # Calculate the starting position of test augmented images in corpus
    # This is core information to determine ground truth
    test_aug_start_idx = len(train_features) + len(val_features) + len(train_aug_features) + len(val_aug_features)
    
    # Create mapping from query (original test) to its augmented versions
    query_to_augmented_mapping = {}
    for query_idx in range(len(query_features)):
        start = test_aug_start_idx + query_idx * num_augmentations
        end = start + num_augmentations
        query_to_augmented_mapping[query_idx] = list(range(start, end))

    print(f"  - Total corpus size: {corpus_features.shape[0]} samples")
    print(f"  - Test augmented (ground truth) start index: {test_aug_start_idx}")
    print(f"  - Example mapping: Query 0 -> Corpus indices {query_to_augmented_mapping.get(0)}")

    # --- 4. Debug & Sanity Check (Important) ---
    # Check if the feature of an original image is "close" to the features of its augmentations.
    # They will never be "equal" due to random augmentations.
    # We expect the cosine similarity to be high.
    print("\n--- Sanity Check: Similarity of Query vs. its Augmentations ---")
    for i in range(min(3, len(query_features))):
        query_emb = F.normalize(query_features[i:i+1], p=2, dim=1)
        
        aug_indices = query_to_augmented_mapping[i]
        aug_embs = F.normalize(corpus_features[aug_indices], p=2, dim=1)
        
        similarities = torch.mm(query_emb, aug_embs.t())
        avg_sim = similarities.mean().item()
        
        # Check top similarities with the entire corpus
        all_sims = torch.mm(query_emb, F.normalize(corpus_features, p=2, dim=1).t())
        top_sim_values, top_sim_indices = torch.topk(all_sims, 10, dim=1)
        
        print(f"  - Query {i} vs. its {num_augmentations} augments - Avg similarity: {avg_sim:.4f}")
        print(f"    Individual similarities: {similarities.squeeze().tolist()}")
        print(f"    Top 10 corpus similarities: {top_sim_values.squeeze()[:5].tolist()}")
        print(f"    Ground truth indices: {aug_indices}")
        print(f"    Top 10 retrieved indices: {top_sim_indices.squeeze()[:5].tolist()}")
        
        # Check if any ground truth is in the top 10
        gt_in_top10 = any(idx in aug_indices for idx in top_sim_indices.squeeze()[:10].tolist())
        print(f"    Ground truth in top 10: {gt_in_top10}")
        print()
    # --- 5. Calculate and Return Results ---
    print("\n--- Calculating Metrics ---")
    metrics = calculate_metrics_with_topk(
        query_features, 
        corpus_features, 
        k_values=[1, 5, 10],
        query_to_augmented_mapping=query_to_augmented_mapping
    )
    
    print(f"\n📊 Results for {model_name}:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    return metrics

def create_summary_table(results):
    """Create a summary table of results."""
    print("\n" + "="*25 + " SUMMARY TABLE " + "="*25)
    
    rows = []
    for model_name, model_results in results.items():
        if model_results.get('pretrained'):
            for metric, value in model_results['pretrained'].items():
                rows.append({'Model': model_name, 'Training': 'Pretrained', 'Metric': metric, 'Value': value})
        
        if model_results.get('finetuned'):
            for metric, value in model_results['finetuned'].items():
                rows.append({'Model': model_name, 'Training': 'Fine-tuned', 'Metric': metric, 'Value': value})
    
    if not rows:
        print("No results to display.")
        return
        
    df = pd.DataFrame(rows)
    
    # Pivot for a more intuitive comparison table
    pivot_df = df.pivot_table(index=['Metric', 'Model'], columns='Training', values='Value')
    
    # Reorder metrics for better readability
    metric_order = ['HitRate@1', 'HitRate@5', 'HitRate@10', 'MRR@1', 'MRR@5', 'MRR@10', 'Recall@1', 'Recall@5', 'Recall@10']
    pivot_df = pivot_df.reindex(metric_order, level='Metric')
    
    print(pivot_df.to_string(float_format="%.4f"))
    print("="*65)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of image retrieval models')
    parser.add_argument('--output', '-o', default='evaluation_results.json', help='Output file for results in JSON format')
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    
    # Define the models to be evaluated
    models_to_evaluate = [
        {
            'name': 'DINOv2-ViT-S/14',
            'config': 'configs/dinov2_vits14.yaml',
            'checkpoint': 'outputs/dinov2_vits14_ntxent/best_model424.pth'
        },
        {
            'name': 'ENT-ViT',
            'config': 'configs/ent-vit.yaml',
            'checkpoint': 'outputs/ent_vit_ntxent/best_model424.pth'
        },
                {
            'name': 'DINOv2-ViT-B/14',
            'config': 'configs/dinov2_vitb14.yaml',
            'checkpoint': 'outputs/dinov2_vitb14_ntxent/best_model424.pth'
        },
                {
            'name': 'DINOv2-ViT-L/14',
            'config': 'configs/dinov2_vitl14.yaml',
            'checkpoint': 'outputs/dinov2_vitl14_ntxent/best_model424.pth'
        },
    ]
    
    all_results = {}
    
    print("🚀 Starting Comprehensive Evaluation...")
    print("=" * 80)
    
    for model_info in models_to_evaluate:
        model_name = model_info['name']
        config_path = Path(model_info['config'])
        checkpoint_path = Path(model_info['checkpoint'])
        
        print(f"\n\n🔍 Evaluating Model: {model_name}")
        print("-" * 50)
        
        # Evaluate the fine-tuned model
        print(f"🎯 Evaluating {model_name} (Fine-tuned)")
        finetuned_results = evaluate_model(
            config_path,
            checkpoint_path=checkpoint_path,
            model_name=f"{model_name} (Fine-tuned)"
        )
        
        # Evaluate the pretrained model (without loading checkpoint)
        print(f"\n📦 Evaluating {model_name} (Pretrained only)")
        pretrained_results = evaluate_model(
            config_path, 
            checkpoint_path=None, 
            model_name=f"{model_name} (Pretrained)"
        )
        
        all_results[model_name] = {
            'pretrained': pretrained_results,
            'finetuned': finetuned_results
        }
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n\n💾 All evaluation results saved to: {output_path}")
    
    # Create summary table
    create_summary_table(all_results)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

