#!/usr/bin/env python3
"""
Script đánh giá toàn diện cho các model với metrics:
- HitRate@1, HitRate@5, HitRate@10
- MRR@1, MRR@5, MRR@10
- Recall@1, Recall@5, Recall@10

Quy trình đánh giá:
- Query: Các ảnh gốc trong tập test.
- Corpus: Toàn bộ ảnh gốc và ảnh đã augment của tập train, val, và test.
- Ground Truth: Với mỗi query (ảnh test gốc), kết quả đúng là 3 phiên bản augment tương ứng của chính nó trong corpus.
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

# --- Các hàm tính toán Metrics ---
# Logic của các hàm này đã được làm gọn lại để tập trung vào mục tiêu chính:
# tìm chính xác các phiên bản augment, thay vì fallback về so sánh class label.

def calculate_metrics_with_topk(query_embeddings: torch.Tensor,
                               corpus_embeddings: torch.Tensor,
                               k_values: list,
                               query_to_augmented_mapping: dict) -> dict:
    """
    Tính toán HitRate@k, MRR@k và Recall@k.

    Args:
        query_embeddings: Embeddings của các ảnh test gốc (queries).
        corpus_embeddings: Embeddings của toàn bộ ảnh trong corpus.
        k_values: Danh sách các giá trị k (ví dụ: [1, 5, 10]).
        query_to_augmented_mapping: Dict map từ index của query đến list các index của
                                    phiên bản augment tương ứng trong corpus.
    """
    results = {}
    
    # Chuẩn hóa embeddings để tính cosine similarity
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)
    
    # Tính ma trận tương đồng (cosine similarity) giữa queries và corpus
    similarity_matrix = torch.mm(query_embeddings, corpus_embeddings.t())
    
    n_queries = query_embeddings.size(0)
    
    # Lấy top-k indices cho tất cả các query cùng một lúc để tăng hiệu quả
    # Lấy top-k lớn nhất để tái sử dụng cho các k nhỏ hơn
    max_k = max(k_values)
    _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)

    for k in k_values:
        # Lấy top-k cho giá trị k hiện tại
        top_k_indices = top_k_indices_all[:, :k]
        
        # --- Tính HitRate@k ---
        hit_count = 0
        for i in range(n_queries):
            query_augmented_indices = set(query_to_augmented_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i].tolist())
            
            # Giao của hai tập hợp không rỗng nghĩa là đã tìm thấy ít nhất 1 ground truth
            if not query_augmented_indices.isdisjoint(retrieved_indices):
                hit_count += 1
        
        results[f"HitRate@{k}"] = hit_count / n_queries
        
        # --- Tính MRR@k ---
        reciprocal_ranks = []
        for i in range(n_queries):
            query_augmented_indices = query_to_augmented_mapping.get(i, [])
            best_rank = float('inf')
            
            # Tìm rank (vị trí) của ground truth đầu tiên được tìm thấy
            for rank, retrieved_idx in enumerate(top_k_indices[i].tolist()):
                if retrieved_idx in query_augmented_indices:
                    best_rank = rank + 1
                    break
            
            if best_rank != float('inf'):
                reciprocal_ranks.append(1.0 / best_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        results[f"MRR@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        # --- Tính Recall@k ---
        recall_scores = []
        for i in range(n_queries):
            query_augmented_indices = set(query_to_augmented_mapping.get(i, []))
            retrieved_indices = set(top_k_indices[i].tolist())
            
            # Tính số lượng ground truth được tìm thấy
            found_gt = len(query_augmented_indices.intersection(retrieved_indices))
            total_gt = len(query_augmented_indices)
            
            # Recall = số ground truth tìm thấy / tổng số ground truth
            recall = found_gt / total_gt if total_gt > 0 else 0.0
            recall_scores.append(recall)
        
        results[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    return results

# --- Các hàm trích xuất đặc trưng ---

def extract_features(model, dataloader, device):
    """Trích xuất đặc trưng cho các ảnh gốc (không augment)."""
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
    """Tạo transform với augmentation mạnh."""
    
    # Determine normalization parameters based on backbone
    if backbone == 'ent_vit':
        # EndoViT-specific normalization parameters
        mean = [0.3464, 0.2280, 0.2228]
        std = [0.2520, 0.2128, 0.2093]
    else:
        # Standard ImageNet normalization for other models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Gợi ý: Sử dụng torchvision.transforms.v2 để có thể chạy augment trên GPU, tăng tốc độ đáng kể
    return transforms.Compose([
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
        transforms.Normalize(mean=mean, std=std)
    ])

def extract_augmented_features(model, dataloader, device, backbone, num_augmentations=3):
    """
    Trích xuất đặc trưng cho các phiên bản augment của ảnh.
    Hàm này chỉ trả về features của các ảnh đã augment.
    """
    model.eval()
    all_features = []
    all_labels = []

    if num_augmentations <= 0:
        return torch.tensor
    
    image_size = model.image_size if hasattr(model, 'image_size') else 224
    strong_transform = get_strong_augmentation_transform(image_size, backbone)
    
    # Lấy transform chuẩn để denormalize ảnh trước khi augment
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
            # `images` là batch ảnh gốc từ dataloader
            batch_size = images.size(0)
            
            # Lặp lại targets cho các phiên bản augment
            augmented_targets = targets.repeat_interleave(num_augmentations)
            all_labels.append(augmented_targets.cpu())

            # Tạo và xử lý các phiên bản augment
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
            
            # Nối các features augment theo đúng thứ tự:
            # [img1_aug1, img2_aug1, ..., img1_aug2, img2_aug2, ...]
            # Cần sắp xếp lại để thành:
            # [img1_aug1, img1_aug2, ..., img2_aug1, img2_aug2, ...]
            reordered_features = torch.cat(batch_augmented_features, dim=0).reshape(num_augmentations, batch_size, -1).transpose(0, 1).reshape(batch_size * num_augmentations, -1)
            all_features.append(reordered_features.cpu())

    if not all_features:
        return None, None

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def evaluate_model(config_path, checkpoint_path=None, model_name=""):
    """
    Hàm chính để đánh giá một model.
    Quy trình đã được làm rõ và logic được đơn giản hóa.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Tải Dataloaders và Model ---
    train_loader, val_loader, test_loader = create_dataloaders(config['data'], config['model']['backbone'])
    model = build_model(config['model'])
    model.to(device)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint loaded. Using pretrained weights from model definition.")
    
    # --- 2. Trích xuất đặc trưng ---
    print("\n--- Feature Extraction ---")
    # Query: Test gốc
    print("📊 Extracting features from test set (Queries)...")
    query_features, query_labels = extract_features(model, test_loader, device)
    
    # Corpus Part 1: Ảnh gốc
    print("📊 Extracting features from train set (Corpus - Original)...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("📊 Extracting features from val set (Corpus - Original)...")
    val_features, val_labels = extract_features(model, val_loader, device)
    
    # Corpus Part 2: Ảnh augment
    num_augmentations = 5
    backbone = config['model']['backbone']
    print(f"📊 Extracting {num_augmentations} augmented features from train set (Corpus - Augmented)...")
    train_aug_features, train_aug_labels = extract_augmented_features(model, train_loader, device, backbone, 1)
    print(f"📊 Extracting {num_augmentations} augmented features from val set (Corpus - Augmented)...")
    val_aug_features, val_aug_labels = extract_augmented_features(model, val_loader, device, backbone, 1)
    print(f"📊 Extracting {num_augmentations} augmented features from test set (Corpus - Ground Truth)...")
    test_aug_features, test_aug_labels = extract_augmented_features(model, test_loader, device, backbone, num_augmentations)

    if query_features is None or train_features is None or test_aug_features is None:
        print(f"❌ Failed to extract necessary features for {model_name}. Skipping evaluation.")
        return None

    # --- 3. Xây dựng Corpus và Ground Truth Mapping ---
    print("\n--- Building Corpus & Ground Truth ---")
    
    # Nối tất cả các features lại để tạo thành corpus hoàn chỉnh
    corpus_features = torch.cat([
        train_features, 
        val_features,
        train_aug_features,
        val_aug_features,
        test_aug_features
    ], dim=0)
    
    # (Tùy chọn) Nối labels nếu cần debug
    corpus_labels = torch.cat([
        train_labels,
        val_labels,
        train_aug_labels,
        val_aug_labels,
        test_aug_labels
    ], dim=0)
    
    # Tính toán vị trí bắt đầu của các ảnh test augment trong corpus
    # Đây là thông tin cốt lõi để xác định ground truth
    test_aug_start_idx = len(train_features) + len(val_features) + len(train_aug_features) + len(val_aug_features)
    
    # Tạo mapping từ query (test gốc) đến các phiên bản augment của nó
    query_to_augmented_mapping = {}
    for query_idx in range(len(query_features)):
        start = test_aug_start_idx + query_idx * num_augmentations
        end = start + num_augmentations
        query_to_augmented_mapping[query_idx] = list(range(start, end))

    print(f"  - Total corpus size: {corpus_features.shape[0]} samples")
    print(f"  - Test augmented (ground truth) start index: {test_aug_start_idx}")
    print(f"  - Example mapping: Query 0 -> Corpus indices {query_to_augmented_mapping.get(0)}")

    # --- 4. Debug & Sanity Check (Quan trọng) ---
    # Kiểm tra xem feature của ảnh gốc có "gần" với feature của các bản augment không.
    # Chúng không bao giờ "bằng nhau" (equal) do có phép augment ngẫu nhiên.
    # Ta kỳ vọng cosine similarity sẽ cao.
    print("\n--- Sanity Check: Similarity of Query vs. its Augmentations ---")
    for i in range(min(3, len(query_features))):
        query_emb = F.normalize(query_features[i:i+1], p=2, dim=1)
        
        aug_indices = query_to_augmented_mapping[i]
        aug_embs = F.normalize(corpus_features[aug_indices], p=2, dim=1)
        
        similarities = torch.mm(query_emb, aug_embs.t())
        avg_sim = similarities.mean().item()
        
        # Kiểm tra top similarities với toàn bộ corpus
        all_sims = torch.mm(query_emb, F.normalize(corpus_features, p=2, dim=1).t())
        top_sim_values, top_sim_indices = torch.topk(all_sims, 10, dim=1)
        
        print(f"  - Query {i} vs. its {num_augmentations} augments - Avg similarity: {avg_sim:.4f}")
        print(f"    Individual similarities: {similarities.squeeze().tolist()}")
        print(f"    Top 10 corpus similarities: {top_sim_values.squeeze()[:5].tolist()}")
        print(f"    Ground truth indices: {aug_indices}")
        print(f"    Top 10 retrieved indices: {top_sim_indices.squeeze()[:5].tolist()}")
        
        # Kiểm tra xem có ground truth nào trong top 10 không
        gt_in_top10 = any(idx in aug_indices for idx in top_sim_indices.squeeze()[:10].tolist())
        print(f"    Ground truth in top 10: {gt_in_top10}")
        print()
    # --- 5. Tính toán và Trả về kết quả ---
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
    """Tạo bảng tổng kết kết quả."""
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
    
    # Pivot để có bảng so sánh trực quan
    pivot_df = df.pivot_table(index=['Metric', 'Model'], columns='Training', values='Value')
    
    # Sắp xếp lại thứ tự metric cho dễ đọc
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
    
    # Định nghĩa các model cần đánh giá
    models_to_evaluate = [
        {
            'name': 'DINOv2-ViT-S/14',
            'config': 'configs/dinov2_vits14.yaml',
            'checkpoint': 'outputs/dinov2_vits14_ntxent/best_model2.pth'
        },
        {
            'name': 'ENT-ViT',
            'config': 'configs/ent-vit.yaml',
            'checkpoint': 'outputs/ent_vit_ntxent/best_model2.pth'
        },
        #         {
        #     'name': 'DINOv2-ViT-B/14',
        #     'config': 'configs/dinov2_vitb14.yaml',
        #     'checkpoint': 'outputs/dinov2_vitb14_ntxent/best_model2.pth'
        # },
        #         {
        #     'name': 'DINOv2-ViT-L/14',
        #     'config': 'configs/dinov2_vitl14.yaml',
        #     'checkpoint': 'outputs/dinov2_vitl14_ntxent/best_model2.pth'
        # },
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
        
        # Đánh giá model đã fine-tune
        print(f"🎯 Evaluating {model_name} (Fine-tuned)")
        finetuned_results = evaluate_model(
            config_path,
            checkpoint_path=checkpoint_path,
            model_name=f"{model_name} (Fine-tuned)"
        )
        
        # Đánh giá model pretrained (không load checkpoint)
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
    
    # Lưu kết quả
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n\n💾 All evaluation results saved to: {output_path}")
    
    # Tạo bảng tổng kết
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

