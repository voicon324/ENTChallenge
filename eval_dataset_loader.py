#!/usr/bin/env python3
"""
Script để load và sử dụng bộ data evaluation đã tạo.
Cung cấp các hàm tiện ích để load query, corpus, ground truth.
"""

import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import pandas as pd


class EvaluationDataset:
    """
    Class để load và quản lý bộ data evaluation.
    """
    
    def __init__(self, eval_data_dir: str):
        """
        Args:
            eval_data_dir: Đường dẫn đến thư mục chứa bộ data evaluation
        """
        self.eval_data_dir = Path(eval_data_dir)
        
        # Load metadata
        self.summary = self._load_json('summary.json')
        self.corpus_metadata = self._load_json('corpus_metadata.json')
        self.query_metadata = self._load_json('query_metadata.json')
        self.ground_truth_metadata = self._load_json('ground_truth.json')
        
        # Paths
        self.query_dir = self.eval_data_dir / 'query'
        self.corpus_dir = self.eval_data_dir / 'corpus'
        
        # Basic info
        self.backbone = self.summary['backbone']
        self.num_augmentations = self.summary['num_augmentations']
        
        print(f"📊 Loaded evaluation dataset:")
        print(f"   - Queries: {len(self.query_metadata['query_paths'])}")
        print(f"   - Corpus: {len(self.corpus_metadata['corpus_paths'])}")
        print(f"   - Backbone: {self.backbone}")
        print(f"   - Augmentations per query: {self.num_augmentations}")
    
    def _load_json(self, filename: str) -> dict:
        """Load JSON file từ thư mục evaluation."""
        filepath = self.eval_data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_transform(self, normalize: bool = True) -> transforms.Compose:
        """
        Tạo transform để load ảnh.
        
        Args:
            normalize: Có normalize hay không
        """
        if self.backbone == 'ent_vit':
            mean = [0.3464, 0.2280, 0.2228]
            std = [0.2520, 0.2128, 0.2093]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        transform_list = [transforms.ToTensor()]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def load_query_images(self, normalize: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """
        Load tất cả ảnh query.
        
        Args:
            normalize: Có normalize hay không
            
        Returns:
            Tuple[torch.Tensor, List[str]]: (tensor ảnh, danh sách paths)
        """
        transform = self.get_transform(normalize)
        images = []
        paths = []
        
        for query_path in self.query_metadata['query_paths']:
            # Load ảnh
            img_pil = Image.open(query_path).convert('RGB')
            img_tensor = transform(img_pil)
            images.append(img_tensor)
            paths.append(query_path)
        
        return torch.stack(images), paths
    
    def load_corpus_images(self, normalize: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """
        Load tất cả ảnh corpus.
        
        Args:
            normalize: Có normalize hay không
            
        Returns:
            Tuple[torch.Tensor, List[str]]: (tensor ảnh, danh sách paths)
        """
        transform = self.get_transform(normalize)
        images = []
        paths = []
        
        for corpus_path in self.corpus_metadata['corpus_paths']:
            # Load ảnh
            img_pil = Image.open(corpus_path).convert('RGB')
            img_tensor = transform(img_pil)
            images.append(img_tensor)
            paths.append(corpus_path)
        
        return torch.stack(images), paths
    
    def get_ground_truth_mapping(self) -> Dict[int, Dict]:
        """
        Lấy ground truth mapping.
        
        Returns:
            Dict mapping từ query index đến ground truth info
        """
        return {
            int(k): v for k, v in 
            self.ground_truth_metadata['ground_truth_mapping'].items()
        }
    
    def get_query_ground_truth_indices(self, query_idx: int) -> List[int]:
        """
        Lấy indices của ground truth cho một query.
        
        Args:
            query_idx: Index của query
            
        Returns:
            List indices của ground truth trong corpus
        """
        gt_mapping = self.get_ground_truth_mapping()
        return gt_mapping[query_idx]['ground_truth_indices']
    
    def evaluate_retrieval(self, query_features: torch.Tensor, 
                          corpus_features: torch.Tensor,
                          k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Đánh giá retrieval với các metric HitRate, MRR, Recall.
        
        Args:
            query_features: Features của queries
            corpus_features: Features của corpus
            k_values: Danh sách k values để đánh giá
            
        Returns:
            Dict chứa các metric
        """
        # Chuẩn hóa features
        query_features = F.normalize(query_features, p=2, dim=1)
        corpus_features = F.normalize(corpus_features, p=2, dim=1)
        
        # Tính similarity matrix
        similarity_matrix = torch.mm(query_features, corpus_features.t())
        
        # Lấy ground truth mapping
        gt_mapping = self.get_ground_truth_mapping()
        
        results = {}
        n_queries = query_features.size(0)
        
        # Lấy top-k lớn nhất để tái sử dụng
        max_k = max(k_values)
        _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)
        
        for k in k_values:
            # Lấy top-k cho giá trị k hiện tại
            top_k_indices = top_k_indices_all[:, :k]
            
            # --- Tính HitRate@k ---
            hit_count = 0
            for i in range(n_queries):
                query_gt_indices = set(gt_mapping[i]['ground_truth_indices'])
                retrieved_indices = set(top_k_indices[i].tolist())
                
                if not query_gt_indices.isdisjoint(retrieved_indices):
                    hit_count += 1
            
            results[f"HitRate@{k}"] = hit_count / n_queries
            
            # --- Tính MRR@k ---
            reciprocal_ranks = []
            for i in range(n_queries):
                query_gt_indices = gt_mapping[i]['ground_truth_indices']
                best_rank = float('inf')
                
                for rank, retrieved_idx in enumerate(top_k_indices[i].tolist()):
                    if retrieved_idx in query_gt_indices:
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
                query_gt_indices = set(gt_mapping[i]['ground_truth_indices'])
                retrieved_indices = set(top_k_indices[i].tolist())
                
                found_gt = len(query_gt_indices.intersection(retrieved_indices))
                total_gt = len(query_gt_indices)
                
                recall = found_gt / total_gt if total_gt > 0 else 0.0
                recall_scores.append(recall)
            
            results[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        return results
    
    def create_results_dataframe(self, results: Dict[str, float]) -> pd.DataFrame:
        """
        Tạo DataFrame từ kết quả evaluation.
        
        Args:
            results: Kết quả từ evaluate_retrieval
            
        Returns:
            DataFrame chứa kết quả
        """
        rows = []
        for metric, value in results.items():
            metric_type, k = metric.split('@')
            rows.append({
                'Metric': metric_type,
                'K': int(k),
                'Value': value
            })
        
        df = pd.DataFrame(rows)
        return df.pivot(index='Metric', columns='K', values='Value')
    
    def print_results(self, results: Dict[str, float], title: str = "Evaluation Results"):
        """
        In kết quả evaluation.
        
        Args:
            results: Kết quả từ evaluate_retrieval
            title: Tiêu đề
        """
        print(f"\n📊 {title}")
        print("=" * 50)
        
        # Group theo metric type
        hitrate_results = {k: v for k, v in results.items() if k.startswith('HitRate')}
        mrr_results = {k: v for k, v in results.items() if k.startswith('MRR')}
        recall_results = {k: v for k, v in results.items() if k.startswith('Recall')}
        
        print("📈 Hit Rate:")
        for metric, value in hitrate_results.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📈 MRR:")
        for metric, value in mrr_results.items():
            print(f"   {metric}: {value:.4f}")
        
        print("\n📈 Recall:")
        for metric, value in recall_results.items():
            print(f"   {metric}: {value:.4f}")
        
        print("=" * 50)
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê về dataset.
        
        Returns:
            Dict chứa thống kê
        """
        return {
            'total_queries': len(self.query_metadata['query_paths']),
            'total_corpus': len(self.corpus_metadata['corpus_paths']),
            'corpus_composition': self.corpus_metadata['corpus_composition'],
            'num_augmentations': self.num_augmentations,
            'backbone': self.backbone,
            'test_augmented_start_index': self.ground_truth_metadata['test_augmented_start_index']
        }
    
    def sample_query_with_ground_truth(self, query_idx: int = 0) -> Dict:
        """
        Lấy sample một query với ground truth để kiểm tra.
        
        Args:
            query_idx: Index của query
            
        Returns:
            Dict chứa thông tin query và ground truth
        """
        gt_mapping = self.get_ground_truth_mapping()
        
        if query_idx not in gt_mapping:
            raise ValueError(f"Query index {query_idx} not found")
        
        query_info = gt_mapping[query_idx]
        
        return {
            'query_index': query_idx,
            'query_path': query_info['query_path'],
            'ground_truth_indices': query_info['ground_truth_indices'],
            'ground_truth_paths': query_info['ground_truth_paths'],
            'num_ground_truth': len(query_info['ground_truth_indices'])
        }


def demo_usage():
    """
    Demo cách sử dụng EvaluationDataset.
    """
    # Load dataset
    eval_dataset = EvaluationDataset('eval_data')
    
    # Print statistics
    stats = eval_dataset.get_statistics()
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Sample query
    sample = eval_dataset.sample_query_with_ground_truth(0)
    print(f"\n📷 Sample Query {sample['query_index']}:")
    print(f"   Query path: {sample['query_path']}")
    print(f"   Ground truth indices: {sample['ground_truth_indices']}")
    print(f"   Number of ground truth: {sample['num_ground_truth']}")
    
    # Load a few images for testing
    print("\n🔄 Loading sample images...")
    query_images, query_paths = eval_dataset.load_query_images(normalize=False)
    corpus_images, corpus_paths = eval_dataset.load_corpus_images(normalize=False)
    
    print(f"   Query images shape: {query_images.shape}")
    print(f"   Corpus images shape: {corpus_images.shape}")
    
    # Note: Để thực hiện evaluation thực tế, bạn cần:
    # 1. Load model
    # 2. Extract features từ query_images và corpus_images
    # 3. Gọi eval_dataset.evaluate_retrieval(query_features, corpus_features)


if __name__ == '__main__':
    demo_usage()
