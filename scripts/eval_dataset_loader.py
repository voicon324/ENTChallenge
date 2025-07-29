#!/usr/bin/env python3
"""
Script to load and use pre-created evaluation dataset.
Provides utility functions to load query, corpus, ground truth.
"""

import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import pandas as pd
from gemini_reranker import GeminiReranker


class EvaluationDataset:
    """
    Class to load and manage evaluation dataset.
    """
    
    def __init__(self, eval_data_dir: str):
        """
        Args:
            eval_data_dir: Path to directory containing evaluation dataset
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
        """Load JSON file from evaluation directory."""
        filepath = self.eval_data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_transform(self, normalize: bool = True) -> transforms.Compose:
        """
        Create transform to load images.
        
        Args:
            normalize: Whether to normalize or not
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
        Load all query images.
        
        Args:
            normalize: Whether to normalize or not
            
        Returns:
            Tuple[torch.Tensor, List[str]]: (image tensor, list of paths)
        """
        transform = self.get_transform(normalize)
        images = []
        paths = []
        
        for query_path in self.query_metadata['query_paths']:
            # Load image
            img_pil = Image.open(query_path).convert('RGB')
            img_tensor = transform(img_pil)
            images.append(img_tensor)
            paths.append(query_path)
        
        return torch.stack(images), paths
    
    def load_corpus_images(self, normalize: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """
        Load all corpus images.
        
        Args:
            normalize: Whether to normalize or not
            
        Returns:
            Tuple[torch.Tensor, List[str]]: (image tensor, list of paths)
        """
        transform = self.get_transform(normalize)
        images = []
        paths = []
        
        for corpus_path in self.corpus_metadata['corpus_paths']:
            # Load image
            img_pil = Image.open(corpus_path).convert('RGB')
            img_tensor = transform(img_pil)
            images.append(img_tensor)
            paths.append(corpus_path)
        
        return torch.stack(images), paths
    
    def get_ground_truth_mapping(self) -> Dict[int, Dict]:
        """
        Get ground truth mapping.
        
        Returns:
            Dict mapping from query index to ground truth info
        """
        return {
            int(k): v for k, v in 
            self.ground_truth_metadata['ground_truth_mapping'].items()
        }
    
    def get_query_ground_truth_indices(self, query_idx: int) -> List[int]:
        """
        Get ground truth indices for a query.
        
        Args:
            query_idx: Index of query
            
        Returns:
            List of ground truth indices in corpus
        """
        gt_mapping = self.get_ground_truth_mapping()
        return gt_mapping[query_idx]['ground_truth_indices']
    
    def evaluate_retrieval(self, query_features: torch.Tensor, 
                          corpus_features: torch.Tensor,
                          k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Evaluate retrieval with HitRate, MRR, Recall metrics.
        
        Args:
            query_features: Features of queries
            corpus_features: Features of corpus
            k_values: List of k values to evaluate
            
        Returns:
            Dict containing metrics
        """
        # Normalize features
        query_features = F.normalize(query_features, p=2, dim=1)
        corpus_features = F.normalize(corpus_features, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.mm(query_features, corpus_features.t())
        
        # Get ground truth mapping
        gt_mapping = self.get_ground_truth_mapping()
        
        results = {}
        n_queries = query_features.size(0)
        
        # Get maximum top-k for reuse
        max_k = max(k_values)
        _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)
        
        for k in k_values:
            # Get top-k for current k value
            top_k_indices = top_k_indices_all[:, :k]
            
            # --- Calculate HitRate@k ---
            hit_count = 0
            for i in range(n_queries):
                query_gt_indices = set(gt_mapping[i]['ground_truth_indices'])
                retrieved_indices = set(top_k_indices[i].tolist())
                
                if not query_gt_indices.isdisjoint(retrieved_indices):
                    hit_count += 1
            
            results[f"HitRate@{k}"] = hit_count / n_queries
            
            # --- Calculate MRR@k ---
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
            
            # --- Calculate Recall@k ---
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
    
    def evaluate_retrieval_with_reranking(self, query_features: torch.Tensor, 
                                          corpus_features: torch.Tensor,
                                          k_values: List[int] = [1, 5, 10],
                                          rerank_top_k: int = 10,
                                          use_gemini_reranking: bool = False,
                                          gemini_api_key: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate retrieval with optional reranking using Gemini LVLM.
        
        Args:
            query_features: Features of queries
            corpus_features: Features of corpus
            k_values: List of k values to evaluate
            rerank_top_k: Number of top results to rerank
            use_gemini_reranking: Whether to use Gemini reranking
            gemini_api_key: API key for Gemini (optional)
            
        Returns:
            Dict containing metrics
        """
        # Normalize features
        query_features = F.normalize(query_features, p=2, dim=1)
        corpus_features = F.normalize(corpus_features, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.mm(query_features, corpus_features.t())
        
        # Get ground truth mapping
        gt_mapping = self.get_ground_truth_mapping()
        
        results = {}
        n_queries = query_features.size(0)
        
        # Initialize reranker if needed
        reranker = None
        if use_gemini_reranking:
            try:
                reranker = GeminiReranker(api_key=gemini_api_key)
                print(f"🤖 Using Gemini reranking for top-{rerank_top_k} results")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini reranker: {e}")
                print("   Falling back to standard evaluation")
                use_gemini_reranking = False
        
        # Get maximum top-k for reuse
        max_k = max(k_values + [rerank_top_k]) if use_gemini_reranking else max(k_values)
        _, top_k_indices_all = torch.topk(similarity_matrix, max_k, dim=1)
        
        # Get initial scores for reranking
        top_k_scores_all = None
        if use_gemini_reranking:
            top_k_scores_all = torch.gather(similarity_matrix, 1, top_k_indices_all)
        
        for k in k_values:
            # Determine which indices to use
            if use_gemini_reranking and k <= rerank_top_k:
                # Use reranked results
                reranked_results = []
                
                for i in range(n_queries):
                    query_path = self.query_metadata['query_paths'][i]
                    
                    # Get top-rerank_top_k corpus paths and scores
                    top_indices = top_k_indices_all[i, :rerank_top_k].tolist()
                    top_scores = top_k_scores_all[i, :rerank_top_k].tolist()
                    top_corpus_paths = [self.corpus_metadata['corpus_paths'][idx] for idx in top_indices]
                    
                    try:
                        # Rerank using Gemini
                        reranked_indices, final_scores, lvlm_scores = reranker.rerank_top_k(
                            query_path, top_corpus_paths, top_scores, k=rerank_top_k
                        )
                        
                        # Map back to original corpus indices
                        reranked_corpus_indices = [top_indices[idx] for idx in reranked_indices]
                        reranked_results.append(reranked_corpus_indices[:k])
                        
                    except Exception as e:
                        print(f"   Error reranking query {i}: {e}")
                        # Fallback to original ranking
                        reranked_results.append(top_k_indices_all[i, :k].tolist())
                
                # Convert to tensor
                top_k_indices = torch.tensor(reranked_results)
                
            else:
                # Use original ranking
                top_k_indices = top_k_indices_all[:, :k]
            
            # --- Calculate HitRate@k ---
            hit_count = 0
            for i in range(n_queries):
                query_gt_indices = set(gt_mapping[i]['ground_truth_indices'])
                retrieved_indices = set(top_k_indices[i].tolist())
                
                if not query_gt_indices.isdisjoint(retrieved_indices):
                    hit_count += 1
            
            suffix = "_reranked" if use_gemini_reranking and k <= rerank_top_k else ""
            results[f"HitRate@{k}{suffix}"] = hit_count / n_queries
            
            # --- Calculate MRR@k ---
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
            
            results[f"MRR@{k}{suffix}"] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
            
            # --- Tính Recall@k ---
            recall_scores = []
            for i in range(n_queries):
                query_gt_indices = set(gt_mapping[i]['ground_truth_indices'])
                retrieved_indices = set(top_k_indices[i].tolist())
                
                found_gt = len(query_gt_indices.intersection(retrieved_indices))
                total_gt = len(query_gt_indices)
                
                recall = found_gt / total_gt if total_gt > 0 else 0.0
                recall_scores.append(recall)
            
            results[f"Recall@{k}{suffix}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        return results
    
    def create_results_dataframe(self, results: Dict[str, float]) -> pd.DataFrame:
        """
        Create DataFrame from evaluation results.
        
        Args:
            results: Results from evaluate_retrieval
            
        Returns:
            DataFrame containing results
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
        Print evaluation results.
        
        Args:
            results: Results from evaluate_retrieval
            title: Title
        """
        print(f"\n📊 {title}")
        print("=" * 50)
        
        # Group by metric type
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
        Get statistics about dataset.
        
        Returns:
            Dict containing statistics
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
        Get sample query with ground truth for verification.
        
        Args:
            query_idx: Index of query
            
        Returns:
            Dict containing query and ground truth information
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
    
    def rerank_with_gemini(self, query_features: torch.Tensor, 
                          corpus_features: torch.Tensor, 
                          top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Rerank retrieval results using Gemini Reranker.
        
        Args:
            query_features: Features of queries
            corpus_features: Features of corpus
            top_k_indices: Top-k indices from initial retrieval results
            
        Returns:
            Tensor containing reranked indices
        """
        # Convert to format suitable for Gemini
        query_features_gemini = query_features.detach().cpu().numpy()
        corpus_features_gemini = corpus_features.detach().cpu().numpy()
        top_k_indices_gemini = top_k_indices.detach().cpu().numpy()
        
        # Initialize Gemini Reranker
        reranker = GeminiReranker()
        
        # Perform reranking
        rerank_scores = reranker.rerank(
            query_features_gemini, 
            corpus_features_gemini, 
            top_k_indices_gemini
        )
        
        # Convert back to tensor and return
        return torch.tensor(rerank_scores).to(query_features.device)
    

def demo_usage():
    """
    Demo how to use EvaluationDataset.
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
    
    # Note: To perform actual evaluation, you need:
    # 1. Load model
    # 2. Extract features from query_images and corpus_images
    # 3. Call eval_dataset.evaluate_retrieval(query_features, corpus_features)


if __name__ == '__main__':
    demo_usage()
