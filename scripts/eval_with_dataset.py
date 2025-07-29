#!/usr/bin/env python3
"""
Model evaluation script using pre-created evaluation dataset.
"""

import yaml
import torch
import argparse
import json
import os
from pathlib import Path
import pandas as pd

from eval_dataset_loader import EvaluationDataset
from src.model_factory import build_model
from src.utils import set_seed, setup_logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def extract_features_from_images(model, images, device, batch_size=32):
    """
    Extract features from image tensors.
    
    Args:
        model: Model to extract features
        images: Image tensor (N, C, H, W)
        device: Device to run on
        batch_size: Batch size
        
    Returns:
        Features tensor
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            features = model.get_features(batch)
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0)


def evaluate_model_with_dataset(config_path, eval_data_dir, checkpoint_path=None, model_name="", 
                               use_gemini_reranking=False, rerank_top_k=10, gemini_api_key=None):
    """
    Evaluate model using evaluation dataset.
    
    Args:
        config_path: Model config path
        eval_data_dir: Directory containing evaluation dataset
        checkpoint_path: Checkpoint path (optional)
        model_name: Model name
        use_gemini_reranking: Whether to use Gemini reranking
        rerank_top_k: Number of top results to rerank
        gemini_api_key: API key for Gemini (optional)
        
    Returns:
        Dict containing evaluation results
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"🔄 Loading model: {model_name}")
    model = build_model(config['model'])
    model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print("ℹ️ No checkpoint loaded. Using pretrained weights.")
    
    # Load evaluation dataset
    print(f"📊 Loading evaluation dataset from: {eval_data_dir}")
    eval_dataset = EvaluationDataset(eval_data_dir)
    
    # Load images
    print("📷 Loading query images...")
    query_images, query_paths = eval_dataset.load_query_images(normalize=True)
    
    print("📷 Loading corpus images...")
    corpus_images, corpus_paths = eval_dataset.load_corpus_images(normalize=True)
    
    print(f"   Query images: {query_images.shape}")
    print(f"   Corpus images: {corpus_images.shape}")
    
    # Extract features
    print("🔄 Extracting features from queries...")
    query_features = extract_features_from_images(model, query_images, device)
    
    print("🔄 Extracting features from corpus...")
    corpus_features = extract_features_from_images(model, corpus_images, device)
    
    print(f"   Query features: {query_features.shape}")
    print(f"   Corpus features: {corpus_features.shape}")
    
    # Evaluate
    print("📊 Evaluating retrieval performance...")
    if use_gemini_reranking:
        print(f"🤖 Using LVLM-based Dual-Score Fusion (LDSF) with Gemini reranking")
        results = eval_dataset.evaluate_retrieval_with_reranking(
            query_features, corpus_features,
            use_gemini_reranking=True,
            rerank_top_k=rerank_top_k,
            gemini_api_key=gemini_api_key
        )
    else:
        results = eval_dataset.evaluate_retrieval(query_features, corpus_features)
    
    # Print results
    eval_dataset.print_results(results, f"Results for {model_name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model using pre-created evaluation dataset')
    parser.add_argument('--config', '-c', required=True, help='Path to model config file')
    parser.add_argument('--eval_data', '-e', default='eval_data', help='Path to evaluation dataset directory')
    parser.add_argument('--checkpoint', '-ckpt', help='Path to model checkpoint (optional)')
    parser.add_argument('--model_name', '-n', default='Model', help='Model name for display')
    parser.add_argument('--output', '-o', help='Output JSON file for results (optional)')
    
    # LVLM reranking options
    parser.add_argument('--use_gemini_reranking', action='store_true', 
                       help='Use Gemini LVLM for reranking top-k results')
    parser.add_argument('--rerank_top_k', type=int, 
                       default=int(os.getenv('DEFAULT_RERANK_TOP_K', '10')),
                       help='Number of top results to rerank with Gemini (default: from .env or 10)')
    parser.add_argument('--gemini_api_key', type=str,
                       help='Google API key for Gemini (or set in .env file as GOOGLE_API_KEY)')
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    
    print("🚀 Starting model evaluation...")
    print("=" * 60)
    print(f"📁 Config: {args.config}")
    print(f"📁 Eval data: {args.eval_data}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"🏷️  Model name: {args.model_name}")
    if args.use_gemini_reranking:
        print(f"🤖 Gemini reranking: Enabled (top-{args.rerank_top_k})")
    else:
        print("🤖 Gemini reranking: Disabled")
    print("=" * 60)
    
    try:
        # Run evaluation
        results = evaluate_model_with_dataset(
            config_path=args.config,
            eval_data_dir=args.eval_data,
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            use_gemini_reranking=args.use_gemini_reranking,
            rerank_top_k=args.rerank_top_k,
            gemini_api_key=args.gemini_api_key
        )
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            evaluation_results = {
                'model_name': args.model_name,
                'config_path': args.config,
                'checkpoint_path': args.checkpoint,
                'eval_data_dir': args.eval_data,
                'gemini_reranking_enabled': args.use_gemini_reranking,
                'rerank_top_k': args.rerank_top_k if args.use_gemini_reranking else None,
                'results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {output_path}")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
