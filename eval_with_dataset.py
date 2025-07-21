#!/usr/bin/env python3
"""
Script đánh giá model sử dụng bộ data evaluation đã tạo sẵn.
"""

import yaml
import torch
import argparse
import json
from pathlib import Path
import pandas as pd

from eval_dataset_loader import EvaluationDataset
from src.model_factory import build_model
from src.utils import set_seed, setup_logging


def extract_features_from_images(model, images, device, batch_size=32):
    """
    Trích xuất features từ tensor ảnh.
    
    Args:
        model: Model để trích xuất features
        images: Tensor ảnh (N, C, H, W)
        device: Device để chạy
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


def evaluate_model_with_dataset(config_path, eval_data_dir, checkpoint_path=None, model_name=""):
    """
    Đánh giá model sử dụng bộ data evaluation.
    
    Args:
        config_path: Đường dẫn config model
        eval_data_dir: Thư mục chứa bộ data evaluation
        checkpoint_path: Đường dẫn checkpoint (optional)
        model_name: Tên model
        
    Returns:
        Dict chứa kết quả evaluation
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    args = parser.parse_args()
    
    setup_logging()
    set_seed(42)
    
    print("🚀 Starting model evaluation...")
    print("=" * 60)
    print(f"📁 Config: {args.config}")
    print(f"📁 Eval data: {args.eval_data}")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"🏷️  Model name: {args.model_name}")
    print("=" * 60)
    
    try:
        # Run evaluation
        results = evaluate_model_with_dataset(
            config_path=args.config,
            eval_data_dir=args.eval_data,
            checkpoint_path=args.checkpoint,
            model_name=args.model_name
        )
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            evaluation_results = {
                'model_name': args.model_name,
                'config_path': args.config,
                'checkpoint_path': args.checkpoint,
                'eval_data_dir': args.eval_data,
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
