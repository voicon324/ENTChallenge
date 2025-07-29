#!/usr/bin/env python3
"""
Script to evaluate model on test set
Run: python eval.py
"""

import yaml
import torch
import wandb
from pathlib import Path
import argparse

from src.data_loader import create_dataloaders
from src.model_factory import build_model
from src.trainer import Trainer
from src.utils import set_seed, setup_logging

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Path to config file')
    parser.add_argument('--checkpoint', '-m', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', '-s', default='test',
                       choices=['test', 'val', 'train'],
                       help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path
    config['model']['pretrained_checkpoint'] = args.checkpoint
    
    # Setup
    setup_logging()
    set_seed(42)
    
    # Initialize W&B for evaluation
    run = wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=f"eval_{Path(args.checkpoint).stem}",
        config=config,
        job_type="evaluation",
        resume="allow"
    )
    
    cfg = wandb.config
    
    print(f"🔍 Evaluating model: {args.checkpoint}")
    print(f"📊 On split: {args.split}")
    print(f"📈 W&B URL: {run.url}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # Create dataloaders
    print("📂 Creating dataloaders...")
    backbone = cfg['model']['backbone']
    train_loader, val_loader, test_loader = create_dataloaders(cfg['data'], backbone)
    
    # Select evaluation loader
    eval_loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }[args.split]
    
    # Build model
    print("🏗️ Building model...")
    model = build_model(cfg['model'])
    
    # Create trainer for evaluation
    trainer = Trainer(model, train_loader, val_loader, test_loader, cfg, run)
    
    # Run evaluation
    print(f"🧪 Evaluating on {args.split} set...")
    if args.split == 'test':
        trainer.evaluate_test()
    else:
        # Custom evaluation for train/val splits
        trainer.model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                if len(batch) == 2:
                    images, targets = batch
                    images = images.to(trainer.device)
                    targets = targets.to(trainer.device)
                    
                    features = trainer.model.get_features(images)
                    
                    all_features.append(features.cpu())
                    all_labels.append(targets.cpu())
        
        if all_features:
            from src.utils import calculate_metrics
            
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate metrics
            metrics = calculate_metrics(
                all_features, 
                all_labels, 
                cfg['evaluation'].get('metrics', ['HitRate@10', 'MRR'])
            )
            
            # Log metrics
            log_data = {f"{args.split}_{k}": v for k, v in metrics.items()}
            run.log(log_data)
            
            # Print results
            print(f"📊 Results on {args.split} set:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
    
    print("✅ Evaluation completed!")
    run.finish()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
