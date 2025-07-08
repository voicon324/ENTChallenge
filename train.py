#!/usr/bin/env python3
"""
Script chính để chạy training - RẤT GỌN
Chỉ cần chạy: python train.py
"""

import yaml
import wandb
import torch
from pathlib import Path

from src.data_loader import create_dataloaders
from src.model_factory import build_model
from src.trainer import Trainer
from src.utils import set_seed, setup_logging

def main():
    # 1. Tải cấu hình từ file YAML
    config_path = Path('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError("File config.yaml không tồn tại!")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 2. Setup logging và seed
    setup_logging()
    set_seed(42)
    
    # 3. Khởi tạo Weights & Biases
    # W&B sẽ tự động đọc config và lưu lại mọi thứ
    run = wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb'].get('run_name'),
        config=config,
        job_type="training",
        resume="allow"
    )
    
    # Lấy lại config từ W&B để đảm bảo tính nhất quán
    cfg = wandb.config
    
    print(f"🚀 Bắt đầu thực nghiệm với backbone: {cfg['model']['backbone']}")
    print(f"📊 Theo dõi tại: {run.url}")
    
    # 4. Kiểm tra GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Sử dụng device: {device}")
    
    # 5. Tạo DataLoaders
    print("📂 Tạo DataLoaders...")
    training_strategy = cfg['training']['strategy']
    backbone = cfg['model']['backbone']
    
    if training_strategy == 'contrastive':
        from src.data_loader import create_contrastive_dataloaders
        train_loader, val_loader = create_contrastive_dataloaders(cfg['data'], backbone)
        # For contrastive learning, we'll use the same val_loader for testing
        test_loader = val_loader
    elif training_strategy == 'ntxent':
        from src.data_loader import create_ntxent_dataloaders
        train_loader, val_loader = create_ntxent_dataloaders(cfg['data'], backbone)
        # For NT-Xent learning, we'll use the same val_loader for testing
        test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = create_dataloaders(cfg['data'], backbone)
    
    # 6. Xây dựng Model
    print("🏗️ Xây dựng Model...")
    model = build_model(cfg['model'])
    
    # 7. [W&B] Theo dõi model (gradients, network topology)
    if cfg['logging']['log_gradients']:
        wandb.watch(model, log='all', log_freq=cfg['logging']['log_frequency'])
    
    # 8. Khởi tạo Trainer và bắt đầu huấn luyện
    print("🎯 Khởi tạo Trainer...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, cfg, run)
    
    # 9. Thực hiện training dựa trên strategy
    if cfg['training']['strategy'] == "test_only":
        print("🔍 Chỉ thực hiện đánh giá...")
        trainer.evaluate_test()
    else:
        print("🔥 Bắt đầu huấn luyện...")
        trainer.train()
    
    # 10. Kết thúc run W&B
    print("✅ Hoàn thành!")
    run.finish()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Dừng training bởi người dùng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise
