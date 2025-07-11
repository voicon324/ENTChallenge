#!/usr/bin/env python3
"""
Script phân tích chi tiết kết quả và đề xuất cải thiện
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results(results_file='evaluation_results.json'):
    """Phân tích chi tiết kết quả đánh giá"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("🔍 PHÂN TÍCH CHI TIẾT KẾT QUẢ")
    print("=" * 80)
    
    # Tính toán độ giảm hiệu suất
    performance_analysis = []
    
    for model_name, model_results in results.items():
        if model_results['pretrained'] and model_results['finetuned']:
            pretrained = model_results['pretrained']
            finetuned = model_results['finetuned']
            
            for metric in pretrained.keys():
                pretrained_score = pretrained[metric]
                finetuned_score = finetuned[metric]
                
                # Tính % thay đổi
                change = finetuned_score - pretrained_score
                change_pct = (change / pretrained_score) * 100
                
                performance_analysis.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Pretrained': pretrained_score,
                    'Finetuned': finetuned_score,
                    'Change': change,
                    'Change %': change_pct
                })
    
    df = pd.DataFrame(performance_analysis)
    
    # Hiển thị kết quả theo từng metric
    metrics = ['HitRate@1', 'HitRate@5', 'HitRate@10', 'MRR@1', 'MRR@5', 'MRR@10']
    
    for metric in metrics:
        print(f"\n📊 {metric} - Performance Change:")
        print("-" * 60)
        
        metric_df = df[df['Metric'] == metric].copy()
        metric_df = metric_df.sort_values('Change %', ascending=False)
        
        for _, row in metric_df.iterrows():
            change_symbol = "📈" if row['Change %'] > 0 else "📉"
            print(f"{change_symbol} {row['Model']:20} | "
                  f"Pre: {row['Pretrained']:.4f} | "
                  f"Fine: {row['Finetuned']:.4f} | "
                  f"Change: {row['Change %']:+6.2f}%")
    
    # Thống kê tổng quan
    print(f"\n📈 THỐNG KÊ TỔNG QUAN:")
    print("=" * 80)
    
    avg_change = df['Change %'].mean()
    worst_change = df['Change %'].min()
    best_change = df['Change %'].max()
    
    print(f"📊 Thay đổi trung bình: {avg_change:.2f}%")
    print(f"📉 Thay đổi tệ nhất: {worst_change:.2f}%")
    print(f"📈 Thay đổi tốt nhất: {best_change:.2f}%")
    
    # Phân tích theo model
    print(f"\n🤖 PHÂN TÍCH THEO MODEL:")
    print("=" * 80)
    
    model_avg = df.groupby('Model')['Change %'].mean().sort_values(ascending=False)
    for model, avg_change in model_avg.items():
        symbol = "📈" if avg_change > 0 else "📉"
        print(f"{symbol} {model:20} | Average change: {avg_change:+6.2f}%")
    
    # Nhận xét và đề xuất
    print(f"\n💡 NHẬN XÉT VÀ ĐỀ XUẤT:")
    print("=" * 80)
    
    suggestions = []
    
    if avg_change < -10:
        suggestions.append("❌ Fine-tuning đang làm giảm hiệu suất đáng kể")
        suggestions.append("🔧 Đề xuất: Giảm learning rate (VD: 1e-5 → 1e-6)")
        suggestions.append("🔧 Đề xuất: Sử dụng frozen backbone + chỉ train classification head")
        suggestions.append("🔧 Đề xuất: Áp dụng gradual unfreezing")
    
    if worst_change < -30:
        suggestions.append("⚠️ Có model bị catastrophic forgetting nghiêm trọng")
        suggestions.append("🔧 Đề xuất: Sử dụng smaller learning rate cho backbone")
        suggestions.append("🔧 Đề xuất: Thêm regularization (weight decay, dropout)")
    
    # Kiểm tra model nào ổn định nhất
    stability_scores = df.groupby('Model')['Change %'].std()
    most_stable = stability_scores.idxmin()
    suggestions.append(f"🏆 Model ổn định nhất: {most_stable}")
    
    for suggestion in suggestions:
        print(suggestion)
    
    return df

def create_improvement_recommendations():
    """Tạo đề xuất cải thiện cụ thể"""
    
    print(f"\n🚀 ĐỀ XUẤT CẢI THIỆN CỤ THỂ:")
    print("=" * 80)
    
    recommendations = [
        {
            "category": "Learning Rate Strategy",
            "suggestions": [
                "Sử dụng learning rate scheduler (cosine annealing)",
                "Differential learning rates: backbone (1e-6), head (1e-4)",
                "Warmup learning rate trong 10% epochs đầu"
            ]
        },
        {
            "category": "Training Strategy", 
            "suggestions": [
                "Frozen backbone training: freeze backbone, chỉ train head",
                "Gradual unfreezing: unfreeze từng layer một",
                "Data augmentation mạnh hơn để tăng diversity"
            ]
        },
        {
            "category": "Regularization",
            "suggestions": [
                "Weight decay: 0.01 → 0.001",
                "Dropout trong classification head",
                "Label smoothing với α=0.1"
            ]
        },
        {
            "category": "Loss Function",
            "suggestions": [
                "Thử ContrastiveLoss thay vì NT-Xent",
                "Triplet loss với hard negative mining",
                "Focal loss để xử lý class imbalance"
            ]
        },
        {
            "category": "Data Strategy",
            "suggestions": [
                "Tăng kích thước dataset nếu có thể",
                "Cross-validation để đánh giá robust",
                "Synthetic data generation từ pretrained model"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n📋 {rec['category']}:")
        print("-" * 40)
        for i, suggestion in enumerate(rec['suggestions'], 1):
            print(f"  {i}. {suggestion}")

def generate_training_configs():
    """Tạo config files cải thiện"""
    
    print(f"\n⚙️ TẠO CONFIG FILES CẢI THIỆN:")
    print("=" * 80)
    
    # Config với frozen backbone
    frozen_config = {
        "model": {
            "freeze_backbone": True,
            "backbone_lr": 0.0,
            "head_lr": 1e-4
        },
        "training": {
            "epochs": 50,
            "warmup_epochs": 5,
            "weight_decay": 0.001,
            "label_smoothing": 0.1
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.999]
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": 50
        }
    }
    
    # Config với gradual unfreezing
    gradual_config = {
        "model": {
            "gradual_unfreezing": True,
            "unfreeze_schedule": [10, 20, 30],  # epochs to unfreeze layers
            "backbone_lr": 1e-6,
            "head_lr": 1e-4
        },
        "training": {
            "epochs": 100,
            "warmup_epochs": 10,
            "weight_decay": 0.01,
            "dropout": 0.1
        }
    }
    
    configs = {
        "frozen_backbone": frozen_config,
        "gradual_unfreezing": gradual_config
    }
    
    for name, config in configs.items():
        config_path = f"configs/improved_{name}.yaml"
        print(f"📄 Generated: {config_path}")
        
        # Tạo YAML content
        yaml_content = f"""# Improved training config - {name}
model:
  backbone: dino_v2
  variant: dinov2_vits14
  num_classes: 4
  feature_dim: 768
  freeze_backbone: {config['model'].get('freeze_backbone', False)}
  
training:
  epochs: {config['training']['epochs']}
  batch_size: 32
  warmup_epochs: {config['training']['warmup_epochs']}
  weight_decay: {config['training']['weight_decay']}
  
optimizer:
  type: AdamW
  backbone_lr: {config['model'].get('backbone_lr', 1e-5)}
  head_lr: {config['model'].get('head_lr', 1e-4)}
  
scheduler:
  type: CosineAnnealingLR
  T_max: {config['training']['epochs']}
  
loss:
  type: ntxent
  temperature: 0.1
  
data:
  path: "data/processed"
  image_size: 224
  normalize: true
  augmentation:
    horizontal_flip: 0.5
    rotation: 15
    color_jitter: 0.2
"""
        
        with open(config_path, 'w') as f:
            f.write(yaml_content)

if __name__ == '__main__':
    # Phân tích kết quả hiện tại
    df = analyze_results()
    
    # Tạo đề xuất cải thiện
    create_improvement_recommendations()
    
    # Tạo config files cải thiện
    generate_training_configs()
    
    print(f"\n✅ Phân tích hoàn tất! Đã tạo config files cải thiện.")
