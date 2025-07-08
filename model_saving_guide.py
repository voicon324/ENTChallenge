#!/usr/bin/env python3
"""
Hướng dẫn sử dụng hệ thống lưu model theo run_name
"""

print("""
🎯 HƯỚNG DẪN SỬ DỤNG HỆ THỐNG LUU MODEL THEO RUN_NAME
====================================================

✅ CÁC THAY ĐỔI ĐÃ THỰC HIỆN:

1. 📁 Trainer tự động tạo thư mục theo run_name:
   - Thay vì lưu tất cả vào "outputs/"
   - Giờ lưu vào "outputs/{run_name}/"

2. 🏷️ Mỗi config có run_name duy nhất:
   - dinov2_vitl14.yaml  → "dinov2_vitl14_ntxent_200epochs"
   - dinov2_vitb14.yaml  → "dinov2_vitb14_ntxent_experiment"  
   - dinov2_vits14.yaml  → "dinov2_vits14_ntxent_experiment"
   - ent-vit.yaml        → "ent_vit_ntxent_experiment"

3. 📄 Các file model sẽ được lưu vào:
   outputs/{run_name}/
   ├── best_model.pth           # Model tốt nhất
   └── checkpoint_epoch_*.pth   # Checkpoint theo epoch

📚 CÁCH SỬ DỤNG:

1. Chạy experiment với config:
   python train.py --config configs/dinov2_vitl14.yaml
   
   → Model sẽ lưu vào: outputs/dinov2_vitl14_ntxent_200epochs/

2. Chạy experiment khác:
   python train.py --config configs/dinov2_vitb14.yaml
   
   → Model sẽ lưu vào: outputs/dinov2_vitb14_ntxent_experiment/

3. Không lo bị ghi đè model của experiment khác!

🔧 TÙY CHỈNH RUN_NAME:

Để tạo experiment mới, chỉnh sửa run_name trong config:

wandb:
  project: "Image_Retrieval_Experiments"  
  entity: "hokhanhduy-none"
  run_name: "ten_experiment_moi_cua_ban"  # ← Thay đổi ở đây

💡 GỢI Ý NAMING CONVENTION:

- {model}_{strategy}_{epochs}epochs
- {model}_{strategy}_{special_config}
- {model}_{dataset}_{timestamp}

Ví dụ:
- "dinov2_vitl14_ntxent_200epochs"
- "dinov2_vitb14_contrastive_experiment"  
- "ent_vit_finetune_20250109"

✅ LỢI ÍCH:

✓ Không ghi đè model của experiment khác
✓ Dễ dàng so sánh kết quả các run khác nhau
✓ Tự động tạo thư mục theo run_name
✓ Dễ backup và chia sẻ model
✓ Tương thích hoàn toàn với W&B tracking

🚀 BẮT ĐẦU TRAINING:

python train.py --config configs/dinov2_vitl14.yaml

Model sẽ được lưu vào: outputs/dinov2_vitl14_ntxent_200epochs/
W&B run: https://wandb.ai/hokhanhduy-none/Image_Retrieval_Experiments/runs/...

""")
