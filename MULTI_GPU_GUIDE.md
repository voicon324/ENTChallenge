# Multi-GPU Training Guide

## 🚀 Cấu hình Multi-GPU Training

### Kiểm tra hệ thống
```bash
# Kiểm tra GPU và khả năng multi-GPU
python check_gpu.py
```

### Chạy training với multi-GPU

#### 1. Chạy config cụ thể
```bash
# Chạy DinoV2 ViT-B/14
bash train_multi_gpu.sh dinov2_vitb14

# Chạy DinoV2 ViT-L/14
bash train_multi_gpu.sh dinov2_vitl14

# Chạy DinoV2 ViT-S/14
bash train_multi_gpu.sh dinov2_vits14

# Chạy ENT-ViT
bash train_multi_gpu.sh ent-vit
```

#### 2. Chạy tất cả config
```bash
# Chạy tất cả config lần lượt
bash train_multi_gpu.sh all
```

#### 3. Chạy interactive mode
```bash
# Chạy và chọn config
bash train_multi_gpu.sh
```

## ⚙️ Cấu hình đã tối ưu cho Multi-GPU

### DinoV2 ViT-B/14
- **Batch size**: 16 (8 per GPU)
- **Learning rate**: 0.0002
- **Num workers**: 8
- **Features**: 768D

### DinoV2 ViT-L/14
- **Batch size**: 12 (6 per GPU)
- **Learning rate**: 0.00015
- **Num workers**: 8
- **Features**: 1024D

### DinoV2 ViT-S/14
- **Batch size**: 20 (10 per GPU)
- **Learning rate**: 0.00025
- **Num workers**: 8
- **Features**: 384D

### ENT-ViT
- **Batch size**: 16 (8 per GPU)
- **Learning rate**: 0.0002
- **Num workers**: 8
- **Features**: 768D

## 🔧 Tính năng Multi-GPU

### 1. DataParallel
- Tự động chia batch cho các GPU
- Synchronize gradients
- Load balancing

### 2. Synchronized Batch Normalization
- Đồng bộ statistics giữa các GPU
- Cải thiện stability cho training

### 3. Automatic batch size adjustment
- Tự động chia batch size cho các GPU
- Giữ nguyên total batch size

## 📊 Monitoring

### Weights & Biases
- Tự động log metrics
- Model gradients tracking
- Real-time visualization

### GPU Utilization
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 🚨 Troubleshooting

### CUDA Out of Memory
```bash
# Giảm batch size trong config
batch_size: 8  # Thay vì 16
```

### Single GPU fallback
Nếu chỉ có 1 GPU, script sẽ tự động chuyển sang single GPU mode.

### CPU fallback
Nếu không có GPU, script sẽ chạy trên CPU (chậm hơn).

## 📝 Logs và Outputs

### Cấu trúc thư mục
```
outputs/
├── dinov2_vitb14_ntxent/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── dinov2_vitl14_ntxent/
├── dinov2_vits14_ntxent/
└── ent_vit_ntxent/
```

### Training logs
- Console output với progress bars
- File logs: `training.log`
- W&B dashboard: Real-time metrics

## 🔍 Performance Tips

1. **Batch size**: Tăng batch size để tận dụng multi-GPU
2. **Learning rate**: Tăng learning rate tương ứng với số GPU
3. **Num workers**: Tăng num_workers cho data loading
4. **SyncBN**: Bật synchronized batch norm cho model lớn
5. **Memory**: Monitor GPU memory usage
