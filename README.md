# 🎯 Image Retrieval Project

Một hệ thống image retrieval đơn giản nhưng mạnh mẽ với **YAML config** và tích hợp **Weights & Biases (W&B)**.

## 🏗️ Triết lý Thiết kế

> **Mọi thứ được điều khiển từ một file config, mã nguồn được chia để dễ đọc, và kết quả được theo dõi tự động.**

## 📁 Cấu trúc Project

```
image_retrieval_project/
├── config.yaml                # <<< File cấu hình DUY NHẤT cho mọi thứ
│
├── src/
│   ├── data_loader.py         # Class Dataset và hàm tạo DataLoader
│   ├── model_factory.py       # "Nhà máy" tạo ra model (DinoV2, EntVit)
│   ├── trainer.py             # Class Trainer chứa logic train/val và log W&B
│   ├── utils.py               # Các hàm tiện ích: grad_cam, lưu checkpoint...
│   └── models/
│       ├── dino_v2.py         # DinoV2 implementation
│       └── ent_vit.py         # EntVit implementation
│
├── train.py                   # <<< Script chính để chạy, rất GỌN
├── eval.py                    # Script để đánh giá trên tập test
├── prepare_data.py            # Script chuẩn bị dữ liệu
│
├── requirements.txt           # Thư viện cần cài
├── outputs/                   # Checkpoints và kết quả
├── data/                      # Dữ liệu training
└── README.md                  # Hướng dẫn này
```

## 🚀 Cài đặt & Khởi chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Đăng nhập W&B

```bash
wandb login
```

### 3. Chuẩn bị dữ liệu

```bash
# Tạo dữ liệu mẫu để test
python prepare_data.py --create-sample --num-classes 10 --images-per-class 50

# Hoặc phân tích dữ liệu có sẵn
python prepare_data.py --analyze --data-dir path/to/your/data
```

### 4. Cấu hình thực nghiệm

Chỉnh sửa file `config.yaml`:

```yaml
# Thay đổi backbone
model:
  backbone: "dino_v2"  # hoặc "ent_vit"

# Thay đổi strategy
training:
  strategy: "contrastive"  # hoặc "test_only"
```

### 5. Chạy training

```bash
python train.py
```

## 📊 Các Thực nghiệm Mẫu

### Thực nghiệm 1: DinoV2 với Contrastive Learning

```yaml
model:
  backbone: "dino_v2"
training:
  strategy: "contrastive"
  epochs: 50
```

### Thực nghiệm 2: EntVit với Contrastive Learning

```yaml
model:
  backbone: "ent_vit"
training:
  strategy: "contrastive"
  epochs: 50
```

### Thực nghiệm 3: Đánh giá Model có sẵn

```yaml
model:
  backbone: "dino_v2"
  pretrained_checkpoint: "outputs/best_model.pth"
training:
  strategy: "test_only"
```

## 🔧 Các Scripts Hỗ trợ

### Chuẩn bị dữ liệu

```bash
# Tạo dữ liệu mẫu
python prepare_data.py --create-sample

# Phân tích dataset
python prepare_data.py --analyze --data-dir data/processed

# Kiểm tra tính hợp lệ
python prepare_data.py --validate --data-dir data/processed

# Tạo annotation files
python prepare_data.py --create-annotations --data-dir data/processed
```

### Đánh giá model

```bash
# Đánh giá trên test set
python eval.py --checkpoint outputs/best_model.pth

# Đánh giá trên validation set
python eval.py --checkpoint outputs/best_model.pth --split val
```

## 📈 Theo dõi với W&B

Sau khi chạy training, bạn sẽ thấy:

1. **Dashboard trực quan** với:
   - Loss curves (train/val)
   - Metrics: HitRate@k, MRR, mAP
   - Grad-CAM visualizations
   - Embedding space visualization

2. **Config tracking**: Mọi thay đổi trong `config.yaml` đều được lưu trữ

3. **Model artifacts**: Checkpoints tự động được upload

## 🎛️ Cấu hình Chi tiết

### Model Configuration

```yaml
model:
  backbone: "dino_v2"           # "dino_v2" hoặc "ent_vit"
  pretrained_checkpoint: null   # Đường dẫn checkpoint
  feature_dim: 768              # Dimension của feature embeddings
  num_classes: 1000             # Số classes
  dropout: 0.1                  # Dropout rate
  freeze_backbone: false        # Đóng băng backbone
```

### Training Configuration

```yaml
training:
  strategy: "contrastive"       # "contrastive", "test_only"
  epochs: 50                    # Số epochs
  optimizer: "AdamW"            # "AdamW", "Adam", "SGD"
  learning_rate: 0.0001         # Learning rate
  weight_decay: 0.01            # Weight decay
  scheduler: "cosine"           # "cosine", "step"
  warmup_epochs: 5              # Warmup epochs
  early_stopping_patience: 10   # Early stopping patience
  
  # Contrastive learning specific
  temperature: 0.07             # Temperature for contrastive loss
  margin: 0.5                   # Margin for contrastive loss
```

### Evaluation Configuration

```yaml
evaluation:
  metrics: ["HitRate@1", "HitRate@5", "HitRate@10", "MRR", "mAP"]
  log_grad_cam: true            # Log Grad-CAM visualizations
  grad_cam_samples: 5           # Number of Grad-CAM samples
  save_embeddings: true         # Save embedding visualizations
```

## 🔍 Metrics Giải thích

- **HitRate@k**: Tỷ lệ query có ít nhất 1 kết quả đúng trong top-k
- **MRR** (Mean Reciprocal Rank): Trung bình nghịch đảo của rank của kết quả đúng đầu tiên
- **mAP** (mean Average Precision): Trung bình của Average Precision cho tất cả queries

## 📝 Cấu trúc Dữ liệu

### Directory Structure

```
data/processed/
├── train/
│   ├── class_01/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── class_02/
│   └── ...
├── val/
└── test/
```

### Annotation Format (Optional)

```json
[
  {
    "image_path": "train/class_01/img_001.jpg",
    "label": "class_01",
    "class_id": 1
  }
]
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Giảm `batch_size` trong config
   - Giảm `image_size`

2. **W&B authentication**:
   ```bash
   wandb login
   ```

3. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Data not found**:
   - Kiểm tra đường dẫn trong config
   - Chạy `python prepare_data.py --validate`

## 🔄 Workflow Điển hình

1. **Setup**: `pip install -r requirements.txt && wandb login`
2. **Data**: `python prepare_data.py --create-sample`
3. **Config**: Chỉnh sửa `config.yaml`
4. **Train**: `python train.py`
5. **Analyze**: Mở W&B dashboard để phân tích
6. **Evaluate**: `python eval.py --checkpoint outputs/best_model.pth`

## 🎯 Mở rộng

### Thêm backbone mới

1. Tạo file trong `src/models/`
2. Thêm vào `src/model_factory.py`
3. Cập nhật config options

### Thêm loss function mới

1. Implement trong `src/trainer.py`
2. Thêm vào `_setup_loss_function()`
3. Cập nhật training strategies

## 🏥 EndoViT Integration

This project includes full support for **EndoViT** (Endoscopic Vision Transformer), a specialized model for medical endoscopic image analysis.

### EndoViT Features

- **Pre-trained weights**: Automatically downloads from HuggingFace (egeozsoy/EndoViT)
- **Specialized normalization**: Uses endoscopy-specific normalization parameters
  - Mean: `[0.3464, 0.2280, 0.2228]`
  - Std: `[0.2520, 0.2128, 0.2093]`
- **Architecture**: Vision Transformer with patch size 16, 768 embedding dimension
- **Optimized for medical images**: Trained specifically on endoscopic datasets

### Using EndoViT

1. **Update config.yaml**:
```yaml
model:
  backbone: "ent_vit"  # Switch to EndoViT
  feature_dim: 768
  freeze_backbone: false  # Set to true for feature extraction only
```

2. **Test the implementation**:
```bash
python test_entvit.py
```

3. **Run training**:
```bash
python train.py
```

### EndoViT vs DinoV2

| Feature | EndoViT | DinoV2 |
|---------|---------|--------|
| **Domain** | Medical/Endoscopic | General Vision |
| **Normalization** | Endoscopy-specific | ImageNet |
| **Pre-training** | Endoscopic datasets | Internet images |
| **Use case** | Medical image analysis | General image tasks |

## 📧 Liên hệ & Đóng góp

- Issues: Tạo issue trên GitHub
- Features: Submit pull request
- Questions: Liên hệ qua email

---

**Happy Training! 🚀**
