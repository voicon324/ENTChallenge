# Evaluation Dataset Creation and Usage

Bộ script này giúp tạo và sử dụng bộ data evaluation cho image retrieval task.

## Tổng quan

Hệ thống bao gồm các thành phần chính:

1. **create_eval_dataset.py** - Tạo bộ data evaluation
2. **eval_dataset_loader.py** - Class helper để load và sử dụng bộ data  
3. **eval_with_dataset.py** - Đánh giá một model duy nhất
4. **compare_models_with_dataset.py** - So sánh nhiều model

## Cấu trúc bộ data evaluation

```
eval_data/
├── query/                      # Ảnh test gốc (queries)
│   ├── query_00000_class0.jpg
│   ├── query_00001_class1.jpg
│   └── ...
├── corpus/                     # Toàn bộ ảnh trong corpus
│   ├── train_orig_00000_class0.jpg      # Ảnh train gốc
│   ├── val_orig_00000_class1.jpg        # Ảnh val gốc
│   ├── train_aug_00000_aug00_class0.jpg # Ảnh train augmented
│   ├── val_aug_00000_aug00_class1.jpg   # Ảnh val augmented
│   ├── test_aug_00000_aug00_class0.jpg  # Ảnh test augmented (ground truth)
│   └── ...
├── corpus_metadata.json        # Thông tin về corpus
├── query_metadata.json         # Thông tin về queries
├── ground_truth.json           # Mapping query → ground truth
└── summary.json                # Tổng kết thông tin
```

## Cách sử dụng

### 1. Tạo bộ data evaluation

```bash
python create_eval_dataset.py \
    --config configs/dinov2_vits14.yaml \
    --output eval_data \
    --num_augmentations 5
```

**Tham số:**
- `--config`: Đường dẫn config model (để xác định backbone và normalization)
- `--output`: Thư mục đích lưu bộ data
- `--num_augmentations`: Số augmentation cho mỗi ảnh test (ground truth)

### 2. Đánh giá một model

```bash
python eval_with_dataset.py \
    --config configs/dinov2_vits14.yaml \
    --eval_data eval_data \
    --checkpoint outputs/dinov2_vits14_ntxent/best_model2.pth \
    --model_name "DINOv2-ViT-S/14" \
    --output results_dinov2_vits14.json
```

**Tham số:**
- `--config`: Config model
- `--eval_data`: Thư mục chứa bộ data evaluation
- `--checkpoint`: Checkpoint model (optional, nếu không có sẽ dùng pretrained)
- `--model_name`: Tên model để hiển thị
- `--output`: File JSON lưu kết quả (optional)

### 3. So sánh nhiều model

```bash
python compare_models_with_dataset.py \
    --eval_data eval_data \
    --output model_comparison_results.json
```

**Tham số:**
- `--eval_data`: Thư mục chứa bộ data evaluation
- `--output`: File JSON lưu kết quả so sánh

## Quy trình đánh giá

### Query và Corpus
- **Query**: Các ảnh gốc từ tập test
- **Corpus**: Bao gồm:
  - Ảnh gốc từ tập train
  - Ảnh gốc từ tập val  
  - Ảnh augmented từ tập train (1 augment/ảnh)
  - Ảnh augmented từ tập val (1 augment/ảnh)
  - Ảnh augmented từ tập test (N augments/ảnh - đây là ground truth)

### Ground Truth
- Với mỗi query (ảnh test gốc), ground truth là N phiên bản augmented tương ứng trong corpus
- Mapping được lưu trong `ground_truth.json`

### Metrics
- **HitRate@k**: Tỷ lệ query tìm thấy ít nhất 1 ground truth trong top-k
- **MRR@k**: Mean Reciprocal Rank của ground truth đầu tiên trong top-k
- **Recall@k**: Tỷ lệ trung bình ground truth được tìm thấy trong top-k

## Ưu điểm của approach này

1. **Tiết kiệm thời gian**: Tạo ảnh augmented một lần, tái sử dụng nhiều lần
2. **Nhất quán**: Tất cả model đều đánh giá trên cùng một bộ data
3. **Dễ debug**: Ảnh được lưu dưới dạng file, có thể xem trực tiếp
4. **Linh hoạt**: Có thể thay đổi số lượng augmentation
5. **Scalable**: Có thể chạy parallel cho nhiều model

## Augmentation Strategy

Augmentation được thiết kế đặc biệt cho ảnh nội soi:

1. **Geometric**: Rotation, translation, scaling (mô phỏng chuyển động ống soi)
2. **Color**: Brightness, contrast, saturation (mô phỏng điều kiện ánh sáng)  
3. **Noise**: Gaussian blur, random erasing (mô phỏng nhiễu và che khuất)
4. **Preprocessing**: Center crop để tập trung vào vùng quan trọng

## Lưu ý quan trọng

1. **Normalization**: Ảnh được lưu KHÔNG normalize để có thể xem trực tiếp
2. **Backbone compatibility**: Hỗ trợ cả `dinov2` và `ent_vit` normalization
3. **Deterministic**: Sử dụng seed để đảm bảo kết quả reproducible
4. **Memory efficient**: Load và xử lý ảnh theo batch

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**: Giảm batch_size trong `extract_features_from_images`
2. **File not found**: Kiểm tra đường dẫn config và checkpoint
3. **Normalization mismatch**: Đảm bảo config đúng với backbone đang sử dụng

### Debug tips:

1. Kiểm tra `summary.json` để xem thống kê bộ data
2. Sử dụng `eval_dataset_loader.py` để load và kiểm tra ảnh mẫu
3. Kiểm tra similarity giữa query và ground truth bằng sanity check

## Ví dụ workflow hoàn chỉnh

```bash
# 1. Tạo bộ data evaluation
python create_eval_dataset.py \
    --config configs/dinov2_vits14.yaml \
    --output eval_data \
    --num_augmentations 5

# 2. So sánh tất cả model
python compare_models_with_dataset.py \
    --eval_data eval_data \
    --output final_comparison.json

# 3. Đánh giá riêng một model cụ thể
python eval_with_dataset.py \
    --config configs/ent-vit.yaml \
    --eval_data eval_data \
    --checkpoint outputs/ent_vit_ntxent/best_model2.pth \
    --model_name "ENT-ViT Fine-tuned" \
    --output ent_vit_results.json
```

Kết quả sẽ hiển thị bảng so sánh chi tiết và được lưu vào file JSON để phân tích sau.
