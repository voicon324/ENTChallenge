#!/bin/bash

# Script để chạy training với multi-GPU cho tất cả các config
# Tác giả: Training Script
# Cách sử dụng: bash train_multi_gpu.sh [config_name]
# Ví dụ: bash train_multi_gpu.sh dinov2_vitb14
# Hoặc: bash train_multi_gpu.sh all (để chạy tất cả config)

echo "🚀 Khởi động training với Multi-GPU..."

# Kiểm tra số lượng GPU
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "🔍 Phát hiện $GPU_COUNT GPU(s)"

# Danh sách tất cả config
CONFIGS=("dinov2_vitb14" "dinov2_vitl14" "dinov2_vits14" "ent-vit")

# Hàm chạy training cho một config
run_training() {
    local config_name=$1
    local config_file="configs/${config_name}.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "❌ Config file $config_file không tồn tại!"
        return 1
    fi
    
    echo "🏃 Chạy training với config: $config_name"
    echo "📁 Config file: $config_file"
    
    if [ $GPU_COUNT -ge 2 ]; then
        echo "✅ Sử dụng Multi-GPU training với $GPU_COUNT GPU(s)"
        export CUDA_VISIBLE_DEVICES=0,1
    else
        echo "⚠️ Chỉ có $GPU_COUNT GPU, sử dụng single GPU training"
        export CUDA_VISIBLE_DEVICES=0
    fi
    
    # Chạy training
    python train.py --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ Training hoàn thành cho config: $config_name"
    else
        echo "❌ Training thất bại cho config: $config_name"
        return 1
    fi
}

# Xử lý tham số dòng lệnh
if [ $# -eq 0 ]; then
    echo "📋 Chọn config để chạy:"
    echo "1. dinov2_vitb14"
    echo "2. dinov2_vitl14" 
    echo "3. dinov2_vits14"
    echo "4. ent-vit"
    echo "5. all (chạy tất cả)"
    echo ""
    read -p "Nhập lựa chọn (1-5): " choice
    
    case $choice in
        1) run_training "dinov2_vitb14" ;;
        2) run_training "dinov2_vitl14" ;;
        3) run_training "dinov2_vits14" ;;
        4) run_training "ent-vit" ;;
        5) 
            echo "🚀 Chạy tất cả config..."
            for config in "${CONFIGS[@]}"; do
                echo "=================="
                run_training "$config"
                echo "=================="
                sleep 2
            done
            ;;
        *) echo "❌ Lựa chọn không hợp lệ!" ;;
    esac
elif [ "$1" = "all" ]; then
    echo "🚀 Chạy tất cả config..."
    for config in "${CONFIGS[@]}"; do
        echo "=================="
        run_training "$config"
        echo "=================="
        sleep 2
    done
else
    # Chạy config cụ thể
    run_training "$1"
fi

echo "🎉 Hoàn thành tất cả training!"
