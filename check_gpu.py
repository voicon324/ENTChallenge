#!/usr/bin/env python3
"""
Script kiểm tra cấu hình GPU và thông tin hệ thống
"""

import torch
import torch.nn as nn
import subprocess
import sys

def check_gpu_info():
    """Kiểm tra thông tin GPU"""
    print("🔍 KIỂM TRA THÔNG TIN GPU")
    print("=" * 50)
    
    # Kiểm tra CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA có sẵn: {gpu_count} GPU(s)")
        
        # Thông tin chi tiết từng GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Kiểm tra memory hiện tại
        print(f"\n📊 MEMORY USAGE:")
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"   GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        print("❌ CUDA không có sẵn")
        return False
    
    return True

def check_multi_gpu_capabilities():
    """Kiểm tra khả năng multi-GPU"""
    print("\n🚀 KIỂM TRA KHẢ NĂNG MULTI-GPU")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA không có sẵn")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"⚠️  Chỉ có {gpu_count} GPU - Không thể chạy multi-GPU")
        return False
    
    print(f"✅ Có {gpu_count} GPU - Có thể chạy multi-GPU")
    
    # Test DataParallel
    try:
        print("\n🧪 Testing DataParallel...")
        model = nn.Linear(10, 1)
        model = nn.DataParallel(model)
        model.cuda()
        
        # Test input
        x = torch.randn(4, 10).cuda()
        output = model(x)
        print(f"✅ DataParallel test thành công: input_shape={x.shape}, output_shape={output.shape}")
        
        # Test SyncBatchNorm
        print("\n🔄 Testing SyncBatchNorm...")
        model = nn.BatchNorm1d(10)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.DataParallel(model)
        model.cuda()
        
        x = torch.randn(4, 10).cuda()
        output = model(x)
        print(f"✅ SyncBatchNorm test thành công: input_shape={x.shape}, output_shape={output.shape}")
        
    except Exception as e:
        print(f"❌ Multi-GPU test thất bại: {e}")
        return False
    
    return True

def check_system_info():
    """Kiểm tra thông tin hệ thống"""
    print("\n💻 THÔNG TIN HỆ THỐNG")
    print("=" * 50)
    
    # Python version
    print(f"🐍 Python version: {sys.version}")
    
    # PyTorch version
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # CUDA version
    if torch.cuda.is_available():
        print(f"⚡ CUDA version: {torch.version.cuda}")
        print(f"🏗️  cuDNN version: {torch.backends.cudnn.version()}")
    
    # CPU info
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"💾 CPU cores: {cpu_count}")
    except:
        print("💾 CPU cores: Unknown")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"🧠 RAM: {memory.total / (1024**3):.1f} GB (available: {memory.available / (1024**3):.1f} GB)")
    except ImportError:
        print("🧠 RAM: Unknown (install psutil for memory info)")

def main():
    print("🔧 KIỂM TRA CẤU HÌNH MULTI-GPU TRAINING")
    print("=" * 60)
    
    # Check system info
    check_system_info()
    
    # Check GPU info
    gpu_available = check_gpu_info()
    
    if gpu_available:
        multi_gpu_ready = check_multi_gpu_capabilities()
        
        print("\n" + "=" * 60)
        print("📋 KẾT LUẬN:")
        if multi_gpu_ready:
            print("✅ Hệ thống SẴN SÀNG cho multi-GPU training!")
            print("🚀 Có thể chạy: bash train_multi_gpu.sh")
        else:
            print("⚠️  Hệ thống chỉ hỗ trợ single-GPU training")
            print("🔥 Có thể chạy: python train.py --config configs/dinov2_vitb14.yaml")
    else:
        print("\n" + "=" * 60)
        print("📋 KẾT LUẬN:")
        print("❌ Hệ thống chỉ hỗ trợ CPU training")
        print("💻 Có thể chạy: python train.py --config configs/dinov2_vitb14.yaml")

if __name__ == "__main__":
    main()
