#!/usr/bin/env python3
"""
Script setup môi trường project
Chạy: python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Chạy command và hiển thị kết quả"""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Kiểm tra phiên bản Python"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Python 3.8+ is recommended")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def install_requirements():
    """Cài đặt requirements"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )

def setup_wandb():
    """Setup W&B"""
    print("🔗 Setting up Weights & Biases...")
    print("   Please login to W&B when prompted")
    
    try:
        import wandb
        wandb.login()
        print("✅ W&B login successful")
        return True
    except Exception as e:
        print(f"❌ W&B setup failed: {e}")
        print("💡 You can run 'wandb login' manually later")
        return False

def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        "data/processed",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def check_gpu():
    """Kiểm tra GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU available: {gpu_name} (x{gpu_count})")
            return True
        else:
            print("💻 No GPU detected, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def create_sample_data():
    """Tạo dữ liệu mẫu"""
    print("📊 Creating sample data...")
    
    return run_command(
        f"{sys.executable} prepare_data.py --create-sample --num-classes 5 --images-per-class 20",
        "Creating sample dataset"
    )

def test_config():
    """Test config file"""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ config.yaml is valid")
        return True
    except Exception as e:
        print(f"❌ config.yaml error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Image Retrieval Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("⚠️  Consider upgrading Python for better compatibility")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Test config
    if not test_config():
        print("❌ Config validation failed")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Setup W&B
    try:
        setup_wandb()
    except KeyboardInterrupt:
        print("\n⚠️  W&B setup skipped")
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Edit config.yaml to customize your experiment")
    print("   2. Run: python train.py")
    print("   3. Monitor training at W&B dashboard")
    print("\n📚 For more info, check README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
