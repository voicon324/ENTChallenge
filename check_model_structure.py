#!/usr/bin/env python3
"""
Script demo để kiểm tra việc lưu model theo run_name
"""

import yaml
from pathlib import Path

def check_model_saving_structure():
    """Kiểm tra cấu trúc lưu model mới"""
    
    print("🔍 Kiểm tra cấu trúc lưu model theo run_name:")
    print("=" * 50)
    
    # Đọc tất cả config files
    config_files = list(Path("configs").glob("*.yaml"))
    
    for config_path in config_files:
        print(f"\n📋 Config: {config_path.name}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        run_name = config.get('wandb', {}).get('run_name', 'default_run')
        output_dir = Path("outputs") / run_name
        
        print(f"   🏷️  Run name: {run_name}")
        print(f"   📁 Output dir: {output_dir}")
        print(f"   📄 Best model: {output_dir / 'best_model.pth'}")
        print(f"   📄 Checkpoints: {output_dir / 'checkpoint_epoch_*.pth'}")
        
        # Kiểm tra xem thư mục đã tồn tại chưa
        if output_dir.exists():
            print(f"   ✅ Directory exists with {len(list(output_dir.glob('*.pth')))} model files")
        else:
            print(f"   📝 Directory will be created when training starts")
    
    print("\n" + "=" * 50)
    print("✅ Cấu trúc mới sẽ tạo thư mục riêng cho mỗi experiment:")
    print("   outputs/")
    print("   ├── dinov2_vitl14_ntxent_experiment/")
    print("   │   ├── best_model.pth")
    print("   │   └── checkpoint_epoch_*.pth")
    print("   ├── dinov2_vitb14_ntxent_experiment/")
    print("   │   ├── best_model.pth")
    print("   │   └── checkpoint_epoch_*.pth")
    print("   ├── dinov2_vits14_ntxent_experiment/")
    print("   │   ├── best_model.pth")
    print("   │   └── checkpoint_epoch_*.pth")
    print("   └── ent_vit_ntxent_experiment/")
    print("       ├── best_model.pth")
    print("       └── checkpoint_epoch_*.pth")

if __name__ == "__main__":
    check_model_saving_structure()
