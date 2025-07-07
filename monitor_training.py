#!/usr/bin/env python3
"""
Simple monitor script to check training progress
"""

import time
import os
from pathlib import Path

def monitor_training():
    """Monitor the training progress"""
    print("🔍 Monitoring EndoViT Contrastive Training")
    print("=" * 50)
    
    wandb_dir = Path("wandb")
    latest_run = None
    
    if wandb_dir.exists():
        # Find the latest run
        runs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        if runs:
            latest_run = max(runs, key=lambda x: x.stat().st_mtime)
            print(f"📊 Latest W&B run: {latest_run.name}")
    
    # Check processes
    os.system("ps aux | grep 'python train.py' | grep -v grep")
    
    # Check GPU usage
    print("\n🖥️  GPU Status:")
    os.system("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU detected'")
    
    # Check training log
    if Path("training.log").exists():
        print(f"\n📝 Latest training log entries:")
        os.system("tail -n 5 training.log")
    
    # Check outputs
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        print(f"\n💾 Output files:")
        os.system("ls -lah outputs/")
    
    print(f"\n⏰ Monitoring at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_training()
