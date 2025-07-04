#!/usr/bin/env python3
"""
Script chạy các experiments khác nhau
Chạy: python run_experiments.py
"""

import yaml
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
import argparse

def load_config(config_path: str) -> Dict:
    """Load config from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: Dict, config_path: str):
    """Save config to YAML file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_experiment(experiment_name: str, config_updates: Dict, base_config_path: str = "config.yaml"):
    """Run a single experiment"""
    print(f"\n🧪 Running experiment: {experiment_name}")
    print("=" * 50)
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Update config with experiment settings
    experiment_config = base_config.copy()
    
    # Deep update
    for key, value in config_updates.items():
        if '.' in key:
            # Handle nested keys like 'model.backbone'
            keys = key.split('.')
            current = experiment_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            experiment_config[key] = value
    
    # Update W&B run name
    if 'wandb' not in experiment_config:
        experiment_config['wandb'] = {}
    experiment_config['wandb']['run_name'] = experiment_name
    
    # Save experiment config
    experiment_config_path = f"config_{experiment_name}.yaml"
    save_config(experiment_config, experiment_config_path)
    
    print(f"💾 Saved experiment config: {experiment_config_path}")
    
    # Run training
    try:
        cmd = [sys.executable, "train.py"]
        # Override config file
        with open("config.yaml", "w") as f:
            yaml.dump(experiment_config, f, default_flow_style=False)
        
        print(f"🚀 Starting training...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Experiment {experiment_name} completed successfully")
            return True
        else:
            print(f"❌ Experiment {experiment_name} failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running experiment {experiment_name}: {e}")
        return False

def run_all_experiments():
    """Run all predefined experiments"""
    
    experiments = [
        {
            "name": "dinov2_contrastive",
            "config": {
                "model.backbone": "dino_v2",
                "training.strategy": "contrastive",
                "training.epochs": 30,
                "training.learning_rate": 0.0001,
            }
        },
        {
            "name": "entvit_contrastive", 
            "config": {
                "model.backbone": "ent_vit",
                "training.strategy": "contrastive",
                "training.epochs": 30,
                "training.learning_rate": 0.0001,
            }
        },
        {
            "name": "dinov2_frozen",
            "config": {
                "model.backbone": "dino_v2",
                "model.freeze_backbone": True,
                "training.strategy": "contrastive",
                "training.epochs": 20,
                "training.learning_rate": 0.001,
            }
        },
        {
            "name": "entvit_frozen",
            "config": {
                "model.backbone": "ent_vit", 
                "model.freeze_backbone": True,
                "training.strategy": "contrastive",
                "training.epochs": 20,
                "training.learning_rate": 0.001,
            }
        },
    ]
    
    # Backup original config
    original_config_path = "config_original.yaml"
    if Path("config.yaml").exists():
        subprocess.run(["cp", "config.yaml", original_config_path])
    
    successful_experiments = []
    failed_experiments = []
    
    for experiment in experiments:
        success = run_experiment(experiment["name"], experiment["config"])
        
        if success:
            successful_experiments.append(experiment["name"])
        else:
            failed_experiments.append(experiment["name"])
        
        # Wait between experiments
        time.sleep(5)
    
    # Restore original config
    if Path(original_config_path).exists():
        subprocess.run(["cp", original_config_path, "config.yaml"])
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EXPERIMENTS SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ Successful experiments ({len(successful_experiments)}):")
    for exp in successful_experiments:
        print(f"   - {exp}")
    
    print(f"\n❌ Failed experiments ({len(failed_experiments)}):")
    for exp in failed_experiments:
        print(f"   - {exp}")
    
    print(f"\n🎯 Total: {len(successful_experiments)}/{len(experiments)} experiments completed")

def run_custom_experiment():
    """Run custom experiment with user input"""
    print("🎛️  Custom Experiment Setup")
    print("=" * 30)
    
    experiment_name = input("Experiment name: ")
    
    # Model settings
    print("\n🏗️  Model Configuration:")
    backbone = input("Backbone (dino_v2/ent_vit) [dino_v2]: ") or "dino_v2"
    freeze_backbone = input("Freeze backbone? (y/n) [n]: ").lower() == 'y'
    
    # Training settings
    print("\n🎯 Training Configuration:")
    strategy = input("Strategy (contrastive/test_only) [contrastive]: ") or "contrastive"
    epochs = int(input("Epochs [30]: ") or "30")
    learning_rate = float(input("Learning rate [0.0001]: ") or "0.0001")
    
    # Create config updates
    config_updates = {
        "model.backbone": backbone,
        "model.freeze_backbone": freeze_backbone,
        "training.strategy": strategy,
        "training.epochs": epochs,
        "training.learning_rate": learning_rate,
    }
    
    print(f"\n📋 Experiment configuration:")
    for key, value in config_updates.items():
        print(f"   {key}: {value}")
    
    confirm = input("\nProceed with experiment? (y/n): ").lower() == 'y'
    
    if confirm:
        run_experiment(experiment_name, config_updates)
    else:
        print("⚠️  Experiment cancelled")

def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--all', action='store_true', 
                       help='Run all predefined experiments')
    parser.add_argument('--custom', action='store_true',
                       help='Run custom experiment')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available Experiments:")
        experiments = [
            "dinov2_contrastive - DinoV2 with contrastive learning",
            "entvit_contrastive - EntVit with contrastive learning", 
            "dinov2_frozen - DinoV2 with frozen backbone",
            "entvit_frozen - EntVit with frozen backbone",
        ]
        for exp in experiments:
            print(f"   - {exp}")
        return
    
    if args.all:
        run_all_experiments()
    elif args.custom:
        run_custom_experiment()
    else:
        print("🧪 Experiment Runner")
        print("=" * 30)
        print("Choose an option:")
        print("   1. Run all experiments")
        print("   2. Run custom experiment")
        print("   3. List available experiments")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            run_all_experiments()
        elif choice == "2":
            run_custom_experiment()
        elif choice == "3":
            print("📋 Available Experiments:")
            experiments = [
                "dinov2_contrastive - DinoV2 with contrastive learning",
                "entvit_contrastive - EntVit with contrastive learning", 
                "dinov2_frozen - DinoV2 with frozen backbone",
                "entvit_frozen - EntVit with frozen backbone",
            ]
            for exp in experiments:
                print(f"   - {exp}")
        else:
            print("⚠️  Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
