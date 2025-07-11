# Image Retrieval System

A modular image retrieval system using Vision Transformers with contrastive learning and integrated Weights & Biases tracking.

## Architecture

```
ENTChanglleng/
├── configs/                   # Configuration files
│   ├── dinov2_vitb14.yaml    # DinoV2 ViT-B/14 config
│   ├── dinov2_vitl14.yaml    # DinoV2 ViT-L/14 config
│   └── ent-vit.yaml          # EndoViT config
├── src/
│   ├── data_loader.py        # Dataset and DataLoader
│   ├── model_factory.py      # Model factory (DinoV2, EntVit)
│   ├── trainer.py            # Training logic with W&B integration
│   ├── utils.py              # Utilities and helper functions
│   └── models/
│       ├── dino_v2.py       # DinoV2 implementation
│       └── ent_vit.py       # EndoViT implementation
├── train.py                  # Main training script
├── eval.py                   # Evaluation script
├── prepare_data.py           # Data preparation utilities
└── outputs/                  # Model checkpoints and results
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
wandb login
```

### 2. Data Preparation

```bash
# Analyze existing data
python prepare_data.py --analyze --data-dir Dataset/

# Create sample data for testing
python prepare_data.py --create-sample --num-classes 7 --images-per-class 100
```

### 3. Training

```bash
# Train with DinoV2 ViT-B/14
python train.py --config configs/dinov2_vitb14.yaml

# Train with EndoViT
python train.py --config configs/ent-vit.yaml
```

### 4. Evaluation

```bash
python eval.py --checkpoint outputs/best_model.pth
```

## Configuration

The system uses YAML configuration files for all experiments. Key parameters:

```yaml
# Model configuration
model:
  backbone: "dino_v2"          # "dino_v2" or "ent_vit"
  model_name: "dinov2_vitb14"  # Specific model variant
  feature_dim: 768             # Feature dimension
  num_classes: 7               # Number of classes
  dropout: 0.1                 # Dropout rate

# Training configuration
training:
  strategy: "ntxent"           # Loss function strategy
  epochs: 200                  # Number of epochs
  learning_rate: 0.0001        # Learning rate
  batch_size: 8                # Batch size
  temperature: 0.07            # Contrastive temperature
```

## Supported Models

### DinoV2
- **dinov2_vits14**: Small model (21M parameters)
- **dinov2_vitb14**: Base model (86M parameters)  
- **dinov2_vitl14**: Large model (300M parameters)

### EndoViT
- Pre-trained on endoscopic datasets
- Specialized for medical image analysis
- 768-dimensional embeddings

## Training Strategies

- **ntxent**: NT-Xent contrastive loss
- **contrastive**: Standard contrastive learning
- **test_only**: Evaluation only mode
