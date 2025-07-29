# ENTRep Challenge Track 3: Medical Text-Image Retrieval

[![Challenge](https://img.shields.io/badge/ENTRep%20Challenge-Track%203-blue)](https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep)
[![Ranking](https://img.shields.io/badge/Ranking-Top%202-gold)](https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep)

## Overview

This repository contains the implementation for **Track 3: Endoscopy Medical Images - Text-to-Image Retrieval** of the ENTRep Challenge, presented at ACM Multimedia 2025. Our solution, developed by **Team ELO**, achieved **2nd place** in the private leaderboard.

![ENTRep Challenge Track 3 Private Leaderboard](public/track3-top2.png)

## Abstract

The task involves developing a robust text-to-image retrieval system for endoscopic medical images. Our approach leverages state-of-the-art vision encoders (DINOv2, EndoViT) combined with CLIP-based text encoders to create an effective cross-modal retrieval system for medical image analysis.

## Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install clip-by-openai
pip install pyyaml
pip install numpy pandas
```

## Folder Structure

Organize your folder according to the following structure:

```
Dataset/
├── test/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── class4/
├── train/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── class4/
├── val/
│   ├── class1/
│   ├── class2/
│   ├── class3/
│   └── class4/
└── splits_info.json

Config/
├── eval/
│   ├── dinob.yaml
│   ├── dinol.yaml
│   ├── dinos.yaml
│   └── endovit.yaml
└── train/
    ├── dino_b.yaml
    ├── dino_l.yaml
    ├── dino_s.yaml
    └── endovit.yaml

Pretrained/
├── backbones/
└── checkpoints/
```

## Configuration

### Evaluation Configuration

The evaluation configuration specifies model parameters and evaluation settings:

```yaml
evaluator:
  batch_size: 32
  image_size: 224
  img_dir: Dataset/test/
  json_path: Dataset/splits_info.json  # Original dataset for building index
  test_path: Dataset/test_set.json     # Enhanced test set for evaluation
  k_values: [1, 5, 10]                # Top-k retrieval metrics

model:
  vision_encoder:
    type: dinov2
    feature_dim: 768
    model_name: dinov2_vits14
    ckp_path: Pretrained/backbones/dinov2_vits14/best_model.pth
  text_encoder:
    type: clip
    feature_dim: 768
    model_name: openai/clip-vit-base-patch32
  ckp_path: Pretrained/checkpoints/dinos_clip/best.pt
```

### Training Configuration

The training configuration defines data preprocessing and model training parameters:

```yaml
data:
  path: Dataset/
  json_path: splits_info.json
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  image_size: 224
  normalize: true

model:
  vision_encoder:
    type: dinov2
    feature_dim: 768
    model_name: dinov2_vits14
    ckp_path: Pretrained/dinov2_vits14/best_model.pth
  text_encoder:
    type: clip
    feature_dim: 768
    model_name: openai/clip-vit-base-patch32
  temperature: 0.07

trainer:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  output_path: checkpoints/
  model_name: dinos
```

## Usage

### Training

To train a model with a specific configuration:

```bash
python train.py --file_config endovit.yaml
```

Available training configurations:
- `dino_s.yaml` - DINOv2 Small variant
- `dino_b.yaml` - DINOv2 Base variant  
- `dino_l.yaml` - DINOv2 Large variant
- `endovit.yaml` - EndoViT specialized for endoscopic images

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --file_config endovit.yaml
```

The evaluation script will compute retrieval metrics including:
- HitRate@1, HitRate@5, HitRate@10
- Mean Reciprocal Rank (MRR)

## Acknowledgments

- ENTRep Challenge organizers for providing the dataset
- ACM Multimedia 2025 for hosting the challenge
- Authors of DINOv2, CLIP, and EndoViT for their foundational work

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this implementation, please open an issue or contact the team through the challenge platform.

---

**Team ELO** | ENTRep Challenge 2025 | ACM Multimedia
