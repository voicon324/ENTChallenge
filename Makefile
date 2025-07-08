# Makefile for Image Retrieval Project

# Default target
.PHONY: help
help:
	@echo "🎯 Image Retrieval Project Commands"
	@echo "=================================="
	@echo "Setup:"
	@echo "  make setup          - Setup project environment"
	@echo "  make install        - Install dependencies"
	@echo "  make wandb-login    - Login to W&B"
	@echo ""
	@echo "Data:"
	@echo "  make sample-data    - Create sample data"
	@echo "  make analyze-data   - Analyze dataset"
	@echo "  make validate-data  - Validate dataset"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Run training"
	@echo "  make train-dinov2   - Train with DinoV2"
	@echo "  make train-entvit   - Train with EntVit"
	@echo "  make experiments    - Run all experiments"
	@echo ""
	@echo "DinoV2 Experiments:"
	@echo "  make dinov2-small   - Run DinoV2 ViT-S/14"
	@echo "  make dinov2-base    - Run DinoV2 ViT-B/14"
	@echo "  make dinov2-large   - Run DinoV2 ViT-L/14"
	@echo "  make dinov2-all     - Run all DinoV2 variants"
	@echo "  make dinov2-test    - Test DinoV2 variants"
	@echo "  make dinov2-menu    - Interactive DinoV2 menu"
	@echo "  make dinov2-compare - Compare DinoV2 results"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval           - Evaluate best model"
	@echo "  make eval-val       - Evaluate on validation set"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Clean output files"
	@echo "  make clean-all      - Clean everything"
	@echo "  make format         - Format code"
	@echo "  make check          - Check code quality"

# Setup targets
.PHONY: setup
setup:
	@echo "🚀 Setting up project..."
	python setup.py

.PHONY: install
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

.PHONY: wandb-login
wandb-login:
	@echo "🔗 Logging into W&B..."
	wandb login

# Data targets
.PHONY: sample-data
sample-data:
	@echo "📊 Creating sample data..."
	python prepare_data.py --create-sample --num-classes 10 --images-per-class 50

.PHONY: analyze-data
analyze-data:
	@echo "🔍 Analyzing dataset..."
	python prepare_data.py --analyze

.PHONY: validate-data
validate-data:
	@echo "✅ Validating dataset..."
	python prepare_data.py --validate

# Training targets
.PHONY: train
train:
	@echo "🎯 Starting training..."
	python train.py

.PHONY: train-dinov2
train-dinov2:
	@echo "🎯 Training with DinoV2..."
	@sed -i 's/backbone: ".*"/backbone: "dino_v2"/' config.yaml
	python train.py

.PHONY: train-entvit
train-entvit:
	@echo "🎯 Training with EntVit..."
	@sed -i 's/backbone: ".*"/backbone: "ent_vit"/' config.yaml
	python train.py

.PHONY: experiments
experiments:
	@echo "🧪 Running all experiments..."
	python run_experiments.py --all

.PHONY: custom-experiment
custom-experiment:
	@echo "🎛️  Running custom experiment..."
	python run_experiments.py --custom

# Evaluation targets
.PHONY: eval
eval:
	@echo "🔍 Evaluating best model..."
	python eval.py --checkpoint outputs/best_model.pth

.PHONY: eval-val
eval-val:
	@echo "🔍 Evaluating on validation set..."
	python eval.py --checkpoint outputs/best_model.pth --split val

# Maintenance targets
.PHONY: clean
clean:
	@echo "🧹 Cleaning output files..."
	rm -rf outputs/*.pth
	rm -rf outputs/*.log
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__

.PHONY: clean-all
clean-all: clean
	@echo "🧹 Cleaning everything..."
	rm -rf outputs/*
	rm -rf data/processed/*
	rm -rf wandb/
	rm -rf *.log

.PHONY: format
format:
	@echo "🎨 Formatting code..."
	black . --line-length 88 --exclude "(venv|env|.git)" || echo "⚠️  black not installed"
	isort . --profile black || echo "⚠️  isort not installed"

.PHONY: check
check:
	@echo "🔍 Checking code quality..."
	flake8 . --max-line-length 88 --exclude venv,env,.git || echo "⚠️  flake8 not installed"
	mypy . --ignore-missing-imports || echo "⚠️  mypy not installed"

# Development targets
.PHONY: dev-setup
dev-setup:
	@echo "🛠️  Setting up development environment..."
	pip install black isort flake8 mypy
	pip install -r requirements.txt

.PHONY: test
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v || echo "⚠️  No tests found"

.PHONY: notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook || echo "⚠️  jupyter not installed"

# Quick commands
.PHONY: quick-start
quick-start: setup sample-data train
	@echo "🚀 Quick start completed!"

.PHONY: status
status:
	@echo "📊 Project Status"
	@echo "================="
	@echo "Config file: $$(test -f config.yaml && echo '✅ Found' || echo '❌ Missing')"
	@echo "Sample data: $$(test -d data/processed && echo '✅ Found' || echo '❌ Missing')"
	@echo "Outputs: $$(test -d outputs && echo '✅ Found' || echo '❌ Missing')"
	@echo "Best model: $$(test -f outputs/best_model.pth && echo '✅ Found' || echo '❌ Missing')"
	@echo "W&B login: $$(wandb status 2>/dev/null && echo '✅ Logged in' || echo '❌ Not logged in')"

# Docker targets (if needed)
.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t image-retrieval .

.PHONY: docker-run
docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --gpus all -v $(PWD):/workspace image-retrieval

# DinoV2 Experiments
.PHONY: dinov2-small dinov2-base dinov2-large dinov2-all dinov2-test
dinov2-small:
	@echo "🚀 Running DinoV2 ViT-S/14 experiment..."
	python train.py --config configs/dinov2_vits14.yaml

dinov2-base:
	@echo "🚀 Running DinoV2 ViT-B/14 experiment..."
	python train.py --config configs/dinov2_vitb14.yaml

dinov2-large:
	@echo "🚀 Running DinoV2 ViT-L/14 experiment..."
	python train.py --config configs/dinov2_vitl14.yaml

dinov2-all:
	@echo "🚀 Running all DinoV2 experiments..."
	python run_dinov2_experiments.py

dinov2-test:
	@echo "🧪 Testing DinoV2 variants..."
	python test_dinov2_variants.py

dinov2-menu:
	@echo "📋 DinoV2 experiment menu..."
	python run_individual_experiments.py

dinov2-compare:
	@echo "📊 Comparing DinoV2 results..."
	python compare_dinov2_results.py
