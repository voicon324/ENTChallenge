#!/bin/bash

# Script to run evaluation for all DinoV2 models with and without checkpoints
# Author: Generated for ENT Challenge

echo "🚀 Starting comprehensive model evaluation..."
echo "=" * 80

# DinoV2-S (Small) evaluations
echo "📊 Evaluating DinoV2-S (Small) models..."
python eval_with_dataset.py -c configs/dinov2_vits14.yaml -e eval_data -n "DinoV2-S-Pretrained" -o results_dinov2_s_pretrained.json
python eval_with_dataset.py -c configs/dinov2_vits14.yaml -e eval_data -ckpt outputs/dinov2_vits14_ntxent/best_model424.pth -n "DinoV2-S-424" -o results_dinov2_s_424.json

# DinoV2-B (Base) evaluations
echo "📊 Evaluating DinoV2-B (Base) models..."
python eval_with_dataset.py -c configs/dinov2_vitb14.yaml -e eval_data -n "DinoV2-B-Pretrained" -o results_dinov2_b_pretrained.json
python eval_with_dataset.py -c configs/dinov2_vitb14.yaml -e eval_data -ckpt outputs/dinov2_vitb14_ntxent/best_model424.pth -n "DinoV2-B-424" -o results_dinov2_b_424.json

# DinoV2-L (Large) evaluations
echo "📊 Evaluating DinoV2-L (Large) models..."
python eval_with_dataset.py -c configs/dinov2_vitl14.yaml -e eval_data -n "DinoV2-L-Pretrained" -o results_dinov2_l_pretrained.json
python eval_with_dataset.py -c configs/dinov2_vitl14.yaml -e eval_data -ckpt outputs/dinov2_vitl14_ntxent/best_model424.pth -n "DinoV2-L-424" -o results_dinov2_l_424.json

echo "🎉 All evaluations completed!"
echo "📁 Results saved to:"
echo "   - results_dinov2_s_pretrained.json"
echo "   - results_dinov2_s_424.json"
echo "   - results_dinov2_b_pretrained.json"
echo "   - results_dinov2_b_424.json"
echo "   - results_dinov2_l_pretrained.json"
echo "   - results_dinov2_l_424.json"
