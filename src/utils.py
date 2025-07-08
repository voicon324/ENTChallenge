"""
Các hàm tiện ích: grad_cam, lưu checkpoint, metrics calculation...
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score
import wandb
from PIL import Image
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import torchvision.transforms as T

def set_seed(seed: int = 42):
    """Đặt seed cho reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    """Thiết lập logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """Lưu checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Đã lưu checkpoint: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Tải checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)

def calculate_metrics(embeddings: torch.Tensor, labels: torch.Tensor, 
                     metrics: List[str], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Tính toán các metrics cho image retrieval
    
    Args:
        embeddings: Feature embeddings (N, D)
        labels: Ground truth labels (N,)
        metrics: List of metrics to calculate
        k_values: Values of k for HitRate@k
    
    Returns:
        Dictionary of calculated metrics
    """
    results = {}
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Tính similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    # Tạo mask để loại bỏ self-similarity
    mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
    similarity_matrix.masked_fill_(mask, -float('inf'))
    
    # Tính toán cho mỗi query
    n_queries = embeddings.size(0)
    
    for metric in metrics:
        if metric.startswith("HitRate@"):
            k = int(metric.split("@")[1])
            hit_rate = calculate_hit_rate(similarity_matrix, labels, k)
            results[metric] = hit_rate
        elif metric == "MRR":
            mrr = calculate_mrr(similarity_matrix, labels)
            results[metric] = mrr
        elif metric == "mAP":
            map_score = calculate_map(similarity_matrix, labels)
            results[metric] = map_score
    
    return results

def calculate_hit_rate(similarity_matrix: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Tính Hit Rate@k"""
    n_queries = similarity_matrix.size(0)
    hit_count = 0
    
    for i in range(n_queries):
        # Lấy top-k similar items
        _, top_k_indices = torch.topk(similarity_matrix[i], k)
        
        # Kiểm tra xem có item nào cùng class không
        query_label = labels[i]
        retrieved_labels = labels[top_k_indices]
        
        if torch.any(retrieved_labels == query_label):
            hit_count += 1
    
    return hit_count / n_queries

def calculate_mrr(similarity_matrix: torch.Tensor, labels: torch.Tensor) -> float:
    """Tính Mean Reciprocal Rank"""
    n_queries = similarity_matrix.size(0)
    reciprocal_ranks = []
    
    for i in range(n_queries):
        # Sắp xếp theo similarity giảm dần
        _, sorted_indices = torch.sort(similarity_matrix[i], descending=True)
        
        query_label = labels[i]
        
        # Tìm rank của item đầu tiên cùng class
        for rank, idx in enumerate(sorted_indices):
            if labels[idx] == query_label:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)

def calculate_map(similarity_matrix: torch.Tensor, labels: torch.Tensor) -> float:
    """Tính Mean Average Precision"""
    n_queries = similarity_matrix.size(0)
    average_precisions = []
    
    for i in range(n_queries):
        # Sắp xếp theo similarity giảm dần
        _, sorted_indices = torch.sort(similarity_matrix[i], descending=True)
        
        query_label = labels[i]
        retrieved_labels = labels[sorted_indices]
        
        # Tạo binary relevance vector
        relevance = (retrieved_labels == query_label).float()
        
        # Tính Average Precision
        if torch.sum(relevance) > 0:
            precision_at_k = torch.cumsum(relevance, dim=0) / torch.arange(1, len(relevance) + 1, dtype=torch.float)
            ap = torch.mean(precision_at_k * relevance)
            average_precisions.append(ap.item())
        else:
            average_precisions.append(0.0)
    
    return np.mean(average_precisions)

def generate_grad_cam_image(model, images: torch.Tensor, device: torch.device, 
                           target_layer: str = None) -> np.ndarray:
    """
    Tạo ảnh Grad-CAM để visualize attention heatmap overlay trên ảnh gốc
    
    Args:
        model: Model để tạo Grad-CAM
        images: Input images (B, C, H, W)
        device: Device
        target_layer: Target layer name
    
    Returns:
        Grad-CAM visualization as numpy array with heatmap overlay
    """
    try:
        # Use the first image for visualization
        single_image = images[0:1].clone().detach().requires_grad_(True)
        model.eval()
        
        # Get gradients and activations from the last convolutional layer
        # Try to find the last conv layer automatically with better detection
        target_layer_found = None
        conv_layers = []
        
        # First, try to find conv layers in the entire model
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        
        # If we found conv layers, use the last one
        if conv_layers:
            target_layer_found = conv_layers[-1][1]
            print(f"🎯 Using conv layer: {conv_layers[-1][0]}")
        else:
            # Fallback: try to get from backbone for ViT models
            if hasattr(model, 'backbone'):
                # For Vision Transformers, get attention from last layer
                if hasattr(model.backbone, 'blocks'):
                    # ViT case
                    return generate_vit_attention_map(model, single_image, device)
                else:
                    # Try to find conv layers in backbone
                    for name, module in model.backbone.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            conv_layers.append((name, module))
                    if conv_layers:
                        target_layer_found = conv_layers[-1][1]
                        print(f"🎯 Using backbone conv layer: {conv_layers[-1][0]}")
        
        # If still no conv layer found, try other layer types
        if target_layer_found is None:
            # Try to find Linear layers or other suitable layers
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.BatchNorm2d)):
                    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                        target_layer_found = module
                        print(f"🎯 Using fallback layer: {name}")
                        break
        
        # If we still can't find a suitable layer, create a simple gradient-based heatmap
        if target_layer_found is None:
            print("🔄 No suitable layer found, using input gradient method")
            return generate_input_gradient_map(model, single_image, device)
        
        # Manual Grad-CAM implementation with proper hook handling
        gradients_dict = {}
        activations_dict = {}
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients_dict['gradients'] = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            activations_dict['activations'] = output.detach()
        
        # Register hooks
        handle_backward = target_layer_found.register_full_backward_hook(backward_hook)
        handle_forward = target_layer_found.register_forward_hook(forward_hook)
        
        try:
            # Forward pass to get activations
            output = model(single_image)
            
            # Get the score to backprop through
            if output.dim() > 1 and output.size(1) > 1:
                # Classification case
                pred_class = torch.argmax(output, dim=1)
                score = output[0, pred_class[0]]
            else:
                # Feature case - use mean of output
                score = output.mean()
            
            # Backward pass to get gradients
            model.zero_grad()
            score.backward(retain_graph=True)
            
            # Get gradients and activations from the stored dictionaries
            if 'gradients' in gradients_dict and 'activations' in activations_dict:
                gradients = gradients_dict['gradients']
                activations = activations_dict['activations']
                
                # Ensure gradients and activations have spatial dimensions
                if len(gradients.shape) >= 4 and len(activations.shape) >= 4:
                    # Calculate Grad-CAM
                    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                    cam = torch.sum(weights * activations, dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # Resize to input image size
                    cam = F.interpolate(cam, size=(single_image.shape[2], single_image.shape[3]), 
                                      mode='bilinear', align_corners=False)
                    
                    # Normalize
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    cam = cam.squeeze().cpu().numpy()
                else:
                    # For non-spatial layers, create a uniform heatmap
                    cam = np.ones((single_image.shape[2], single_image.shape[3])) * 0.5
            else:
                # Fallback if hooks didn't capture anything
                print("⚠️ Hooks didn't capture gradients/activations, using input gradient method")
                return generate_input_gradient_map(model, single_image, device)
            
        finally:
            # Always remove hooks
            handle_backward.remove()
            handle_forward.remove()
        
        # Convert image to numpy
        image_np = single_image.squeeze().cpu().detach().numpy()
        if image_np.shape[0] == 3:  # RGB
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Normalize image to [0, 1]
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Blend with original image
        alpha = 0.3  # Transparency factor - reduced for less overlay
        overlay = alpha * heatmap + (1 - alpha) * image_np
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}, using simple gradient method")
        try:
            return generate_input_gradient_map(model, images[0:1], device)
        except Exception as e2:
            print(f"⚠️ Input gradient method also failed: {e2}")
            # Ultimate fallback - return original image
            try:
                img = images[0].cpu().detach().numpy()
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return (img * 255).astype(np.uint8)
            except:
                # Final fallback
                return np.zeros((224, 224, 3), dtype=np.uint8)

def generate_input_gradient_map(model, image: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Generate a simple gradient-based attention map using input gradients
    """
    try:
        image = image.clone().detach().requires_grad_(True)
        model.eval()
        
        # Forward pass
        output = model(image)
        
        # Get score
        if output.dim() > 1 and output.size(1) > 1:
            pred_class = torch.argmax(output, dim=1)
            score = output[0, pred_class[0]]
        else:
            score = output.mean()
        
        # Backward pass
        model.zero_grad()
        score.backward()
        
        # Get gradients
        gradients = image.grad.data
        
        # Create attention map from gradients
        gradient_map = torch.abs(gradients).mean(dim=1, keepdim=True)  # Average over channels
        gradient_map = F.interpolate(gradient_map, size=(image.shape[2], image.shape[3]), 
                                   mode='bilinear', align_corners=False)
        
        # Normalize
        gradient_map = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-8)
        cam = gradient_map.squeeze().cpu().numpy()
        
        # Convert image to numpy
        image_np = image.squeeze().cpu().detach().numpy()
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Blend
        alpha = 0.3  # Reduced transparency for clearer original image
        overlay = alpha * heatmap + (1 - alpha) * image_np
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Input gradient method failed: {e}")
        # Return original image
        img = image.squeeze().cpu().detach().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)

def generate_vit_attention_map(model, image: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Generate attention map for Vision Transformer models
    """
    try:
        model.eval()
        
        # Get attention weights from the last transformer block
        def hook_fn(module, input, output):
            # For ViT, output usually contains attention weights
            if hasattr(module, 'attn'):
                module.attention_weights = module.attn.attention_weights
        
        # Register hook on the last transformer block
        if hasattr(model.backbone, 'blocks'):
            last_block = model.backbone.blocks[-1]
            handle = last_block.register_forward_hook(hook_fn)
            
            # Forward pass
            with torch.no_grad():
                _ = model(image)
            
            # Get attention weights
            if hasattr(last_block, 'attention_weights'):
                attn_weights = last_block.attention_weights  # Shape: [B, heads, seq_len, seq_len]
                
                # Average over heads and take attention to CLS token
                attn_weights = attn_weights.mean(dim=1)  # [B, seq_len, seq_len]
                attn_map = attn_weights[0, 0, 1:]  # Attention from CLS to patches
                
                # Reshape to spatial dimensions
                grid_size = int(np.sqrt(len(attn_map)))
                attn_map = attn_map.reshape(grid_size, grid_size)
                
                # Resize to image size
                attn_map = F.interpolate(
                    attn_map.unsqueeze(0).unsqueeze(0),
                    size=(image.shape[2], image.shape[3]),
                    mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()
            else:
                # Fallback
                attn_map = np.ones((image.shape[2], image.shape[3])) * 0.5
            
            handle.remove()
        else:
            attn_map = np.ones((image.shape[2], image.shape[3])) * 0.5
        
        # Convert image to numpy
        image_np = image.squeeze().cpu().numpy()
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Normalize attention map
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Blend with original image
        alpha = 0.3  # Reduced transparency for clearer original image
        overlay = alpha * heatmap + (1 - alpha) * image_np
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
        
    except Exception as e:
        print(f"⚠️ Could not generate ViT attention map: {e}")
        img = image.squeeze().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)

def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        class_names: List[str], wandb_run) -> None:
    """Log confusion matrix to W&B"""
    try:
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Log to W&B
        wandb_run.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Không thể log confusion matrix: {e}")

def create_embedding_plot(embeddings: torch.Tensor, labels: torch.Tensor, 
                         wandb_run, method: str = "tsne") -> None:
    """Tạo plot embedding space với t-SNE hoặc PCA"""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels_np, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Embedding Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        # Log to W&B
        wandb_run.log({f"embedding_{method}": wandb.Image(plt)})
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Không thể tạo embedding plot: {e}")

def log_sample_images(images: torch.Tensor, labels: torch.Tensor, 
                     predictions: torch.Tensor, wandb_run, num_samples: int = 8) -> None:
    """Log sample images với predictions"""
    try:
        # Chọn random samples
        indices = torch.randperm(len(images))[:num_samples]
        
        sample_images = images[indices]
        sample_labels = labels[indices]
        sample_preds = predictions[indices]
        
        # Tạo wandb Images
        wandb_images = []
        for i in range(num_samples):
            img = sample_images[i].cpu().numpy()
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
            
            # Normalize
            img = (img - img.min()) / (img.max() - img.min())
            
            true_label = sample_labels[i].item()
            pred_label = sample_preds[i].item()
            
            wandb_images.append(
                wandb.Image(img, caption=f"True: {true_label}, Pred: {pred_label}")
            )
        
        wandb_run.log({"sample_predictions": wandb_images})
        
    except Exception as e:
        print(f"⚠️ Không thể log sample images: {e}")

def get_model_summary(model, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict:
    """Tạo summary của model"""
    try:
        from torchsummary import summary
        
        # Tạo dummy input
        dummy_input = torch.randn(input_size)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
    except Exception as e:
        print(f"⚠️ Không thể tạo model summary: {e}")
        return {}

def process_entvit_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228], dataset_std=[0.2520, 0.2128, 0.2093]):
    """
    Process a single image for EndoViT model with proper normalization.
    
    Args:
        image_path: Path to the image file
        input_size: Target image size (default: 224)
        dataset_mean: EndoViT-specific mean values for normalization
        dataset_std: EndoViT-specific std values for normalization
    
    Returns:
        Processed image tensor
    """
    # Define the transformations
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations
    processed_image = transform(image)

    return processed_image

def get_entvit_transforms(input_size=224, is_train=True):
    """
    Get EndoViT-specific transforms for training or validation.
    
    Args:
        input_size: Target image size
        is_train: Whether to apply training augmentations
        
    Returns:
        Transform pipeline
    """
    # EndoViT-specific normalization
    dataset_mean = [0.3464, 0.2280, 0.2228]
    dataset_std = [0.2520, 0.2128, 0.2093]
    
    if is_train:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ])
    else:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=dataset_mean, std=dataset_std)
        ])
