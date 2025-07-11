"""
Visualization tools for model analysis - PCA visualization of patch features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path
import json
from collections import defaultdict
import seaborn as sns

class PCAVisualizer:
    """PCA Visualization for Vision Transformer models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PCA visualizer
        
        Args:
            model: Vision transformer model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Standard transforms for DinoV2
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_patch_features(self, imgs: List[Image.Image], patch_size: int = 14) -> List[np.ndarray]:
        """
        Extract patch features from a list of images
        
        Args:
            imgs: List of PIL images
            patch_size: Patch size for ViT
            
        Returns:
            List of patch features arrays
        """
        all_features = []
        
        with torch.no_grad():
            for img in imgs:
                # Transform image
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # Extract features - handle different model types
                features = None
                
                try:
                    # For DinoV2 models - use forward_features from backbone
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                        if hasattr(self.model.model.backbone, 'backbone'):
                            # DinoV2 hub model - use forward_features
                            dino_model = self.model.model.backbone.backbone
                            features = dino_model.forward_features(img_tensor)
                        else:
                            # Regular backbone
                            features = self.model.model.backbone(img_tensor)
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'get_backbone_features'):
                        features = self.model.model.get_backbone_features(img_tensor)
                    elif hasattr(self.model, 'get_features'):
                        features = self.model.get_features(img_tensor)
                    else:
                        # Direct model access
                        features = self.model(img_tensor)
                
                except Exception as e:
                    print(f"⚠️ Error extracting features: {e}")
                    # Try getting features using model's get_features method
                    if hasattr(self.model, 'get_features'):
                        features = self.model.get_features(img_tensor)
                    else:
                        # Final fallback
                        features = self.model(img_tensor)
                
                # Handle different output formats
                patch_features = None
                
                if features is None:
                    print(f"⚠️ No features extracted for image")
                    continue
                
                if isinstance(features, dict):
                    # DinoV2 style output
                    if 'x_norm_patchtokens' in features:
                        patch_features = features['x_norm_patchtokens']
                    elif 'patch_tokens' in features:
                        patch_features = features['patch_tokens']
                    elif 'x_norm_tokens' in features:
                        patch_features = features['x_norm_tokens']
                    elif 'last_hidden_state' in features:
                        patch_features = features['last_hidden_state']
                    else:
                        # Take the first available feature
                        patch_features = list(features.values())[0]
                else:
                    # Direct tensor output
                    patch_features = features
                
                # Handle different tensor shapes
                if patch_features is None:
                    print(f"⚠️ No patch features found")
                    continue
                    
                if len(patch_features.shape) == 3:
                    # Shape: [batch, num_patches, dim]
                    patch_features = patch_features.squeeze(0)
                elif len(patch_features.shape) == 2:
                    # Shape: [num_patches, dim] - already squeezed
                    pass
                elif len(patch_features.shape) == 4:
                    # Shape: [batch, height, width, dim] - spatial features
                    patch_features = patch_features.squeeze(0).view(-1, patch_features.shape[-1])
                else:
                    # Unexpected shape, try to handle
                    print(f"⚠️ Unexpected feature shape: {patch_features.shape}")
                    patch_features = patch_features.view(-1, patch_features.shape[-1])
                
                # For models that include class token, remove it
                if patch_features.shape[0] == 197:  # 14x14 + 1 class token
                    patch_features = patch_features[1:]  # Remove class token
                elif patch_features.shape[0] == 257:  # 16x16 + 1 class token
                    patch_features = patch_features[1:]  # Remove class token
                
                all_features.append(patch_features.cpu().numpy())
        
        return all_features
    
    def compute_pca_visualization(self, features_list: List[np.ndarray], 
                                 n_components: int = 3) -> Tuple[List[np.ndarray], PCA]:
        """
        Compute PCA on patch features
        
        Args:
            features_list: List of patch features
            n_components: Number of PCA components
            
        Returns:
            Tuple of (PCA features for each image, fitted PCA object)
        """
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=0)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(all_features)
        
        # Normalize PCA features to [0, 1] range
        pca_normalized = np.zeros_like(pca_features)
        for i in range(n_components):
            pca_normalized[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                                  (pca_features[:, i].max() - pca_features[:, i].min())
        
        # Split back into individual images
        pca_images = []
        start_idx = 0
        for features in features_list:
            end_idx = start_idx + len(features)
            pca_images.append(pca_normalized[start_idx:end_idx])
            start_idx = end_idx
        
        return pca_images, pca
    
    def reshape_to_spatial(self, pca_features: np.ndarray, 
                          patch_size: int, img_size: int = 224) -> np.ndarray:
        """
        Reshape 1D patch features to 2D spatial grid
        
        Args:
            pca_features: PCA features array
            patch_size: Size of each patch
            img_size: Input image size
            
        Returns:
            Reshaped features as spatial grid
        """
        num_patches_per_side = img_size // patch_size
        expected_patches = num_patches_per_side * num_patches_per_side
        
        # Handle different numbers of patches (some models may have class tokens)
        if len(pca_features) == expected_patches:
            # Perfect match
            return pca_features.reshape(num_patches_per_side, num_patches_per_side, -1)
        elif len(pca_features) == expected_patches + 1:
            # Likely has class token, remove it
            return pca_features[1:].reshape(num_patches_per_side, num_patches_per_side, -1)
        else:
            # Try to handle other cases - automatically detect the grid size
            num_patches = len(pca_features)
            
            # Try common grid sizes
            grid_sizes = [14, 16, 32]  # Common patch grid sizes
            for grid_size in grid_sizes:
                if num_patches == grid_size * grid_size:
                    return pca_features.reshape(grid_size, grid_size, -1)
                elif num_patches == grid_size * grid_size + 1:  # With class token
                    return pca_features[1:].reshape(grid_size, grid_size, -1)
            
            # If no perfect match, try to find the closest square
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size == num_patches:
                return pca_features.reshape(grid_size, grid_size, -1)
            
            # Last resort - use the closest square and pad/trim
            print(f"Warning: Expected {expected_patches} patches, got {num_patches}")
            print(f"Using {grid_size}x{grid_size} grid")
            patches_to_use = grid_size * grid_size
            return pca_features[:patches_to_use].reshape(grid_size, grid_size, -1)
    
    def create_pca_visualization(self, imgs: List[Image.Image], 
                               class_names: List[str],
                               patch_size: int = 14,
                               threshold_percentile: float = 25,
                               save_path: Optional[str] = None) -> Tuple[List[np.ndarray], PCA]:
        """
        Create PCA visualization for a list of images (PCA only, no original images)
        
        Args:
            imgs: List of PIL images
            class_names: List of class names for each image
            patch_size: Patch size
            threshold_percentile: Percentile for thresholding
            save_path: Path to save the visualization
            
        Returns:
            Tuple of (PCA images, PCA object)
        """
        # Extract features
        print(f"🔍 Extracting patch features from {len(imgs)} images...")
        features_list = self.extract_patch_features(imgs, patch_size)
        
        # Compute PCA
        print("📊 Computing PCA...")
        pca_images, pca = self.compute_pca_visualization(features_list, n_components=3)
        
        # Create visualization (PCA only)
        print("🎨 Creating PCA visualization...")
        fig, axes = plt.subplots(1, len(imgs), figsize=(len(imgs) * 4, 4))
        if len(imgs) == 1:
            axes = [axes]
        
        for i, (pca_features, class_name) in enumerate(zip(pca_images, class_names)):
            # Show PCA visualization only
            pca_spatial = self.reshape_to_spatial(pca_features, patch_size)
            
            # Apply threshold on first component
            first_component = pca_spatial[:, :, 0]
            threshold = np.percentile(first_component, threshold_percentile)
            mask = first_component > threshold
            
            pca_masked = pca_spatial.copy()
            pca_masked[~mask] = 0
            
            axes[i].imshow(pca_masked)
            axes[i].set_title(f'PCA - {class_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved PCA visualization to {save_path}")
        
        plt.show()
        
        return pca_images, pca
    
    def create_class_comparison_pca(self, class_images: Dict[str, List[Image.Image]], 
                                   patch_size: int = 14,
                                   samples_per_class: int = 5,
                                   threshold_percentile: float = 25,
                                   save_path: Optional[str] = None) -> Dict[str, Tuple[List[np.ndarray], PCA]]:
        """
        Create PCA visualization comparing different classes (PCA only)
        
        Args:
            class_images: Dictionary mapping class names to lists of images
            patch_size: Patch size
            samples_per_class: Number of samples to visualize per class
            threshold_percentile: Percentile for thresholding
            save_path: Path to save the visualization
            
        Returns:
            Dictionary mapping class names to (PCA images, PCA object)
        """
        results = {}
        
        # Create a large figure for all classes (PCA only)
        num_classes = len(class_images)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class * 4, num_classes * 4))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        for class_name, images in class_images.items():
            print(f"\n🔍 Processing class: {class_name}")
            
            # Sample images for this class
            sample_images = images[:samples_per_class]
            if len(sample_images) < samples_per_class:
                print(f"⚠️ Only {len(sample_images)} images available for class {class_name}")
                # Pad with repeated images if needed
                while len(sample_images) < samples_per_class:
                    sample_images.extend(images[:min(len(images), samples_per_class - len(sample_images))])
            
            # Extract features and compute PCA
            features_list = self.extract_patch_features(sample_images, patch_size)
            pca_images, pca = self.compute_pca_visualization(features_list, n_components=3)
            
            results[class_name] = (pca_images, pca)
            
            # Plot PCA visualizations only
            for i, pca_features in enumerate(pca_images):
                pca_spatial = self.reshape_to_spatial(pca_features, patch_size)
                
                # Apply threshold
                first_component = pca_spatial[:, :, 0]
                threshold = np.percentile(first_component, threshold_percentile)
                mask = first_component > threshold
                
                pca_masked = pca_spatial.copy()
                pca_masked[~mask] = 0
                
                axes[row_idx, i].imshow(pca_masked)
                axes[row_idx, i].set_title(f'{class_name} - PCA {i+1}')
                axes[row_idx, i].axis('off')
            
            row_idx += 1
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved class comparison to {save_path}")
        
        plt.show()
        
        return results
    
    def analyze_pca_components(self, pca: PCA, save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze PCA components and their explained variance
        
        Args:
            pca: Fitted PCA object
            save_path: Path to save the analysis plot
            
        Returns:
            Dictionary with analysis results
        """
        # Get explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of explained variance
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.set_xticks(range(1, len(explained_variance) + 1))
        
        # Cumulative variance plot
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Explained Variance')
        ax2.set_xticks(range(1, len(cumulative_variance) + 1))
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved PCA analysis to {save_path}")
        
        plt.show()
        
        # Return analysis results
        results = {
            'explained_variance_ratio': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'total_variance_explained': float(cumulative_variance[-1])
        }
        
        return results
    
    def save_individual_pca_images(self, imgs: List[Image.Image], 
                                  image_names: List[str],
                                  output_dir: str,
                                  patch_size: int = 14,
                                  threshold_percentile: float = 25) -> List[str]:
        """
        Save individual PCA images with original names
        
        Args:
            imgs: List of PIL images
            image_names: List of original image names (without extension)
            output_dir: Output directory
            patch_size: Patch size
            threshold_percentile: Percentile for thresholding
            
        Returns:
            List of saved file paths
        """
        # Extract features
        print(f"🔍 Extracting patch features from {len(imgs)} images...")
        features_list = self.extract_patch_features(imgs, patch_size)
        
        # Compute PCA
        print("📊 Computing PCA...")
        pca_images, pca = self.compute_pca_visualization(features_list, n_components=3)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        
        # Save each image individually
        for i, (pca_features, image_name) in enumerate(zip(pca_images, image_names)):
            # Create single image figure
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Show PCA visualization
            pca_spatial = self.reshape_to_spatial(pca_features, patch_size)
            
            # Apply threshold on first component
            first_component = pca_spatial[:, :, 0]
            threshold = np.percentile(first_component, threshold_percentile)
            mask = first_component > threshold
            
            pca_masked = pca_spatial.copy()
            pca_masked[~mask] = 0
            
            ax.imshow(pca_masked)
            ax.set_title(f'PCA - {image_name}')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save with original name
            save_path = os.path.join(output_dir, f"{image_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(save_path)
            print(f"💾 Saved {image_name}.png")
        
        return saved_paths
