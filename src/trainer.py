"""
Class Trainer chứa logic train/val và log W&B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import os

from .utils import (
    calculate_metrics, 
    generate_grad_cam_image, 
    save_checkpoint, 
    load_checkpoint,
    log_confusion_matrix,
    create_embedding_plot,
    log_sample_images
)

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for similarity learning - pushes different images apart"""
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features1, features2):
        """
        Args:
            features1: Features from first image (B, D)
            features2: Features from second image (B, D)
        """
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(features1, features2)
        
        # Contrastive loss - always push different images apart
        # We want to maximize the distance, so minimize the similarity
        loss = torch.clamp(similarity + self.margin, min=0).pow(2)
        
        return loss.mean()

class InfoNCELoss(nn.Module):
    """InfoNCE Loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features1, features2, labels):
        """
        Args:
            features1: Features from first image (B, D)
            features2: Features from second image (B, D)
            labels: Binary labels (1 for same class, 0 for different class)
        """
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # Create labels for InfoNCE
        batch_size = features1.size(0)
        labels_matrix = torch.zeros(batch_size, batch_size).to(features1.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if labels[i] == labels[j]:
                    labels_matrix[i, j] = 1
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = exp_sim * labels_matrix
        neg_sim = exp_sim * (1 - labels_matrix)
        
        loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1)))
        return loss.mean()

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = F.normalize(torch.cat((z_i, z_j), dim=0), p=2, dim=1)
        similarity_matrix = torch.matmul(z, z.T)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(self.device)
        denominator = mask * torch.exp(similarity_matrix / self.temperature)
        
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss

class Trainer:
    """Main Trainer class"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 config: Dict[str, Any],
                 wandb_run):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.wandb_run = wandb_run
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.loss_fn = self._setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"💻 Training on device: {self.device}")
        print(f"📚 Training batches: {len(train_loader)}")
        print(f"📊 Validation batches: {len(val_loader)}")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        optimizer_name = self.config.training.get('optimizer', 'AdamW')
        lr = self.config.training.get('learning_rate', 1e-4)
        weight_decay = self.config.training.get('weight_decay', 0.01)
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print(f"🔧 Using optimizer: {optimizer_name} with lr={lr}")
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_name = self.config.training.get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.get('epochs', 50)
            )
        elif scheduler_name == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.get('epochs', 50) // 3,
                gamma=0.1
            )
        else:
            scheduler = None
        
        print(f"📈 Using scheduler: {scheduler_name}")
        return scheduler
    
    def _setup_loss_function(self):
        """Setup loss function based on training strategy"""
        strategy = self.config.training.get('strategy', 'contrastive')
        
        if strategy == 'contrastive':
            loss_fn = ContrastiveLoss(
                margin=self.config.training.get('margin', 0.5),
                temperature=self.config.training.get('temperature', 0.07)
            )
        elif strategy == 'info_nce':
            loss_fn = InfoNCELoss(
                temperature=self.config.training.get('temperature', 0.07)
            )
        elif strategy == 'ntxent':
            loss_fn = NTXentLoss(
                temperature=self.config.training.get('temperature', 0.07),
                device=self.device
            )
        else:
            # Default to CrossEntropy for classification
            loss_fn = nn.CrossEntropyLoss()
        
        print(f"📝 Using loss function: {strategy}")
        return loss_fn
    
    def _train_one_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 2:
                # Standard classification
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                
            elif len(batch) == 3:
                # Contrastive learning
                images1, images2, labels = batch
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                # labels = labels.to(self.device)  # No longer needed
                
                self.optimizer.zero_grad()
                features1 = self.model.get_features(images1)
                features2 = self.model.get_features(images2)
                loss = self.loss_fn(features1, features2)  # Only pass features
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to W&B every N steps
            if batch_idx % self.config.logging.get('log_frequency', 100) == 0:
                self.wandb_run.log({
                    "train_step_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "epoch": self.current_epoch
                })
        
        return total_loss / num_batches
    
    def _validate_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_features = []
        all_labels = []
        all_outputs = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch + 1}")
        
        with torch.no_grad():
            for batch in progress_bar:
                if len(batch) == 2:
                    # Standard classification
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(images)
                    features = self.model.get_features(images)
                    loss = self.loss_fn(outputs, targets)
                    
                    all_outputs.append(outputs.cpu())
                    all_features.append(features.cpu())
                    all_labels.append(targets.cpu())
                    
                elif len(batch) == 3:
                    # Contrastive learning
                    images1, images2, labels = batch
                    images1 = images1.to(self.device)
                    images2 = images2.to(self.device)
                    # labels = labels.to(self.device)  # No longer needed for loss
                    
                    features1 = self.model.get_features(images1)
                    features2 = self.model.get_features(images2)
                    loss = self.loss_fn(features1, features2)  # Only pass features
                    
                    all_features.append(features1.cpu())
                    all_labels.append(labels.cpu())  # Still collect labels for metrics
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss = total_loss / num_batches
        
        # Calculate metrics
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate retrieval metrics
            val_metrics = calculate_metrics(
                all_features, 
                all_labels, 
                self.config.evaluation.get('metrics', ['HitRate@10', 'MRR'])
            )
            
            # Log Grad-CAM if enabled
            if self.config.evaluation.get('log_grad_cam', False):
                try:
                    grad_cam_img = generate_grad_cam_image(
                        self.model, 
                        images[:self.config.evaluation.get('grad_cam_samples', 1)], 
                        self.device
                    )
                    val_metrics["grad_cam_example"] = wandb.Image(grad_cam_img)
                except Exception as e:
                    print(f"⚠️ Could not generate Grad-CAM: {e}")
            
            # Log embedding visualization
            if self.config.evaluation.get('save_embeddings', False):
                try:
                    create_embedding_plot(
                        all_features[:1000],  # Subsample for visualization
                        all_labels[:1000],
                        self.wandb_run,
                        method="tsne"
                    )
                except Exception as e:
                    print(f"⚠️ Could not create embedding plot: {e}")
        
        else:
            val_metrics = {}
        
        return val_loss, val_metrics
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        
        total_epochs = self.config.training.get('epochs', 50)
        save_every = self.config.training.get('save_every', 10)
        early_stopping_patience = self.config.training.get('early_stopping_patience', 10)
        
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self._train_one_epoch()
            
            # Validate one epoch
            val_loss, val_metrics = self._validate_one_epoch()
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics to W&B
            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                **val_metrics
            }
            self.wandb_run.log(log_data)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{total_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    checkpoint_path
                )
            
            # Check for best model
            primary_metric = val_metrics.get('HitRate@10', val_metrics.get('MRR', 0))
            if primary_metric > self.best_val_metric:
                self.best_val_metric = primary_metric
                self.patience_counter = 0
                
                # Save best model
                best_model_path = self.output_dir / "best_model.pth"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    best_model_path
                )
                print(f"💾 Saved best model with {primary_metric:.4f}")
                
                # Log best model to W&B
                self.wandb_run.log({"best_metric": primary_metric})
                
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print(f"⏰ Early stopping after {early_stopping_patience} epochs without improvement")
                    break
        
        print("✅ Training completed!")
        
        # Final evaluation on test set
        if self.test_loader:
            print("🔍 Evaluating on test set...")
            self.evaluate_test()
    
    def evaluate_test(self):
        """Evaluate on test set"""
        self.model.eval()
        
        all_features = []
        all_labels = []
        all_outputs = []
        
        print("🧪 Running test evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Evaluation"):
                if len(batch) == 2:
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(images)
                    features = self.model.get_features(images)
                    
                    all_outputs.append(outputs.cpu())
                    all_features.append(features.cpu())
                    all_labels.append(targets.cpu())
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate test metrics
            test_metrics = calculate_metrics(
                all_features, 
                all_labels, 
                self.config.evaluation.get('metrics', ['HitRate@10', 'MRR'])
            )
            
            # Log test metrics
            test_log_data = {f"test_{k}": v for k, v in test_metrics.items()}
            self.wandb_run.log(test_log_data)
            
            # Print test results
            print("📊 Test Results:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            
            # Create final embedding visualization
            if self.config.evaluation.get('save_embeddings', False):
                try:
                    create_embedding_plot(
                        all_features[:1000],
                        all_labels[:1000],
                        self.wandb_run,
                        method="tsne"
                    )
                except Exception as e:
                    print(f"⚠️ Could not create final embedding plot: {e}")
        
        print("✅ Test evaluation completed!")
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        print(f"📥 Resuming training from: {checkpoint_path}")
        
        epoch, loss = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler
        )
        
        self.current_epoch = epoch
        print(f"🔄 Resumed from epoch {epoch + 1}")
        
        # Continue training
        self.train()
