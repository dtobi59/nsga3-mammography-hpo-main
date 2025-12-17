"""
Training module for mammography classification.
Includes loss functions, optimizers, and evaluation metrics.

Author: David
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from dataset import mixup_data, mixup_criterion


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Lin et al., 2017)."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        # Store alpha on CPU, will move to device in forward()
        self.register_buffer('alpha', alpha)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # alpha is already on correct device due to register_buffer
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def get_class_weights(labels: List[int], strategy: str = 'inverse_freq') -> torch.Tensor:
    """Compute class weights."""
    counts = np.bincount(labels)
    
    if strategy == 'inverse_freq':
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
    elif strategy == 'effective_num':
        beta = 0.9999
        effective = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective
        weights = weights / weights.sum() * len(counts)
    else:
        weights = np.ones(len(counts))
    
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(config: Dict, class_weights: Optional[torch.Tensor] = None, device: str = 'cpu') -> nn.Module:
    """Build loss function from config."""
    loss_type = config.get('loss_function', 'ce')
    
    if loss_type == 'focal':
        gamma = config.get('focal_gamma', 2.0)
        criterion = FocalLoss(gamma=gamma, alpha=class_weights)
        return criterion.to(device)
    elif loss_type == 'weighted_ce' and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_type = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    wd = config.get('weight_decay', 1e-4)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if opt_type == 'adam':
        return Adam(params, lr=lr, weight_decay=wd)
    elif opt_type == 'sgd':
        return SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        return AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, config: Dict, epochs: int):
    """Build learning rate scheduler."""
    sched_type = config.get('scheduler', 'cosine')
    
    if sched_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_type == 'step':
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif sched_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        return None


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            _, preds = outputs.max(1)
            correct += (lam * preds.eq(targets_a).sum().float() + 
                       (1 - lam) * preds.eq(targets_b).sum().float()).item()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total += targets.size(0)
    
    return {'loss': total_loss / len(loader), 'accuracy': correct / total}


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_targets = [], [], []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = outputs.max(1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Metrics
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    
    return {
        'loss': total_loss / len(loader),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc
    }


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    train_labels: List[int],
    device: str = 'cuda',
    patience: int = 10,
    verbose: bool = False
) -> Dict:
    """Complete training pipeline."""
    model = model.to(device)
    epochs = config.get('epochs', 30)
    
    # Class weights - compute if needed
    weight_strategy = config.get('class_weight_strategy', 'none')
    class_weights = None
    if weight_strategy != 'none':
        class_weights = get_class_weights(train_labels, weight_strategy)
    
    # Build criterion - pass device so weights are moved correctly
    criterion = build_criterion(config, class_weights, device)
    
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    
    use_mixup = config.get('use_mixup', False)
    mixup_alpha = config.get('mixup_alpha', 0.2)
    
    # Training loop
    best_auc = 0
    best_metrics = None
    no_improve = 0
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup, mixup_alpha)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, "
                  f"Sens: {val_metrics['sensitivity']:.4f}, "
                  f"Spec: {val_metrics['specificity']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_metrics if best_metrics else val_metrics


def full_evaluation(
    config: Dict,
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    device: str = 'cuda',
    verbose: bool = False
) -> Dict:
    """
    Full evaluation pipeline: build model, train, evaluate.
    Returns dict with sensitivity, specificity, auc, model_size, inference_time.
    """
    from models import build_model, measure_inference_time, get_param_count
    from dataset import create_dataloaders
    
    # Build model
    model = build_model(config)
    
    # Get model metrics
    total_params, trainable_params = get_param_count(model)
    model_size = total_params / 1e6  # In millions
    
    # Create dataloaders
    aug_config = {
        'horizontal_flip': config.get('horizontal_flip', True),
        'rotation_range': config.get('rotation_range', 10),
        'brightness_contrast': config.get('brightness_contrast', 0.2)
    }
    
    train_loader, val_loader = create_dataloaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=config.get('batch_size', 16),
        image_size=config.get('image_size', 224),
        num_workers=2,
        aug_config=aug_config,
        use_weighted_sampling=config.get('oversampling_ratio', 1.0) > 1.0
    )
    
    # Train and evaluate
    metrics = train_and_evaluate(
        model, train_loader, val_loader, config, train_labels,
        device=device, patience=10, verbose=verbose
    )
    
    # Measure inference time
    model.eval()
    inference_time = measure_inference_time(model, device=device, image_size=config.get('image_size', 224))
    
    return {
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'auc': metrics['auc'],
        'model_size': model_size,
        'inference_time': inference_time
    }
