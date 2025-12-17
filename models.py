"""
CNN Models for Mammography Classification
Supports EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny

Author: David
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple
import time


class MammographyCNN(nn.Module):
    """
    CNN for mammography classification with configurable backbone and fine-tuning.
    """
    
    BACKBONES = {
        'efficientnet_b0': {'model': models.efficientnet_b0, 'weights': models.EfficientNet_B0_Weights.IMAGENET1K_V1, 'features': 1280},
        'resnet50': {'model': models.resnet50, 'weights': models.ResNet50_Weights.IMAGENET1K_V2, 'features': 2048},
        'densenet121': {'model': models.densenet121, 'weights': models.DenseNet121_Weights.IMAGENET1K_V1, 'features': 1024},
        'convnext_tiny': {'model': models.convnext_tiny, 'weights': models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 'features': 768},
    }
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        num_classes: int = 2,
        unfreeze_strategy: str = 'last_block',
        dropout_rate: float = 0.5,
        fc_hidden_size: int = 512,
        use_additional_fc: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        cfg = self.BACKBONES[backbone]
        self.backbone = cfg['model'](weights=cfg['weights'])
        self.feature_dim = cfg['features']
        
        # Remove original classifier
        self._remove_classifier()
        
        # Apply freezing strategy
        self._apply_freeze_strategy(unfreeze_strategy)
        
        # Build new classifier
        if use_additional_fc:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(fc_hidden_size, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, num_classes)
            )
    
    def _remove_classifier(self):
        """Remove the original classification head."""
        if self.backbone_name == 'resnet50':
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == 'densenet121':
            self.backbone.classifier = nn.Identity()
        elif self.backbone_name in ['efficientnet_b0', 'convnext_tiny']:
            self.backbone.classifier = nn.Identity()
    
    def _apply_freeze_strategy(self, strategy: str):
        """Apply freezing strategy."""
        if strategy == 'none':
            # Freeze all
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        elif strategy == 'all':
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
                
        elif strategy in ['last_block', 'last_2_blocks']:
            # Freeze all first
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            n_blocks = 1 if strategy == 'last_block' else 2
            
            if self.backbone_name == 'resnet50':
                layers = ['layer4', 'layer3'][:n_blocks]
                for name in layers:
                    for param in getattr(self.backbone, name).parameters():
                        param.requires_grad = True
                        
            elif self.backbone_name == 'densenet121':
                # Unfreeze last dense blocks
                features = list(self.backbone.features.children())
                for module in features[-n_blocks:]:
                    for param in module.parameters():
                        param.requires_grad = True
                        
            elif self.backbone_name in ['efficientnet_b0', 'convnext_tiny']:
                features = self.backbone.features
                for i in range(len(features) - n_blocks, len(features)):
                    for param in features[i].parameters():
                        param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        if self.backbone_name == 'resnet50':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
        elif self.backbone_name == 'densenet121':
            x = self.backbone.features(x)
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
        elif self.backbone_name in ['efficientnet_b0', 'convnext_tiny']:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        
        return self.classifier(x)
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get (total, trainable) parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(config: Dict[str, Any]) -> MammographyCNN:
    """Build model from config dict."""
    return MammographyCNN(
        backbone=config.get('backbone', 'efficientnet_b0'),
        num_classes=2,
        unfreeze_strategy=config.get('unfreeze_strategy', 'last_block'),
        dropout_rate=config.get('dropout_rate', 0.5),
        fc_hidden_size=config.get('fc_hidden_size', 512),
        use_additional_fc=config.get('use_additional_fc', True)
    )


def get_param_count(model: nn.Module) -> Tuple[int, int]:
    """Get (total, trainable) parameter counts for any model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model: nn.Module, device: str = 'cuda', image_size: int = 224, n_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    model = model.to(device)
    
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    
    return sum(times) / len(times)
