"""
Configuration for NSGA-III Surrogate-Assisted Hyperparameter Optimization
for Mammography-based Breast Cancer Detection

Author: David
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


@dataclass
class HyperparameterSpace:
    """Defines the search space for all hyperparameters."""
    
    # Architecture hyperparameters
    backbone: List[str] = field(default_factory=lambda: [
        "efficientnet_b0", "resnet50", "densenet121", "convnext_tiny"
    ])
    unfreeze_strategy: List[str] = field(default_factory=lambda: [
        "none", "last_block", "last_2_blocks", "all"
    ])
    dropout_rate: Tuple[float, float] = (0.2, 0.7)
    fc_hidden_size: List[int] = field(default_factory=lambda: [256, 512, 1024])
    use_additional_fc: List[bool] = field(default_factory=lambda: [True, False])
    
    # Optimization hyperparameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)  # log-uniform
    optimizer: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    weight_decay: Tuple[float, float] = (1e-6, 1e-2)  # log-uniform
    scheduler: List[str] = field(default_factory=lambda: ["cosine", "step", "plateau", "none"])
    batch_size: List[int] = field(default_factory=lambda: [8, 16, 32])
    epochs: Tuple[int, int] = (10, 50)
    
    # Class imbalance handling
    loss_function: List[str] = field(default_factory=lambda: ["ce", "weighted_ce", "focal"])
    focal_gamma: Tuple[float, float] = (0.5, 5.0)
    class_weight_strategy: List[str] = field(default_factory=lambda: [
        "none", "inverse_freq", "effective_num"
    ])
    oversampling_ratio: Tuple[float, float] = (1.0, 3.0)
    
    # Data augmentation
    horizontal_flip: List[bool] = field(default_factory=lambda: [True, False])
    rotation_range: Tuple[int, int] = (0, 15)
    brightness_contrast: Tuple[float, float] = (0.0, 0.3)
    zoom_range: Tuple[float, float] = (0.9, 1.1)
    use_mixup: List[bool] = field(default_factory=lambda: [True, False])
    mixup_alpha: Tuple[float, float] = (0.1, 0.4)


@dataclass
class NSGAIIIConfig:
    """Configuration for NSGA-III algorithm."""
    
    n_objectives: int = 5
    n_partitions: int = 4
    pop_size: int = 50
    n_generations: int = 30
    
    # Genetic operators
    crossover_eta: float = 15.0
    crossover_prob: float = 0.9
    mutation_eta: float = 20.0
    
    # Surrogate settings
    surrogate_ratio: float = 0.7
    min_samples_for_surrogate: int = 15
    
    # Checkpointing
    checkpoint_frequency: int = 5


@dataclass
class ExperimentConfig:
    """Master configuration."""
    
    hyperparameter_space: HyperparameterSpace = field(default_factory=HyperparameterSpace)
    nsga3: NSGAIIIConfig = field(default_factory=NSGAIIIConfig)
    
    experiment_name: str = "nsga3_mammography"
    output_dir: str = "/content/outputs"
    device: str = "cuda"
    seed: int = 42


# Objective definitions
OBJECTIVES = {
    "sensitivity": {"direction": "maximize", "index": 0},
    "specificity": {"direction": "maximize", "index": 1},
    "auc": {"direction": "maximize", "index": 2},
    "model_size": {"direction": "minimize", "index": 3},
    "inference_time": {"direction": "minimize", "index": 4}
}
