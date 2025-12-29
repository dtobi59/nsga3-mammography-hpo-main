"""
Example: Using the Kaggle VinDr-Mammo PNG Dataset

Dataset: https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png

Download instructions:
1. Install Kaggle API: pip install kaggle
2. Set up Kaggle credentials: https://www.kaggle.com/docs/api
3. Download dataset:
   kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png
4. Unzip to your desired location

Author: David
"""

from dataset import prepare_dataset
from training import full_evaluation
from optimization import run_optimization
from config import ExperimentConfig

# ============================================================================
# STEP 1: Download the Kaggle dataset
# ============================================================================
"""
# Run these commands in your terminal:

# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png

# Unzip
unzip vindr-mammogram-dataset-dicom-to-png.zip -d /path/to/kaggle_vindr_data
"""

# ============================================================================
# STEP 2: Load dataset
# ============================================================================

# Update this path to where you extracted the dataset
DATA_ROOT = "/path/to/kaggle_vindr_data"  # Change this!

# For Google Colab:
# DATA_ROOT = "/content/drive/MyDrive/kaggle_vindr_data"

# For local Windows:
# DATA_ROOT = "C:/Users/YourName/Downloads/kaggle_vindr_data"

print("Loading Kaggle VinDr-Mammo PNG dataset...")
train_paths, train_labels, val_paths, val_labels = prepare_dataset(
    dataset_name="kaggle_vindr_png",
    data_root=DATA_ROOT,
    val_split=0.2,
    seed=42
)

print(f"\nDataset loaded successfully!")
print(f"Train samples: {len(train_paths)}")
print(f"Val samples: {len(val_paths)}")
print(f"Class distribution - Train: Benign={train_labels.count(0)}, Malignant={train_labels.count(1)}")
print(f"Class distribution - Val: Benign={val_labels.count(0)}, Malignant={val_labels.count(1)}")

# ============================================================================
# STEP 3: Test with a single hyperparameter configuration
# ============================================================================

print("\n" + "="*60)
print("Testing single configuration...")
print("="*60)

test_config = {
    'backbone': 'efficientnet_b0',
    'unfreeze_strategy': 'last_block',
    'dropout_rate': 0.3,
    'fc_hidden_size': 512,
    'use_additional_fc': True,
    'learning_rate': 1e-4,
    'optimizer': 'adamw',
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'batch_size': 16,
    'epochs': 3,  # Small for testing
    'loss_function': 'focal',
    'focal_gamma': 2.0,
    'class_weight_strategy': 'inverse_freq',
    'oversampling_ratio': 1.5,
    'horizontal_flip': True,
    'rotation_range': 10,
    'brightness_contrast': 0.2,
    'use_mixup': False,
    'mixup_alpha': 0.2
}

results = full_evaluation(
    test_config,
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    device='cuda',  # Change to 'cpu' if no GPU
    verbose=True
)

print(f"\nResults:")
print(f"  Sensitivity: {results['sensitivity']:.4f}")
print(f"  Specificity: {results['specificity']:.4f}")
print(f"  AUC: {results['auc']:.4f}")
print(f"  Model Size: {results['model_size']:.2f}M parameters")

# ============================================================================
# STEP 4: Run full NSGA-III optimization
# ============================================================================

print("\n" + "="*60)
print("Starting NSGA-III Hyperparameter Optimization...")
print("="*60)

# Create evaluation function
def make_eval_fn(tp, tl, vp, vl):
    def eval_fn(hp_config):
        hp_config = hp_config.copy()
        hp_config['epochs'] = 5  # Reduce for faster optimization
        return full_evaluation(hp_config, tp, tl, vp, vl, device='cuda', verbose=True)
    return eval_fn

# Configure optimization
config = ExperimentConfig()
config.nsga3.pop_size = 20          # Population size
config.nsga3.n_generations = 10     # Number of generations
config.nsga3.surrogate_ratio = 0.7  # Use surrogate for 70% of evaluations

# Run optimization
OUTPUT_DIR = "/path/to/output"  # Change this!

optimization_results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=make_eval_fn(train_paths, train_labels, val_paths, val_labels),
    output_dir=OUTPUT_DIR,
    seed=42,
    verbose=True
)

# ============================================================================
# STEP 5: Analyze results
# ============================================================================

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)

print(f"\nFound {len(optimization_results['pareto_configs'])} Pareto-optimal solutions")
print(f"True evaluations: {optimization_results['n_true_evals']}")
print(f"Surrogate evaluations: {optimization_results['n_surrogate_evals']}")

print("\nTop 5 solutions:")
for i, (cfg, obj) in enumerate(zip(
    optimization_results['pareto_configs'][:5],
    optimization_results['pareto_F'][:5]
)):
    print(f"\nSolution {i+1}:")
    print(f"  Backbone: {cfg['backbone']}")
    print(f"  Optimizer: {cfg['optimizer']}, LR: {cfg['learning_rate']:.2e}")
    print(f"  Batch size: {cfg['batch_size']}, Epochs: {cfg['epochs']}")
    print(f"  Results:")
    print(f"    Sensitivity: {obj[0]:.4f}")
    print(f"    Specificity: {obj[1]:.4f}")
    print(f"    AUC: {obj[2]:.4f}")
    print(f"    Model Size: {obj[3]:.2f}M")

# ============================================================================
# STEP 6: Visualize Pareto front (optional)
# ============================================================================

try:
    import matplotlib.pyplot as plt
    import numpy as np

    F = optimization_results['pareto_F']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pareto Front - Trade-offs', fontsize=16)

    # Sensitivity vs Specificity
    axes[0, 0].scatter(F[:, 0], F[:, 1], alpha=0.6)
    axes[0, 0].set_xlabel('Sensitivity')
    axes[0, 0].set_ylabel('Specificity')
    axes[0, 0].grid(True)

    # AUC vs Model Size
    axes[0, 1].scatter(F[:, 2], F[:, 3], alpha=0.6)
    axes[0, 1].set_xlabel('AUC')
    axes[0, 1].set_ylabel('Model Size (M)')
    axes[0, 1].grid(True)

    # Sensitivity vs Model Size
    axes[0, 2].scatter(F[:, 0], F[:, 3], alpha=0.6)
    axes[0, 2].set_xlabel('Sensitivity')
    axes[0, 2].set_ylabel('Model Size (M)')
    axes[0, 2].grid(True)

    # Specificity vs Model Size
    axes[1, 0].scatter(F[:, 1], F[:, 3], alpha=0.6)
    axes[1, 0].set_xlabel('Specificity')
    axes[1, 0].set_ylabel('Model Size (M)')
    axes[1, 0].grid(True)

    # AUC vs Sensitivity
    axes[1, 1].scatter(F[:, 2], F[:, 0], alpha=0.6)
    axes[1, 1].set_xlabel('AUC')
    axes[1, 1].set_ylabel('Sensitivity')
    axes[1, 1].grid(True)

    # AUC vs Specificity
    axes[1, 2].scatter(F[:, 2], F[:, 1], alpha=0.6)
    axes[1, 2].set_xlabel('AUC')
    axes[1, 2].set_ylabel('Specificity')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto_front.png", dpi=150)
    print(f"\nPareto front visualization saved to: {OUTPUT_DIR}/pareto_front.png")

except ImportError:
    print("\nMatplotlib not available for visualization")
