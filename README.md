# NSGA-III Surrogate-Assisted Hyperparameter Optimization for Mammography Classification

Multi-objective hyperparameter optimization for deep learning models in breast cancer detection using mammography images.

## Features

- **NSGA-III optimization** with 5 objectives: sensitivity, specificity, AUC, model size, inference time
- **Surrogate-assisted evaluation** using Gaussian Process to reduce computational cost
- **Multiple CNN backbones**: EfficientNet-B0, ResNet50, DenseNet121, ConvNeXt-Tiny
- **DICOM support**: Direct loading of medical imaging formats
- **Class imbalance handling**: Focal loss, weighted CE, oversampling

## Installation

```bash
git clone https://github.com/dtobi59/nsga3-mammography-hpo.git
cd nsga3-mammography-hpo
pip install -r requirements.txt
```

## Quick Start (Google Colab)

```python
# Clone and setup
!git clone https://github.com/dtobi59/nsga3-mammography-hpo.git
%cd nsga3-mammography-hpo
!pip install -q -r requirements.txt

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
from dataset import prepare_dataset
train_paths, train_labels, val_paths, val_labels = prepare_dataset(
    dataset_name="vindr",  # or "inbreast"
    data_root="/content/drive/MyDrive/vindr-mammo"
)

# Create evaluation function
from training import full_evaluation

def make_eval_fn(tp, tl, vp, vl):
    def eval_fn(hp_config):
        hp_config = hp_config.copy()
        hp_config['epochs'] = 5  # Reduce for testing
        return full_evaluation(hp_config, tp, tl, vp, vl, device='cuda', verbose=True)
    return eval_fn

# Run optimization
from optimization import run_optimization
from config import ExperimentConfig

config = ExperimentConfig()
config.nsga3.pop_size = 20
config.nsga3.n_generations = 5

results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=make_eval_fn(train_paths, train_labels, val_paths, val_labels),
    output_dir="/content/drive/MyDrive/nsga3_outputs"
)

# View Pareto front
print(f"Found {len(results['pareto_configs'])} Pareto-optimal solutions")
for i, (cfg, obj) in enumerate(zip(results['pareto_configs'], results['pareto_F'])):
    print(f"\nSolution {i+1}:")
    print(f"  Backbone: {cfg['backbone']}")
    print(f"  AUC: {obj[2]:.4f}, Sens: {obj[0]:.4f}, Spec: {obj[1]:.4f}")
    print(f"  Size: {obj[3]:.2f}M, Time: {obj[4]:.2f}ms")
```

## Dataset Structure

### VinDr-Mammo
```
vindr-mammo/
├── images/
│   └── {study_id}/
│       └── {image_id}.dicom
└── metadata/
    └── breast-level_annotations.csv
```

### INbreast
```
inbreast/
└── INbreast Release 1.0/
    ├── AllDICOMs/
    │   └── {id}_{hash}_MG_{L/R}_{CC/MLO}_ANON.dcm
    └── INbreast.csv
```

## Hyperparameter Search Space

| Category | Parameter | Range/Options |
|----------|-----------|---------------|
| Architecture | backbone | efficientnet_b0, resnet50, densenet121, convnext_tiny |
| | unfreeze_strategy | none, last_block, last_2_blocks, all |
| | dropout_rate | [0.2, 0.7] |
| | fc_hidden_size | 256, 512, 1024 |
| Optimization | learning_rate | [1e-5, 1e-2] (log) |
| | optimizer | adam, adamw, sgd |
| | batch_size | 8, 16, 32 |
| | epochs | [10, 50] |
| Loss | loss_function | ce, weighted_ce, focal |
| | focal_gamma | [0.5, 5.0] |
| Augmentation | horizontal_flip | True, False |
| | rotation_range | [0, 15] |
| | use_mixup | True, False |

## Objectives

1. **Sensitivity** (maximize) - True positive rate for malignant cases
2. **Specificity** (maximize) - True negative rate for benign cases  
3. **AUC** (maximize) - Area under ROC curve
4. **Model Size** (minimize) - Parameters in millions
5. **Inference Time** (minimize) - Milliseconds per image

## Files

- `config.py` - Configuration dataclasses
- `dataset.py` - Data loading with DICOM support
- `models.py` - CNN architectures
- `training.py` - Training and evaluation
- `surrogate.py` - Gaussian Process surrogate
- `optimization.py` - NSGA-III implementation

## Citation

If you use this code, please cite:

```bibtex
@thesis{nsga3_mammography_2024,
  author = {David},
  title = {Automated Hyperparameter Tuning for Deep Learning Models in Breast Cancer Detection},
  school = {Białystok University of Technology},
  year = {2024}
}
```

## License

MIT License
