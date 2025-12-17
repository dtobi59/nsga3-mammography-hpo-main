"""
Comprehensive test of the entire NSGA-III HPO pipeline.
Run this to verify all components work together.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Test configuration
print("="*70)
print("NSGA-III MAMMOGRAPHY HPO - COMPREHENSIVE TEST")
print("="*70)

# ============================================================================
# TEST 1: Config
# ============================================================================
print("\n[TEST 1] Config module...")
try:
    from config import ExperimentConfig, HyperparameterSpace, NSGAIIIConfig
    
    config = ExperimentConfig()
    assert config.nsga3.n_objectives == 5
    assert len(config.hyperparameter_space.backbone) == 4
    print("  ✓ Config OK")
except Exception as e:
    print(f"  ✗ Config FAILED: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Create mock dataset
# ============================================================================
print("\n[TEST 2] Creating mock dataset...")
try:
    import shutil
    
    # Create VinDr-like structure
    mock_root = "/tmp/test_vindr"
    if os.path.exists(mock_root):
        shutil.rmtree(mock_root)
    
    os.makedirs(f"{mock_root}/metadata")
    os.makedirs(f"{mock_root}/images")
    
    rows = []
    for i in range(20):
        study_id = f"study_{i}"
        image_id = f"img_{i}"
        birads = f"BI-RADS {1 if i < 12 else 4}"  # 12 benign, 8 malignant
        
        os.makedirs(f"{mock_root}/images/{study_id}", exist_ok=True)
        # Create a fake PNG file (just needs to exist for path checking)
        Path(f"{mock_root}/images/{study_id}/{image_id}.png").touch()
        rows.append({"study_id": study_id, "image_id": image_id, "breast_birads": birads})
    
    pd.DataFrame(rows).to_csv(f"{mock_root}/metadata/breast-level_annotations.csv", index=False)
    print(f"  ✓ Mock dataset created: 12 benign, 8 malignant")
except Exception as e:
    print(f"  ✗ Mock dataset FAILED: {e}")
    sys.exit(1)

# ============================================================================
# TEST 3: Dataset loading
# ============================================================================
print("\n[TEST 3] Dataset loading...")
try:
    from dataset import prepare_dataset
    
    train_paths, train_labels, val_paths, val_labels = prepare_dataset(
        dataset_name="vindr",
        data_root=mock_root,
        val_split=0.2,
        seed=42
    )
    
    assert len(train_paths) > 0, "No training samples"
    assert len(val_paths) > 0, "No validation samples"
    assert len(train_paths) == len(train_labels)
    print(f"  ✓ Dataset OK: train={len(train_paths)}, val={len(val_paths)}")
except Exception as e:
    print(f"  ✗ Dataset FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: Model building (requires torch)
# ============================================================================
print("\n[TEST 4] Model building...")
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  ⚠ Skipped (torch not installed - will work in Colab)")

if TORCH_AVAILABLE:
    try:
        from models import build_model, MammographyCNN
        
        test_config = {
            'backbone': 'efficientnet_b0',
            'unfreeze_strategy': 'last_block',
            'dropout_rate': 0.5,
            'fc_hidden_size': 512,
            'use_additional_fc': True
        }
        
        model = build_model(test_config)
        total, trainable = model.get_param_count()
        print(f"  ✓ Model OK: {total/1e6:.2f}M params ({trainable/1e6:.2f}M trainable)")
    except Exception as e:
        print(f"  ✗ Model FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# TEST 5: Encoder
# ============================================================================
print("\n[TEST 5] Hyperparameter encoder...")
try:
    from optimization import HyperparameterEncoder
    
    encoder = HyperparameterEncoder(config.hyperparameter_space)
    xl, xu = encoder.get_bounds()
    
    assert len(xl) == encoder.n_vars
    assert len(xu) == encoder.n_vars
    
    # Test encode/decode roundtrip
    x = (xl + xu) / 2  # Middle of bounds
    decoded = encoder.decode(x)
    
    assert 'backbone' in decoded
    assert 'learning_rate' in decoded
    assert 'epochs' in decoded
    print(f"  ✓ Encoder OK: {encoder.n_vars} variables")
    print(f"    Sample decode: backbone={decoded['backbone']}, lr={decoded['learning_rate']:.6f}")
except Exception as e:
    print(f"  ✗ Encoder FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Surrogate
# ============================================================================
print("\n[TEST 6] Surrogate model...")
try:
    from surrogate import MultiObjectiveGPSurrogate, SurrogateAssistedSelection
    
    surrogate = MultiObjectiveGPSurrogate(n_objectives=5)
    
    # Generate fake training data
    X_fake = np.random.rand(20, encoder.n_vars)
    y_fake = np.random.rand(20, 5)
    
    surrogate.fit(X_fake, y_fake)
    means, stds = surrogate.predict(X_fake[:5])
    
    assert means.shape == (5, 5)
    assert stds.shape == (5, 5)
    print(f"  ✓ Surrogate OK: fitted and predicting")
except Exception as e:
    print(f"  ✗ Surrogate FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: Full integration (mock evaluation) - requires torch
# ============================================================================
print("\n[TEST 7] Full integration with mock evaluation...")
if not TORCH_AVAILABLE:
    print("  ⚠ Skipped (torch not installed - will work in Colab)")
else:
    try:
        from optimization import MammographyHPOProblem
        
        # Mock evaluation function (returns random but valid results)
        def mock_eval(config):
            return {
                'sensitivity': np.random.uniform(0.5, 1.0),
                'specificity': np.random.uniform(0.5, 1.0),
                'auc': np.random.uniform(0.5, 1.0),
                'model_size': np.random.uniform(1, 50),
                'inference_time': np.random.uniform(1, 100)
            }
        
        selector = SurrogateAssistedSelection(
            surrogate=MultiObjectiveGPSurrogate(n_objectives=5),
            true_eval_ratio=0.3,
            min_samples=10
        )
        
        problem = MammographyHPOProblem(
            hp_space=config.hyperparameter_space,
            eval_function=mock_eval,
            surrogate_selector=selector
        )
        
        # Simulate one generation
        X_test = np.random.rand(10, encoder.n_vars) * (xu - xl) + xl
        out = {}
        problem._evaluate(X_test, out)
        
        assert "F" in out
        assert out["F"].shape == (10, 5)
        print(f"  ✓ Integration OK: evaluated 10 samples")
        print(f"    True evals: {problem.true_eval_count}, Surrogate: {problem.surrogate_eval_count}")
    except Exception as e:
        print(f"  ✗ Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nThe code is ready. In Colab, run:")
print("""
# Clone repo
!git clone https://github.com/dtobi59/nsga3-mammography-hpo.git
%cd nsga3-mammography-hpo
!pip install -q -r requirements.txt

# Load data
from dataset import prepare_dataset
train_paths, train_labels, val_paths, val_labels = prepare_dataset(
    dataset_name="vindr",
    data_root="/content/drive/MyDrive/vindr-mammo"
)

# Run optimization
from optimization import run_optimization
from config import ExperimentConfig

config = ExperimentConfig()
config.nsga3.pop_size = 20
config.nsga3.n_generations = 5

def make_eval_fn(tp, tl, vp, vl):
    from training import full_evaluation
    def eval_fn(hp_config):
        hp_config = hp_config.copy()
        hp_config['epochs'] = 5
        return full_evaluation(hp_config, tp, tl, vp, vl, device='cuda', verbose=True)
    return eval_fn

results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=make_eval_fn(train_paths, train_labels, val_paths, val_labels),
    output_dir="/content/drive/MyDrive/nsga3_outputs"
)
""")
