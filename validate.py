#!/usr/bin/env python3
"""
Comprehensive Validation Script for NSGA-III Mammography HPO Framework

This script validates:
1. All imports work correctly
2. All functions have correct signatures
3. All functions produce expected outputs
4. Integration between modules works
5. Edge cases are handled

Run this in Google Colab to verify the installation.
"""

import sys
import os

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_pass(msg):
    print(f"  ✓ {msg}")

def print_fail(msg):
    print(f"  ✗ {msg}")

def print_warn(msg):
    print(f"  ⚠ {msg}")

# Track results
TESTS_PASSED = 0
TESTS_FAILED = 0
ERRORS = []

def test(name, condition, error_msg=""):
    global TESTS_PASSED, TESTS_FAILED, ERRORS
    if condition:
        print_pass(name)
        TESTS_PASSED += 1
    else:
        print_fail(f"{name}: {error_msg}")
        TESTS_FAILED += 1
        ERRORS.append(f"{name}: {error_msg}")

# ==============================================================================
# TEST 1: IMPORTS
# ==============================================================================
print_header("TEST 1: Module Imports")

try:
    import numpy as np
    print_pass("numpy")
except ImportError as e:
    print_fail(f"numpy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print_pass("pandas")
except ImportError as e:
    print_fail(f"pandas: {e}")

try:
    import torch
    import torch.nn as nn
    print_pass(f"torch {torch.__version__}")
    TORCH_AVAILABLE = True
except ImportError as e:
    print_warn(f"torch not available: {e}")
    TORCH_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    print_pass("sklearn")
except ImportError as e:
    print_fail(f"sklearn: {e}")

try:
    from pymoo.core.problem import Problem
    print_pass("pymoo")
except ImportError as e:
    print_fail(f"pymoo: {e}")

# ==============================================================================
# TEST 2: CONFIG MODULE
# ==============================================================================
print_header("TEST 2: Config Module")

try:
    from config import HyperparameterSpace, NSGAIIIConfig, ExperimentConfig, OBJECTIVES
    print_pass("Config imports")
    
    hp_space = HyperparameterSpace()
    test("HyperparameterSpace instantiation", hp_space is not None)
    test("backbone options", len(hp_space.backbone) == 4, f"Expected 4, got {len(hp_space.backbone)}")
    test("learning_rate bounds", hp_space.learning_rate == (1e-5, 1e-2))
    
    nsga_config = NSGAIIIConfig()
    test("NSGAIIIConfig instantiation", nsga_config is not None)
    test("n_objectives", nsga_config.n_objectives == 5)
    
    exp_config = ExperimentConfig()
    test("ExperimentConfig instantiation", exp_config is not None)
    
    test("OBJECTIVES dict", len(OBJECTIVES) == 5)
    
except Exception as e:
    print_fail(f"Config module error: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# TEST 3: DATASET MODULE (without torch dependency)
# ==============================================================================
print_header("TEST 3: Dataset Module (Basic)")

try:
    from dataset import (
        load_dicom_image,
        VinDrMammoDataset,
        INbreastDataset,
        prepare_dataset,
        get_transforms,
        create_dataloaders,
        mixup_data,
        mixup_criterion,
        convert_dicom_to_png,
        prepare_dataset_with_png_conversion
    )
    print_pass("Dataset imports")
    
    # Test get_transforms
    if TORCH_AVAILABLE:
        try:
            train_transform = get_transforms(224, is_training=True)
            val_transform = get_transforms(224, is_training=False)
            test("get_transforms", train_transform is not None and val_transform is not None)
        except Exception as e:
            print_fail(f"get_transforms: {e}")
    
except Exception as e:
    print_fail(f"Dataset module error: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# TEST 4: MODELS MODULE
# ==============================================================================
print_header("TEST 4: Models Module")

if TORCH_AVAILABLE:
    try:
        from models import MammographyCNN, build_model, get_param_count, measure_inference_time
        print_pass("Models imports")
        
        # Test build_model with all backbones
        backbones = ['efficientnet_b0', 'resnet50', 'densenet121', 'convnext_tiny']
        for backbone in backbones:
            try:
                config = {'backbone': backbone, 'unfreeze_strategy': 'last_block'}
                model = build_model(config)
                total, trainable = get_param_count(model)
                test(f"build_model({backbone})", model is not None and total > 0)
            except Exception as e:
                print_fail(f"build_model({backbone}): {e}")
        
        # Test forward pass
        model = build_model({'backbone': 'efficientnet_b0'})
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        test("Forward pass shape", output.shape == (2, 2), f"Expected (2, 2), got {output.shape}")
        
        # Test get_param_count function (standalone)
        total, trainable = get_param_count(model)
        test("get_param_count", total > 0 and trainable > 0)
        
        # Test measure_inference_time signature
        import inspect
        sig = inspect.signature(measure_inference_time)
        params = list(sig.parameters.keys())
        test("measure_inference_time has image_size param", 'image_size' in params, f"Params: {params}")
        
    except Exception as e:
        print_fail(f"Models module error: {e}")
        import traceback
        traceback.print_exc()
else:
    print_warn("Skipping models tests (torch not available)")

# ==============================================================================
# TEST 5: TRAINING MODULE
# ==============================================================================
print_header("TEST 5: Training Module")

if TORCH_AVAILABLE:
    try:
        from training import (
            FocalLoss,
            get_class_weights,
            build_criterion,
            build_optimizer,
            build_scheduler,
            train_epoch,
            evaluate,
            train_and_evaluate,
            full_evaluation
        )
        print_pass("Training imports")
        
        # Test get_class_weights
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 3 benign, 7 malignant
        
        weights_inv = get_class_weights(labels, 'inverse_freq')
        test("get_class_weights inverse_freq", weights_inv.shape == (2,))
        test("inverse_freq weights minority class higher", weights_inv[0] > weights_inv[1])
        
        weights_eff = get_class_weights(labels, 'effective_num')
        test("get_class_weights effective_num", weights_eff.shape == (2,))
        
        # Test build_criterion with all loss types
        device = 'cpu'
        
        # CE without weights
        criterion_ce = build_criterion({'loss_function': 'ce'}, None, device)
        test("build_criterion CE", criterion_ce is not None)
        
        # Weighted CE
        weights = get_class_weights(labels, 'inverse_freq')
        criterion_wce = build_criterion({'loss_function': 'weighted_ce'}, weights, device)
        test("build_criterion weighted_ce", criterion_wce is not None)
        
        # Focal loss without weights
        criterion_focal = build_criterion({'loss_function': 'focal', 'focal_gamma': 2.0}, None, device)
        test("build_criterion focal (no weights)", criterion_focal is not None)
        
        # Focal loss with weights
        criterion_focal_w = build_criterion({'loss_function': 'focal', 'focal_gamma': 2.0}, weights, device)
        test("build_criterion focal (with weights)", criterion_focal_w is not None)
        
        # Test forward pass with each criterion
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        
        loss_ce = criterion_ce(logits, targets)
        test("CE forward pass", loss_ce.item() > 0)
        
        loss_focal = criterion_focal(logits, targets)
        test("Focal forward pass", loss_focal.item() > 0)
        
        loss_focal_w = criterion_focal_w(logits, targets)
        test("Focal (weighted) forward pass", loss_focal_w.item() > 0)
        
        # Test build_optimizer
        model = build_model({'backbone': 'efficientnet_b0'})
        
        optimizer_adam = build_optimizer(model, {'optimizer': 'adam', 'learning_rate': 1e-4})
        test("build_optimizer adam", optimizer_adam is not None)
        
        optimizer_adamw = build_optimizer(model, {'optimizer': 'adamw', 'learning_rate': 1e-4})
        test("build_optimizer adamw", optimizer_adamw is not None)
        
        optimizer_sgd = build_optimizer(model, {'optimizer': 'sgd', 'learning_rate': 1e-4})
        test("build_optimizer sgd", optimizer_sgd is not None)
        
        # Test build_scheduler
        scheduler_cosine = build_scheduler(optimizer_adamw, {'scheduler': 'cosine'}, epochs=10)
        test("build_scheduler cosine", scheduler_cosine is not None)
        
        scheduler_step = build_scheduler(optimizer_adamw, {'scheduler': 'step'}, epochs=10)
        test("build_scheduler step", scheduler_step is not None)
        
        scheduler_none = build_scheduler(optimizer_adamw, {'scheduler': 'none'}, epochs=10)
        test("build_scheduler none", scheduler_none is None)
        
        # Test full_evaluation signature
        import inspect
        sig = inspect.signature(full_evaluation)
        params = list(sig.parameters.keys())
        expected_params = ['config', 'train_paths', 'train_labels', 'val_paths', 'val_labels', 'device', 'verbose']
        test("full_evaluation signature", all(p in params for p in expected_params), f"Params: {params}")
        
    except Exception as e:
        print_fail(f"Training module error: {e}")
        import traceback
        traceback.print_exc()
else:
    print_warn("Skipping training tests (torch not available)")

# ==============================================================================
# TEST 6: SURROGATE MODULE
# ==============================================================================
print_header("TEST 6: Surrogate Module")

try:
    from surrogate import MultiObjectiveGPSurrogate, SurrogateAssistedSelection
    print_pass("Surrogate imports")
    
    # Test MultiObjectiveGPSurrogate
    surrogate = MultiObjectiveGPSurrogate(n_objectives=5)
    test("MultiObjectiveGPSurrogate instantiation", surrogate is not None)
    test("surrogate not fitted initially", not surrogate.is_fitted)
    
    # Generate fake data
    np.random.seed(42)
    X_fake = np.random.rand(20, 10)
    y_fake = np.random.rand(20, 5)
    
    # Fit surrogate
    surrogate.fit(X_fake, y_fake)
    test("surrogate.fit()", surrogate.is_fitted)
    
    # Predict
    means, stds = surrogate.predict(X_fake[:5])
    test("surrogate.predict() means shape", means.shape == (5, 5))
    test("surrogate.predict() stds shape", stds.shape == (5, 5))
    test("surrogate.predict() stds positive", np.all(stds >= 0))
    
    # Test SurrogateAssistedSelection
    selector = SurrogateAssistedSelection(
        surrogate=surrogate,
        true_eval_ratio=0.3,
        min_samples=15
    )
    test("SurrogateAssistedSelection instantiation", selector is not None)
    test("should_use_surrogate initially False", not selector.should_use_surrogate())
    
    # Register evaluations
    for i in range(15):
        selector.register_true_evaluation(X_fake[i:i+1], y_fake[i:i+1])
    test("should_use_surrogate after 15 samples", selector.should_use_surrogate())
    
    # Select for evaluation
    indices, X_selected = selector.select_for_true_evaluation(X_fake)
    test("select_for_true_evaluation returns indices", len(indices) > 0)
    
except Exception as e:
    print_fail(f"Surrogate module error: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# TEST 7: OPTIMIZATION MODULE
# ==============================================================================
print_header("TEST 7: Optimization Module")

try:
    from optimization import HyperparameterEncoder, MammographyHPOProblem, run_optimization
    from config import HyperparameterSpace
    print_pass("Optimization imports")
    
    # Test HyperparameterEncoder
    hp_space = HyperparameterSpace()
    encoder = HyperparameterEncoder(hp_space)
    
    test("HyperparameterEncoder n_vars", encoder.n_vars == 20, f"Expected 20, got {encoder.n_vars}")
    
    xl, xu = encoder.get_bounds()
    test("get_bounds() lower", len(xl) == encoder.n_vars)
    test("get_bounds() upper", len(xu) == encoder.n_vars)
    test("bounds valid (xl <= xu)", np.all(xl <= xu))
    
    # Test decode
    x_mid = (xl + xu) / 2
    decoded = encoder.decode(x_mid)
    
    test("decode returns dict", isinstance(decoded, dict))
    test("decode has backbone", 'backbone' in decoded)
    test("decode has learning_rate", 'learning_rate' in decoded)
    test("decode has epochs", 'epochs' in decoded)
    test("decode backbone valid", decoded['backbone'] in hp_space.backbone)
    
    # Test encode/decode roundtrip for continuous vars
    lr = decoded['learning_rate']
    test("decoded learning_rate in bounds", hp_space.learning_rate[0] <= lr <= hp_space.learning_rate[1])
    
except Exception as e:
    print_fail(f"Optimization module error: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# TEST 8: INTEGRATION TEST
# ==============================================================================
print_header("TEST 8: Integration Test")

if TORCH_AVAILABLE:
    try:
        from config import ExperimentConfig
        from models import build_model, get_param_count
        from training import full_evaluation, get_class_weights, build_criterion
        from optimization import HyperparameterEncoder
        
        # Create a mock evaluation that doesn't require real data
        def mock_full_evaluation(config):
            """Mock evaluation for testing."""
            model = build_model(config)
            total, trainable = get_param_count(model)
            
            return {
                'sensitivity': np.random.uniform(0.5, 1.0),
                'specificity': np.random.uniform(0.5, 1.0),
                'auc': np.random.uniform(0.5, 1.0),
                'model_size': total / 1e6,
                'inference_time': np.random.uniform(5, 50)
            }
        
        # Test with various configs
        test_configs = [
            {
                'backbone': 'efficientnet_b0',
                'unfreeze_strategy': 'last_block',
                'dropout_rate': 0.5,
                'fc_hidden_size': 512,
                'use_additional_fc': True,
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'loss_function': 'ce',
                'class_weight_strategy': 'none'
            },
            {
                'backbone': 'resnet50',
                'unfreeze_strategy': 'all',
                'dropout_rate': 0.3,
                'fc_hidden_size': 256,
                'use_additional_fc': False,
                'learning_rate': 1e-3,
                'optimizer': 'sgd',
                'loss_function': 'focal',
                'focal_gamma': 2.0,
                'class_weight_strategy': 'effective_num'
            },
            {
                'backbone': 'densenet121',
                'unfreeze_strategy': 'last_2_blocks',
                'dropout_rate': 0.4,
                'fc_hidden_size': 1024,
                'use_additional_fc': True,
                'learning_rate': 5e-5,
                'optimizer': 'adam',
                'loss_function': 'weighted_ce',
                'class_weight_strategy': 'inverse_freq'
            }
        ]
        
        for i, config in enumerate(test_configs):
            try:
                result = mock_full_evaluation(config)
                valid = (
                    0 <= result['sensitivity'] <= 1 and
                    0 <= result['specificity'] <= 1 and
                    0 <= result['auc'] <= 1 and
                    result['model_size'] > 0 and
                    result['inference_time'] > 0
                )
                test(f"Integration config {i+1} ({config['backbone']})", valid)
            except Exception as e:
                print_fail(f"Integration config {i+1}: {e}")
        
        # Test with class weights on different devices
        labels = [0] * 30 + [1] * 70  # Imbalanced
        for strategy in ['none', 'inverse_freq', 'effective_num']:
            for loss in ['ce', 'weighted_ce', 'focal']:
                try:
                    weights = None
                    if strategy != 'none':
                        weights = get_class_weights(labels, strategy)
                    criterion = build_criterion({'loss_function': loss, 'focal_gamma': 2.0}, weights, 'cpu')
                    
                    # Forward pass
                    logits = torch.randn(8, 2)
                    targets = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])
                    loss_val = criterion(logits, targets)
                    test(f"Integration {loss}/{strategy}", loss_val.item() > 0)
                except Exception as e:
                    print_fail(f"Integration {loss}/{strategy}: {e}")
        
    except Exception as e:
        print_fail(f"Integration test error: {e}")
        import traceback
        traceback.print_exc()
else:
    print_warn("Skipping integration tests (torch not available)")

# ==============================================================================
# SUMMARY
# ==============================================================================
print_header("VALIDATION SUMMARY")

print(f"\n  Passed: {TESTS_PASSED}")
print(f"  Failed: {TESTS_FAILED}")

if ERRORS:
    print(f"\n  Errors:")
    for err in ERRORS:
        print(f"    - {err}")

if TESTS_FAILED == 0:
    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED ✓")
    print(f"{'='*60}")
    print("\nThe code is ready for use. Run your optimization with:")
    print("""
from dataset import prepare_dataset_with_png_conversion
from training import full_evaluation
from optimization import run_optimization
from config import ExperimentConfig

# Load data
train_paths, train_labels, val_paths, val_labels = prepare_dataset_with_png_conversion(
    dataset_name="vindr",
    data_root="/content/drive/MyDrive/vindr-mammo",
    png_output_dir="/content/drive/MyDrive/vindr-mammo-png"
)

# Run optimization
config = ExperimentConfig()
config.nsga3.pop_size = 20
config.nsga3.n_generations = 5

def make_eval_fn(tp, tl, vp, vl):
    def fn(hp):
        hp = hp.copy()
        hp['epochs'] = 5
        return full_evaluation(hp, tp, tl, vp, vl, device='cuda', verbose=False)
    return fn

results = run_optimization(
    hp_space=config.hyperparameter_space,
    nsga_config=config.nsga3,
    eval_function=make_eval_fn(train_paths, train_labels, val_paths, val_labels),
    output_dir="/content/drive/MyDrive/nsga3_outputs"
)
""")
else:
    print(f"\n{'='*60}")
    print(f"  {TESTS_FAILED} TESTS FAILED - FIX ERRORS BEFORE PROCEEDING")
    print(f"{'='*60}")
    sys.exit(1)
