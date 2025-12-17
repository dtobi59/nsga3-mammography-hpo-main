"""
Test training.py components thoroughly
"""
import sys
import numpy as np

print("="*60)
print("TESTING TRAINING.PY COMPONENTS")
print("="*60)

# Test 1: Import without torch
print("\n[TEST 1] Import check...")
try:
    import torch
    TORCH_AVAILABLE = True
    print("  torch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("  torch NOT available - skipping training tests")
    sys.exit(0)

# Test 2: Import training module
print("\n[TEST 2] Import training module...")
try:
    from training import (
        FocalLoss, get_class_weights, build_criterion,
        build_optimizer, build_scheduler, train_and_evaluate
    )
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: get_class_weights
print("\n[TEST 3] get_class_weights...")
try:
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 3 benign, 7 malignant
    
    weights_inv = get_class_weights(labels, 'inverse_freq')
    print(f"  inverse_freq: {weights_inv}")
    assert weights_inv.shape == (2,)
    assert weights_inv[0] > weights_inv[1]  # minority class gets higher weight
    
    weights_eff = get_class_weights(labels, 'effective_num')
    print(f"  effective_num: {weights_eff}")
    assert weights_eff.shape == (2,)
    
    weights_none = get_class_weights(labels, 'none')
    print(f"  none: {weights_none}")
    
    print("  ✓ get_class_weights OK")
except Exception as e:
    print(f"  ✗ get_class_weights FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: build_criterion with different configs
print("\n[TEST 4] build_criterion...")
try:
    device = 'cpu'
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
    # Test 4a: CE without weights
    config_ce = {'loss_function': 'ce', 'class_weight_strategy': 'none'}
    criterion_ce = build_criterion(config_ce, None)
    print(f"  CE (no weights): {type(criterion_ce).__name__}")
    
    # Test 4b: Weighted CE
    config_wce = {'loss_function': 'weighted_ce', 'class_weight_strategy': 'inverse_freq'}
    weights = get_class_weights(labels, 'inverse_freq').to(device)
    criterion_wce = build_criterion(config_wce, weights)
    print(f"  Weighted CE: {type(criterion_wce).__name__}, weight={criterion_wce.weight}")
    
    # Test 4c: Focal loss without weights
    config_focal = {'loss_function': 'focal', 'focal_gamma': 2.0, 'class_weight_strategy': 'none'}
    criterion_focal = build_criterion(config_focal, None)
    print(f"  Focal (no weights): {type(criterion_focal).__name__}, alpha={criterion_focal.alpha}")
    
    # Test 4d: Focal loss with weights
    config_focal_w = {'loss_function': 'focal', 'focal_gamma': 2.0, 'class_weight_strategy': 'effective_num'}
    weights_focal = get_class_weights(labels, 'effective_num').to(device)
    criterion_focal_w = build_criterion(config_focal_w, weights_focal)
    print(f"  Focal (weights): {type(criterion_focal_w).__name__}, alpha={criterion_focal_w.alpha}")
    
    print("  ✓ build_criterion OK")
except Exception as e:
    print(f"  ✗ build_criterion FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass with each criterion
print("\n[TEST 5] Criterion forward pass...")
try:
    device = 'cpu'
    batch_size = 4
    num_classes = 2
    
    # Fake logits and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 1, 0])
    
    labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
    # CE
    criterion_ce = build_criterion({'loss_function': 'ce'}, None)
    loss_ce = criterion_ce(logits, targets)
    print(f"  CE loss: {loss_ce.item():.4f}")
    
    # Weighted CE
    weights = get_class_weights(labels, 'inverse_freq').to(device)
    criterion_wce = build_criterion({'loss_function': 'weighted_ce'}, weights)
    loss_wce = criterion_wce(logits, targets)
    print(f"  Weighted CE loss: {loss_wce.item():.4f}")
    
    # Focal without weights
    criterion_focal = build_criterion({'loss_function': 'focal', 'focal_gamma': 2.0}, None)
    loss_focal = criterion_focal(logits, targets)
    print(f"  Focal loss (no weights): {loss_focal.item():.4f}")
    
    # Focal with weights
    weights_f = get_class_weights(labels, 'effective_num').to(device)
    criterion_focal_w = build_criterion({'loss_function': 'focal', 'focal_gamma': 2.0}, weights_f)
    loss_focal_w = criterion_focal_w(logits, targets)
    print(f"  Focal loss (weights): {loss_focal_w.item():.4f}")
    
    print("  ✓ Forward pass OK")
except Exception as e:
    print(f"  ✗ Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Simulate train_and_evaluate setup (the problematic part)
print("\n[TEST 6] Simulating train_and_evaluate setup...")
try:
    device = 'cpu'
    train_labels = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    
    test_configs = [
        {'loss_function': 'ce', 'class_weight_strategy': 'none'},
        {'loss_function': 'ce', 'class_weight_strategy': 'inverse_freq'},
        {'loss_function': 'weighted_ce', 'class_weight_strategy': 'inverse_freq'},
        {'loss_function': 'focal', 'focal_gamma': 2.0, 'class_weight_strategy': 'none'},
        {'loss_function': 'focal', 'focal_gamma': 2.0, 'class_weight_strategy': 'effective_num'},
    ]
    
    for i, config in enumerate(test_configs):
        # Replicate train_and_evaluate logic
        weight_strategy = config.get('class_weight_strategy', 'none')
        class_weights = None
        if weight_strategy != 'none':
            class_weights = get_class_weights(train_labels, weight_strategy)
            class_weights = class_weights.to(device)
        
        criterion = build_criterion(config, class_weights)
        
        # Test forward pass
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        loss = criterion(logits, targets)
        
        print(f"  Config {i+1}: loss={config['loss_function']}, weights={weight_strategy} -> loss={loss.item():.4f} ✓")
    
    print("  ✓ All configs OK")
except Exception as e:
    print(f"  ✗ Simulation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TRAINING TESTS PASSED ✓")
print("="*60)
