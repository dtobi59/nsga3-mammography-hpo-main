"""
Test Single Hyperparameter Configuration

Tests the full training pipeline with one configuration.
This is faster than optimization and helps verify everything works.

Usage:
    python test_single_config.py --data_root /path/to/kaggle_vindr_data

Author: David
"""

import argparse
import os
from dataset import prepare_dataset
from training import full_evaluation

def main():
    parser = argparse.ArgumentParser(description='Test single configuration')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Kaggle VinDr dataset')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs (default: 2 for quick test)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    args = parser.parse_args()

    print("="*70)
    print("SINGLE CONFIGURATION TEST")
    print("="*70)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    train_paths, train_labels, val_paths, val_labels = prepare_dataset(
        dataset_name="kaggle_vindr_png",
        data_root=args.data_root,
        val_split=0.2,
        seed=42
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_paths)} samples")
    print(f"  Val: {len(val_paths)} samples")

    # Define test configuration
    print("\n[2/3] Defining test configuration...")

    test_config = {
        # Architecture
        'backbone': 'efficientnet_b0',
        'unfreeze_strategy': 'last_block',
        'dropout_rate': 0.3,
        'fc_hidden_size': 512,
        'use_additional_fc': True,

        # Optimization
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'batch_size': args.batch_size,
        'epochs': args.epochs,

        # Loss
        'loss_function': 'focal',
        'focal_gamma': 2.0,
        'class_weight_strategy': 'inverse_freq',
        'oversampling_ratio': 1.5,

        # Augmentation
        'horizontal_flip': True,
        'rotation_range': 10,
        'brightness_contrast': 0.2,
        'use_mixup': False,
        'mixup_alpha': 0.2
    }

    print("\nConfiguration:")
    for key, value in test_config.items():
        print(f"  {key}: {value}")

    # Train and evaluate
    print(f"\n[3/3] Training model ({args.epochs} epochs)...")
    print("This may take a few minutes...\n")

    try:
        results = full_evaluation(
            test_config,
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            device=args.device,
            verbose=True
        )

        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nPerformance Metrics:")
        print(f"  Sensitivity:  {results['sensitivity']:.4f} (recall for malignant cases)")
        print(f"  Specificity:  {results['specificity']:.4f} (recall for benign cases)")
        print(f"  AUC:          {results['auc']:.4f} (area under ROC curve)")
        print(f"  Accuracy:     {results.get('accuracy', 0):.4f}")

        print(f"\nModel Characteristics:")
        print(f"  Model Size:   {results['model_size']:.2f}M parameters")

        print(f"\nConfusion Matrix:")
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            print(f"                Predicted")
            print(f"                Benign  Malignant")
            print(f"  Actual Benign   {cm[0][0]:4d}    {cm[0][1]:4d}")
            print(f"        Malignant {cm[1][0]:4d}    {cm[1][1]:4d}")

        print("\n" + "="*70)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("="*70)

        print("\nNext steps:")
        print("  1. Try different configurations by editing this script")
        print("  2. Increase --epochs for better results (default is 2 for speed)")
        print("  3. Run mini optimization: python test_mini_optimization.py")
        print("  4. Run full optimization: python example_kaggle_dataset.py")

    except Exception as e:
        print("\n" + "="*70)
        print("✗ ERROR DURING TRAINING")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - If CUDA out of memory: reduce --batch_size (try 8)")
        print("  - If CPU is too slow: use Google Colab with GPU")
        print("  - If dataset errors: run python test_kaggle_setup.py")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
