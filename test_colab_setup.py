"""
Colab Setup Test - Non-interactive version for Google Colab

This script tests your setup without requiring user input.
Designed for Jupyter/Colab environments.

Usage in Colab:
    !python test_colab_setup.py --data_root /content/drive/MyDrive/kaggle_vindr_data

Author: David
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Test Colab setup')
    parser.add_argument('--data_root', type=str,
                        default='/content/drive/MyDrive/kaggle_vindr_data',
                        help='Path to dataset')
    args = parser.parse_args()

    print("="*70)
    print("COLAB SETUP TEST (Non-Interactive)")
    print("="*70)

    # ============================================================================
    # TEST 1: Check Dependencies
    # ============================================================================

    print("\n[TEST 1] Checking dependencies...")

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'pymoo': 'pymoo',
    }

    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [X] {package} (install with: pip install {pip_name})")
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"\n[!] Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return 1
    else:
        print("\n[OK] All dependencies installed!")

    # ============================================================================
    # TEST 2: Check GPU
    # ============================================================================

    print("\n[TEST 2] Checking GPU availability...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  [!] No GPU available - will use CPU (slower)")
    except Exception as e:
        print(f"  [X] Error checking GPU: {e}")

    # ============================================================================
    # TEST 3: Dataset Path
    # ============================================================================

    print("\n[TEST 3] Checking dataset path...")

    DATA_ROOT = args.data_root
    print(f"  Dataset path: {DATA_ROOT}")

    data_path = Path(DATA_ROOT)

    if not data_path.exists():
        print(f"  [X] Path does not exist: {DATA_ROOT}")
        print("\nDownload the dataset:")
        print("  1. Install Kaggle: !pip install kaggle")
        print("  2. Upload kaggle.json")
        print("  3. Download: !kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png")
        print(f"  4. Extract: !unzip -q vindr-mammogram-dataset-dicom-to-png.zip -d {DATA_ROOT}")
        return 1

    print(f"  [OK] Path exists: {DATA_ROOT}")

    # Check for CSV file
    csv_candidates = [
        data_path / "vindr_detection_v1_folds.csv",
        data_path / "breast-level_annotations.csv",
        data_path / "metadata.csv",
    ]

    csv_found = None
    for csv_path in csv_candidates:
        if csv_path.exists():
            csv_found = csv_path
            break

    if csv_found:
        print(f"  [OK] CSV file found: {csv_found.name}")
    else:
        print(f"  [X] No CSV file found. Expected one of:")
        for csv_path in csv_candidates:
            print(f"    - {csv_path.name}")
        return 1

    # Check for images (supports multiple layouts)
    images_dir = data_path / "images"
    if images_dir.exists():
        png_files = list(images_dir.glob("**/*.png"))
        print(f"  [OK] Images in 'images/' subfolder: {len(png_files)} PNG files")
        if len(png_files) == 0:
            print("  [!] Warning: No PNG files found in images directory")
    else:
        # Check for PNGs in subdirectories (organized by patient/study ID)
        png_files = list(data_path.glob("**/*.png"))
        direct_png = list(data_path.glob("*.png"))

        if len(png_files) > 0:
            if len(direct_png) > 0:
                print(f"  [OK] Images in root directory: {len(png_files)} PNG files")
            else:
                print(f"  [OK] Images in subdirectories: {len(png_files)} PNG files")
        else:
            print(f"  [X] No PNG files found in {data_path}")
            return 1

    # ============================================================================
    # TEST 4: Load Dataset Class
    # ============================================================================

    print("\n[TEST 4] Testing dataset class...")

    try:
        from dataset import KaggleVinDrPNGDataset
        print("  [OK] Dataset class imported")
    except ImportError as e:
        print(f"  [X] Failed to import dataset class: {e}")
        return 1

    # ============================================================================
    # TEST 5: Initialize Dataset
    # ============================================================================

    print("\n[TEST 5] Initializing dataset...")

    try:
        dataset = KaggleVinDrPNGDataset(DATA_ROOT)
        print("  [OK] Dataset initialized")
    except Exception as e:
        print(f"  [X] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # TEST 6: Load CSV
    # ============================================================================

    print("\n[TEST 6] Loading CSV...")

    try:
        df = dataset.load_and_process()
        print(f"  [OK] CSV processed: {len(df)} samples")
    except Exception as e:
        print(f"  [X] Failed to process CSV: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # TEST 7: Verify Image Paths
    # ============================================================================

    print("\n[TEST 7] Verifying image paths...")

    sample_size = min(10, len(df))
    valid_paths = 0

    for idx, row in df.head(sample_size).iterrows():
        path = dataset.get_image_path(row)
        if path and os.path.exists(path):
            valid_paths += 1

    print(f"  [OK] Found {valid_paths}/{sample_size} sample images")

    if valid_paths == 0:
        print("  [X] No valid image paths found!")
        return 1

    # ============================================================================
    # TEST 8: Create Splits
    # ============================================================================

    print("\n[TEST 8] Creating train/val splits...")

    try:
        train_paths, train_labels, val_paths, val_labels = dataset.prepare_splits(
            val_split=0.2,
            seed=42
        )
        print(f"  [OK] Splits created")
        print(f"    Train: {len(train_paths)} samples")
        print(f"    Val: {len(val_paths)} samples")
    except Exception as e:
        print(f"  [X] Failed to create splits: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # TEST 9: Load Single Image
    # ============================================================================

    print("\n[TEST 9] Testing image loading...")

    try:
        from dataset import MammographyDataset, get_transforms

        transform = get_transforms(image_size=224, is_training=False)
        test_dataset = MammographyDataset(
            val_paths[:1],
            val_labels[:1],
            transform=transform,
            image_size=224
        )

        image, label = test_dataset[0]
        print(f"  [OK] Image loaded successfully")
        print(f"    Shape: {image.shape}")
        print(f"    Label: {label}")
    except Exception as e:
        print(f"  [X] Failed to load image: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # TEST 10: Model Creation
    # ============================================================================

    print("\n[TEST 10] Testing model creation...")

    try:
        from models import create_model

        model = create_model(
            backbone='efficientnet_b0',
            num_classes=2,
            dropout_rate=0.3,
            unfreeze_strategy='none'
        )
        print(f"  [OK] Model created")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {total_params/1e6:.2f}M")
    except Exception as e:
        print(f"  [X] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # TEST 11: Forward Pass
    # ============================================================================

    print("\n[TEST 11] Testing forward pass...")

    try:
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"    Using device: {device}")

        model = model.to(device)
        image = image.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image)

        print(f"  [OK] Forward pass successful")
        print(f"    Output shape: {output.shape}")
    except Exception as e:
        print(f"  [X] Failed forward pass: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ============================================================================
    # SUCCESS
    # ============================================================================

    print("\n" + "="*70)
    print("[OK] ALL TESTS PASSED!")
    print("="*70)

    print("\nYour Colab environment is ready!")
    print("\nNext steps:")
    print("  1. Run the cells in the notebook to load your dataset")
    print("  2. Configure optimization parameters")
    print("  3. Run the optimization")

    print("\n" + "="*70)

    return 0

if __name__ == '__main__':
    exit(main())
