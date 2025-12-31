"""
Test Script for Kaggle VinDr-Mammo PNG Dataset Setup

This script tests your setup step-by-step before running full optimization.

Usage:
    python test_kaggle_setup.py

Author: David
"""

import os
import sys
from pathlib import Path

print("="*70)
print("KAGGLE VINDR-MAMMO DATASET SETUP TEST")
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
    sys.exit(1)
else:
    print("\n[OK] All dependencies installed!")

# ============================================================================
# TEST 2: Dataset Path Configuration
# ============================================================================

print("\n[TEST 2] Dataset path configuration...")
print("\nPlease enter the path to your Kaggle VinDr dataset:")
print("Example: C:/Users/YourName/Downloads/kaggle_vindr_data")
print("         /content/drive/MyDrive/kaggle_vindr_data")

DATA_ROOT = input("\nDataset path: ").strip()

if not DATA_ROOT:
    print("[!] No path provided. Using default: ./kaggle_vindr_data")
    DATA_ROOT = "./kaggle_vindr_data"

data_path = Path(DATA_ROOT)

if not data_path.exists():
    print(f"\n[X] Path does not exist: {DATA_ROOT}")
    print("\nDid you download the dataset? Run:")
    print("  kaggle datasets download -d shantanughosh/vindr-mammogram-dataset-dicom-to-png")
    print(f"  unzip vindr-mammogram-dataset-dicom-to-png.zip -d {DATA_ROOT}")
    sys.exit(1)

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
    sys.exit(1)

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
        input("\nPress Enter to exit...")
        sys.exit(1)

# ============================================================================
# TEST 3: Load Dataset Class
# ============================================================================

print("\n[TEST 3] Loading dataset class...")

try:
    from dataset import KaggleVinDrPNGDataset
    print("  [OK] Dataset class imported successfully")
except ImportError as e:
    print(f"  [X] Failed to import dataset class: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: Initialize Dataset
# ============================================================================

print("\n[TEST 4] Initializing dataset...")

try:
    dataset = KaggleVinDrPNGDataset(DATA_ROOT)
    print("  [OK] Dataset initialized")
except Exception as e:
    print(f"  [X] Failed to initialize dataset: {e}")
    sys.exit(1)

# ============================================================================
# TEST 5: Load and Process CSV
# ============================================================================

print("\n[TEST 5] Loading and processing CSV...")

try:
    df = dataset.load_and_process()
    print(f"  [OK] CSV processed: {len(df)} samples after filtering")
except Exception as e:
    print(f"  [X] Failed to process CSV: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: Verify Image Paths
# ============================================================================

print("\n[TEST 6] Verifying image paths...")

sample_size = min(10, len(df))
valid_paths = 0

for idx, row in df.head(sample_size).iterrows():
    path = dataset.get_image_path(row)
    if path and os.path.exists(path):
        valid_paths += 1

print(f"  [OK] Found {valid_paths}/{sample_size} sample images")

if valid_paths == 0:
    print("  [X] No valid image paths found!")
    print("\nDebug info:")
    print(f"  Image column: {dataset.image_col}")
    print(f"  Sample ID: {df.iloc[0][dataset.image_col]}")
    print(f"  Looking in: {dataset.images_dir}")
    sys.exit(1)
elif valid_paths < sample_size:
    print(f"  [!] Warning: Only {valid_paths}/{sample_size} images found")
    print("  Some images may be missing from the dataset")

# ============================================================================
# TEST 7: Prepare Train/Val Splits
# ============================================================================

print("\n[TEST 7] Preparing train/val splits...")

try:
    train_paths, train_labels, val_paths, val_labels = dataset.prepare_splits(
        val_split=0.2,
        seed=42
    )
    print(f"  [OK] Splits created successfully")
    print(f"    Train: {len(train_paths)} samples")
    print(f"    Val: {len(val_paths)} samples")
except Exception as e:
    print(f"  [X] Failed to create splits: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 8: Test Single Image Loading
# ============================================================================

print("\n[TEST 8] Testing image loading...")

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
    sys.exit(1)

# ============================================================================
# TEST 9: Test Model Creation
# ============================================================================

print("\n[TEST 9] Testing model creation...")

try:
    from models import create_model

    model = create_model(
        backbone='efficientnet_b0',
        num_classes=2,
        dropout_rate=0.3,
        unfreeze_strategy='none'
    )
    print(f"  [OK] Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params/1e6:.2f}M")

except Exception as e:
    print(f"  [X] Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 10: Test Forward Pass
# ============================================================================

print("\n[TEST 10] Testing forward pass...")

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
    sys.exit(1)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[OK] ALL TESTS PASSED!")
print("="*70)

print("\nYour setup is ready! You can now:")
print("\n1. Test with a single configuration:")
print(f"   python test_single_config.py --data_root {DATA_ROOT}")

print("\n2. Run mini optimization (fast test):")
print(f"   python test_mini_optimization.py --data_root {DATA_ROOT}")

print("\n3. Run full optimization:")
print(f"   python example_kaggle_dataset.py")
print(f"   (Edit DATA_ROOT in the file to: {DATA_ROOT})")

print("\n" + "="*70)
