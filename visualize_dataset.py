"""
Visualize Training and Validation Dataset Samples

This script displays sample mammography images from your dataset
to verify data loading and preprocessing.

Usage:
    python visualize_dataset.py

Author: David
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_dataset, load_dicom_image


def visualize_samples(image_paths, labels, dataset_name, n_samples=6, save_path=None):
    """
    Visualize sample mammography images.

    Args:
        image_paths: List of image file paths
        labels: List of labels (0=benign, 1=malignant)
        dataset_name: Name for the plot title (e.g., 'Training', 'Validation')
        n_samples: Number of samples to display
        save_path: Optional path to save the figure
    """
    # Select random samples (stratified by label)
    benign_indices = [i for i, label in enumerate(labels) if label == 0]
    malignant_indices = [i for i, label in enumerate(labels) if label == 1]

    n_benign = min(n_samples // 2, len(benign_indices))
    n_malignant = min(n_samples - n_benign, len(malignant_indices))

    selected_benign = np.random.choice(benign_indices, n_benign, replace=False)
    selected_malignant = np.random.choice(malignant_indices, n_malignant, replace=False)
    indices = np.concatenate([selected_benign, selected_malignant])
    np.random.shuffle(indices)

    # Create subplot grid
    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{dataset_name} Dataset Samples', fontsize=16, fontweight='bold')

    for idx, sample_idx in enumerate(indices):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        img_path = image_paths[sample_idx]
        label = labels[sample_idx]

        try:
            # Load DICOM image
            image = load_dicom_image(img_path, apply_clahe=True)

            # Display
            ax.imshow(image, cmap='gray')
            label_text = 'MALIGNANT' if label == 1 else 'BENIGN'
            label_color = 'red' if label == 1 else 'green'
            ax.set_title(f'{label_text}', fontsize=12, fontweight='bold', color=label_color)
            ax.axis('off')

            # Add filename as subtitle
            filename = os.path.basename(img_path)
            display_name = filename[:30] + '...' if len(filename) > 30 else filename
            ax.text(0.5, -0.05, display_name,
                   ha='center', va='top', transform=ax.transAxes,
                   fontsize=8, style='italic', wrap=True)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading image:\n{str(e)[:100]}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=8, color='red')
            ax.axis('off')

    # Hide extra subplots
    for idx in range(len(indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def print_dataset_statistics(train_paths, train_labels, val_paths, val_labels):
    """Print detailed statistics about the dataset."""
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total_samples = len(train_paths) + len(val_paths)
    train_benign = sum(1 for label in train_labels if label == 0)
    train_malignant = sum(1 for label in train_labels if label == 1)
    val_benign = sum(1 for label in val_labels if label == 0)
    val_malignant = sum(1 for label in val_labels if label == 1)

    print(f"\nTotal Samples: {total_samples}")
    print(f"  Training: {len(train_paths)} ({len(train_paths)/total_samples*100:.1f}%)")
    print(f"  Validation: {len(val_paths)} ({len(val_paths)/total_samples*100:.1f}%)")

    print(f"\nTraining Set:")
    print(f"  Benign: {train_benign} ({train_benign/len(train_labels)*100:.1f}%)")
    print(f"  Malignant: {train_malignant} ({train_malignant/len(train_labels)*100:.1f}%)")

    print(f"\nValidation Set:")
    print(f"  Benign: {val_benign} ({val_benign/len(val_labels)*100:.1f}%)")
    print(f"  Malignant: {val_malignant} ({val_malignant/len(val_labels)*100:.1f}%)")

    print("\n" + "=" * 70)


def main():
    """Main function to visualize dataset."""

    # Configuration - UPDATE THESE PATHS
    DATASET_NAME = "vindr"  # or "inbreast"
    DATA_ROOT = "/path/to/your/dataset"  # UPDATE THIS!

    print("=" * 70)
    print("MAMMOGRAPHY DATASET VISUALIZATION")
    print("=" * 70)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Data Root: {DATA_ROOT}")

    # Check if path exists
    if not os.path.exists(DATA_ROOT):
        print(f"\n❌ ERROR: Data root does not exist: {DATA_ROOT}")
        print("\nPlease update the DATA_ROOT variable in this script.")
        return

    print("\nLoading dataset...")

    try:
        # Load dataset
        train_paths, train_labels, val_paths, val_labels = prepare_dataset(
            dataset_name=DATASET_NAME,
            data_root=DATA_ROOT
        )

        # Print statistics
        print_dataset_statistics(train_paths, train_labels, val_paths, val_labels)

        # Visualize training samples
        print("\n📊 Visualizing TRAINING samples...")
        visualize_samples(
            train_paths, train_labels,
            'Training',
            n_samples=6,
            save_path='train_samples.png'
        )

        # Visualize validation samples
        print("\n📊 Visualizing VALIDATION samples...")
        visualize_samples(
            val_paths, val_labels,
            'Validation',
            n_samples=6,
            save_path='val_samples.png'
        )

        print("\n✅ Visualization complete!")
        print("Figures saved: train_samples.png, val_samples.png")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
