"""
Dataset Loading for Mammography Classification
Supports VinDr-Mammo and INbreast datasets in DICOM format.

Author: David
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# DICOM support
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def load_dicom_image(dicom_path: str, apply_clahe: bool = True) -> np.ndarray:
    """Load and preprocess a DICOM image."""
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required: pip install pydicom")
    
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)
    
    # Handle inverted images
    if hasattr(dicom, 'PhotometricInterpretation'):
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            image = image.max() - image
    
    # Normalize to 0-255
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    
    # Apply CLAHE
    if apply_clahe and CV2_AVAILABLE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    
    # Convert to RGB (3 channels)
    image = np.stack([image, image, image], axis=-1)
    
    return image


class MammographyDataset:
    """PyTorch Dataset for mammography images (DICOM or PNG)."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        image_size: int = 224,
        apply_clahe: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        import torch
        
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            if image_path.lower().endswith(('.dcm', '.dicom')):
                image = load_dicom_image(image_path, self.apply_clahe)
            else:
                from PIL import Image
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = np.array(img)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        if self.transform is not None:
            if ALBUMENTATIONS_AVAILABLE and hasattr(self.transform, 'transforms'):
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                from PIL import Image as PILImage
                image = PILImage.fromarray(image)
                image = self.transform(image)
        else:
            if CV2_AVAILABLE:
                image = cv2.resize(image, (self.image_size, self.image_size))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
        return image, label


class VinDrMammoDataset:
    """
    VinDr-Mammo dataset handler.
    
    Structure:
    vindr-mammo/
    ├── images/{study_id}/{image_id}.dicom
    ├── metadata/breast-level_annotations.csv
    └── downloaded_files.json (optional - to filter to downloaded images only)
    
    BI-RADS: 1,2 -> benign (0), 4,5,6 -> malignant (1), 3 -> excluded
    """
    
    def __init__(self, data_root: str, downloaded_json: str = None):
        """
        Args:
            data_root: Path to vindr-mammo folder
            downloaded_json: Optional path to JSON file with {"downloaded_files": [...]}
        """
        self.data_root = Path(data_root)
        self.csv_path = self.data_root / "metadata" / "breast-level_annotations.csv"
        self.image_folder = self.data_root / "images"
        self.df = None
        self.downloaded_files = None
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        # Load downloaded files filter if provided
        if downloaded_json:
            json_path = Path(downloaded_json)
            if json_path.exists():
                import json
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.downloaded_files = set(data.get('downloaded_files', []))
                print(f"VinDr-Mammo: {self.data_root}")
                print(f"  Filtering to {len(self.downloaded_files)} downloaded files")
            else:
                print(f"Warning: JSON file not found: {downloaded_json}")
                print(f"VinDr-Mammo: {self.data_root}")
        else:
            # Auto-detect JSON in data_root
            auto_json = self.data_root / "downloaded_files.json"
            if auto_json.exists():
                import json
                with open(auto_json, 'r') as f:
                    data = json.load(f)
                self.downloaded_files = set(data.get('downloaded_files', []))
                print(f"VinDr-Mammo: {self.data_root}")
                print(f"  Auto-detected filter: {len(self.downloaded_files)} downloaded files")
            else:
                print(f"VinDr-Mammo: {self.data_root}")
    
    def _is_downloaded(self, study_id: str, image_id: str) -> bool:
        """Check if image is in downloaded files list."""
        if self.downloaded_files is None:
            return True  # No filter, assume all available
        
        # Check various path formats
        for ext in ['.dicom', '.dcm', '.png']:
            path = f"images/{study_id}/{image_id}{ext}"
            if path in self.downloaded_files:
                return True
        return False
    
    def load_and_process(self) -> pd.DataFrame:
        """Load annotations and create binary labels."""
        self.df = pd.read_csv(self.csv_path)
        print(f"  Loaded {len(self.df)} rows from CSV")
        
        # Filter to downloaded files first (if filter exists)
        if self.downloaded_files is not None:
            before_count = len(self.df)
            self.df = self.df[self.df.apply(
                lambda row: self._is_downloaded(row['study_id'], row['image_id']), 
                axis=1
            )]
            print(f"  After download filter: {len(self.df)} rows (from {before_count})")
        
        birads = self.df['breast_birads'].str.extract(r'BI-RADS (\d+)', expand=False)
        birads = pd.to_numeric(birads, errors='coerce')
        
        # Show BI-RADS distribution before filtering
        print(f"  BI-RADS distribution in downloaded images:")
        for val in [1, 2, 3, 4, 5, 6]:
            count = (birads == val).sum()
            if count > 0:
                print(f"    BI-RADS {val}: {count}")
        
        mask = birads.isin([1, 2, 4, 5, 6])
        self.df = self.df[mask].copy()
        self.df['label'] = birads[mask].apply(lambda x: 0 if x <= 2 else 1)
        
        print(f"  After BI-RADS filter: {len(self.df)} (Benign: {(self.df['label']==0).sum()}, Malignant: {(self.df['label']==1).sum()})")
        return self.df
    
    def get_image_path(self, row) -> Optional[str]:
        study_id = row['study_id']
        image_id = row['image_id']
        
        for ext in ['.dicom', '.dcm', '.png']:
            path = self.image_folder / study_id / f"{image_id}{ext}"
            if path.exists():
                return str(path)
        return None
    
    def prepare_splits(self, val_split: float = 0.2, seed: int = 42):
        if self.df is None:
            self.load_and_process()
        
        paths, labels = [], []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Scanning"):
            path = self.get_image_path(row)
            if path and os.path.exists(path):
                paths.append(path)
                labels.append(row['label'])
        
        print(f"  Valid images: {len(paths)}")
        
        if len(paths) == 0:
            raise ValueError("No images found!")
        
        n_samples = len(paths)
        counts = pd.Series(labels).value_counts()
        can_stratify = counts.min() >= 2 and int(n_samples * val_split) >= 2
        
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels,
            test_size=val_split,
            stratify=labels if can_stratify else None,
            random_state=seed
        )
        
        print(f"  Train: {len(train_p)}, Val: {len(val_p)}")
        return train_p, train_l, val_p, val_l


class INbreastDataset:
    """
    INbreast dataset handler.
    
    Structure:
    inbreast/
    └── INbreast Release 1.0/
        ├── AllDICOMs/{id}_{hash}_MG_{L/R}_{CC/MLO}_ANON.dcm
        └── INbreast.csv (semicolon-separated)
    """
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        
        release = self.data_root / "INbreast Release 1.0"
        self.release_path = release if release.exists() else self.data_root
        
        self.dicoms_dir = self.release_path / "AllDICOMs"
        
        csv = self.release_path / "INbreast.csv"
        xls = self.release_path / "INbreast.xls"
        self.csv_path = csv if csv.exists() else xls
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata not found in {self.release_path}")
        
        self.df = None
        print(f"INbreast: {self.release_path}")
    
    def load_and_process(self) -> pd.DataFrame:
        if self.csv_path.suffix == '.xls':
            self.df = pd.read_excel(self.csv_path)
        else:
            try:
                self.df = pd.read_csv(self.csv_path, sep=';')
                if len(self.df.columns) == 1:
                    self.df = pd.read_csv(self.csv_path, sep=',')
            except:
                self.df = pd.read_csv(self.csv_path)
        
        print(f"  Loaded {len(self.df)} rows, Columns: {list(self.df.columns)}")
        
        birads_col = None
        for col in self.df.columns:
            if 'BI-RADS' in col.upper() or 'BIRADS' in col.upper() or col.upper() == 'ACR':
                birads_col = col
                break
        
        if birads_col is None:
            raise ValueError(f"BI-RADS column not found. Available: {list(self.df.columns)}")
        
        birads = self.df[birads_col].astype(str).str.extract(r'^(\d+)', expand=False)
        birads = pd.to_numeric(birads, errors='coerce')
        
        mask = birads.isin([1, 2, 4, 5, 6])
        self.df = self.df[mask].copy()
        self.df['label'] = birads[mask].apply(lambda x: 0 if x <= 2 else 1)
        
        print(f"  After filtering: {len(self.df)} (Benign: {(self.df['label']==0).sum()}, Malignant: {(self.df['label']==1).sum()})")
        return self.df
    
    def find_dicom(self, file_id: str) -> Optional[str]:
        file_id = str(file_id).strip()
        if not self.dicoms_dir.exists():
            return None
        
        for f in self.dicoms_dir.iterdir():
            if f.name.startswith(f"{file_id}_"):
                return str(f)
        return None
    
    def prepare_splits(self, val_split: float = 0.2, seed: int = 42):
        if self.df is None:
            self.load_and_process()
        
        file_col = None
        for col in self.df.columns:
            if 'file' in col.lower():
                file_col = col
                break
        
        if file_col is None:
            raise ValueError(f"File column not found. Available: {list(self.df.columns)}")
        
        paths, labels = [], []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Scanning"):
            path = self.find_dicom(row[file_col])
            if path and os.path.exists(path):
                paths.append(path)
                labels.append(row['label'])
        
        print(f"  Valid images: {len(paths)}")
        
        if len(paths) == 0:
            raise ValueError("No images found!")
        
        n_samples = len(paths)
        counts = pd.Series(labels).value_counts()
        can_stratify = counts.min() >= 2 and int(n_samples * val_split) >= 2
        
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels,
            test_size=val_split,
            stratify=labels if can_stratify else None,
            random_state=seed
        )
        
        print(f"  Train: {len(train_p)}, Val: {len(val_p)}")
        return train_p, train_l, val_p, val_l


def get_transforms(image_size: int = 224, is_training: bool = True, aug_config: Dict = None):
    """Get augmentation transforms."""
    aug = aug_config or {}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if ALBUMENTATIONS_AVAILABLE:
        if is_training:
            transform_list = [A.Resize(image_size, image_size)]
            
            if aug.get('horizontal_flip', True):
                transform_list.append(A.HorizontalFlip(p=0.5))
            
            rot = aug.get('rotation_range', 15)
            if rot > 0:
                transform_list.append(A.Rotate(limit=rot, p=0.5))
            
            bc = aug.get('brightness_contrast', 0.2)
            if bc > 0:
                transform_list.append(A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc, p=0.5))
            
            transform_list.extend([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
            return A.Compose(transform_list)
        else:
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
    else:
        from torchvision import transforms as T
        if is_training:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])


def create_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    aug_config: Dict = None,
    use_weighted_sampling: bool = False
):
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    train_transform = get_transforms(image_size, is_training=True, aug_config=aug_config)
    val_transform = get_transforms(image_size, is_training=False)
    
    train_dataset = MammographyDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = MammographyDataset(val_paths, val_labels, val_transform, image_size)
    
    sampler = None
    shuffle = True
    if use_weighted_sampling:
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(list(weights), len(train_labels), replacement=True)
        shuffle = False
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def prepare_dataset(dataset_name: str, data_root: str, val_split: float = 0.2, seed: int = 42, downloaded_json: str = None):
    """
    Unified dataset preparation.
    
    Args:
        dataset_name: "vindr" or "inbreast"
        data_root: Path to dataset folder
        val_split: Validation split ratio
        seed: Random seed
        downloaded_json: (VinDr only) Path to JSON with downloaded file list
    """
    if dataset_name.lower() == "vindr":
        ds = VinDrMammoDataset(data_root, downloaded_json=downloaded_json)
    elif dataset_name.lower() == "inbreast":
        ds = INbreastDataset(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return ds.prepare_splits(val_split, seed)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    import torch
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# DICOM TO PNG CONVERSION UTILITIES
# ============================================================================

def convert_dicom_to_png(
    dicom_path: str,
    output_path: str,
    size: Tuple[int, int] = (512, 512),
    apply_clahe: bool = True
) -> bool:
    """
    Convert a single DICOM to PNG.
    
    Args:
        dicom_path: Path to input DICOM
        output_path: Path for output PNG
        size: Output image size (width, height)
        apply_clahe: Apply CLAHE contrast enhancement
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = load_dicom_image(dicom_path, apply_clahe=apply_clahe)
        
        if CV2_AVAILABLE:
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save grayscale (first channel since all 3 are identical)
            cv2.imwrite(output_path, img_resized[:, :, 0])
        else:
            from PIL import Image
            pil_img = Image.fromarray(img[:, :, 0])
            pil_img = pil_img.resize(size, Image.LANCZOS)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pil_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False


def convert_dataset_to_png(
    image_paths: List[str],
    labels: List[int],
    output_dir: str,
    size: Tuple[int, int] = (512, 512),
    apply_clahe: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Convert a list of DICOM files to PNG.
    
    Args:
        image_paths: List of DICOM file paths
        labels: Corresponding labels
        output_dir: Directory for output PNGs
        size: Output image size
        apply_clahe: Apply CLAHE contrast enhancement
    
    Returns:
        (png_paths, labels) - only successfully converted files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    png_paths = []
    valid_labels = []
    
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Converting DICOMs to PNG"):
        # Create unique filename from path
        path_obj = Path(path)
        # Use study_id + image_id for unique naming
        parts = path_obj.parts
        if len(parts) >= 2:
            filename = f"{parts[-2]}_{path_obj.stem}.png"
        else:
            filename = f"{path_obj.stem}.png"
        
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already converted
        if os.path.exists(output_path):
            png_paths.append(output_path)
            valid_labels.append(label)
            continue
        
        if convert_dicom_to_png(path, output_path, size, apply_clahe):
            png_paths.append(output_path)
            valid_labels.append(label)
    
    print(f"\nConversion complete:")
    print(f"  Successful: {len(png_paths)}/{len(image_paths)}")
    print(f"  Output dir: {output_dir}")
    
    return png_paths, valid_labels


def prepare_dataset_with_png_conversion(
    dataset_name: str,
    data_root: str,
    png_output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    size: Tuple[int, int] = (512, 512),
    downloaded_json: str = None
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Prepare dataset with automatic DICOM to PNG conversion.
    
    This is the recommended function for faster training.
    
    Args:
        dataset_name: "vindr" or "inbreast"
        data_root: Path to original dataset
        png_output_dir: Directory to save converted PNGs
        val_split: Validation split ratio
        seed: Random seed
        size: PNG output size (width, height)
        downloaded_json: (VinDr only) Path to JSON filter file
    
    Returns:
        train_paths, train_labels, val_paths, val_labels (all as PNG paths)
    
    Example:
        train_paths, train_labels, val_paths, val_labels = prepare_dataset_with_png_conversion(
            dataset_name="vindr",
            data_root="/content/drive/MyDrive/vindr-mammo",
            png_output_dir="/content/drive/MyDrive/vindr-mammo-png",
            downloaded_json="/content/drive/MyDrive/vindr-mammo/downloaded_files.json"
        )
    """
    # First get the original paths (DICOM)
    train_paths, train_labels, val_paths, val_labels = prepare_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        val_split=val_split,
        seed=seed,
        downloaded_json=downloaded_json
    )
    
    print(f"\n{'='*60}")
    print("Converting DICOMs to PNG for faster training...")
    print(f"{'='*60}")
    
    # Convert train set
    train_png_dir = os.path.join(png_output_dir, "train")
    train_png_paths, train_labels = convert_dataset_to_png(
        train_paths, train_labels, train_png_dir, size
    )
    
    # Convert val set
    val_png_dir = os.path.join(png_output_dir, "val")
    val_png_paths, val_labels = convert_dataset_to_png(
        val_paths, val_labels, val_png_dir, size
    )
    
    print(f"\n{'='*60}")
    print("Dataset ready!")
    print(f"  Train: {len(train_png_paths)} (Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)})")
    print(f"  Val: {len(val_png_paths)} (Benign: {val_labels.count(0)}, Malignant: {val_labels.count(1)})")
    print(f"{'='*60}")
    
    return train_png_paths, train_labels, val_png_paths, val_labels


def load_prepared_png_dataset(
    png_dir: str,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load a previously converted PNG dataset.
    
    Expects structure:
    png_dir/
    ├── train/
    │   └── *.png
    └── val/
        └── *.png
    
    And CSV files with labels:
    png_dir/train_labels.csv
    png_dir/val_labels.csv
    
    Or will re-split if CSVs don't exist.
    """
    train_dir = os.path.join(png_dir, "train")
    val_dir = os.path.join(png_dir, "val")
    
    train_csv = os.path.join(png_dir, "train_labels.csv")
    val_csv = os.path.join(png_dir, "val_labels.csv")
    
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        # Load from CSVs
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        return (
            train_df['path'].tolist(),
            train_df['label'].tolist(),
            val_df['path'].tolist(),
            val_df['label'].tolist()
        )
    else:
        raise FileNotFoundError(
            f"Label CSVs not found. Use prepare_dataset_with_png_conversion() first."
        )

