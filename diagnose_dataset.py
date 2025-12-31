"""
Dataset Structure Diagnostic Tool

Analyzes the CSV file and directory structure to understand how to match them.
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_dataset.py /path/to/kaggle_vindr_data")
        sys.exit(1)

    data_root = Path(sys.argv[1])

    print("="*70)
    print("DATASET STRUCTURE DIAGNOSTIC")
    print("="*70)
    print(f"\nData root: {data_root}")

    # Find CSV file
    csv_candidates = [
        data_root / "vindr_detection_v1_folds.csv",
        data_root / "breast-level_annotations.csv",
        data_root / "metadata.csv",
    ]

    csv_path = None
    for candidate in csv_candidates:
        if candidate.exists():
            csv_path = candidate
            break

    if not csv_path:
        print("\n[X] No CSV file found!")
        return

    print(f"\n[OK] CSV file: {csv_path.name}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nCSV Info:")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

    # Show first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())

    # Check for image-related columns
    print(f"\nImage-related columns:")
    for col in df.columns:
        if any(kw in col.lower() for kw in ['image', 'file', 'path', 'study', 'patient']):
            print(f"  - {col}")
            # Show sample values
            sample_vals = df[col].head(3).tolist()
            print(f"    Sample values: {sample_vals}")

    # Analyze directory structure
    print(f"\n" + "="*70)
    print("DIRECTORY STRUCTURE")
    print("="*70)

    # Check for images folder
    images_dir = data_root / "images"
    if images_dir.exists():
        print(f"\n[OK] Found 'images/' subfolder")
        png_files = list(images_dir.glob("**/*.png"))[:5]
    else:
        print(f"\n[!] No 'images/' subfolder, checking root...")
        png_files = list(data_root.glob("**/*.png"))[:5]

    print(f"\nSample PNG file paths (first 5):")
    for i, png_file in enumerate(png_files, 1):
        rel_path = png_file.relative_to(data_root)
        print(f"  {i}. {rel_path}")
        print(f"     Filename: {png_file.name}")
        print(f"     Parent dir: {png_file.parent.name}")

    # List subdirectories
    subdirs = [d for d in data_root.iterdir() if d.is_dir() and d.name not in ['images', '__pycache__']]
    print(f"\nSubdirectories in root ({len(subdirs)} total):")
    for subdir in subdirs[:5]:
        png_count = len(list(subdir.glob("*.png")))
        print(f"  - {subdir.name}/ ({png_count} PNG files)")

    if len(subdirs) > 5:
        print(f"  ... and {len(subdirs) - 5} more")

    # Try to match a sample
    print(f"\n" + "="*70)
    print("MATCHING TEST")
    print("="*70)

    print("\nAttempting to match first 5 CSV rows to files...")
    for idx, row in df.head(5).iterrows():
        print(f"\nRow {idx}:")
        for col in df.columns:
            if any(kw in col.lower() for kw in ['image', 'file', 'study', 'patient']):
                print(f"  {col}: {row[col]}")

        # Try to find the file
        # Check different patterns
        found = False
        for col in df.columns:
            if 'image' in col.lower() or 'file' in col.lower():
                image_id = str(row[col]).strip()
                # Pattern 1: direct
                candidates = [
                    data_root / f"{image_id}.png",
                ]
                # Pattern 2: with subdirectory
                for subdir_col in ['study_id', 'patient_id', 'StudyInstanceUID', 'PatientID']:
                    if subdir_col in df.columns:
                        subdir = str(row[subdir_col]).strip()
                        candidates.append(data_root / subdir / f"{image_id}.png")

                for candidate in candidates:
                    if candidate.exists():
                        rel_path = candidate.relative_to(data_root)
                        print(f"  [OK] FOUND: {rel_path}")
                        found = True
                        break
                if found:
                    break

        if not found:
            print(f"  [X] NOT FOUND")

if __name__ == '__main__':
    main()
