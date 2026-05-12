"""
Stanford Cars Dataset download helper.

Option A (Kaggle API):
  export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy
  python src/download_data.py --method kaggle

Option B (manual): Download from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
  and extract into data/raw/stanford_cars/
"""

import os
import argparse
import subprocess
import zipfile
import tarfile
from pathlib import Path


DATA_DIR = Path("data/raw/stanford_cars")

KAGGLE_DATASET = "rickyyyyyyy/torchvision-stanford-cars"


def download_kaggle(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Stanford Cars from Kaggle ({KAGGLE_DATASET})...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir), "--unzip"],
        check=True,
    )
    print(f"Downloaded to {data_dir}")


def verify_structure(data_dir: Path):
    required = [data_dir / "devkit" / "cars_annos.mat"]
    alt_required = [data_dir / "train", data_dir / "test"]

    mat_ok = all(p.exists() for p in required)
    folder_ok = all(p.exists() for p in alt_required)

    if mat_ok:
        print("Dataset structure OK (mat-based layout)")
    elif folder_ok:
        print("Dataset structure OK (folder-based layout)")
    else:
        print("Dataset structure not recognized. Expected one of:")
        print(f"  {data_dir}/devkit/cars_annos.mat")
        print(f"  {data_dir}/train/ and {data_dir}/test/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["kaggle", "verify"], default="verify")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.method == "kaggle":
        download_kaggle(data_dir)

    verify_structure(data_dir)


if __name__ == "__main__":
    main()
