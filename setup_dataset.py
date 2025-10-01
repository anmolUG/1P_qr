#!/usr/bin/env python3
"""Dataset setup helper script."""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/train_images',
        'data/test_images',
        'data/demo_images',
        'outputs',
        'runs/detect/qr_clean_training2/weights'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def download_dataset():
    """Provide instructions for downloading the dataset"""
    print("Dataset Setup Instructions")
    print("=" * 30)
    print("1. Download the dataset from the provided link:")
    print("   https://drive.google.com/file/d/1YCQggB6DdBEeIeBJy_odCW8ma_dq6Fg9/view?usp=sharing")
    print()
    print("2. Extract the downloaded file")
    print("3. Copy the train images to: data/train_images/")
    print("4. Copy the test images to: data/test_images/")
    print()
    print("After setting up the dataset, you can:")
    print("- Train the model: python train.py --data_dir data/train_images")
    print("- Run detection: python infer.py --input data/test_images --output outputs/submission_detection_1.json")
    print("- Run detection and decoding: python infer.py --input data/test_images --output outputs/submission_decoding_2.json --decode")

def main():
    """Main setup function"""
    print("Multi-QR Code Recognition System - Dataset Setup")
    print("=" * 50)
    
    create_directories()
    download_dataset()
    
    print("\n" + "=" * 50)
    print("Setup completed! Follow the instructions above to download and set up the dataset.")

if __name__ == "__main__":
    main()