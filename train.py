#!/usr/bin/env python3
"""
Training script for QR detection model
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import json
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import QRDetector, create_data_yaml, convert_annotations_to_yolo
from src.utils import setup_logging
from src.datasets import create_sample_annotations

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train QR Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--annotations', type=str, default='annotations.json',
                       help='Annotations file name (relative to data_dir)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='runs/train',
                       help='Directory to save training results')
    parser.add_argument('--name', type=str, default='qr_detection',
                       help='Experiment name')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--cache', action='store_true',
                       help='Cache images for faster training')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample annotations for testing')
    
    return parser.parse_args()

def prepare_data(data_dir: str, annotations_file: str, train_split: float = 0.8):
    """Prepare data in YOLO format"""
    data_dir = Path(data_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Check for images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_dir.glob(f"*{ext}")))
        image_files.extend(list(data_dir.glob(f"**/*{ext}")))
    
    if len(image_files) == 0:
        logging.warning(f"No images found in {data_dir}")
        return None, None, None
    
    logging.info(f"Found {len(image_files)} images")
    
    # Check annotations file
    annotations_path = data_dir / annotations_file
    if not annotations_path.exists():
        logging.warning(f"Annotations file not found: {annotations_path}")
        return None, None, None
    
    # Create train/val split directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create images and labels subdirectories
    for split_dir in [train_dir, val_dir]:
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'labels').mkdir(exist_ok=True)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Convert annotations to YOLO format
    convert_annotations_to_yolo(
        str(annotations_path),
        str(data_dir),
        str(train_dir / 'labels')
    )
    
    # Split data into train/val
    import random
    random.seed(42)
    
    image_ids = [ann['image_id'] for ann in annotations]
    random.shuffle(image_ids)
    
    split_idx = int(len(image_ids) * train_split)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    logging.info(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")
    
    # Copy/link images and labels to appropriate splits
    import shutil
    
    for image_id in train_ids:
        # Find image file
        image_path = None
        for ext in image_extensions:
            candidate = data_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path:
            shutil.copy2(image_path, train_dir / 'images' / image_path.name)
        
        # Copy label file
        label_path = train_dir / 'labels' / f"{image_id}.txt"
        if label_path.exists():
            # Already created by convert_annotations_to_yolo
            pass
    
    for image_id in val_ids:
        # Find image file
        image_path = None
        for ext in image_extensions:
            candidate = data_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path:
            shutil.copy2(image_path, val_dir / 'images' / image_path.name)
        
        # Copy label file
        src_label = train_dir / 'labels' / f"{image_id}.txt"
        dst_label = val_dir / 'labels' / f"{image_id}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
    
    return str(train_dir), str(val_dir), len(train_ids) + len(val_ids)

def train_model(args):
    """Train the QR detection model"""
    # Setup logging
    setup_logging()
    logging.info("Starting QR detection training")
    
    # Prepare data
    train_dir, val_dir, total_samples = prepare_data(
        args.data_dir, args.annotations, args.train_split
    )
    
    if train_dir is None:
        logging.error("Failed to prepare data")
        return
    
    # Create data.yaml
    data_yaml_path = Path(args.data_dir) / 'data.yaml'
    create_data_yaml(train_dir, val_dir, str(data_yaml_path))
    
    # Initialize model
    detector = QRDetector(
        model_size=args.model_size,
        num_classes=1,
        device=args.device
    )
    
    # Load pretrained weights
    detector.load_pretrained(args.pretrained)
    
    # Set detection parameters
    detector.set_detection_params(
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100
    )
    
    # Training arguments
    train_kwargs = {
        'lr0': args.lr,
        'patience': args.patience,
        'workers': args.workers,
        'cache': args.cache,
        'name': args.name,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,  # Automatic Mixed Precision
        'mosaic': 0.5,  # Mosaic augmentation probability
        'mixup': 0.1,   # Mixup augmentation probability
        'copy_paste': 0.1,  # Copy-paste augmentation probability
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
    }
    
    # Start training
    try:
        results = detector.train(
            train_data_yaml=str(data_yaml_path),
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            save_dir=args.save_dir,
            resume=args.resume,
            **train_kwargs
        )
        
        logging.info("Training completed successfully")
        
        # Validate model
        logging.info("Running validation...")
        val_results = detector.validate(data=str(data_yaml_path))
        
        # Print results summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total samples: {total_samples}")
        print(f"Model size: YOLOv8{args.model_size}")
        print(f"Epochs trained: {args.epochs}")
        print(f"Final mAP50: {val_results.box.map50:.4f}")
        print(f"Final mAP50-95: {val_results.box.map:.4f}")
        print(f"Best weights saved to: runs/train/{args.name}/weights/best.pt")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

def main():
    """Main function"""
    args = parse_args()
    
    # Create sample annotations if requested
    if args.create_sample:
        data_dir = Path(args.data_dir)
        images_dir = data_dir / 'demo_images'
        if not images_dir.exists():
            images_dir.mkdir(parents=True)
            print(f"Created demo images directory: {images_dir}")
            print("Please add some sample images and run again.")
            return
        
        annotations_file = data_dir / args.annotations
        create_sample_annotations(str(images_dir), str(annotations_file))
        print(f"Created sample annotations: {annotations_file}")
        print("Please modify the annotations with real bounding boxes.")
        return
    
    # Train model
    train_model(args)

if __name__ == '__main__':
    main()