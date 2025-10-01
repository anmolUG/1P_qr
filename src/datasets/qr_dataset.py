"""Dataset utilities for QR detection training."""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

class QRDataset(Dataset):
    """Dataset class for QR detection training."""
    
    def __init__(self, 
                 images_dir: str,
                 annotations_file: str,
                 image_size: Tuple[int, int] = (640, 640),
                 augmentations: Optional[A.Compose] = None,
                 mode: str = 'train'):
        """Initialize dataset with paths, size, augmentations, mode."""
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        self.image_ids = list(self.annotations.keys())
        
        # Setup augmentations
        if augmentations is None:
            self.augmentations = self._get_default_augmentations()
        else:
            self.augmentations = augmentations
    
    def _load_annotations(self, annotations_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load annotations from JSON file."""
        try:
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            
            # Convert to image_id -> annotations mapping
            annotations = {}
            for item in data:
                image_id = item['image_id']
                annotations[image_id] = item.get('qrs', [])
            
            self.logger.info(f"Loaded annotations for {len(annotations)} images")
            return annotations
            
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {e}")
            raise
    
    def _get_default_augmentations(self) -> A.Compose:
        """Default augmentation pipeline."""
        if self.mode == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            return A.Compose([
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.images_dir / f"{image_id}"
        if not image_path.suffix:
            # Try common image extensions
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                if (self.images_dir / f"{image_id}{ext}").exists():
                    image_path = self.images_dir / f"{image_id}{ext}"
                    break
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        qr_annotations = self.annotations.get(image_id, [])
        
        # Extract bounding boxes and labels
        bboxes = []
        labels = []
        
        for qr in qr_annotations:
            bbox = qr['bbox']  # [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            labels.append(1)  # QR code class = 1 (0 is background)
        
        # Handle case with no annotations
        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]  # Dummy bbox
            labels = [0]  # Background class
        
        # Apply augmentations
        try:
            augmented = self.augmentations(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            
            image = augmented['image']
            bboxes = augmented['bboxes']
            labels = augmented['labels']
            
        except Exception as e:
            self.logger.warning(f"Augmentation failed for {image_id}: {e}")
            # Fallback to basic resize and normalization
            image = cv2.resize(image, self.image_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Convert to YOLO format if needed
        if len(bboxes) > 0 and labels[0] != 0:  # Skip dummy boxes
            yolo_targets = self._convert_to_yolo_format(bboxes, labels)
        else:
            yolo_targets = torch.zeros((0, 5))  # Empty tensor for no detections
        
        return {
            'image': image,
            'targets': yolo_targets,
            'image_id': image_id,
            'original_size': (image.shape[2], image.shape[1])  # (width, height)
        }
    
    def _convert_to_yolo_format(self, bboxes: List[List[float]], labels: List[int]) -> torch.Tensor:
        """Convert bounding boxes to YOLO format."""
        yolo_targets = []
        
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to center coordinates and normalize
            x_center = (x_min + x_max) / 2.0 / self.image_size[0]
            y_center = (y_min + y_max) / 2.0 / self.image_size[1]
            width = (x_max - x_min) / self.image_size[0]
            height = (y_max - y_min) / self.image_size[1]
            
            yolo_targets.append([label, x_center, y_center, width, height])
        
        return torch.tensor(yolo_targets, dtype=torch.float32)

class QRDataModule:
    """Data module for organizing train/val/test datasets."""
    
    def __init__(self,
                 train_images_dir: str,
                 train_annotations: str,
                 val_images_dir: Optional[str] = None,
                 val_annotations: Optional[str] = None,
                 test_images_dir: Optional[str] = None,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (640, 640)):
        
        self.train_images_dir = train_images_dir
        self.train_annotations = train_annotations
        self.val_images_dir = val_images_dir or train_images_dir
        self.val_annotations = val_annotations or train_annotations
        self.test_images_dir = test_images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets."""
        # Training dataset with augmentations
        train_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Resize(self.image_size[1], self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        # Validation dataset without augmentations
        val_augmentations = A.Compose([
            A.Resize(self.image_size[1], self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        self.train_dataset = QRDataset(
            self.train_images_dir,
            self.train_annotations,
            self.image_size,
            train_augmentations,
            mode='train'
        )
        
        self.val_dataset = QRDataset(
            self.val_images_dir,
            self.val_annotations,
            self.image_size,
            val_augmentations,
            mode='val'
        )
        
        if self.test_images_dir:
            self.test_dataset = QRDataset(
                self.test_images_dir,
                self.val_annotations,  # Use val annotations as placeholder
                self.image_size,
                val_augmentations,
                mode='test'
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching."""
        images = torch.stack([item['image'] for item in batch])
        
        # Handle variable number of targets per image
        targets = []
        for i, item in enumerate(batch):
            target = item['targets']
            if len(target) > 0:
                # Add batch index to targets
                batch_targets = torch.cat([
                    torch.full((len(target), 1), i),  # Batch index
                    target
                ], dim=1)
                targets.append(batch_targets)
        
        if targets:
            targets = torch.cat(targets, dim=0)
        else:
            targets = torch.zeros((0, 6))  # Empty tensor
        
        return {
            'images': images,
            'targets': targets,
            'image_ids': [item['image_id'] for item in batch],
            'original_sizes': [item['original_size'] for item in batch]
        }

def create_sample_annotations(images_dir: str, output_file: str):
    """Create sample annotations file for testing."""
    import glob
    
    images_dir = Path(images_dir)
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(str(images_dir / ext)))
    
    sample_annotations = []
    
    for img_path in image_files[:5]:  # Create annotations for first 5 images
        img_name = Path(img_path).stem
        
        # Create dummy annotations (you should replace with real annotations)
        sample_annotations.append({
            'image_id': img_name,
            'qrs': [
                {'bbox': [100, 100, 200, 200]},  # Sample QR location
                {'bbox': [300, 150, 400, 250]}   # Another sample QR location
            ]
        })
    
    with open(output_file, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    print(f"Created sample annotations file: {output_file}")