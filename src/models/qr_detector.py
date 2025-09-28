"""
QR Detection Model using YOLOv8
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import logging

class QRDetector:
    """QR Detection model wrapper using YOLOv8"""
    
    def __init__(self, 
                 model_size: str = 'n',  # n, s, m, l, x
                 num_classes: int = 1,   # QR code class
                 device: str = 'auto'):
        """
        Initialize QR Detector
        
        Args:
            model_size: YOLOv8 model size (nano, small, medium, large, xlarge)
            num_classes: Number of classes (1 for QR codes)
            device: Device to run on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.is_trained = False
        
        # Detection parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 100
        
    def load_pretrained(self, weights_path: Optional[str] = None):
        """Load pretrained YOLOv8 model"""
        if weights_path and Path(weights_path).exists():
            self.logger.info(f"Loading custom weights from {weights_path}")
            self.model = YOLO(weights_path)
            self.is_trained = True
        else:
            # Load pretrained COCO model
            model_name = f'yolov8{self.model_size}.pt'
            self.logger.info(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
            
            # Modify for our number of classes if needed
            if self.num_classes != 80:  # COCO has 80 classes
                self._modify_model_classes()
    
    def _modify_model_classes(self):
        """Modify model for custom number of classes"""
        # This will be handled during training configuration
        pass
    
    def train(self,
              train_data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              save_dir: str = 'runs/train',
              resume: bool = False,
              **kwargs):
        """
        Train the QR detection model
        
        Args:
            train_data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            save_dir: Directory to save results
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments
        """
        if self.model is None:
            self.load_pretrained()
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        # Training arguments
        train_args = {
            'data': train_data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'save_dir': save_dir,
            'device': self.device,
            'resume': resume,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': True,
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': True,  # Automatic Mixed Precision
            **kwargs
        }
        
        # Start training
        results = self.model.train(**train_args)
        self.is_trained = True
        
        self.logger.info("Training completed")
        return results
    
    def predict(self,
                source,
                conf: float = None,
                iou: float = None,
                save: bool = False,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Run inference on images
        
        Args:
            source: Image source (path, array, etc.)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save results
            **kwargs: Additional prediction arguments
            
        Returns:
            List of detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Use instance thresholds if not provided
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        # Prediction arguments
        pred_args = {
            'source': source,
            'conf': conf,
            'iou': iou,
            'device': self.device,
            'save': save,
            'max_det': self.max_detections,
            **kwargs
        }
        
        # Run prediction
        results = self.model.predict(**pred_args)
        
        # Process results
        processed_results = []
        for result in results:
            processed_result = self._process_result(result)
            processed_results.append(processed_result)
        
        return processed_results
    
    def _process_result(self, result) -> Dict[str, Any]:
        """Process a single prediction result"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(score),
                    'class': int(cls),
                    'class_name': 'qr_code'
                }
                detections.append(detection)
        
        return {
            'image_path': result.path,
            'image_shape': result.orig_shape,
            'detections': detections
        }
    
    def validate(self, data_yaml: str, **kwargs):
        """Validate model performance"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        self.logger.info("Starting validation")
        results = self.model.val(data=data_yaml, **kwargs)
        
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """Export model to different formats"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        self.logger.info(f"Exporting model to {format}")
        self.model.export(format=format, **kwargs)
    
    def set_detection_params(self, 
                           conf_threshold: float = None,
                           iou_threshold: float = None,
                           max_detections: int = None):
        """Set detection parameters"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if max_detections is not None:
            self.max_detections = max_detections
    
    def detect_from_image(self, image: np.ndarray, return_image: bool = False) -> Dict[str, Any]:
        """
        Detect QR codes from a single image array
        
        Args:
            image: Input image (RGB format)
            return_image: Whether to return annotated image
            
        Returns:
            Detection results
        """
        # Convert RGB to BGR for YOLO
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run prediction
        results = self.predict(bgr_image)
        
        if len(results) == 0:
            return {'detections': [], 'annotated_image': image if return_image else None}
        
        result = results[0]
        detections = result['detections']
        
        annotated_image = None
        if return_image:
            annotated_image = self._annotate_image(image.copy(), detections)
        
        return {
            'detections': detections,
            'annotated_image': annotated_image
        }
    
    def _annotate_image(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Annotate image with detection results"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"QR: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return image
    
    def benchmark_speed(self, image_size: Tuple[int, int] = (640, 640), num_runs: int = 100):
        """Benchmark inference speed"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            _ = self.detect_from_image(dummy_image)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.detect_from_image(dummy_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        self.logger.info(f"Average inference time: {avg_time*1000:.2f}ms")
        self.logger.info(f"Average FPS: {fps:.2f}")
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'times': times
        }

def create_data_yaml(train_dir: str, val_dir: str, save_path: str):
    """Create data.yaml file for YOLO training"""
    import os
    
    # Use absolute paths for YOLO
    train_abs = os.path.abspath(os.path.join(train_dir, 'images'))
    val_abs = os.path.abspath(os.path.join(val_dir, 'images'))
    
    data_yaml_content = f"""
# QR Detection Dataset Configuration

# Paths (absolute)
train: {train_abs}
val: {val_abs}

# Number of classes
nc: 1

# Class names
names:
  0: qr_code
"""
    
    with open(save_path, 'w') as f:
        f.write(data_yaml_content.strip())
    
    print(f"Created data.yaml at: {save_path}")

def convert_annotations_to_yolo(annotations_file: str, 
                               images_dir: str,
                               output_dir: str,
                               image_size: Tuple[int, int] = (640, 640)):
    """Convert JSON annotations to YOLO format"""
    import json
    from pathlib import Path
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for annotation in annotations:
        image_id = annotation['image_id']
        qrs = annotation.get('qrs', [])
        
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = Path(images_dir) / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            continue
        
        # Get image dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        img_h, img_w = image.shape[:2]
        
        # Convert bounding boxes to YOLO format
        yolo_annotations = []
        for qr in qrs:
            bbox = qr['bbox']  # [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox
            
            # Convert to YOLO format (normalized center coordinates + width/height)
            x_center = (x_min + x_max) / 2.0 / img_w
            y_center = (y_min + y_max) / 2.0 / img_h
            width = (x_max - x_min) / img_w
            height = (y_max - y_min) / img_h
            
            # Class 0 for QR code
            yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save YOLO annotation file
        annotation_file = output_dir / f"{image_id}.txt"
        with open(annotation_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    print(f"Converted annotations saved to: {output_dir}")