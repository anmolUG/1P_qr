"""
Utility functions for QR code detection and processing
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Any
from pathlib import Path
import logging

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multiqr.log'),
            logging.StreamHandler()
        ]
    )

def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def apply_nms(boxes: List[List[float]], scores: List[float], threshold: float = 0.5) -> List[int]:
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        ious = [calculate_iou(boxes[current], boxes[idx]) for idx in indices[1:]]
        indices = indices[1:][np.array(ious) < threshold]
    
    return keep

def preprocess_image_for_detection(image: np.ndarray, size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Preprocess image for object detection"""
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to target size
    pad_w = (size[0] - new_w) // 2
    pad_h = (size[1] - new_h) // 2
    
    padded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded, scale, (pad_w, pad_h)

def postprocess_detections(detections: np.ndarray, scale: float, padding: Tuple[int, int], 
                         conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Postprocess detection results"""
    results = []
    pad_w, pad_h = padding
    
    for detection in detections:
        confidence = detection[4]
        if confidence < conf_threshold:
            continue
            
        # Convert from padded coordinates to original coordinates
        x_center, y_center, width, height = detection[:4]
        
        # Adjust for padding and scaling
        x_center = (x_center - pad_w) / scale
        y_center = (y_center - pad_h) / scale
        width = width / scale
        height = height / scale
        
        # Convert to x_min, y_min, x_max, y_max
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        
        results.append({
            'bbox': [x_min, y_min, x_max, y_max],
            'confidence': float(confidence)
        })
    
    return results

def augment_image(image: np.ndarray, rotation_range: int = 15, brightness_range: float = 0.2) -> np.ndarray:
    """Apply data augmentation to image"""
    h, w = image.shape[:2]
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Random brightness adjustment
    brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    
    return image

def create_submission_format(detections: List[Dict[str, Any]], image_id: str, 
                           include_values: bool = False) -> Dict[str, Any]:
    """Create submission format for hackathon"""
    qrs = []
    for detection in detections:
        qr_data = {'bbox': detection['bbox']}
        if include_values and 'value' in detection:
            qr_data['value'] = detection['value']
        qrs.append(qr_data)
    
    return {
        'image_id': image_id,
        'qrs': qrs
    }

def validate_submission_format(submission_data: List[Dict[str, Any]], stage: int = 1) -> bool:
    """Validate submission format"""
    required_keys = ['image_id', 'qrs']
    
    for entry in submission_data:
        if not all(key in entry for key in required_keys):
            return False
        
        for qr in entry['qrs']:
            if 'bbox' not in qr:
                return False
            
            if len(qr['bbox']) != 4:
                return False
                
            if stage == 2 and 'value' not in qr:
                return False
    
    return True