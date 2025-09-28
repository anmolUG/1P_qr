#!/usr/bin/env python3
"""
Visualization script for QR code detections
"""

import cv2
import json
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_detections(image_dir, json_file, output_dir):
    """
    Visualize QR code detections by drawing bounding boxes on images
    
    Args:
        image_dir (str): Path to directory containing images
        json_file (str): Path to JSON file with detection results
        output_dir (str): Path to output directory for visualization images
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load detection results
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded detection results from {json_file}")
    except Exception as e:
        logger.error(f"Failed to load detection results: {e}")
        return
    
    # Process each image
    processed_count = 0
    for item in results:
        image_id = item['image_id']
        qrs = item['qrs']
        
        # Try different image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_path = None
        
        for ext in image_extensions:
            candidate_path = Path(image_dir) / f"{image_id}{ext}"
            if candidate_path.exists():
                image_path = candidate_path
                break
            
            # Try uppercase extension
            candidate_path = Path(image_dir) / f"{image_id}{ext.upper()}"
            if candidate_path.exists():
                image_path = candidate_path
                break
        
        if image_path is None:
            logger.warning(f"Image file not found for {image_id}")
            continue
        
        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                continue
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            continue
        
        # Draw bounding boxes
        qr_count = 0
        for qr in qrs:
            if 'bbox' in qr:
                bbox = qr['bbox']
                x_min, y_min, x_max, y_max = bbox
                
                # Draw red rectangle
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                # Add QR number label in red
                qr_count += 1
                label = f"QR {qr_count}"
                cv2.putText(image, label, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save visualization
        output_path = Path(output_dir) / f"{image_id}_detections.jpg"
        try:
            cv2.imwrite(str(output_path), image)
            logger.debug(f"Saved visualization for {image_id} with {qr_count} QR codes")
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to save visualization for {image_id}: {e}")
    
    logger.info(f"Processed {processed_count} images. Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize QR code detections")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory containing images")
    parser.add_argument("--json", "-j", required=True,
                       help="JSON file with detection results")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for visualization images")
    
    args = parser.parse_args()
    
    logger.info("🚀 QR Detection Visualization")
    logger.info("=" * 40)
    
    visualize_detections(args.input, args.json, args.output)
    
    logger.info("🎉 Visualization completed!")

if __name__ == "__main__":
    main()