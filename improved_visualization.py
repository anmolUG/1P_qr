#!/usr/bin/env python3
"""
Improved visualization script for QR code detection and decoding results
Generates visualizations with thicker borders, larger fonts, and no coordinates
"""

import cv2
import json
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_improved_visualization(image_dir, detection_file, decoding_file, output_dir):
    """
    Create improved visualization with thick borders and large fonts, without coordinates
    
    Args:
        image_dir (str): Path to directory containing images
        detection_file (str): JSON file with detection results
        decoding_file (str): JSON file with decoding results
        output_dir (str): Path to output directory for visualization images
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load detection and decoding results
    try:
        with open(detection_file, 'r') as f:
            detection_results = json.load(f)
        logger.info(f"Loaded detection results from {detection_file}")
        
        with open(decoding_file, 'r') as f:
            decoding_results = json.load(f)
        logger.info(f"Loaded decoding results from {decoding_file}")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return
    
    # Create dictionaries for easy lookup
    detection_dict = {item['image_id']: item['qrs'] for item in detection_results}
    decoding_dict = {item['image_id']: item['qrs'] for item in decoding_results}
    
    # Process each image
    processed_count = 0
    total_images = len(detection_dict)
    
    # Import tqdm for progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_images, desc="Generating Improved Visualizations", unit="image",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    except ImportError:
        # Fallback to simple logging if tqdm is not available
        progress_bar = None
        logger.info("Generating improved visualizations... (install tqdm for a progress bar)")
    
    for image_id in detection_dict.keys():
        # Get detection and decoding results for this image
        detections = detection_dict.get(image_id, [])
        decodings = decoding_dict.get(image_id, [])
        
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
            if progress_bar is not None:
                progress_bar.update(1)
            continue
        
        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                if progress_bar is not None:
                    progress_bar.update(1)
                continue
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            if progress_bar is not None:
                progress_bar.update(1)
            continue
        
        # Draw detection results (blue borders with thick lines)
        detection_count = 0
        for qr in detections:
            if 'bbox' in qr:
                bbox = qr['bbox']
                x_min, y_min, x_max, y_max = bbox
                
                # Draw very thick blue rectangle for detections (10px border)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 10)
                
                # Add detection number label with large, bold text
                detection_count += 1
                label = f"DET {detection_count}"
                
                # Get text size for background rectangle (larger font)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                
                # Draw background rectangle for text
                cv2.rectangle(image, (x_min, y_min - text_height - 15), 
                             (x_min + text_width, y_min), (255, 0, 0), -1)
                
                # Draw bold white text
                cv2.putText(image, label, (x_min, y_min - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Draw decoding results (red borders with thick lines)
        decoding_count = 0
        for qr in decodings:
            if 'bbox' in qr:
                bbox = qr['bbox']
                x_min, y_min, x_max, y_max = bbox
                
                # Draw very thick red rectangle for decodings (10px border)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)
                
                # Add decoding number label with large, bold text
                decoding_count += 1
                label = f"DEC {decoding_count}"
                
                # Get text size for background rectangle (larger font)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                
                # Draw background rectangle for text
                cv2.rectangle(image, (x_min, y_max), 
                             (x_min + text_width, y_max + text_height + 15), (0, 0, 255), -1)
                
                # Draw bold white text
                cv2.putText(image, label, (x_min, y_max + text_height + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                
                # Add decoded value if available with large text
                if 'value' in qr and qr['value']:
                    value_text = qr['value'][:20] + "..." if len(qr['value']) > 20 else qr['value']
                    # Get text size for value background (larger font)
                    (value_width, value_height), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                    
                    # Draw background rectangle for value
                    cv2.rectangle(image, (x_min, y_max + text_height + 20), 
                                 (x_min + value_width, y_max + text_height + value_height + 30), (0, 255, 0), -1)
                    
                    # Draw bold black text for value
                    cv2.putText(image, value_text, (x_min, y_max + text_height + value_height + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        
        # Add summary text at top of image with bold formatting
        summary_text = f"Image: {image_id} | Detections: {len(detections)} | Decodings: {len(decodings)}"
        (summary_width, summary_height), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(image, (0, 0), (summary_width + 30, summary_height + 30), (0, 0, 0), -1)
        cv2.putText(image, summary_text, (15, summary_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Add legend with bold text
        legend_y = image.shape[0] - 120
        # Detection legend (blue)
        cv2.rectangle(image, (20, legend_y), (40, legend_y + 30), (255, 0, 0), 10)
        cv2.putText(image, "Detection", (60, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        # Decoding legend (red)
        cv2.rectangle(image, (250, legend_y), (270, legend_y + 30), (0, 0, 255), 10)
        cv2.putText(image, "Decoding", (290, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # Decoded value legend (green)
        cv2.rectangle(image, (450, legend_y), (470, legend_y + 30), (0, 255, 0), -1)
        cv2.putText(image, "Decoded Value", (490, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Save visualization
        output_path = Path(output_dir) / f"{image_id}_improved_visualization.jpg"
        try:
            cv2.imwrite(str(output_path), image)
            if progress_bar is not None:
                progress_bar.set_postfix({"Processed": processed_count + 1, "Detections": len(detections), "Decodings": len(decodings)})
                progress_bar.update(1)
            else:
                logger.debug(f"Saved improved visualization for {image_id} with {len(detections)} detections and {len(decodings)} decodings")
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to save improved visualization for {image_id}: {e}")
            if progress_bar is not None:
                progress_bar.update(1)
    
    # Close progress bar if used
    if progress_bar is not None:
        progress_bar.close()
    
    logger.info(f"Processed {processed_count} images. Improved visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Improved visualization of QR detection and decoding results")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory containing images")
    parser.add_argument("--detection", "-d", required=True,
                       help="JSON file with detection results")
    parser.add_argument("--decoding", "-c", required=True,
                       help="JSON file with decoding results")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for improved visualization images")
    
    args = parser.parse_args()
    
    logger.info("Improved QR Detection and Decoding Visualization")
    logger.info("=" * 50)
    
    create_improved_visualization(args.input, args.detection, args.decoding, args.output)
    
    logger.info("Improved visualization completed!")

if __name__ == "__main__":
    main()