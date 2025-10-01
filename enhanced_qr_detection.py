#!/usr/bin/env python3
"""Enhanced QR detection with multi-threshold fusion and IoU-based de-dup."""

import os
import sys
import argparse
from pathlib import Path
import json
import logging
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from models.qr_detector import QRDetector
    from utils.qr_decoder_advanced import AdvancedQRDecoder
    ADVANCED_DECODER_AVAILABLE = True
except ImportError as e:
    print(f"Advanced decoder not available: {e}")
    from models.qr_detector import QRDetector
    from utils.qr_decoder import QRDecoder
    ADVANCED_DECODER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedQRDetector:
    def __init__(self, model_path: str = None):
        self.detector = QRDetector(device='cpu')
        
        # Load best available model
        if model_path and Path(model_path).exists():
            self.detector.load_pretrained(model_path)
            logger.info(f"Loaded custom model: {model_path}")
        else:
            # Try to find best available model
            model_candidates = [
                "runs/detect/qr_clean_training2/weights/best.pt",
                "runs/detect/qr_advanced_training/weights/best.pt",
                "runs/detect/qr_detection_improved/weights/best.pt",
            ]
            
            for candidate in model_candidates:
                if Path(candidate).exists():
                    self.detector.load_pretrained(candidate)
                    logger.info(f"Loaded best available model: {candidate}")
                    break
            else:
                logger.warning("No trained model found, using default YOLOv8")
        
        # Initialize decoder
        if ADVANCED_DECODER_AVAILABLE:
            self.decoder = AdvancedQRDecoder()
            logger.info("Using advanced QR decoder")
        else:
            from utils.qr_decoder import QRDecoder
            self.decoder = QRDecoder()
            logger.info("Using standard QR decoder")
    
    def calculate_iou(self, bbox1: list, bbox2: list) -> float:
        """Compute IoU between two boxes [x_min, y_min, x_max, y_max]."""
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Calculate intersection area
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            intersection = x_overlap * y_overlap
            
            # Calculate union area
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            
            if union == 0:
                return 0
            
            return intersection / union
            
        except Exception:
            return 0
    
    def remove_duplicate_detections(self, detections: list, iou_threshold: float = 0.3) -> list:
        """Remove duplicates via IoU threshold with confidence priority."""
        if not detections:
            return []
        
        # Sort by confidence in descending order (highest confidence first)
        sorted_detections = sorted(detections, 
                                 key=lambda x: x.get('confidence', 0.5), 
                                 reverse=True)
        
        unique_detections = []
        
        for current_det in sorted_detections:
            current_bbox = current_det['bbox']
            is_duplicate = False
            
            # Check against all previously accepted detections
            for unique_det in unique_detections:
                unique_bbox = unique_det['bbox']
                iou = self.calculate_iou(current_bbox, unique_bbox)
                
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(current_det)
        
        return unique_detections
    
    def detect_qr_codes_enhanced(self, image_path: str) -> list:
        """Multi-threshold detection with IoU-based consolidation."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return []
            
            all_detections = []
            
            # Strategy 1: High confidence baseline (proven system parameters)
            logger.debug("Running detection strategy 1 (conf=0.3, iou=0.45)")
            results1 = self.detector.predict(image, conf=0.3, iou=0.45, verbose=False)
            if results1 and len(results1) > 0:
                detections1 = results1[0].get('detections', [])
                all_detections.extend(detections1)
                logger.debug(f"Strategy 1 found {len(detections1)} detections")
            
            # Strategy 2: Medium confidence for missed QRs
            logger.debug("Running detection strategy 2 (conf=0.15, iou=0.4)")
            results2 = self.detector.predict(image, conf=0.15, iou=0.4, verbose=False)
            if results2 and len(results2) > 0:
                detections2 = results2[0].get('detections', [])
                all_detections.extend(detections2)
                logger.debug(f"Strategy 2 found {len(detections2)} detections")
            
            # Strategy 3: Low confidence aggressive detection
            logger.debug("Running detection strategy 3 (conf=0.1, iou=0.3)")
            results3 = self.detector.predict(image, conf=0.1, iou=0.3, verbose=False)
            if results3 and len(results3) > 0:
                detections3 = results3[0].get('detections', [])
                all_detections.extend(detections3)
                logger.debug(f"Strategy 3 found {len(detections3)} detections")
            
            # Strategy 4: Very low confidence for edge cases
            logger.debug("Running detection strategy 4 (conf=0.05, iou=0.25)")
            results4 = self.detector.predict(image, conf=0.05, iou=0.25, verbose=False)
            if results4 and len(results4) > 0:
                detections4 = results4[0].get('detections', [])
                all_detections.extend(detections4)
                logger.debug(f"Strategy 4 found {len(detections4)} detections")
            
            # Remove duplicates with conservative threshold to minimize false positives
            unique_detections = self.remove_duplicate_detections(all_detections, iou_threshold=0.3)
            logger.debug(f"Combined: {len(all_detections)} raw detections -> {len(unique_detections)} unique detections")
            
            return unique_detections
            
        except Exception as e:
            logger.error(f"Enhanced detection failed for {image_path}: {e}")
            return []
    
    def detect_qr_codes_baseline(self, image_path: str) -> list:
        """Baseline detection (single threshold)."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return []
            
            # Use the proven detection method
            results = self.detector.predict(image, conf=0.3, iou=0.45, verbose=False)
            
            if results and len(results) > 0:
                detections = results[0].get('detections', [])
                logger.debug(f"Baseline detected {len(detections)} QR codes in {Path(image_path).name}")
                return detections
            else:
                logger.debug(f"No QR codes detected in {Path(image_path).name}")
                return []
                
        except Exception as e:
            logger.error(f"Baseline detection failed for {image_path}: {e}")
            return []
    
    def process_single_image(self, image_path: str, use_enhanced: bool = True) -> dict:
        """Process a single image for QR detection."""
        image_id = Path(image_path).stem
        
        # Detect QR codes
        if use_enhanced:
            detections = self.detect_qr_codes_enhanced(image_path)
        else:
            detections = self.detect_qr_codes_baseline(image_path)
        
        # Return detection-only results
        qr_results = [{"bbox": det["bbox"]} for det in detections]
        
        return {
            "image_id": image_id,
            "qrs": qr_results
        }
    
    def process_directory(self, input_dir: str, output_file: str, use_enhanced: bool = True) -> dict:
        """Process all images in a directory."""
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {"total_images": 0, "total_qrs": 0}
        
        logger.info(f"Processing {len(image_files)} images with {'enhanced' if use_enhanced else 'baseline'} method...")
        
        # Process images
        results = []
        total_qrs = 0
        
        for image_file in image_files:
            try:
                result = self.process_single_image(str(image_file), use_enhanced=use_enhanced)
                results.append(result)
                
                # Update statistics
                qr_count = len(result["qrs"])
                total_qrs += qr_count
                
                if qr_count > 0:
                    logger.debug(f"{result['image_id']}: {qr_count} QRs")
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                # Add empty result to maintain consistency
                results.append({
                    "image_id": image_file.stem,
                    "qrs": []
                })
        
        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate statistics
        stats = {
            "total_images": len(image_files),
            "images_with_qrs": sum(1 for r in results if len(r["qrs"]) > 0),
            "total_qrs": total_qrs,
            "average_qrs_per_image": total_qrs / len(image_files) if image_files else 0,
        }
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Statistics: {stats}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="Enhanced QR Code Detection System")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSON file path")
    parser.add_argument("--baseline", action="store_true",
                       help="Use baseline detection (single threshold) instead of enhanced method")
    parser.add_argument("--model", "-m", 
                       help="Path to trained model weights")
    
    args = parser.parse_args()
    
    logger.info("🚀 Enhanced QR Detection System")
    logger.info("=" * 50)
    
    try:
        # Initialize enhanced detector
        detector = EnhancedQRDetector(model_path=args.model)
        
        # Process images
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Detection method: {'BASELINE' if args.baseline else 'ENHANCED'}")
        
        stats = detector.process_directory(
            input_dir=args.input,
            output_file=args.output,
            use_enhanced=not args.baseline
        )
        
        # Print final statistics
        logger.info("\n🎉 Processing completed successfully!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   - Total images processed: {stats['total_images']}")
        logger.info(f"   - Images with QR codes: {stats['images_with_qrs']}")
        logger.info(f"   - Total QR codes detected: {stats['total_qrs']}")
        logger.info(f"   - Average QRs per image: {stats['average_qrs_per_image']:.2f}")
        
        logger.info(f"\n📁 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()