#!/usr/bin/env python3
"""
Inference script for QR code detection and decoding
"""

import os
import sys
import argparse
from pathlib import Path
import json
import logging

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

class QRInference:
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
    
    def detect_qr_codes(self, image_path: str) -> list:
        """Detect QR codes in an image using the proven reliable method"""
        try:
            import cv2
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return []
            
            # Use the proven detection method
            results = self.detector.predict(image, conf=0.3, iou=0.45, verbose=False)
            
            if results and len(results) > 0:
                detections = results[0].get('detections', [])
                logger.debug(f"Detected {len(detections)} QR codes in {Path(image_path).name}")
                return detections
            else:
                logger.debug(f"No QR codes detected in {Path(image_path).name}")
                return []
                
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []
    
    def decode_qr_codes(self, image_path: str, detections: list) -> list:
        """Decode detected QR codes"""
        if not detections:
            return []
        
        try:
            import cv2
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Convert to RGB for decoder
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            decoded_results = []
            
            for detection in detections:
                bbox = detection['bbox']
                
                # Decode QR code
                decoded_value = self.decoder.decode_qr_from_region(image_rgb, bbox, enhance=True)
                
                # Create result
                qr_result = {
                    "bbox": bbox,
                    "value": decoded_value.strip() if decoded_value else "",
                }
                
                decoded_results.append(qr_result)
            
            return decoded_results
            
        except Exception as e:
            logger.error(f"Decoding failed for {image_path}: {e}")
            # Return detections without decoding
            return [{"bbox": det["bbox"], "value": ""} for det in detections]
    
    def process_single_image(self, image_path: str, decode: bool = False) -> dict:
        """Process a single image for QR detection and optional decoding"""
        image_id = Path(image_path).stem
        
        # Detect QR codes
        detections = self.detect_qr_codes(image_path)
        
        if decode and detections:
            # Decode QR codes
            qr_results = self.decode_qr_codes(image_path, detections)
        else:
            # Detection only
            qr_results = [{"bbox": det["bbox"]} for det in detections]
        
        return {
            "image_id": image_id,
            "qrs": qr_results
        }
    
    def process_directory(self, input_dir: str, output_file: str, decode: bool = False) -> dict:
        """Process all images in a directory"""
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
        
        logger.info(f"Processing {len(image_files)} images...")
        
        # Process images
        results = []
        total_qrs = 0
        
        for image_file in image_files:
            try:
                result = self.process_single_image(str(image_file), decode=decode)
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
    parser = argparse.ArgumentParser(description="QR Code Detection and Decoding")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSON file path")
    parser.add_argument("--decode", action="store_true",
                       help="Enable QR code decoding")
    parser.add_argument("--model", "-m", 
                       help="Path to trained model weights")
    
    args = parser.parse_args()
    
    logger.info("🚀 QR Detection and Decoding Pipeline")
    logger.info("=" * 50)
    
    try:
        # Initialize inference pipeline
        inference = QRInference(model_path=args.model)
        
        # Process images
        logger.info(f"Input directory: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Decode mode: {'ON' if args.decode else 'OFF'}")
        
        stats = inference.process_directory(
            input_dir=args.input,
            output_file=args.output,
            decode=args.decode
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