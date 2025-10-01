#!/usr/bin/env python3
"""Fast inference for QR detection and decoding (accuracy-focused, faster)."""

import os
import sys
import argparse
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.qr_detector import QRDetector
from utils.qr_decoder_fast import FastQRDecoder
from utils.qr_decoder import QRDecoder as BackupQRDecoder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QRInferenceFast:
    """Faster pipeline using fewer detection passes and FastQRDecoder."""

    def __init__(self, model_path: str = None):
        self.detector = QRDetector(device='cpu')

        # Load best available model
        if model_path and Path(model_path).exists():
            self.detector.load_pretrained(model_path)
            logger.info(f"Loaded custom model: {model_path}")
        else:
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

        # Fast decoder + lightweight backup
        self.decoder = FastQRDecoder()
        self.backup_decoder = BackupQRDecoder()

    def calculate_iou(self, bbox1: list, bbox2: list) -> float:
        try:
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            intersection = x_overlap * y_overlap
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - intersection
            if union == 0:
                return 0
            return intersection / union
        except Exception:
            return 0

    def remove_duplicate_detections(self, detections: list, iou_threshold: float = 0.4) -> list:
        if not detections:
            return []
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0.5), reverse=True)
        unique_detections = []
        for current_det in sorted_detections:
            current_bbox = current_det['bbox']
            is_duplicate = False
            for unique_det in unique_detections:
                if self.calculate_iou(current_bbox, unique_det['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_detections.append(current_det)
        return unique_detections

    def detect_qr_codes(self, image_path: str) -> list:
        """Two-pass detection (balanced + sensitive) for speed and recall."""
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return []

            all_detections = []

            # Pass 1: balanced (good precision)
            r1 = self.detector.predict(image, conf=0.25, iou=0.4, verbose=False)
            if r1 and len(r1) > 0:
                all_detections.extend(r1[0].get('detections', []))

            # Pass 2: sensitive (recall boost, still moderate)
            r2 = self.detector.predict(image, conf=0.18, iou=0.35, verbose=False)
            if r2 and len(r2) > 0:
                all_detections.extend(r2[0].get('detections', []))

            # De-dup
            unique = self.remove_duplicate_detections(all_detections, iou_threshold=0.35)

            # Minimal filtering (very light)
            filtered = []
            h, w = image.shape[:2]
            for d in unique:
                x1, y1, x2, y2 = d['bbox']
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                if x2 - x1 > 0.95 * w or y2 - y1 > 0.95 * h:
                    continue
                filtered.append(d)

            return filtered
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return []

    def decode_qr_codes(self, image_path: str, detections: list) -> list:
        if not detections:
            return []
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return []
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = [d['bbox'] for d in detections]

            # Fast batch decode first
            try:
                fast_values = self.decoder.decode_multiple_regions(image_rgb, bboxes, enhance=True)
            except Exception:
                # Fallback to per-region fast decoding
                fast_values = []
                for bbox in bboxes:
                    v = self.decoder.decode_qr_from_region(image_rgb, bbox, enhance=True)
                    fast_values.append(v if v else "")

            # Fill missing with lightweight backup decoder (target only empties)
            filled_values = []
            for v, bbox in zip(fast_values, bboxes):
                if v:
                    filled_values.append(v.strip())
                else:
                    # Quick backup: try without heavy enhancement first
                    vb = self.backup_decoder.decode_qr_from_region(image_rgb, bbox, enhance=False)
                    if not vb:
                        vb = self.backup_decoder.decode_qr_from_region(image_rgb, bbox, enhance=True)
                    filled_values.append(vb.strip() if vb else "")

            return [{"bbox": bbox, "value": val} for bbox, val in zip(bboxes, filled_values)]
        except Exception as e:
            logger.error(f"Decoding failed for {image_path}: {e}")
            return [{"bbox": d["bbox"], "value": ""} for d in detections]

    def process_single_image(self, image_path: str, decode: bool = False) -> dict:
        image_id = Path(image_path).stem
        detections = self.detect_qr_codes(image_path)
        if decode and detections:
            qrs = self.decode_qr_codes(image_path, detections)
        else:
            qrs = [{"bbox": d["bbox"]} for d in detections]
        return {"image_id": image_id, "qrs": qrs}

    def process_directory(self, input_dir: str, output_file: str, decode: bool = False) -> dict:
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(list(set(image_files)))
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {"total_images": 0, "total_qrs": 0}

        logger.info(f"Processing {len(image_files)} images (fast mode)...")

        results = []
        total_qrs = 0

        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(image_files), desc="Processing", unit="image",
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        except ImportError:
            progress_bar = None

        for image_file in image_files:
            try:
                result = self.process_single_image(str(image_file), decode=decode)
                results.append(result)
                qr_count = len(result["qrs"]) 
                total_qrs += qr_count
                if progress_bar is not None:
                    progress_bar.set_postfix({"QRs": total_qrs, "Current": qr_count})
                    progress_bar.update(1)
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                results.append({"image_id": image_file.stem, "qrs": []})
                if progress_bar is not None:
                    progress_bar.set_postfix({"QRs": total_qrs, "Current": 0, "Errors": 1})
                    progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

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
    parser = argparse.ArgumentParser(description="Fast QR Detection and Decoding")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--decode", action="store_true", help="Enable QR code decoding")
    parser.add_argument("--model", "-m", help="Path to trained model weights")

    args = parser.parse_args()

    logger.info("Fast QR Detection and Decoding Pipeline")
    logger.info("=" * 50)

    try:
        inference = QRInferenceFast(model_path=args.model)
        stats = inference.process_directory(input_dir=args.input, output_file=args.output, decode=args.decode)
        logger.info("\nProcessing completed successfully!")
        logger.info("Final Statistics:")
        logger.info(f"   - Total images processed: {stats['total_images']}")
        logger.info(f"   - Images with QR codes: {stats['images_with_qrs']}")
        logger.info(f"   - Total QR codes detected: {stats['total_qrs']}")
        logger.info(f"   - Average QRs per image: {stats['average_qrs_per_image']:.2f}")
        logger.info(f"\nResults saved to: {args.output}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


