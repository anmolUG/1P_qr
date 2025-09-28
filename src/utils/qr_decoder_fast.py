#!/usr/bin/env python3
"""
Fast QR Code Decoder
Optimized for speed while maintaining good accuracy
"""

import cv2
import numpy as np
import logging
from typing import Optional, List
import time

logger = logging.getLogger(__name__)

class FastQRDecoder:
    """Fast QR decoder with optimized processing pipeline"""
    
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.logger = logger
        
    def decode_qr_from_region(self, image: np.ndarray, bbox: List[int], 
                             enhance: bool = True) -> Optional[str]:
        """
        Fast QR decoding with optimized pipeline
        
        Args:
            image: Input image (RGB format)
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            enhance: Whether to use enhanced decoding (still fast)
            
        Returns:
            Decoded QR value or empty string if decoding fails
        """
        try:
            start_time = time.time()
            
            x_min, y_min, x_max, y_max = bbox
            
            # Minimal padding for speed
            padding = 8
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return ""
            
            # Convert to BGR for OpenCV
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            
            # FAST STRATEGY 1: Direct decode (works for 60-70% of QRs)
            retval, decoded_info, _ = self.detector.detectAndDecode(roi_bgr)
            if retval and decoded_info and len(decoded_info.strip()) > 0:
                elapsed = time.time() - start_time
                logger.debug(f"Direct decode success in {elapsed*1000:.1f}ms")
                return decoded_info.strip()
            
            # FAST STRATEGY 2: Grayscale conversion (quick color fix)
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            retval, decoded_info, _ = self.detector.detectAndDecode(gray_bgr)
            if retval and decoded_info and len(decoded_info.strip()) > 0:
                elapsed = time.time() - start_time
                logger.debug(f"Grayscale decode success in {elapsed*1000:.1f}ms")
                return decoded_info.strip()
            
            # Skip enhancement if not requested (for ultra-fast mode)
            if not enhance:
                return ""
            
            # FAST STRATEGY 3: 2x scaling only (most effective scaling)
            h, w = gray.shape
            if h > 20 and w > 20 and h < 300 and w < 300:  # Size check for speed
                try:
                    scaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                    scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                    retval, decoded_info, _ = self.detector.detectAndDecode(scaled_bgr)
                    if retval and decoded_info and len(decoded_info.strip()) > 0:
                        elapsed = time.time() - start_time
                        logger.debug(f"Scaled decode success in {elapsed*1000:.1f}ms")
                        return decoded_info.strip()
                        
                    # Also try 1.5x scaling (good middle ground)
                    scaled = cv2.resize(gray, (int(w*1.5), int(h*1.5)), interpolation=cv2.INTER_CUBIC)
                    scaled_bgr = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
                    retval, decoded_info, _ = self.detector.detectAndDecode(scaled_bgr)
                    if retval and decoded_info and len(decoded_info.strip()) > 0:
                        elapsed = time.time() - start_time
                        logger.debug(f"1.5x Scaled decode success in {elapsed*1000:.1f}ms")
                        return decoded_info.strip()
                except:
                    pass
            
            # FAST STRATEGY 4: OTSU thresholding only (most effective threshold)
            try:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                retval, decoded_info, _ = self.detector.detectAndDecode(thresh_bgr)
                if retval and decoded_info and len(decoded_info.strip()) > 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"OTSU decode success in {elapsed*1000:.1f}ms")
                    return decoded_info.strip()
            except:
                pass
            
            # FAST STRATEGY 5: Quick contrast enhancement (try multiple levels)
            contrast_levels = [1.8, 1.5, 2.2]  # Most effective contrast levels
            for alpha in contrast_levels:
                try:
                    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    retval, decoded_info, _ = self.detector.detectAndDecode(enhanced_bgr)
                    if retval and decoded_info and len(decoded_info.strip()) > 0:
                        elapsed = time.time() - start_time
                        logger.debug(f"Contrast {alpha} decode success in {elapsed*1000:.1f}ms")
                        return decoded_info.strip()
                except:
                    continue
            
            # FAST STRATEGY 6: Adaptive threshold (gaussian only)
            try:
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
                retval, decoded_info, _ = self.detector.detectAndDecode(adaptive_bgr)
                if retval and decoded_info and len(decoded_info.strip()) > 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"Adaptive decode success in {elapsed*1000:.1f}ms")
                    return decoded_info.strip()
            except:
                pass
            
            # FAST STRATEGY 7: Simple blur + threshold (noise reduction)
            try:
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                _, blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                blur_bgr = cv2.cvtColor(blur_thresh, cv2.COLOR_GRAY2BGR)
                retval, decoded_info, _ = self.detector.detectAndDecode(blur_bgr)
                if retval and decoded_info and len(decoded_info.strip()) > 0:
                    elapsed = time.time() - start_time
                    logger.debug(f"Blur decode success in {elapsed*1000:.1f}ms")
                    return decoded_info.strip()
            except:
                pass
            
            elapsed = time.time() - start_time
            logger.debug(f"All strategies failed in {elapsed*1000:.1f}ms")
            return ""
            
        except Exception as e:
            logger.debug(f"Fast decode error: {e}")
            return ""
    
    def decode_multiple_regions(self, image: np.ndarray, bboxes: List[List[int]], 
                               enhance: bool = True) -> List[str]:
        """
        Fast decode multiple QR regions in batch
        
        Args:
            image: Input image (RGB format)
            bboxes: List of bounding boxes
            enhance: Whether to use enhanced decoding
            
        Returns:
            List of decoded values (empty string for failed decodes)
        """
        start_time = time.time()
        
        results = []
        for bbox in bboxes:
            decoded = self.decode_qr_from_region(image, bbox, enhance)
            results.append(decoded if decoded else "")
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r)
        logger.debug(f"Batch decode: {success_count}/{len(bboxes)} in {elapsed*1000:.1f}ms")
        
        return results
    
    def classify_qr_content(self, content: str) -> str:
        """Fast QR content classification"""
        if not content or len(content.strip()) < 3:
            return "unknown"
        
        content_upper = content.upper().strip()
        
        # Quick pattern matching
        if len(content_upper) >= 8 and content_upper.isalnum():
            if content_upper.startswith(('5A0', 'BA', 'CA', 'DA')):
                return "serial_number"
            elif 'BATCH' in content_upper or 'LOT' in content_upper:
                return "batch_number"
            elif 'EXP' in content_upper or '/' in content:
                return "expiry_date"
            elif 'MFR' in content_upper or 'MANU' in content_upper:
                return "manufacturer"
            elif 'DIST' in content_upper:
                return "distributor"
            else:
                return "serial_number"  # Default for alphanumeric codes
        
        return "unknown"