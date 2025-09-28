"""
QR Code decoding and classification utilities
"""

import cv2
import numpy as np
# from pyzbar import pyzbar  # Disabled due to Windows DLL issues
from typing import List, Dict, Any, Tuple, Optional
import re
import logging

class QRDecoder:
    """QR Code decoder and classifier"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Classification patterns for different QR types
        self.classification_patterns = {
            'batch_number': [
                r'^B[0-9]{4,8}$',  # Batch number pattern
                r'^BATCH[0-9]{4,8}$',
                r'^LOT[0-9]{4,8}$'
            ],
            'manufacturer': [
                r'^MFR[A-Z0-9]{4,8}$',  # Manufacturer code
                r'^MANU[A-Z0-9]{4,8}$',
                r'^COMP[A-Z0-9]{4,8}$'
            ],
            'distributor': [
                r'^DIST[A-Z0-9]{4,8}$',  # Distributor code
                r'^DISTR[A-Z0-9]{4,8}$'
            ],
            'regulator': [
                r'^REG[A-Z0-9]{4,8}$',  # Regulator code
                r'^FDA[A-Z0-9]{4,8}$',
                r'^REGUL[A-Z0-9]{4,8}$'
            ],
            'expiry_date': [
                r'^\d{2}/\d{2}/\d{4}$',  # Date format MM/DD/YYYY
                r'^\d{4}-\d{2}-\d{2}$',  # Date format YYYY-MM-DD
                r'^EXP\d{8}$'           # Expiry date format
            ],
            'serial_number': [
                r'^SN[0-9]{6,12}$',     # Serial number
                r'^SERIAL[0-9]{6,12}$'
            ]
        }
    
    def decode_qr_from_region(self, image: np.ndarray, bbox: List[int], 
                             enhance: bool = True) -> Optional[str]:
        """
        Decode QR code from a specific region of the image
        
        Args:
            image: Input image (RGB format)
            bbox: Bounding box [x_min, y_min, x_max, y_max]
            enhance: Whether to apply image enhancement
            
        Returns:
            Decoded QR value or None if decoding fails
        """
        try:
            x_min, y_min, x_max, y_max = bbox
            
            # Extract region with some padding
            padding = 20  # Increased padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                return None
            
            # Convert to BGR for OpenCV operations
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            
            # Try multiple decoding strategies
            # 1. Direct decoding
            decoded_value = self._decode_with_opencv(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 2. Enhanced decoding
            if enhance:
                enhanced_roi = self._enhance_qr_region(roi_bgr)
                decoded_value = self._decode_with_opencv(enhanced_roi)
                if decoded_value:
                    return decoded_value
            
            # 3. Try with different preprocessing
            decoded_value = self._decode_with_multiple_preprocessing(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 4. Try with rotation
            decoded_value = self._decode_with_rotation(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 5. Try with scaling
            decoded_value = self._decode_with_scaling(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 6. Try with perspective correction
            decoded_value = self._decode_with_perspective_correction(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 7. Try with contour-based refinement
            decoded_value = self._decode_with_contour_refinement(roi_bgr)
            if decoded_value:
                return decoded_value
            
            # 8. Try extreme enhancement for challenging cases
            decoded_value = self._decode_with_extreme_enhancement(roi_bgr)
            if decoded_value:
                return decoded_value
            
            return None
            
        except Exception as e:
            self.logger.warning(f"QR decoding failed for bbox {bbox}: {e}")
            return None
    
    def _enhance_qr_region(self, roi: np.ndarray) -> np.ndarray:
        """Enhance QR region for better decoding"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _decode_with_multiple_preprocessing(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Try decoding with multiple advanced preprocessing techniques"""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Advanced preprocessing approaches
        preprocessing_methods = [
            # 1. Simple binary threshold with multiple values
            lambda img: cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
            lambda img: cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1],
            
            # 2. Otsu's threshold
            lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            
            # 3. Adaptive thresholds with different parameters
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3),
            lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3),
            
            # 4. Morphological operations
            lambda img: cv2.morphologyEx(
                cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
                cv2.MORPH_OPEN, 
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            ),
            lambda img: cv2.morphologyEx(
                cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
                cv2.MORPH_CLOSE, 
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            ),
            
            # 5. Enhanced contrast methods
            lambda img: cv2.threshold(
                cv2.equalizeHist(img), 127, 255, cv2.THRESH_BINARY
            )[1],
            
            # 6. CLAHE with different parameters
            lambda img: cv2.threshold(
                cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img),
                127, 255, cv2.THRESH_BINARY
            )[1],
            
            # 7. Gaussian blur + threshold
            lambda img: cv2.threshold(
                cv2.GaussianBlur(img, (3, 3), 0), 127, 255, cv2.THRESH_BINARY
            )[1],
            
            # 8. Bilateral filter + threshold (noise reduction)
            lambda img: cv2.threshold(
                cv2.bilateralFilter(img, 9, 75, 75), 127, 255, cv2.THRESH_BINARY
            )[1],
            
            # 9. Unsharp masking for edge enhancement
            lambda img: self._unsharp_mask(img),
        ]
        
        for method in preprocessing_methods:
            try:
                processed = method(gray)
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                result = self._decode_with_opencv(processed_bgr)
                if result:
                    return result
            except Exception:
                continue
                
        return None
    
    def _decode_with_perspective_correction(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Try to correct perspective distortion and decode"""
        try:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Find contours to detect potential QR code corners
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If we found a quadrilateral, try perspective correction
                if len(approx) == 4:
                    # Sort corners
                    corners = approx.reshape(4, 2).astype(np.float32)
                    
                    # Define target square
                    size = 200
                    target_corners = np.array([
                        [0, 0], [size, 0], [size, size], [0, size]
                    ], dtype=np.float32)
                    
                    # Get perspective transformation matrix
                    matrix = cv2.getPerspectiveTransform(corners, target_corners)
                    corrected = cv2.warpPerspective(roi_bgr, matrix, (size, size))
                    
                    result = self._decode_with_opencv(corrected)
                    if result:
                        return result
            
            return None
            
        except Exception:
            return None
    
    def _decode_with_contour_refinement(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Refine QR region using contour detection"""
        try:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Apply different thresholding methods to find contours
            thresholding_methods = [
                lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
                lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ]
            
            for threshold_method in thresholding_methods:
                try:
                    binary = threshold_method(gray)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Find the largest contour (likely the QR code)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        # Extract refined region
                        refined_roi = roi_bgr[y:y+h, x:x+w]
                        if refined_roi.size > 0:
                            result = self._decode_with_opencv(refined_roi)
                            if result:
                                return result
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _decode_with_extreme_enhancement(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Apply extreme enhancement techniques for challenging QR codes"""
        try:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Extreme enhancement techniques
            enhancement_methods = [
                # 1. Super contrast enhancement
                lambda img: cv2.convertScaleAbs(img, alpha=3.0, beta=50),
                
                # 2. Gamma correction variants
                lambda img: self._gamma_correction(img, 0.3),
                lambda img: self._gamma_correction(img, 0.5),
                lambda img: self._gamma_correction(img, 1.5),
                lambda img: self._gamma_correction(img, 2.0),
                
                # 3. Advanced CLAHE with extreme parameters
                lambda img: cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(img),
                lambda img: cv2.createCLAHE(clipLimit=8.0, tileGridSize=(16, 16)).apply(img),
                
                # 4. Edge enhancement
                lambda img: self._enhance_edges(img),
                
                # 5. Noise reduction + sharpening
                lambda img: self._denoise_and_sharpen(img),
            ]
            
            for method in enhancement_methods:
                try:
                    enhanced = method(gray)
                    # Apply threshold and convert back to BGR
                    _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
                    enhanced_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    
                    result = self._decode_with_opencv(enhanced_bgr)
                    if result:
                        return result
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges using Laplacian filter"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        enhanced = cv2.addWeighted(image, 0.7, laplacian, 0.3, 0)
        return enhanced
    
    def _denoise_and_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising followed by sharpening"""
        # Denoise
        denoised = cv2.fastNlMeansDenoising(image)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement"""
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return cv2.threshold(unsharp, 127, 255, cv2.THRESH_BINARY)[1]
    
    def _decode_with_opencv(self, roi_bgr: np.ndarray) -> Optional[str]:
        """OpenCV QR code decoding with enhanced error handling"""
        try:
            detector = cv2.QRCodeDetector()
            
            # Try detectAndDecodeMulti first
            retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(roi_bgr)
            
            if retval and decoded_info:
                # Return first non-empty decoded string
                for info in decoded_info:
                    if info and len(info.strip()) > 0:
                        return info.strip()
            
            # Try simple detect and decode if multi fails
            retval, decoded_info, points = detector.detectAndDecode(roi_bgr)
            
            if retval and decoded_info and len(decoded_info.strip()) > 0:
                return decoded_info.strip()
            
            # Try on grayscale version
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            retval, decoded_info, points = detector.detectAndDecode(gray)
            
            if retval and decoded_info and len(decoded_info.strip()) > 0:
                return decoded_info.strip()
            
            return None
            
        except Exception as e:
            self.logger.debug(f"OpenCV QR decoding failed: {e}")
            return None
    
    def _decode_with_pyzbar(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Standard pyzbar decoding - disabled on Windows"""
        # Disabled due to Windows DLL compatibility issues
        return None
    
    def _decode_with_rotation(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Try decoding with comprehensive rotation angles"""
        # More comprehensive rotation angles including fine adjustments
        angles = [0, 45, 90, 135, 180, 225, 270, 315, -45, -90, -135]
        
        # Add fine-tuning angles for slightly skewed QR codes
        fine_angles = [5, 10, 15, -5, -10, -15, 30, -30]
        angles.extend(fine_angles)
        
        for angle in angles:
            try:
                if angle == 0:
                    rotated = roi_bgr
                else:
                    h, w = roi_bgr.shape[:2]
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(roi_bgr, rotation_matrix, (w, h), 
                                           borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=(255, 255, 255))
                
                result = self._decode_with_opencv(rotated)
                if result:
                    return result
            except Exception:
                continue
        
        return None
    
    def _decode_with_scaling(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Try decoding with comprehensive scaling and interpolation methods"""
        # More comprehensive scaling factors
        scales = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
        
        # Different interpolation methods
        interpolations = [
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
            cv2.INTER_NEAREST
        ]
        
        for scale in scales:
            h, w = roi_bgr.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h < 15 or new_w < 15:  # Skip if too small
                continue
            if new_h > 2000 or new_w > 2000:  # Skip if too large
                continue
                
            for interp in interpolations:
                try:
                    scaled = cv2.resize(roi_bgr, (new_w, new_h), interpolation=interp)
                    result = self._decode_with_opencv(scaled)
                    if result:
                        return result
                except Exception:
                    continue
        
        return None
    
    def classify_qr_content(self, qr_value: str) -> str:
        """
        Classify QR content based on patterns
        
        Args:
            qr_value: Decoded QR string
            
        Returns:
            Classification type
        """
        if not qr_value:
            return 'unknown'
        
        qr_upper = qr_value.upper().strip()
        
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if re.match(pattern, qr_upper):
                    return category
        
        # Additional heuristic classification
        if any(keyword in qr_upper for keyword in ['BATCH', 'LOT']):
            return 'batch_number'
        elif any(keyword in qr_upper for keyword in ['MFR', 'MANU', 'MANUFACTURER']):
            return 'manufacturer'
        elif any(keyword in qr_upper for keyword in ['DIST', 'DISTRIBUTOR']):
            return 'distributor'
        elif any(keyword in qr_upper for keyword in ['REG', 'FDA', 'REGULATOR']):
            return 'regulator'
        elif any(keyword in qr_upper for keyword in ['EXP', 'EXPIRY', 'DATE']):
            return 'expiry_date'
        elif any(keyword in qr_upper for keyword in ['SN', 'SERIAL']):
            return 'serial_number'
        
        return 'unknown'
    
    def process_multiple_qrs(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple QR detections in an image
        
        Args:
            image: Input image (RGB format)
            detections: List of detection results with bboxes
            
        Returns:
            Updated detections with decoded values and classifications
        """
        enhanced_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Decode QR
            qr_value = self.decode_qr_from_region(image, bbox)
            
            # Update detection with decoded info
            enhanced_detection = detection.copy()
            if qr_value:
                enhanced_detection['value'] = qr_value
                enhanced_detection['type'] = self.classify_qr_content(qr_value)
                enhanced_detection['decoded'] = True
            else:
                enhanced_detection['decoded'] = False
                enhanced_detection['type'] = 'unknown'
            
            enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def validate_qr_detection(self, image: np.ndarray, bbox: List[int]) -> bool:
        """
        Validate if a bounding box likely contains a QR code
        
        Args:
            image: Input image
            bbox: Bounding box to validate
            
        Returns:
            True if likely contains QR code
        """
        try:
            x_min, y_min, x_max, y_max = bbox
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                return False
            
            # Check aspect ratio (QR codes are roughly square)
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height if height > 0 else 0
            
            if not (0.5 <= aspect_ratio <= 2.0):
                return False
            
            # Check minimum size
            if width < 20 or height < 20:
                return False
            
            # Try to decode - if successful, it's likely a QR code
            qr_value = self.decode_qr_from_region(image, bbox, enhance=False)
            return qr_value is not None
            
        except Exception:
            return False