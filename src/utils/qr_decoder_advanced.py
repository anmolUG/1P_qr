"""Advanced QR decoding with multiple CV strategies and fallbacks."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import logging
import math
from scipy import ndimage
from scipy.spatial.distance import cdist

# Optional dependencies - graceful fallback if not available
try:
    from skimage import filters, morphology, segmentation, feature
    from skimage.restoration import denoise_nl_means
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    # Create mock functions for graceful fallback
    def denoise_nl_means(image, **kwargs):
        return cv2.fastNlMeansDenoising(image)

import warnings
warnings.filterwarnings('ignore')

class AdvancedQRDecoder:
    """Advanced QR decoder with multiple strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced classification patterns
        self.classification_patterns = {
            'batch_number': [
                r'^B[0-9A-Z]{4,12}$',
                r'^BATCH[0-9A-Z]{4,12}$',
                r'^LOT[0-9A-Z]{4,12}$',
                r'^[0-9]{4,12}B$',
                r'^BN[0-9A-Z]{4,10}$'
            ],
            'manufacturer': [
                r'^MFR[A-Z0-9]{4,12}$',
                r'^MANU[A-Z0-9]{4,12}$',
                r'^COMP[A-Z0-9]{4,12}$',
                r'^[A-Z]{2,4}MFG[0-9A-Z]{2,8}$'
            ],
            'distributor': [
                r'^DIST[A-Z0-9]{4,12}$',
                r'^DISTR[A-Z0-9]{4,12}$',
                r'^DIS[A-Z0-9]{4,12}$'
            ],
            'regulator': [
                r'^REG[A-Z0-9]{4,12}$',
                r'^FDA[A-Z0-9]{4,12}$',
                r'^REGUL[A-Z0-9]{4,12}$',
                r'^[A-Z]{2,3}REG[0-9A-Z]{2,8}$'
            ],
            'expiry_date': [
                r'^\d{2}/\d{2}/\d{4}$',
                r'^\d{4}-\d{2}-\d{2}$',
                r'^EXP\d{6,8}$',
                r'^\d{2}\d{2}\d{4}$',
                r'^[0-9]{6,8}EXP$'
            ],
            'serial_number': [
                r'^SN[0-9A-Z]{6,16}$',
                r'^SERIAL[0-9A-Z]{6,16}$',
                r'^[0-9A-Z]{8,16}$',
                r'^S[0-9A-Z]{7,15}$'
            ]
        }
        
        # Initialize advanced processors
        self.setup_advanced_processors()
    
    def setup_advanced_processors(self):
        """Setup advanced image processing components."""
        
        # Create various kernels for morphological operations
        self.kernels = {
            'small_rect': cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            'medium_rect': cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            'large_rect': cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            'cross': cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
            'ellipse': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        }
        
        # Setup advanced filters
        self.setup_advanced_filters()
    
    def setup_advanced_filters(self):
        """Setup advanced filtering kernels."""
        
        # Sobel filters for edge detection
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Laplacian filter for edge enhancement
        self.laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        
        # Gaussian kernels for multi-scale processing
        self.gaussian_kernels = {
            'small': cv2.getGaussianKernel(5, 1),
            'medium': cv2.getGaussianKernel(9, 2),
            'large': cv2.getGaussianKernel(15, 3)
        }
    
    def decode_qr_from_region(self, image: np.ndarray, bbox: List[int], 
                             enhance: bool = True) -> Optional[str]:
        """Decode a QR from bbox using advanced multi-stage pipeline."""
        try:
            x_min, y_min, x_max, y_max = bbox
            
            # Extract region with adaptive padding
            padding = self.calculate_adaptive_padding(bbox)
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                return None
            
            # Convert to BGR for OpenCV operations
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            
            # Multi-stage advanced decoding pipeline
            decoding_strategies = [
                self.direct_opencv_decode,
                self.advanced_preprocessing_decode,
                self.multi_scale_decode,
                self.perspective_correction_decode,
                self.frequency_domain_decode,
                self.template_matching_decode,
                self.machine_learning_enhance_decode,
                self.extreme_recovery_decode
            ]
            
            for strategy in decoding_strategies:
                try:
                    result = strategy(roi_bgr)
                    if result and len(result.strip()) > 2:  # Minimum viable QR content
                        # Validate result
                        if self.validate_qr_content(result):
                            return result.strip()
                except Exception as e:
                    self.logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Advanced QR decoding failed for bbox {bbox}: {e}")
            return None
    
    def calculate_adaptive_padding(self, bbox: List[int]) -> int:
        """Calculate adaptive padding based on QR size."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        avg_size = (width + height) // 2
        
        if avg_size < 50:
            return 25
        elif avg_size < 100:
            return 20
        elif avg_size < 200:
            return 15
        else:
            return 10
    
    def direct_opencv_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Direct OpenCV decoding with multiple attempts."""
        detector = cv2.QRCodeDetector()
        
        # Try multiple color spaces
        color_variants = [
            roi_bgr,
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV),
        ]
        
        for variant in color_variants:
            try:
                if len(variant.shape) == 3:
                    variant_bgr = variant
                else:
                    variant_bgr = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
                
                # Try detectAndDecodeMulti
                retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(variant_bgr)
                if retval and decoded_info:
                    for info in decoded_info:
                        if info and len(info.strip()) > 0:
                            return info.strip()
                
                # Try simple detectAndDecode
                retval, decoded_info, points = detector.detectAndDecode(variant_bgr)
                if retval and decoded_info and len(decoded_info.strip()) > 0:
                    return decoded_info.strip()
                    
            except Exception:
                continue
        
        return None
    
    def advanced_preprocessing_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Advanced preprocessing with multiple enhancement techniques."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Advanced preprocessing pipeline
        preprocessing_methods = [
            # Noise reduction first
            lambda img: self.apply_non_local_means_denoising(img),
            lambda img: self.apply_bilateral_filtering(img),
            lambda img: self.apply_guided_filter(img),
            
            # Contrast enhancement
            lambda img: self.apply_adaptive_histogram_equalization(img),
            lambda img: self.apply_contrast_stretching(img),
            lambda img: self.apply_gamma_correction_variants(img),
            
            # Edge enhancement
            lambda img: self.apply_unsharp_masking(img),
            lambda img: self.apply_edge_preserving_smoothing(img),
            
            # Frequency domain enhancement
            lambda img: self.apply_frequency_enhancement(img),
            
            # Advanced thresholding
            lambda img: self.apply_multi_otsu_thresholding(img),
            lambda img: self.apply_local_adaptive_thresholding(img),
        ]
        
        for method in preprocessing_methods:
            try:
                enhanced = method(gray)
                if enhanced is not None:
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    result = self.direct_opencv_decode(enhanced_bgr)
                    if result:
                        return result
            except Exception:
                continue
        
        return None
    
    def apply_non_local_means_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        try:
            # Use scikit-image for better NLM denoising
            denoised = denoise_nl_means(image, patch_size=5, patch_distance=6, h=0.1)
            return (denoised * 255).astype(np.uint8)
        except:
            # Fallback to OpenCV
            return cv2.fastNlMeansDenoising(image)
    
    def apply_bilateral_filtering(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving smoothing."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def apply_guided_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply guided filter approximation via repeated bilateral filters."""
        # Simulate guided filter with bilateral filtering
        filtered = image.copy()
        for _ in range(3):
            filtered = cv2.bilateralFilter(filtered, 5, 50, 50)
        return filtered
    
    def apply_adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE with multiple configurations."""
        best_result = image
        
        clahe_configs = [
            (2.0, (8, 8)),
            (3.0, (8, 8)),
            (4.0, (16, 16)),
            (5.0, (8, 8)),
        ]
        
        for clip_limit, tile_size in clahe_configs:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                enhanced = clahe.apply(image)
                # Use the one with best contrast
                if np.std(enhanced) > np.std(best_result):
                    best_result = enhanced
            except:
                continue
        
        return best_result
    
    def apply_contrast_stretching(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast stretching."""
        p2, p98 = np.percentile(image, (2, 98))
        return np.clip((image - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)
    
    def apply_gamma_correction_variants(self, image: np.ndarray) -> np.ndarray:
        """Apply multiple gamma corrections and return best."""
        gamma_values = [0.3, 0.5, 0.7, 1.3, 1.7, 2.0]
        best_result = image
        best_std = np.std(image)
        
        for gamma in gamma_values:
            try:
                inv_gamma = 1.0 / gamma
                table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
                gamma_corrected = cv2.LUT(image, table)
                
                # Choose gamma that gives best contrast
                current_std = np.std(gamma_corrected)
                if current_std > best_std:
                    best_result = gamma_corrected
                    best_std = current_std
                    
            except:
                continue
        
        return best_result
    
    def apply_unsharp_masking(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for edge enhancement."""
        gaussian = cv2.GaussianBlur(image, (9, 9), 2.0)
        unsharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def apply_edge_preserving_smoothing(self, image: np.ndarray) -> np.ndarray:
        """Apply edge-preserving smoothing."""
        return cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)
    
    def apply_frequency_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply frequency domain enhancement."""
        try:
            # FFT-based enhancement
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            
            # Create high-pass filter
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            
            # High-pass filter mask
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0
            
            # Apply filter
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            # Combine with original
            enhanced = cv2.addWeighted(image.astype(np.float32), 0.7, 
                                     img_back.astype(np.float32), 0.3, 0)
            
            return np.clip(enhanced, 0, 255).astype(np.uint8)
            
        except:
            return image
    
    def apply_multi_otsu_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-level Otsu thresholding."""
        try:
            if SKIMAGE_AVAILABLE:
                from skimage.filters import threshold_multiotsu
                thresholds = threshold_multiotsu(image, classes=3)
                # Create binary image using middle threshold
                binary = image > thresholds[0]
                return (binary * 255).astype(np.uint8)
            else:
                # Fallback to regular Otsu
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
        except Exception:
            # Fallback to regular Otsu
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
    
    def apply_local_adaptive_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply local adaptive thresholding with multiple parameters."""
        block_sizes = [11, 15, 19, 25]
        C_values = [2, 5, 8, 11]
        
        best_result = None
        best_score = 0
        
        for block_size in block_sizes:
            for C in C_values:
                try:
                    adaptive = cv2.adaptiveThreshold(
                        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, block_size, C
                    )
                    
                    # Score based on number of connected components (simple heuristic)
                    num_labels, _ = cv2.connectedComponents(adaptive)
                    score = min(num_labels, 50)  # Cap to avoid noise
                    
                    if score > best_score:
                        best_result = adaptive
                        best_score = score
                        
                except:
                    continue
        
        return best_result if best_result is not None else image
    
    def multi_scale_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Multi-scale decoding with scaling and interpolation variants."""
        
        # Comprehensive scaling factors
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        interpolations = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]
        
        h, w = roi_bgr.shape[:2]
        
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Skip if too small or too large
            if new_h < 20 or new_w < 20 or new_h > 3000 or new_w > 3000:
                continue
            
            for interp in interpolations:
                try:
                    scaled = cv2.resize(roi_bgr, (new_w, new_h), interpolation=interp)
                    
                    # Try multiple rotation angles on scaled image
                    for angle in [0, 5, -5, 10, -10, 15, -15, 30, -30, 45, -45, 90, -90, 180]:
                        try:
                            if angle != 0:
                                center = (new_w // 2, new_h // 2)
                                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                                rotated = cv2.warpAffine(scaled, rotation_matrix, (new_w, new_h),
                                                       borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=(255, 255, 255))
                            else:
                                rotated = scaled
                            
                            result = self.direct_opencv_decode(rotated)
                            if result:
                                return result
                                
                        except Exception:
                            continue
                            
                except Exception:
                    continue
        
        return None
    
    def extreme_recovery_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Extreme recovery techniques for damaged QRs."""
        try:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # Multiple extreme enhancement techniques
            recovery_methods = [
                lambda img: cv2.equalizeHist(img),
                lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernels['small_rect']),
            ]
            
            detector = cv2.QRCodeDetector()
            
            for method in recovery_methods:
                try:
                    processed = method(gray)
                    if len(processed.shape) == 2:
                        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    else:
                        processed_bgr = processed
                    
                    retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(processed_bgr)
                    if retval and decoded_info:
                        for info in decoded_info:
                            if info and len(info.strip()) > 0:
                                return info.strip()
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Extreme recovery decode failed: {e}")
            return None
    
    def classify_qr_content(self, content: str) -> str:
        """Classify QR content based on patterns."""
        if not content:
            return "unknown"
        
        content_upper = content.upper().strip()
        
        # Check against classification patterns
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if re.match(pattern, content_upper):
                    return category
        
        # Default classification based on content characteristics
        if re.match(r'^[0-9A-Z]{8,16}$', content_upper):
            return "serial_number"
        elif 'BATCH' in content_upper or 'LOT' in content_upper:
            return "batch_number"
        elif 'EXP' in content_upper or re.search(r'\d{2}/\d{2}/\d{4}', content):
            return "expiry_date"
        elif len(content_upper) >= 6 and content_upper.isalnum():
            return "product_code"
        else:
            return "unknown"
    
    # Additional utility methods for completeness
    def validate_qr_content(self, content: str) -> bool:
        """Validate decoded content heuristically."""
        if not content or len(content.strip()) < 3:
            return False
        
        content_upper = content.upper().strip()
        
        # Valid if it matches any of our known patterns
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if re.match(pattern, content_upper):
                    return True
        
        # Valid if it contains alphanumeric characters
        if re.match(r'^[A-Za-z0-9]+$', content.strip()):
            return True
        
        # Valid if it's a reasonable length and contains meaningful characters
        if 4 <= len(content.strip()) <= 50 and content.strip().isalnum():
            return True
        
        return False
    
    def quick_qr_check(self, image_bgr: np.ndarray) -> bool:
        """Quick check if QR code might be decodable"""
        try:
            detector = cv2.QRCodeDetector()
            retval, _, _ = detector.detectAndDecode(image_bgr)
            return retval
        except Exception:
            return False
    
    def perspective_correction_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Advanced perspective correction and decoding."""
        
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        edge_methods = [
            lambda img: cv2.Canny(img, 50, 150),
            lambda img: cv2.Canny(img, 30, 100),
            lambda img: cv2.Canny(img, 100, 200),
        ]
        
        # Add scikit-image Canny if available
        if SKIMAGE_AVAILABLE:
            try:
                edge_methods.append(lambda img: (feature.canny(img, sigma=1.0) * 255).astype(np.uint8))
            except Exception:
                pass
        
        for edge_method in edge_methods:
            try:
                edges = edge_method(gray).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Look for quadrilaterals
                    if len(approx) == 4:
                        try:
                            # Sort corners
                            corners = approx.reshape(4, 2).astype(np.float32)
                            corners = self.sort_corners(corners)
                            
                            # Define target square sizes
                            target_sizes = [200, 300, 400, 500]
                            
                            for size in target_sizes:
                                target_corners = np.array([
                                    [0, 0], [size, 0], [size, size], [0, size]
                                ], dtype=np.float32)
                                
                                # Get perspective transformation
                                matrix = cv2.getPerspectiveTransform(corners, target_corners)
                                corrected = cv2.warpPerspective(roi_bgr, matrix, (size, size))
                                
                                result = self.direct_opencv_decode(corrected)
                                if result:
                                    return result
                                    
                        except Exception:
                            continue
                            
            except Exception:
                continue
        
        return None
    
    def sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners clockwise starting from top-left."""
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        return corners[sorted_indices]
    
    def frequency_domain_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Frequency domain enhancement and decoding."""
        
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        try:
            # Apply multiple frequency domain filters
            frequency_filters = [
                self.apply_high_pass_filter,
                self.apply_low_pass_filter,
                self.apply_band_pass_filter,
                self.apply_notch_filter,
            ]
            
            for filter_func in frequency_filters:
                try:
                    filtered = filter_func(gray)
                    filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
                    
                    result = self.direct_opencv_decode(filtered_bgr)
                    if result:
                        return result
                        
                except Exception:
                    continue
                    
        except Exception:
            pass
        
        return None
    
    def apply_high_pass_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply high-pass filter in frequency domain."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create high-pass filter
        mask = np.ones((rows, cols), np.uint8)
        r = min(rows, cols) // 8
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0
        
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return np.clip(img_back, 0, 255).astype(np.uint8)
    
    def apply_low_pass_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply low-pass filter in frequency domain."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create low-pass filter
        mask = np.zeros((rows, cols), np.uint8)
        r = min(rows, cols) // 4
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return np.clip(img_back, 0, 255).astype(np.uint8)
    
    def apply_band_pass_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply band-pass filter in frequency domain."""
        # Combine high-pass and low-pass
        high_pass = self.apply_high_pass_filter(image)
        low_pass = self.apply_low_pass_filter(image)
        
        # Band-pass is the difference
        band_pass = cv2.absdiff(low_pass, high_pass)
        return band_pass
    
    def apply_notch_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove specific frequencies."""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create notch filter (remove center frequencies)
        mask = np.ones((rows, cols), np.uint8)
        r1, r2 = min(rows, cols) // 16, min(rows, cols) // 8
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        
        mask_area1 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r1*r1
        mask_area2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r2*r2
        
        mask[mask_area2] = 0
        mask[mask_area1] = 1
        
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        return np.clip(img_back, 0, 255).astype(np.uint8)
    
    def template_matching_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """Template matching and pattern-based enhancement."""
        
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # QR code has specific patterns - try to enhance them
        # This is a simplified approach - real QR template matching would be more complex
        
        # Create QR pattern templates (finder patterns)
        templates = self.create_qr_templates()
        
        for template in templates:
            try:
                # Match template
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= 0.3)
                
                if len(locations[0]) > 0:
                    # Enhance regions around matches
                    enhanced = self.enhance_around_matches(gray, locations, template.shape)
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    
                    decode_result = self.direct_opencv_decode(enhanced_bgr)
                    if decode_result:
                        return decode_result
                        
            except Exception:
                continue
        
        return None
    
    def create_qr_templates(self) -> List[np.ndarray]:
        """Create QR finder pattern templates."""
        templates = []
        
        # Create basic finder pattern template
        sizes = [7, 9, 11, 13, 15]
        
        for size in sizes:
            template = np.ones((size, size), dtype=np.uint8) * 255
            
            # Create finder pattern (7x7 with specific pattern)
            border = size // 7
            if border > 0:
                template[border:-border, border:-border] = 0
                center = size // 2
                template[center-border:center+border+1, center-border:center+border+1] = 255
            
            templates.append(template)
        
        return templates
    
    def enhance_around_matches(self, image: np.ndarray, locations: Tuple, template_shape: Tuple) -> np.ndarray:
        """Enhance image around template matches."""
        enhanced = image.copy()
        h, w = template_shape
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            roi = enhanced[y:y+h, x:x+w]
            if roi.size > 0:
                # Apply local enhancement
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced_roi = clahe.apply(roi)
                enhanced[y:y+h, x:x+w] = enhanced_roi
        
        return enhanced
    
    def machine_learning_enhance_decode(self, roi_bgr: np.ndarray) -> Optional[str]:
        """ML-inspired enhancement (no external models)."""
        
        # This would typically use trained models, but we'll use advanced heuristics
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Feature-based enhancement
        enhanced_variants = [
            self.enhance_based_on_texture_analysis(gray),
            self.enhance_based_on_gradient_analysis(gray),
            self.enhance_based_on_symmetry_analysis(gray),
        ]
        
        for variant in enhanced_variants:
            if variant is not None:
                try:
                    variant_bgr = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
                    result = self.direct_opencv_decode(variant_bgr)
                    if result:
                        return result
                except Exception:
                    continue
        
        return None