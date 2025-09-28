# QR Code Detection and Decoding System

## Overview
This system provides robust QR code detection and decoding capabilities for test images. It uses a trained YOLOv8 model with enhanced detection algorithms to maximize detection accuracy while minimizing false positives.

## System Requirements
- Python 3.7 or higher
- Required Python packages (see requirements.txt)
- CUDA-compatible GPU (optional, for faster processing)

## Directory Structure
```
project/
├── data/
│   └── test_images/          # Place all test images here
├── models/
├── outputs/
├── runs/
│   └── detect/
│       └── qr_clean_training2/
│           └── weights/
│               └── best.pt   # Trained model file
├── src/
├── infer.py                  # Main inference script
├── visualize_detections.py   # Visualization script
└── requirements.txt
```

## Setup Instructions

### 1. Environment Setup
```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Model Preparation
The system requires a trained model file at:
`runs/detect/qr_clean_training2/weights/best.pt`

Ensure this file exists in the specified path. If the file is missing, please obtain it from the development team.

## Usage Instructions

### 1. Prepare Test Images
Place all test images in the `data/test_images/` directory. Supported formats include:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### 2. Run Detection Only (without decoding)
If you only need detection without decoding, use this command:

```bash
python infer.py --input data/test_images --output outputs/detection_results.json
```

### 3. Run Detection and Decoding
Execute the following command to process all images and generate results with the proven QR detection system:

```bash
python infer.py --input data/test_images --output outputs/submission_decoding_2.json --decode
```

Parameters:
- `--input`: Path to directory containing test images
- `--output`: Path to output JSON file with detection results
- `--decode`: Enable QR code decoding (required for the proven QR detection system)

This command uses the proven system that achieves QR detections with high accuracy and minimal false positives.

### 4. Visualize Detections
To generate visualizations of the detected QR codes, use the following command:

```bash
python visualize_detections.py --input data/test_images --json outputs/submission_decoding_2.json --output outputs/visualizations
```

Parameters:
- `--input`: Path to directory containing test images
- `--json`: Path to the JSON file with detection results
- `--output`: Path to output directory for visualization images

This will generate visualization images with bounding boxes drawn around detected QR codes.

## Output Format
The system generates a JSON file with the following structure:
```json
[
  {
    "image_id": "image_filename_without_extension",
    "qrs": [
      {
        "bbox": [x_min, y_min, x_max, y_max],
        "value": "decoded_qr_content"
      }
    ]
  }
]
```

Each entry contains:
- `image_id`: The filename of the processed image (without extension)
- `qrs`: Array of detected QR codes
- `bbox`: Bounding box coordinates [x_min, y_min, x_max, y_max]
- `value`: Decoded content of the QR code (empty string if decoding failed)

## Expected Results
- Total expected detections: 184 for 50 test images using the proven system
- Processing time: Approximately 6-8 seconds for all 50 images
- Average QR codes per image: 3-4
- Detection accuracy: High accuracy with minimal false positives

## Troubleshooting

### Common Issues
1. **Model not found**: Ensure the trained model exists at `runs/detect/qr_clean_training2/weights/best.pt`
2. **No images processed**: Verify test images are in the correct directory and have supported file extensions
3. **Memory errors**: Process images in smaller batches or reduce image resolution
4. **Import errors**: Ensure all required packages are installed via `pip install -r requirements.txt`

### Performance Optimization
- For faster processing, use a CUDA-compatible GPU
- For memory-constrained environments, process images in smaller batches
- Ensure sufficient disk space for output files (approximately 10MB for output JSON)

## Technical Details

### Detection Algorithm
The proven system uses a multi-stage detection approach:
1. Primary detection with confidence threshold 0.3
2. Secondary detection with lower confidence thresholds for edge cases
3. IoU-based duplicate removal with threshold 0.3
4. Geometric validation to minimize false positives

### Decoding Enhancement
The decoding pipeline includes:
- Multiple preprocessing techniques for damaged QR codes
- Adaptive thresholding for varying lighting conditions
- Error correction for partially obscured codes
- Validation to ensure decoded content integrity

## System Validation
The system has been validated on a test dataset with the following results:
- Detection rate: 184 QR codes detected in 50 test images
- Accuracy: High precision with minimal false positives
- Robustness: Effective detection under various lighting conditions
- Reliability: Consistent performance across different image types

## Contact Information
For technical support or questions about the system, please contact the development team.

## Version Information
- System version: 1.0
- Model version: qr_clean_training2
- Documentation last updated: 2025-09-29