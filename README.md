# Multi-QR Code Recognition System for Medicine Packs

## Overview
This system detects and decodes multiple QR codes on medicine pack images. It uses a trained YOLOv8 model to identify QR codes and advanced decoding algorithms to extract their contents. The system is designed to work with tilted, blurred, or partially covered images commonly found in medical packaging.

## System Requirements
- Python 3.7 or higher
- Required Python packages (see [requirements.txt](file:///C:/Users/Dell/Desktop/1P/requirements.txt))
- CUDA-compatible GPU (optional, for faster processing)

## Directory Structure
```
project/
├── data/
│   └── test_images/          # Place all test images here
├── outputs/
│   ├── submission_detection_1.json    # Detection results
│   └── submission_decoding_2.json     # Decoding results
├── src/                      # Source code
├── infer.py                  # Main inference script
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── visualize_detections.py   # Visualization script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start Guide

### Step 1: Environment Setup
1. Install Python 3.7 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Test Images
1. Download the dataset from the provided link
2. Extract the test images (50 images)
3. Place all test images in the `data/test_images/` directory

Supported image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Step 3: Run QR Code Detection
Execute the following command to detect QR codes in your test images:
```bash
python infer.py --input data/test_images --output outputs/submission_detection_1.json
```

This will:
- Process all images in the `data/test_images/` directory
- Generate bounding box coordinates for each detected QR code
- Save results to `outputs/submission_detection_1.json`

### Step 4: Run QR Code Detection and Decoding
To additionally decode the content of detected QR codes, use the `--decode` flag:
```bash
python infer.py --input data/test_images --output outputs/submission_decoding_2.json --decode
```

This will:
- Detect all QR codes as in Step 3
- Decode the content of each detected QR code
- Save results to `outputs/submission_decoding_2.json`

### Step 5: Visualize Results (Optional)
To generate visual representations of the detection results:
```bash
python visualize_detections.py --input data/test_images --json outputs/submission_detection_1.json --output outputs/visualizations
```

This will:
- Create visualization images showing detected QR codes with bounding boxes
- Save visualization images to the `outputs/visualizations/` directory

## Detailed Usage Instructions

### Detection Only
Command:
```bash
python infer.py --input data/test_images --output outputs/submission_detection_1.json
```

Parameters:
- `--input`: Path to directory containing test images
- `--output`: Path to output JSON file with detection results

Expected output format:
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

### Detection and Decoding
Command:
```bash
python infer.py --input data/test_images --output outputs/submission_decoding_2.json --decode
```

Parameters:
- `--input`: Path to directory containing test images
- `--output`: Path to output JSON file with detection and decoding results
- `--decode`: Enable QR code decoding

Expected output format:
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max], "value": "B12345"},
      {"bbox": [x_min, y_min, x_max, y_max], "value": "MFR56789"}
    ]
  }
]
```

### Visualization
Command:
```bash
python improved_visualization.py --input data/test_images --detection outputs/submission_detection_1.json --decoding outputs/submission_decoding_2.json --output outputs/visualizations
```

Parameters:
- `--input`: Path to directory containing test images
- `--detection`: Path to the JSON file with detection results
- `--decoding`: Path to the JSON file with decoding results
- `--output`: Path to output directory for visualization images

Visualization features:
- Extremely thick borders (10px) for better visibility
- Larger font sizes for all text elements
- No bounding box coordinates for cleaner look
- Detection results shown in blue, decoding results in red
- Enhanced legend and summary information


## Output Files

### submission_detection_1.json
Contains bounding box coordinates for all detected QR codes:
- Format: JSON array with image IDs and QR bounding boxes
- Location: `outputs/submission_detection_1.json`
- Size: ~25KB

### submission_decoding_2.json
Contains both bounding box coordinates and decoded QR values:
- Format: JSON array with image IDs, QR bounding boxes, and decoded values
- Location: `outputs/submission_decoding_2.json`
- Size: ~30KB

## Performance Metrics

### Detection Performance
- Total QR codes detected: 184
- Images processed: 50
- Average QR codes per image: 3.68
- Processing time: ~6-8 seconds

### System Capabilities
- Works with tilted, blurred, or partially covered images
- Handles multiple QR codes per image
- Robust detection under various lighting conditions
- High accuracy with minimal false positives

## Training Your Own Model (Advanced)

To train the model on your own dataset:
```bash
python train.py --data_dir path/to/your/data --epochs 100
```

Parameters:
- `--data_dir`: Directory containing training data
- `--epochs`: Number of training epochs

## Evaluation and Analysis

To evaluate and analyze your results:
```bash
python evaluate.py --images data/test_images --results outputs/submission_decoding_2.json --output evaluation_results
```

Parameters:
- `--images`: Directory containing test images
- `--results`: JSON file with detection results
- `--output`: Directory for evaluation results

## Troubleshooting

### Common Issues
1. **No images processed**: Verify test images are in the correct directory and have supported file extensions
2. **Import errors**: Ensure all required packages are installed via `pip install -r requirements.txt`
3. **Memory errors**: Process images in smaller batches or reduce image resolution
4. **Model not found**: Ensure the trained model file is in the correct location

### Support
For technical support or questions about the system, please contact the development team.

## Version Information
- System version: 1.0
- Documentation last updated: 2025-10-01