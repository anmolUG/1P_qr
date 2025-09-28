#!/usr/bin/env python3
"""
QR Detection Evaluation and Visualization Script
Visualizes detection results with bounding boxes for verification
"""

import cv2
import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from tqdm import tqdm

def visualize_detections(image_dir, results_file, output_dir, max_images=10):
    """
    Visualize QR detection results with bounding boxes
    
    Args:
        image_dir: Directory containing test images
        results_file: JSON file with detection results
        output_dir: Directory to save visualization images
        max_images: Maximum number of images to visualize
    """
    # Load detection results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process results (limit to max_images)
    results_to_process = results[:max_images] if max_images else results
    
    print(f"🎨 Visualizing {len(results_to_process)} detection results...")
    
    for result in tqdm(results_to_process, desc="Creating visualizations"):
        image_id = result['image_id']
        qrs = result['qrs']
        
        # Find corresponding image file
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None
        for ext in image_extensions:
            potential_path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"⚠️ Image not found for {image_id}")
            continue
        
        # Load and display image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Draw bounding boxes
        for i, qr in enumerate(qrs):
            bbox = qr['bbox']
            x_min, y_min, x_max, y_max = bbox
            
            # Create rectangle
            rect = patches.Rectangle(
                (x_min, y_min), 
                x_max - x_min, 
                y_max - y_min,
                linewidth=2, 
                edgecolor='red', 
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add QR index label
            ax.text(x_min, y_min - 5, f'QR{i+1}', 
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add decoded value if available
            if 'value' in qr and qr['value']:
                value_text = qr['value'][:20] + "..." if len(qr['value']) > 20 else qr['value']
                ax.text(x_min, y_max + 15, value_text, 
                       color='blue', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Set title and remove axes
        title = f"{image_id} - {len(qrs)} QR code(s) detected"
        if qrs and 'value' in qrs[0]:
            decoded_count = sum(1 for qr in qrs if qr.get('value', ''))
            title += f" ({decoded_count} decoded)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{image_id}_detection.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")
    print(f"📁 Generated {len(results_to_process)} visualization files")

def analyze_results(results_file):
    """
    Analyze detection results and print statistics
    
    Args:
        results_file: JSON file with detection results
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_images = len(results)
    total_qrs = sum(len(result['qrs']) for result in results)
    images_with_qrs = sum(1 for result in results if len(result['qrs']) > 0)
    
    print(f"\n📊 DETECTION ANALYSIS:")
    print(f"   📸 Total images: {total_images}")
    print(f"   📱 Total QR codes: {total_qrs}")
    print(f"   ✅ Images with QRs: {images_with_qrs}")
    print(f"   📈 Average QRs per image: {total_qrs/total_images:.2f}")
    print(f"   🎯 Detection rate: {images_with_qrs/total_images*100:.1f}%")
    
    # Check if decoding results are available
    has_decoding = any('value' in qr for result in results for qr in result['qrs'])
    if has_decoding:
        decoded_qrs = sum(1 for result in results for qr in result['qrs'] if qr.get('value', ''))
        print(f"   🔤 Decoded QRs: {decoded_qrs}")
        print(f"   📊 Decoding success: {decoded_qrs/total_qrs*100:.1f}%")
    
    # QR distribution
    qr_counts = [len(result['qrs']) for result in results]
    print(f"   📊 QR distribution: min={min(qr_counts)}, max={max(qr_counts)}, avg={np.mean(qr_counts):.1f}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="QR Detection Evaluation and Visualization")
    parser.add_argument("--images", "-i", required=True,
                       help="Directory containing test images")
    parser.add_argument("--results", "-r", required=True,
                       help="JSON file with detection results")
    parser.add_argument("--output", "-o", default="visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--max-images", "-m", type=int, default=10,
                       help="Maximum number of images to visualize (0 = all)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze results without creating visualizations")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.images):
        print(f"❌ Image directory not found: {args.images}")
        return
    
    if not os.path.exists(args.results):
        print(f"❌ Results file not found: {args.results}")
        return
    
    print(f"🔍 QR Detection Evaluation")
    print(f"📂 Images: {args.images}")
    print(f"📋 Results: {args.results}")
    
    # Analyze results
    analyze_results(args.results)
    
    # Create visualizations (unless analyze-only)
    if not args.analyze_only:
        print(f"\n🎨 Creating visualizations...")
        visualize_detections(
            image_dir=args.images,
            results_file=args.results,
            output_dir=args.output,
            max_images=args.max_images if args.max_images > 0 else None
        )

if __name__ == "__main__":
    main()