#!/usr/bin/env python3
"""
Complete pipeline script to run detection and visualization in one command
"""

import subprocess
import sys
import os
from pathlib import Path

def run_detection():
    """Run the main detection pipeline"""
    print("🚀 Running QR Detection Pipeline...")
    cmd = [
        "python", "infer.py",
        "--input", "data/test_images",
        "--output", "outputs/final_submission.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Detection completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_visualization():
    """Run the visualization pipeline"""
    print("🎨 Generating Visualizations...")
    cmd = [
        "python", "visualize_detections.py",
        "--input", "data/test_images",
        "--json", "outputs/final_submission.json",
        "--output", "outputs/final_visualizations"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Visualizations generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def count_detections():
    """Count the number of detected QR codes"""
    try:
        import json
        with open("outputs/final_submission.json", "r") as f:
            data = json.load(f)
        
        total = sum(len(item['qrs']) for item in data)
        print(f"📊 Total QR codes detected: {total}")
        return total
    except Exception as e:
        print(f"❌ Failed to count detections: {e}")
        return 0

def main():
    """Run the complete pipeline"""
    print("=" * 60)
    print("QR CODE DETECTION AND VISUALIZATION COMPLETE PIPELINE")
    print("=" * 60)
    
    # Run detection
    if not run_detection():
        sys.exit(1)
    
    # Count detections
    count_detections()
    
    # Run visualization
    if not run_visualization():
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
    print("=" * 60)
    print("📁 Output files generated:")
    print("   - outputs/final_submission.json (detection results)")
    print("   - outputs/final_visualizations/ (visualization images)")
    print("\n💡 Next steps:")
    print("   - Check outputs/final_submission.json for detection results")
    print("   - View visualizations in outputs/final_visualizations/")
    print("=" * 60)

if __name__ == "__main__":
    main()