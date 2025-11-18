"""
Debug segmentation chi tiết - xem từng ảnh bị lỗi như thế nào
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer

def debug_segmentation_detailed():
    """Debug segmentation chi tiết"""
    print("=" * 70)
    print("🔍 DETAILED SEGMENTATION DEBUG")
    print("=" * 70)
    
    # Load test set
    test_csv = "datasets/kaggle_foreign/test_annotations.csv"
    df = pd.read_csv(test_csv)
    
    recognizer = CharacterRecognizer()
    
    print(f"\n{'Image':<20} {'Ground Truth':<15} {'Detected':<15} {'Match?':<10}")
    print("-" * 60)
    
    correct = 0
    
    for idx, row in df.iterrows():
        image_name = row['filename']
        ground_truth = row['plate_text']
        
        img_path = f"datasets/kaggle_foreign/test/{image_name}"
        if not Path(img_path).exists():
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        try:
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            
            plate_img = img[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # Segment characters
            char_images = recognizer.segment_characters(plate_img)
            detected_count = len(char_images)
            
            match = "✅" if detected_count == len(ground_truth) else "❌"
            if detected_count == len(ground_truth):
                correct += 1
            
            print(f"{image_name:<20} {ground_truth:<15} {detected_count:<15} {match:<10}")
        
        except Exception as e:
            print(f"{image_name:<20} {ground_truth:<15} ERROR: {str(e)[:20]:<15}")
    
    print("-" * 60)
    print(f"Correct character count: {correct}/17 = {correct/17*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    debug_segmentation_detailed()
