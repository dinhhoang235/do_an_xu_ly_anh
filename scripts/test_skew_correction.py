"""
Test Skew Correction - Visualize trước/sau hiệu chỉnh
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.skew_corrector import SkewCorrector
from src.plate_detector import PlateDetector
from src.preprocessor import Preprocessor


def visualize_skew_correction(image_path):
    """Visualize hiệu chỉnh góc trước/sau"""
    
    # Load components
    preprocessor = Preprocessor()
    plate_detector = PlateDetector()
    skew_corrector = SkewCorrector()
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc: {image_path}")
        return
    
    print(f"🔍 Xử lý: {Path(image_path).name}")
    print("="*60)
    
    # Step 1: Preprocess
    preprocessed = preprocessor.preprocess(image)
    
    # Step 2: Detect plates
    plates = plate_detector.detect_plates(preprocessed)
    print(f"🎯 Phát hiện: {len(plates)} biển số\n")
    
    if len(plates) == 0:
        print("⚠️  Không phát hiện biển số")
        return
    
    # Vẽ bounding box lên ảnh
    result_image = image.copy()
    
    for plate_idx, (x, y, w, h) in enumerate(plates, 1):
        # Crop plate
        plate_roi = image[y:y+h, x:x+w]
        
        print(f"📍 Biển số #{plate_idx}: ({x}, {y}) - {w}x{h}")
        
        # Skew correction
        corrected_roi, angle = skew_corrector.correct_skew(plate_roi)
        
        print(f"   Góc hiệu chỉnh: {angle:.1f}°")
        print(f"   Trạng thái: {'🔄 Đã xoay' if abs(angle) > 0.5 else '✅ Đã thẳng'}")
        
        # Visualize
        fig = np.hstack([
            cv2.copyMakeBorder(plate_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 255)),
            cv2.copyMakeBorder(corrected_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        ])
        
        cv2.putText(fig, "TRUOC", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(fig, "SAU", (w+20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(fig, f"Goc: {angle:.1f}", (10, h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow(f"Plate #{plate_idx} - Skew Correction", fig)
        cv2.waitKey(0)
        
        # Draw bounding box on result
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, f"#{plate_idx}: {angle:.1f}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow("Plate Detection Results", result_image)
    print("\n⌨️  Nhấn phím để tiếp tục...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test trên một số ảnh
    test_images = [
        "datasets/kaggle_foreign/test/Cars1.png",
        "datasets/kaggle_foreign/test/Cars4.png",
        "datasets/kaggle_foreign/test/Cars10.png",
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            visualize_skew_correction(img_path)
            print()
        else:
            print(f"⚠️  Không tìm thấy: {img_path}")
