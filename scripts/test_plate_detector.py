"""
Test script để kiểm tra PlateDetector trên ảnh từ Kaggle dataset
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np

# Thêm path để import src
sys.path.append(str(Path(__file__).parent.parent))

from src.plate_detector import PlateDetector

def test_single_image(image_path, detector):
    """Test phát hiện biển số trên một ảnh"""
    print(f"\n{'='*60}")
    print(f"🔍 Kiểm tra ảnh: {os.path.basename(image_path)}")
    print('='*60)
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None
    
    print(f"📏 Kích thước ảnh: {image.shape}")
    
    # Phát hiện biển số
    plates = detector.detect_plates(image)
    
    print(f"🎯 Phát hiện được {len(plates)} biển số")
    
    if len(plates) > 0:
        for idx, (x, y, w, h) in enumerate(plates, 1):
            print(f"   Biển số #{idx}: vị trí ({x}, {y}), kích thước {w}x{h}")
            print(f"              tỷ lệ: {w/h:.2f}")
    
    # Visualize kết quả
    result_image = detector.visualize_detection(image, plates)
    
    # Hiển thị ảnh
    cv2.imshow(f"PlateDetector - {os.path.basename(image_path)}", result_image)
    print("⏳ Nhấn phím bất kỳ để xem ảnh tiếp theo...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return plates

def test_batch_images(image_dir, num_images=10):
    """Test phát hiện trên nhiều ảnh"""
    detector = PlateDetector()
    
    # Lấy danh sách ảnh
    image_paths = sorted(Path(image_dir).glob("*.png"))[:num_images]
    
    if not image_paths:
        print(f"❌ Không tìm thấy ảnh trong: {image_dir}")
        return
    
    print(f"\n🚀 Bắt đầu test PlateDetector")
    print(f"📁 Thư mục: {image_dir}")
    print(f"📷 Số ảnh test: {len(image_paths)}")
    
    results = []
    for image_path in image_paths:
        plates = test_single_image(str(image_path), detector)
        results.append({
            'image': image_path.name,
            'num_plates': len(plates) if plates else 0,
            'plates': plates
        })
    
    # Thống kê
    print(f"\n{'='*60}")
    print("📊 THỐNG KÊ KẾT QUẢ")
    print('='*60)
    total_plates = sum(r['num_plates'] for r in results)
    print(f"Tổng ảnh test: {len(results)}")
    print(f"Tổng biển số phát hiện: {total_plates}")
    print(f"Trung bình/ảnh: {total_plates/len(results):.2f}")
    
    # Chi tiết từng ảnh
    print("\n📋 Chi tiết:")
    for r in results:
        status = "✅" if r['num_plates'] > 0 else "⚠️"
        print(f"{status} {r['image']:<20} → {r['num_plates']} biển số")
    
    print(f"\n💾 Hoàn thành test {len(results)} ảnh")

if __name__ == "__main__":
    # Đường dẫn dataset
    kaggle_images_dir = Path(__file__).parent.parent / "datasets" / "kaggle_foreign" / "images"
    
    if not kaggle_images_dir.exists():
        print(f"❌ Không tìm thấy thư mục: {kaggle_images_dir}")
        sys.exit(1)
    
    # Test 10 ảnh đầu tiên
    test_batch_images(str(kaggle_images_dir), num_images=10)
