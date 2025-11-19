"""
Full Pipeline: Plate Detection → Character Segmentation → Recognition
"""

import cv2
import sys
import os
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.plate_detector import PlateDetector
from src.character_recognizer import CharacterRecognizer
from src.preprocessor import Preprocessor
from src.skew_corrector import SkewCorrector
import pickle

class LicensePlateRecognitionPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.plate_detector = PlateDetector()
        self.char_recognizer = CharacterRecognizer()
        self.skew_corrector = SkewCorrector()
        
        # Load hybrid model
        hybrid_model_path = Path(__file__).parent.parent / "models" / "knn_character_recognizer_hybrid.pkl"
        if hybrid_model_path.exists():
            with open(hybrid_model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data.get('model', data)
            print("✅ Hybrid KNN model loaded")
        else:
            print(f"❌ Model not found: {hybrid_model_path}")
            self.model = None
    
    def extract_features(self, char_img):
        """Extract features từ character image"""
        resized = cv2.resize(char_img, (20, 30))
        _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
        features = binary.flatten().astype(np.float32) / 255.0
        features = np.clip(features, 0, 1)
        return features
    
    def process_image(self, image_path, visualize=True):
        """
        Pipeline hoàn chỉnh từng bước:
        1. Tiền xử lý
        2. Phát hiện biển số
        3. Hiệu chỉnh góc nghiêng
        4. Segment ký tự
        5. Nhận dạng ký tự
        """
        print(f"\n🔍 Xử lý: {Path(image_path).name}")
        print("="*60)
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Không thể đọc ảnh")
            return None
        
        h, w = image.shape[:2]
        print(f"📏 Kích thước ảnh: {w}x{h}")
        
        # STEP 1: Tiền xử lý ảnh
        print("\n[Step 1] Tiền xử lý ảnh...")
        preprocessed = self.preprocessor.preprocess(image)
        print("  ✅ Đã preprocess (grayscale, blur)")
        
        # STEP 2: Phát hiện biển số
        print("\n[Step 2] Phát hiện biển số...")
        plates = self.plate_detector.detect_plates(preprocessed)
        print(f"  🎯 Phát hiện: {len(plates)} biển số")
        
        if len(plates) == 0:
            print("  ⚠️  Không phát hiện biển số nào")
            return None
        
        results = []
        
        # STEP 3-5: Với mỗi biển số
        print("\n[Step 3-5] Hiệu chỉnh góc - Segment ký tự - Nhận dạng...")
        for plate_idx, (x, y, w_plate, h_plate) in enumerate(plates, 1):
            # Crop biển số từ ảnh gốc
            plate_roi = image[y:y+h_plate, x:x+w_plate]
            
            print(f"\n  Biển số #{plate_idx}: ({x}, {y}) - {w_plate}x{h_plate}")
            
            # Step 3: Hiệu chỉnh góc nghiêng
            corrected_roi, skew_angle = self.skew_corrector.correct_skew(plate_roi)
            if abs(skew_angle) > 0.5:
                print(f"    🔄 Hiệu chỉnh góc: {skew_angle:.1f}°")
                plate_roi = corrected_roi
            else:
                print(f"    ✅ Góc đã thẳng ({skew_angle:.1f}°)")
            
            # Step 4: Segment ký tự
            char_images = self.char_recognizer.segment_characters(plate_roi)
            
            if len(char_images) == 0:
                print(f"    ⚠️  Không segment được ký tự")
                continue
            
            print(f"    📋 Segment: {len(char_images)} ký tự")
            
            # Nhận dạng
            if self.model is not None:
                features_list = [self.extract_features(char) for char in char_images]
                features_array = np.array(features_list)
                predictions = self.model.predict(features_array)
                plate_text = ''.join(predictions)
                print(f"    ✅ Kết quả: {plate_text}")
            else:
                plate_text = "N/A"
                print(f"    ⚠️  Model không sẵn sàng")
            
            results.append({
                'position': (x, y, w_plate, h_plate),
                'text': plate_text,
                'char_count': len(char_images)
            })
        
        # Visualize
        if visualize:
            result_image = image.copy()
            
            # Vẽ biển số bounding boxes
            for plate_idx, (x, y, w_p, h_p) in enumerate(plates, 1):
                cv2.rectangle(result_image, (x, y), (x + w_p, y + h_p), (0, 255, 0), 2)
                
                # Thêm text nhận dạng nếu có
                if plate_idx <= len(results):
                    plate_text = results[plate_idx - 1]['text']
                    cv2.putText(result_image, plate_text, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(f"Pipeline Result - {Path(image_path).name}", result_image)
            print("\n⏳ Nhấn phím bất kỳ để tiếp tục...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print("="*60)
        return results
    
    def process_batch(self, image_dir, num_images=None):
        """Process nhiều ảnh"""
        image_paths = sorted(Path(image_dir).glob("*.png"))
        if num_images:
            image_paths = image_paths[:num_images]
        
        print(f"\n🚀 PIPELINE HOÀN CHỈNH")
        print(f"📁 Thư mục: {image_dir}")
        print(f"📷 Số ảnh: {len(image_paths)}\n")
        
        all_results = []
        
        for image_path in image_paths:
            results = self.process_image(str(image_path), visualize=True)
            if results:
                all_results.append({
                    'image': image_path.name,
                    'plates': results
                })
        
        # Thống kê
        print(f"\n{'='*60}")
        print("📊 THỐNG KÊ")
        print(f"{'='*60}")
        print(f"Tổng ảnh xử lý: {len(image_paths)}")
        print(f"Ảnh phát hiện biển số: {len(all_results)}")
        
        if all_results:
            total_plates = sum(len(r['plates']) for r in all_results)
            total_chars = sum(sum(p['char_count'] for p in r['plates']) for r in all_results)
            print(f"Tổng biển số: {total_plates}")
            print(f"Tổng ký tự nhận dạng: {total_chars}")
        
        return all_results

if __name__ == "__main__":
    pipeline = LicensePlateRecognitionPipeline()
    
    # Test folder với ảnh có gán nhãn
    image_dir = Path(__file__).parent.parent / "datasets" / "kaggle_foreign" / "images"
    
    results = pipeline.process_batch(str(image_dir), num_images=10)
    
    print("\n✅ Hoàn thành!")
