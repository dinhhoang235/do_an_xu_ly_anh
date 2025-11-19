"""
License Plate Recognition System - Using Hybrid KNN Model
Best performance: 57.81% accuracy on foreign plates
"""

import cv2
import os
import sys
import argparse
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Thêm path để import các module
sys.path.append(str(Path(__file__).parent / "src"))

from src.character_recognizer import CharacterRecognizer
from src.preprocessor import Preprocessor
from src.plate_detector import PlateDetector
from src.skew_corrector import SkewCorrector

class LicensePlateSystem:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.char_recognizer = CharacterRecognizer()
        self.plate_detector = PlateDetector()
        self.skew_corrector = SkewCorrector()
        
        # Khởi tạo hệ thống
        self._initialize_system()
    
    def _initialize_system(self):
        """Khởi tạo hệ thống với hybrid model"""
        print("🚗 Đang khởi tạo hệ thống nhận dạng biển số xe...")
        
        # Load hybrid model (best performance)
        hybrid_model_path = "models/knn_character_recognizer_hybrid.pkl"
        if os.path.exists(hybrid_model_path):
            print("🤖 Đang tải Hybrid KNN model...")
            try:
                with open(hybrid_model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model', data)
                self.recognition_method = 'hybrid'
                print("✅ Đã tải Hybrid model thành công! (57.81% accuracy)")
            except Exception as e:
                print(f"❌ Lỗi tải model: {e}")
                sys.exit(1)
        else:
            print(f"❌ Không tìm thấy model tại: {hybrid_model_path}")
            print("💡 Vui lòng chạy: python scripts/train_knn_hybrid.py")
            sys.exit(1)
        
        print("✅ Hệ thống đã sẵn sàng!")
    
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
        print(f"🔍 Đang xử lý ảnh: {image_path}")
        print("="*60)
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return None
        
        start_time = time.time()
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
        
        processing_time = time.time() - start_time
        
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
            
            cv2.imshow(f"License Plate Recognition - {Path(image_path).name}", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"\n⏱️  Thời gian xử lý: {processing_time:.3f}s")
        print("="*60)
        
        return results
    
    def process_video(self, video_path, output_path=None):
        """
        Xử lý video đầu vào
        """
        print(f"🎥 Đang xử lý video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Thiết lập output video nếu có
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        results_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Xử lý mỗi 3 frame để tăng tốc độ
            if frame_count % 3 != 0:
                continue
            
            start_time = time.time()
            
            # Segment và recognize
            char_images = self.char_recognizer.segment_characters(frame)
            
            plate_text = ""
            if len(char_images) > 0:
                features_list = [self.extract_features(char) for char in char_images]
                features_array = np.array(features_list)
                predictions = self.model.predict(features_array)
                plate_text = ''.join(predictions)
                results_list.append({'frame': frame_count, 'plate': plate_text})
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Vẽ kết quả lên frame
            if plate_text:
                cv2.rectangle(frame, (10, 10), (300, 50), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate: {plate_text}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Hiển thị FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hiển thị frame
            cv2.imshow('License Plate Recognition', frame)
            
            # Ghi video output
            if output_path:
                out.write(frame)
            
            # Thoát nếu nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Tính toán thống kê
        if processing_times:
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
            print(f"\n📊 THỐNG KÊ VIDEO:")
            print(f"   - Tổng frame: {frame_count}")
            print(f"   - FPS trung bình: {avg_fps:.1f}")
            print(f"   - Biển số phát hiện: {len(results_list)}")
            if results_list:
                print(f"   - Ví dụ: {results_list[0]['plate']}")
        
        # Giải phóng tài nguyên
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def process_batch(self, image_dir, output_csv=None):
        """
        Xử lý batch ảnh từ folder
        """
        print(f"📁 Đang xử lý batch từ: {image_dir}")
        
        image_dir = Path(image_dir)
        image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
        
        results = []
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] {img_path.name}")
            
            result = self.process_image(str(img_path), visualize=False)
            
            if result:
                results.append({
                    'filename': img_path.name,
                    'plate_text': result['plate_text'],
                    'char_count': result['char_count'],
                    'processing_time': result['processing_time']
                })
        
        # Lưu kết quả
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\n💾 Đã lưu kết quả tại: {output_csv}")
        
        # Thống kê
        print(f"\n📊 THỐNG KÊ:")
        print(f"   - Tổng ảnh: {len(image_files)}")
        print(f"   - Xử lý thành công: {len(results)}")
        print(f"   - Thời gian trung bình: {np.mean([r['processing_time'] for r in results]):.3f}s")
        
        return results
    
    def evaluate_on_annotations(self, image_dir, annotation_csv):
        """
        Đánh giá hệ thống trên dataset với annotations
        """
        print(f"📊 Đang đánh giá trên dataset: {image_dir}")
        
        # Load annotations
        df = pd.read_csv(annotation_csv)
        print(f"✅ Loaded {len(df)} annotations")
        
        results = []
        detected_count = 0
        
        for idx, row in df.iterrows():
            filename = row['filename']
            ground_truth = row['plate_text']
            img_path = f"{image_dir}/{filename}"
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Get bounding box
            try:
                x1, y1 = int(row['xmin']), int(row['ymin'])
                x2, y2 = int(row['xmax']), int(row['ymax'])
                plate_img = img[y1:y2, x1:x2]
            except:
                continue
            
            # Segment characters
            char_images = self.char_recognizer.segment_characters(plate_img)
            
            if len(char_images) == 0:
                continue
            
            detected_count += 1
            
            # Recognize
            features_list = [self.extract_features(char) for char in char_images]
            features_array = np.array(features_list)
            predictions = self.model.predict(features_array)
            predicted = ''.join(predictions)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(predicted, ground_truth)
            
            results.append({
                'filename': filename,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'accuracy': accuracy
            })
            
            status = "✅" if accuracy == 100 else "⚠️ "
            print(f"{status} {filename:20s} GT: {ground_truth:15s} Pred: {predicted:15s} Acc: {accuracy:6.1f}%")
        
        # Summary
        if results:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            perfect = sum(1 for r in results if r['accuracy'] == 100)
            partial = sum(1 for r in results if r['accuracy'] > 0)
            
            print(f"\n📈 KẾT QUẢ:")
            print(f"   - Phát hiện ký tự: {detected_count}/{len(df)}")
            print(f"   - Độ chính xác trung bình: {avg_accuracy:.2f}%")
            print(f"   - Perfect (100%): {perfect}/{len(results)}")
            print(f"   - Partial (>0%): {partial}/{len(results)}")
    
    def _calculate_accuracy(self, predicted, ground_truth):
        """Tính accuracy ký tự"""
        if len(ground_truth) == 0:
            return 0.0
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        return (correct / len(ground_truth)) * 100
    
    def _visualize_result(self, image, plate_text, char_images, predictions):
        """Visualize kết quả lên ảnh"""
        result_img = image.copy()
        h, w = result_img.shape[:2]
        
        # Add title
        title = f"Recognized Plate: {plate_text}"
        cv2.putText(result_img, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw segmented characters at bottom
        char_display_height = 100
        char_height = h - char_display_height
        char_width = w // len(char_images) if len(char_images) > 0 else w
        
        for i, (char_img, pred) in enumerate(zip(char_images, predictions)):
            x = i * char_width
            # Draw character box
            cv2.rectangle(result_img, (x, char_height), (x + char_width, h), (0, 255, 0), 2)
            # Draw predicted character
            cv2.putText(result_img, pred, (x + 15, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # Show
        cv2.imshow('License Plate Recognition - Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Hàm main để chạy hệ thống"""
    parser = argparse.ArgumentParser(description='Hệ thống nhận dạng biển số xe (Hybrid Model)')
    parser.add_argument('--image', type=str, help='Đường dẫn ảnh đầu vào')
    parser.add_argument('--batch', type=str, help='Xử lý batch từ folder')
    parser.add_argument('--video', type=str, help='Đường dẫn video đầu vào')
    parser.add_argument('--output', type=str, help='Đường dẫn output')
    parser.add_argument('--eval', type=str, help='Folder để đánh giá')
    parser.add_argument('--annotations', type=str, help='File CSV annotations')
    
    args = parser.parse_args()
    
    # Khởi tạo hệ thống
    lpr_system = LicensePlateSystem()
    
    # Chạy chế độ tương ứng
    if args.image:
        lpr_system.process_image(args.image)
    elif args.batch:
        lpr_system.process_batch(args.batch, args.output)
    elif args.video:
        lpr_system.process_video(args.video, args.output)
    elif args.eval and args.annotations:
        lpr_system.evaluate_on_annotations(args.eval, args.annotations)
    else:
        print("\n🎯 USAGE:")
        print("  # Process single image:")
        print("  python main.py --image path/to/image.jpg")
        print("\n  # Process batch:")
        print("  python main.py --batch path/to/folder --output results.csv")
        print("\n  # Process video:")
        print("  python main.py --video input.mp4 --output output.mp4")
        print("\n  # Evaluate on dataset:")
        print("  python main.py --eval datasets/kaggle_foreign/test --annotations test_annotations.csv")

if __name__ == "__main__":
    main()