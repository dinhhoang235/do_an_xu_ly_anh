import cv2
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from .preprocessor import Preprocessor

class CharacterRecognizer:
    def __init__(self):
        self.knn_model = None
        self.is_trained = False
        
        # Kích thước chuẩn cho ký tự
        self.char_width = 20
        self.char_height = 30
    
    def train_knn(self, character_dataset_path, n_neighbors=5, test_size=0.2):
        print("🔄 Đang tải dataset ký tự...")
        
        X = []  # Features
        y = []  # Labels
        
        char_path = Path(character_dataset_path)
        
        if not char_path.exists():
            print(f"❌ Không tìm thấy thư mục: {character_dataset_path}")
            return None
        
        # Duyệt qua từng thư mục ký tự
        char_count = {}
        for char_dir in char_path.iterdir():
            if not char_dir.is_dir():
                continue
            
            char_label = char_dir.name
            char_images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            
            if len(char_images) == 0:
                continue
            
            char_count[char_label] = len(char_images)
            
            for img_path in char_images:
                # Đọc ảnh
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Trích xuất features
                features = self.extract_features(img)
                
                X.append(features)
                y.append(char_label)
        
        if len(X) == 0:
            print("❌ Không tìm thấy ảnh ký tự nào!")
            return None
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Normalize X to prevent numerical issues
        X = np.clip(X, 0, 1)
        
        print(f"📊 Thống kê dataset:")
        print(f"   - Tổng số ký tự: {len(X)}")
        print(f"   - Số loại ký tự: {len(char_count)}")
        for char, count in sorted(char_count.items()):
            print(f"     {char}: {count} ảnh")
        
        # Kiểm tra nếu có class với ít hơn 2 samples
        min_samples = min(char_count.values())
        use_stratify = min_samples >= 2
        
        if not use_stratify:
            print(f"\n⚠️  Cảnh báo: Một số ký tự chỉ có 1 mẫu, không thể dùng stratified split")
        
        # Chia train/test
        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"\n🔄 Đang huấn luyện KNN với {n_neighbors} neighbors...")
        
        # Train KNN với weights='distance' để giảm ảnh hưởng của class imbalance
        self.knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors, 
            weights='distance',  # Weighted by distance
            metric='euclidean'
        )
        self.knn_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Đánh giá trên tập test
        y_pred = self.knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Huấn luyện hoàn tất!")
        print(f"   - Accuracy trên tập test: {accuracy:.2%}")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")
        
        return self.knn_model
    
    def extract_features(self, char_image):
        # Resize về kích thước chuẩn
        resized = cv2.resize(char_image, (self.char_width, self.char_height))
        
        # Chuyển về ảnh nhị phân
        _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Trích xuất features và normalize
        features = binary.flatten().astype(np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        
        # Clip để tránh numerical instability
        features = np.clip(features, 0, 1)
        
        return features
    
    def segment_characters(self, plate_image):
        # Chuyển ảnh xám nếu cần
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 9, 15)
        
        # Morphological operations - giống repository
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Tính diện tích biển số
        height, width = gray.shape
        roi_area = height * width
        
        # Parameters giống repository
        Min_char = 0.005  # Optimized for foreign plates (was 0.01)
        Max_char = 0.12   # Optimized for foreign plates (was 0.09)
        
        char_data = []  # Lưu (x_position, char_image)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            ratio_char = w / h
            char_area = w * h
            
            # Filter - optimized for foreign plates
            if (Min_char * roi_area < char_area < Max_char * roi_area) and \
               (0.25 < ratio_char < 0.9):
                # Cắt ký tự từ ảnh threshold
                char_img = thre_mor[y:y+h, x:x+w]
                char_data.append((x, char_img))
        
        # Sắp xếp theo vị trí x (trái sang phải)
        char_data.sort(key=lambda item: item[0])
        
        # Chỉ lấy ảnh ký tự
        sorted_chars = [char_img for _, char_img in char_data]
        
        return sorted_chars
    
    def save_model(self, filepath):
        """Lưu model KNN"""
        if self.is_trained:
            with open(filepath, 'wb') as f:
                pickle.dump(self.knn_model, f)
            print(f"✅ Đã lưu model tại: {filepath}")
    
    def load_model(self, filepath):
        """Tải model KNN"""
        try:
            with open(filepath, 'rb') as f:
                self.knn_model = pickle.load(f)
            self.is_trained = True
            print(f"✅ Đã tải model từ: {filepath}")
        except FileNotFoundError:
            print("❌ Không tìm thấy file model")
    
    def post_process(self, plate_text):
        if not plate_text:
            return ""
        
        # Chuyển thành chữ hoa
        plate_text = plate_text.upper().strip()
        
        # Loại bỏ khoảng trắng
        plate_text = plate_text.replace(" ", "")
        
        # Thay thế ký tự nhầm lẫn thường gặp
        replacements = {
            'O': '0',  # Letter O → Number 0 (nếu là số)
            'I': '1',  # Letter I → Number 1 (nếu là số)
            'Z': '2',  # Letter Z → Number 2 (nếu là số)
            'S': '5',  # Letter S → Number 5 (nếu là số)
        }
        
        # Chỉ thay thế nếu ở vị trí số
        for old_char, new_char in replacements.items():
            plate_text = plate_text.replace(old_char, new_char)
        
        return plate_text