import cv2
import numpy as np
import os
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
        
    def load_knn_from_files(self, classifications_file="classifications.txt", 
                            flattened_images_file="flattened_images.txt"):
        """
        Load KNN model từ file giống repository VIETNAMESE_LICENSE_PLATE
        """
        print("🔄 Đang load KNN model từ file...")
        
        if not Path(classifications_file).exists():
            print(f"❌ Không tìm thấy {classifications_file}")
            return False
        
        if not Path(flattened_images_file).exists():
            print(f"❌ Không tìm thấy {flattened_images_file}")
            return False
        
        # Load data
        classifications = np.loadtxt(classifications_file, np.float32)
        flattened_images = np.loadtxt(flattened_images_file, np.float32)
        
        # Reshape classifications
        classifications = classifications.reshape((classifications.size, 1))
        
        # Tạo KNN model
        self.knn_model = cv2.ml.KNearest_create()
        self.knn_model.train(flattened_images, cv2.ml.ROW_SAMPLE, classifications)
        
        self.is_trained = True
        
        print(f"✅ Đã load KNN model")
        print(f"   - Số lượng mẫu: {flattened_images.shape[0]}")
        print(f"   - Feature dimension: {flattened_images.shape[1]}")
        
        return True
        
    def create_template_dataset(self, vn_plates_folder):
        """
        Tạo bộ template từ 22 ảnh Việt Nam
        """
        print("🔄 Đang tạo bộ template ký tự...")
        
        # Ký tự cần nhận dạng (biển số VN)
        chars = "0123456789ABCDEFGHKLMNPRSTUVXYZ"
        
        # Tạo thư mục template nếu chưa có
        template_dir = "datasets/character_templates"
        os.makedirs(template_dir, exist_ok=True)
        
        # Dictionary lưu template
        templates = {}
        
        # Với mỗi ký tự, tạo template đơn giản (có thể thay bằng ảnh thật sau)
        for char in chars:
            # Tạo ảnh trắng
            template = np.ones((self.char_height, self.char_width), dtype=np.uint8) * 255
            
            # Vẽ ký tự lên ảnh (giả lập - thực tế sẽ dùng ảnh thật từ dataset)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Tính toán vị trí để căn giữa
            text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
            text_x = (self.char_width - text_size[0]) // 2
            text_y = (self.char_height + text_size[1]) // 2
            
            # Vẽ ký tự màu đen
            cv2.putText(template, char, (text_x, text_y), font, font_scale, 0, thickness)
            
            templates[char] = template
            
            # Lưu template ra file
            cv2.imwrite(f"{template_dir}/{char}.png", template)
        
        self.char_templates = templates
        print(f"✅ Đã tạo {len(templates)} template ký tự")
        
        return templates
    
    def train_knn(self, character_dataset_path, n_neighbors=5, test_size=0.2):
        """
        Huấn luyện mô hình KNN từ dataset ký tự đã cắt
        
        Args:
            character_dataset_path: Đường dẫn đến thư mục chứa các ký tự đã phân loại
            n_neighbors: Số lượng neighbors cho KNN
            test_size: Tỷ lệ dữ liệu test
        """
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
        """
        Trích xuất đặc trưng từ ảnh ký tự
        """
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
        """
        Phân tách ký tự từ ảnh biển số - giống VIETNAMESE_LICENSE_PLATE
        """
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
    
    def segment_characters_improved(self, plate_image):
        """
        Phân tách ký tự cải tiến - dùng Preprocessor pipeline
        """
        preprocessor = Preprocessor()
        
        # Preprocess với Canny + Morphology
        _, processed = preprocessor.preprocess(plate_image)
        
        # Tìm contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Lấy kích thước để tính diện tích
        height, width = processed.shape
        roi_area = height * width
        
        # Parameters tối ưu
        Min_char = 0.003
        Max_char = 0.15
        
        char_data = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if h == 0:
                continue
            
            ratio_char = w / h
            char_area = w * h
            
            # Filter
            if (Min_char * roi_area < char_area < Max_char * roi_area) and \
               (0.15 < ratio_char < 0.9):
                # Cắt từ ảnh processed
                char_img = processed[y:y+h, x:x+w]
                char_data.append((x, char_img))
        
        # Sắp xếp theo vị trí x
        char_data.sort(key=lambda item: item[0])
        sorted_chars = [char_img for _, char_img in char_data]
        
        return sorted_chars
    
    def recognize_template_matching(self, char_image):
        """
        Nhận dạng ký tự sử dụng Template Matching
        """
        best_char = '?'
        best_score = -1
        
        # Resize ký tự đầu vào
        resized_char = cv2.resize(char_image, (self.char_width, self.char_height))
        
        for char, template in self.char_templates.items():
            # Template matching
            result = cv2.matchTemplate(resized_char, template, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(result)[1]  # Lấy điểm số tốt nhất
            
            if score > best_score:
                best_score = score
                best_char = char
        
        return best_char, best_score
    
    def recognize_knn(self, char_image):
        """
        Nhận dạng ký tự sử dụng KNN (OpenCV style như VIETNAMESE_LICENSE_PLATE)
        """
        if not self.is_trained:
            return '?', 0.0
        
        # Resize về kích thước chuẩn
        char_resized = cv2.resize(char_image, (self.char_width, self.char_height))
        
        # Chuyển sang grayscale nếu cần
        if len(char_resized.shape) == 3:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_BGR2GRAY)
        
        # Flatten thành 1D array và normalize
        char_flattened = char_resized.flatten().astype(np.float32) / 255.0
        char_flattened = np.clip(char_flattened, 0, 1).reshape(1, -1)
        
        # Predict với sklearn KNN
        try:
            probabilities = self.knn_model.predict_proba(char_flattened)[0]
            predicted_label = self.knn_model.predict(char_flattened)[0]
            
            # Lấy confidence (xác suất cao nhất)
            confidence = np.max(probabilities)
            
            predicted_char = predicted_label
        except Exception as e:
            # Fallback nếu có lỗi
            predicted_char = '?'
            confidence = 0.0
        
        return predicted_char, confidence
    
    def recognize_plate(self, plate_image, method='template'):
        """
        Nhận dạng toàn bộ biển số
        """
        # Phân tách ký tự
        characters = self.segment_characters(plate_image)
        
        if not characters:
            return "", []
        
        plate_text = ""
        recognition_results = []
        
        for i, char_img in enumerate(characters):
            if method == 'knn' and self.is_trained:
                char, confidence = self.recognize_knn(char_img)
            else:
                char, confidence = self.recognize_template_matching(char_img)
            
            plate_text += char
            recognition_results.append((char, confidence))
        
        return plate_text, recognition_results
    
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
    
    def validate_plate_format(self, plate_text):
        """
        Kiểm tra tính hợp lệ của biển số
        Hỗ trợ biển số Việt Nam và nước ngoài
        """
        if not plate_text or len(plate_text.strip()) == 0:
            return False, "Biển số trống"
        
        plate_text = plate_text.upper().strip()
        
        # Loại bỏ ký tự không hợp lệ
        valid_chars = "0123456789ABCDEFGHKLMNPRSTUVXYZ-"
        
        # Kiểm tra ký tự
        for char in plate_text:
            if char not in valid_chars:
                return False, f"Ký tự '{char}' không hợp lệ"
        
        # Kiểm tra độ dài (biển số thường 6-10 ký tự)
        if len(plate_text) < 6 or len(plate_text) > 10:
            return False, f"Độ dài biển số không hợp lệ: {len(plate_text)}"
        
        # Nếu có dấu gạch ngang, kiểm tra vị trí
        if '-' in plate_text:
            # Format: XXX-YYYY hoặc XXXX-YY
            parts = plate_text.split('-')
            if len(parts) != 2:
                return False, "Định dạng dấu gạch ngang không hợp lệ"
        
        return True, "Hợp lệ"
    
    def post_process(self, plate_text):
        """
        Hậu xử lý kết quả nhận dạng
        - Loại bỏ ký tự nhiễu
        - Chuẩn hóa định dạng
        """
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