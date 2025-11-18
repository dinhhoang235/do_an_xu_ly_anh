import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        
        # Tiền xử lý - Lọc nhiễu
        self.MEDIAN_BLUR_SIZE = 3
        self.GAUSSIAN_BLUR_SIZE = (5, 5)
        
        # Canny Edge Detection
        self.CANNY_THRESH1 = 50
        self.CANNY_THRESH2 = 150
        
        # Morphological Operations
        self.MORPH_KERNEL_SIZE = (5, 5)
    
    def preprocess(self, img_original):
        """
        Preprocessing pipeline theo báo cáo:
        Bước 1: Chuyển ảnh xám
        Bước 2: Lọc Median để khử nhiễu Salt & Pepper
        Bước 3: Lọc Gaussian để làm mịn
        Bước 4: Canny edge detection để phát hiện biên
        Bước 5: Phép toán hình thái học (Closing + Opening) để nối/tách vùng
        
        Returns:
            - img_grayscale: Ảnh xám
            - img_processed: Ảnh sau tiền xử lý (canny + morphology)
        """
        # Bước 1: Chuyển ảnh sang xám
        img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # Bước 2: Lọc Median để khử nhiễu Salt & Pepper
        img_median = cv2.medianBlur(img_grayscale, self.MEDIAN_BLUR_SIZE)
        
        # Bước 3: Lọc Gaussian để làm mịn ảnh
        img_blurred = cv2.GaussianBlur(img_median, self.GAUSSIAN_BLUR_SIZE, 0)
        
        # Bước 4: Canny edge detection để phát hiện biên
        img_canny = cv2.Canny(img_blurred, self.CANNY_THRESH1, self.CANNY_THRESH2)
        
        # Bước 5: Phép toán hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
        
        # Closing: nối biên đứt đoạn (Dilate -> Erode)
        img_closed = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
        
        # Opening: loại bỏ nhiễu nhỏ (Erode -> Dilate)
        img_processed = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)
        
        return img_grayscale, img_processed
    
    def resize_image(self, image):
        """Chuẩn hóa kích thước ảnh"""
        return cv2.resize(image, (self.config.IMG_WIDTH, self.config.IMG_HEIGHT))
    
    def correct_skew(self, img_gray):
        """
        Hiệu chỉnh góc nghiêng của biển số
        Sử dụng Hough Line Transform để phát hiện góc
        """
        try:
            # Canny edge detection
            edges = cv2.Canny(img_gray, 50, 150)
            
            # Hough Line Transform để phát hiện các đường thẳng
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            
            if lines is None or len(lines) == 0:
                return img_gray  # Không phát hiện được đường, trả về ảnh gốc
            
            # Tính toán góc trung bình từ các đường phát hiện được
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90  # Chuyển từ radians sang degrees
                angles.append(angle)
            
            # Lấy góc trung bình
            mean_angle = np.mean(angles)
            
            # Nếu góc quá nhỏ, không cần hiệu chỉnh
            if abs(mean_angle) < 1.0:
                return img_gray
            
            # Thực hiện xoay ảnh
            h, w = img_gray.shape
            rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), mean_angle, 1.0)
            rotated = cv2.warpAffine(img_gray, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT)
            
            return rotated
        
        except Exception as e:
            # Nếu có lỗi, trả về ảnh gốc
            print(f"⚠️  Lỗi hiệu chỉnh nghiêng: {e}")
            return img_gray