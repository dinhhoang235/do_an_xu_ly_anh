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