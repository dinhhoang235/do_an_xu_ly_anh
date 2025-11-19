import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        # Tiền xử lý - Lọc nhiễu
        self.MEDIAN_BLUR_SIZE = 3
        self.GAUSSIAN_BLUR_SIZE = (5, 5)
    
    def preprocess(self, img_original):
        """
        Tiền xử lý ảnh: chuyển xám, lọc nhiễu, làm mịn
        """
        # Bước 1: Chuyển ảnh sang xám
        img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        
        # Bước 2: Lọc Median để khử nhiễu Salt & Pepper
        img_median = cv2.medianBlur(img_grayscale, self.MEDIAN_BLUR_SIZE)
        
        # Bước 3: Lọc Gaussian để làm mịn ảnh
        img_blurred = cv2.GaussianBlur(img_median, self.GAUSSIAN_BLUR_SIZE, 0)
        
        return img_blurred