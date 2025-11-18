import cv2
import numpy as np
from .preprocessor import Preprocessor

class PlateDetector:
    def __init__(self):
        self.preprocessor = Preprocessor()
        
        # Tham số cho Canny Edge Detection
        self.canny_thresh1 = 40
        self.canny_thresh2 = 120
        
        # Tham số Morphological Operations
        self.kernel_size = (5, 5)
        
    def detect_plates(self, image):
        """
        Phát hiện biển số xe sử dụng xử lý ảnh truyền thống
        Trả về list các bounding boxes (x, y, w, h)
        """
        # Bước 1: Tiền xử lý ảnh
        gray, thresh = self.preprocessor.preprocess(image)
        
        # Bước 2: Canny Edge Detection
        edges = self._canny_edge_detection(thresh)
        
        # Bước 3: Morphological Operations
        morph = self._morphological_operations(edges)
        
        # Bước 4: Tìm contours và lọc biển số
        plates = self._find_plate_contours(morph, image)
        
        return plates
    
    def _canny_edge_detection(self, gray_image):
        """Phát hiện biên sử dụng Canny"""
        # Làm mịn ảnh trước khi detect biên
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_thresh1, self.canny_thresh2)
        return edges
    
    def _morphological_operations(self, edges):
        """Morphological operations để nối biên đứt đoạn"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        
        # Dilate để nối các biên gần nhau
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Erode để loại bỏ noise nhỏ
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    
    def _find_plate_contours(self, morph_image, original_image):
        """
        Tìm contours và lọc theo tiêu chí hình học của biển số
        """
        plates = []
        
        # Tìm contours
        contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Lọc contour theo diện tích
            area = cv2.contourArea(contour)
            if area < 500 or area > 80000:  # Extended range for foreign plates
                continue
            
            # Lấy bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            
            # Lọc theo tỷ lệ (biển số thường có tỷ lệ ~ 2:1 đến 5:1)
            aspect_ratio = w / h
            if aspect_ratio < 1.2 or aspect_ratio > 7:
                continue
            
            # Lọc theo solidity (độ đặc của contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.25:
                    continue
            
            plates.append((x, y, w, h))
        
        return plates
    
    def visualize_detection(self, image, plates):
        """Visualize kết quả phát hiện biển số"""
        result = image.copy()
        
        for (x, y, w, h) in plates:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result