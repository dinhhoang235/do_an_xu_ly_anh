import cv2
import numpy as np
from .preprocessor import Preprocessor

class PlateDetector:
    def __init__(self):
        # Tham số cho Canny Edge Detection (tuned: 60, 180)
        self.canny_thresh1 = 60
        self.canny_thresh2 = 180
        
        # Tham số Morphological Operations
        self.kernel_size = (5, 5)
        
        # Tham số lọc contour (cân bằng tốt hơn)
        self.area_min = 400       # Giảm để bao gồm biển số rất nhỏ
        self.area_max = 30000     
        self.aspect_ratio_min = 1.5   # Giảm để bao gồm biển số compact
        self.aspect_ratio_max = 6.5   # Tăng để bao gồm biển số ngoại hẹp
        self.solidity_min = 0.25
        
        # Bộ lọc vị trí: biển số thường ở phần dưới của ảnh
        self.max_y_ratio = 0.95  # Biển số không nằm ở 5% trên cùng
        
    def detect_plates(self, preprocessed_image):
        """
        Phát hiện biển số từ ảnh đã tiền xử lý
        Input: Ảnh grayscale, blurred (từ Preprocessor)
        """
        # Bước 1: Canny Edge Detection
        edges = self._canny_edge_detection(preprocessed_image)
        
        # Bước 2: Morphological Operations
        morph = self._morphological_operations(edges)
        
        # Bước 3: Tìm contours và lọc biển số
        # Cần ảnh gốc để lấy kích thước, nên truyền riêng
        plates = self._find_plate_contours(morph, preprocessed_image)
        
        return plates
    
    def _canny_edge_detection(self, gray_image):
        """Phát hiện biên sử dụng Canny"""
        edges = cv2.Canny(gray_image, self.canny_thresh1, self.canny_thresh2)
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
        plates = []
        h, w = original_image.shape[:2]
        
        # Tìm contours
        contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Lọc contour theo diện tích
            area = cv2.contourArea(contour)
            if area < self.area_min or area > self.area_max:
                continue
            
            # Lấy bounding rect
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Bộ lọc vị trí: biển số không nằm quá trên
            if y > h * self.max_y_ratio:
                continue
            
            # Lọc theo tỷ lệ (biển số thường có tỷ lệ 1.5:1 đến 6.5:1)
            aspect_ratio = w_box / h_box
            if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
                continue
            
            # Lọc theo solidity (độ đặc của contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < self.solidity_min:
                    continue
            
            plates.append((x, y, w_box, h_box))
        
        return plates
    
    def visualize_detection(self, image, plates):
        """Visualize kết quả phát hiện biển số"""
        result = image.copy()
        
        for (x, y, w, h) in plates:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result