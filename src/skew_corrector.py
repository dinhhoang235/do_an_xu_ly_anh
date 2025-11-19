import cv2
import numpy as np
from scipy import ndimage


class SkewCorrector:
    """Hiệu chỉnh góc nghiêng (skew) của biển số xe"""
    
    def __init__(self):
        self.method = 'moments'  # 'moments', 'hough', 'contour'
    
    def correct_skew(self, plate_image):
        """
        Hiệu chỉnh góc nghiêng của biển số
        
        Args:
            plate_image: Ảnh biển số (BGR hoặc grayscale)
        
        Returns:
            corrected_image: Ảnh sau khi hiệu chỉnh
            angle: Góc hiệu chỉnh (độ)
        """
        if plate_image is None or plate_image.size == 0:
            return plate_image, 0
        
        # Chuyển về grayscale nếu cần
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Thử 3 phương pháp, chọn cái nào thành công
        if self.method == 'moments':
            corrected, angle = self._correct_by_moments(plate_image, gray)
        elif self.method == 'hough':
            corrected, angle = self._correct_by_hough(plate_image, gray)
        else:  # contour
            corrected, angle = self._correct_by_contour(plate_image, gray)
        
        return corrected, angle
    
    def _correct_by_moments(self, original_image, gray_image):
        """
        Hiệu chỉnh bằng image moments (PCA-based)
        Phương pháp nhanh và ổn định nhất
        """
        try:
            # Binary threshold
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            
            # Tìm tất cả pixel trắng
            coords = np.column_stack(np.where(binary > 0))
            
            if len(coords) < 10:  # Quá ít pixel, không cần hiệu chỉnh
                return original_image, 0
            
            # Tính PCA
            mean = np.mean(coords, axis=0)
            coords_centered = coords - mean
            cov_matrix = np.cov(coords_centered.T)
            
            # Lấy eigenvector tương ứng eigenvalue lớn nhất
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            max_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Tính góc từ eigenvector
            angle = np.degrees(np.arctan2(max_eigenvector[1], max_eigenvector[0]))
            
            # Góc có thể âm, cần hiệu chỉnh để trong range [-45, 45]
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            # Nếu góc quá nhỏ, không cần xoay
            if abs(angle) < 0.5:
                return original_image, 0
            
            # Xoay ảnh
            corrected = self._rotate_image(original_image, angle)
            
            return corrected, angle
            
        except Exception as e:
            print(f"⚠️  Lỗi moments method: {e}")
            return original_image, 0
    
    def _correct_by_hough(self, original_image, gray_image):
        """
        Hiệu chỉnh bằng Hough Line Transform
        Xác định các đường thẳng trong ảnh (cạnh biển số)
        """
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
            
            if lines is None or len(lines) < 3:
                return original_image, 0
            
            # Tính góc trung bình của các đường
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = np.degrees(theta)
                # Chuyển theta (0-180) thành angle (-90 to 90)
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            
            # Loại bỏ outliers
            mean_angle = np.mean(angles)
            angles = [a for a in angles if abs(a - mean_angle) < 20]
            
            if not angles:
                return original_image, 0
            
            angle = np.median(angles)
            
            # Nếu góc quá nhỏ, không cần xoay
            if abs(angle) < 0.5:
                return original_image, 0
            
            corrected = self._rotate_image(original_image, angle)
            
            return corrected, angle
            
        except Exception as e:
            print(f"⚠️  Lỗi Hough method: {e}")
            return original_image, 0
    
    def _correct_by_contour(self, original_image, gray_image):
        """
        Hiệu chỉnh bằng contour analysis
        Tìm contour của biển số rồi tính góc
        """
        try:
            # Binary threshold
            _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            
            # Morphological operations để kết nối các phần
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Tìm contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return original_image, 0
            
            # Lấy contour lớn nhất
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) < 100:
                return original_image, 0
            
            # Fit rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Hiệu chỉnh góc
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            # Nếu góc quá nhỏ, không cần xoay
            if abs(angle) < 0.5:
                return original_image, 0
            
            corrected = self._rotate_image(original_image, angle)
            
            return corrected, angle
            
        except Exception as e:
            print(f"⚠️  Lỗi contour method: {e}")
            return original_image, 0
    
    def _rotate_image(self, image, angle):
        """Xoay ảnh quanh tâm"""
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Tạo rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Xoay ảnh
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def set_method(self, method):
        """Chọn phương pháp hiệu chỉnh ('moments', 'hough', 'contour')"""
        if method in ['moments', 'hough', 'contour']:
            self.method = method
        else:
            print(f"❌ Phương pháp không hợp lệ: {method}")
