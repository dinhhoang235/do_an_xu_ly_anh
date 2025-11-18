import cv2
import numpy as np

class Preprocessor:
    """
    Preprocessor cho License Plate:
    - enhance (CLAHE, gamma)
    - denoise (bilateral)
    - remove shadows (morph open / top-hat)
    - adaptive threshold / morphological clean
    - deskew / crop theo bounding box
    - normalise character images for recognizer
    """

    def __init__(self, target_width=400):
        self.target_width = target_width

    # --- Utilities ---
    def resize_keep_aspect(self, image, width=None):
        if width is None:
            width = self.target_width
        h, w = image.shape[:2]
        if w == 0:
            return image
        scale = width / float(w)
        new_h = int(h * scale)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_LINEAR)

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                         for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_clahe(self, gray, clip=2.0, tile=(8,8)):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        return clahe.apply(gray)

    def denoise(self, image, d=9, sigmaColor=75, sigmaSpace=75):
        # Bilateral filter preserves edges
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    def remove_shadows(self, gray):
        # Estimate background with morphological opening and subtract
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        diff = cv2.absdiff(gray, background)
        # Normalize
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        return norm

    def deskew_by_contour(self, img):
        # Find largest contour and rotate to upright using minAreaRect
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # --- Main pipelines ---
    def preprocess_plate(self, plate_img, do_deskew=True, gamma=0.9, clahe_clip=2.0):
        """
        Input: plate crop (could be whole image or detected plate bbox)
        Output: dict with 'gray', 'binary', 'enhanced' images
        """
        if plate_img is None:
            return {'enhanced': None, 'gray': None, 'binary': None}

        # 1. Resize for consistent processing
        img = self.resize_keep_aspect(plate_img, width=self.target_width)

        # 2. Convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 3. Denoise
        den = self.denoise(gray)

        # 4. Remove shadows / background
        no_shadows = self.remove_shadows(den)

        # 5. CLAHE for contrast enhancement
        clahe = self.apply_clahe(no_shadows, clip=clahe_clip)

        # 6. Gamma correction to handle under/over exposure
        gamma_corr = self.adjust_gamma(clahe, gamma=gamma)

        # 7. Adaptive threshold (Gaussian works better for uneven light)
        binary = cv2.adaptiveThreshold(gamma_corr, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 10)

        # 8. Morphological clean: remove small noise, close gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 9. Optionally deskew/rotate based on contours
        if do_deskew:
            # deskew using inverted image (so characters are bright on dark for contour)
            rotated = self.deskew_by_contour(cv2.bitwise_not(clean))
            rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
            _, rotated_bin = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            clean = cv2.bitwise_not(rotated_bin)
            gamma_corr = rotated_gray

        return {
            'enhanced': gamma_corr,
            'gray': gray,
            'binary': clean
        }

    def preprocess_for_recognizer(self, char_img):
        """
        Normalize a single character image to the size expected by the recognizer (20x30),
        apply binarization and centering.
        """
        if char_img is None:
            return np.zeros((30,20), dtype=np.uint8)

        try:
            resized = cv2.resize(char_img, (20, 30), interpolation=cv2.INTER_AREA)
        except Exception:
            resized = cv2.resize(char_img, (20, 30))

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # center by mass (tight bounding box) then pad
        coords = cv2.findNonZero(binary)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            char_crop = binary[y:y+h, x:x+w]
            # protect against tiny crop
            if char_crop.size == 0:
                char_crop = binary
            char_crop = cv2.resize(char_crop, (16, 26), interpolation=cv2.INTER_LINEAR)
            out = np.full((30,20), 0, dtype=np.uint8)
            y_off = (30 - 26) // 2
            x_off = (20 - 16) // 2
            out[y_off:y_off+26, x_off:x_off+16] = char_crop
            return out

        # fallback
        return binary
