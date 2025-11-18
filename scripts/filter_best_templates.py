"""
Filter best templates từ characters_auto_labeled/
Chọn 1 ảnh tốt nhất cho mỗi ký tự và lưu vào character_templates/

Input: characters_auto_labeled/{CLASS}/*.png (3100+ ảnh có noise)
Output: character_templates/{CLASS}.png (31 ảnh chất lượng cao)

Criteria for best template:
1. Kích thước chuẩn (20x30 hoặc gần đó)
2. Contrast cao (dễ nhận dạng)
3. Không bị mờ/nhiễu
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

class FilterBestTemplates:
    def __init__(self):
        self.input_base = "datasets/kaggle_foreign/characters_auto_labeled"
        self.output_dir = "datasets/kaggle_foreign/character_templates"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✅ Created output directory: {self.output_dir}")
    
    def calculate_image_quality(self, img):
        """
        Calculate quality score của ảnh ký tự
        
        Factors:
        - Contrast (cao càng tốt)
        - Sharpness (cao càng tốt)
        - Non-blank area ratio (nên có ký tự, không trắng)
        
        Returns: Quality score (0-1)
        """
        if img is None or img.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 1. Contrast score (standard deviation of pixel values)
        contrast = np.std(gray)
        contrast_score = min(contrast / 100, 1.0)  # Normalize to 0-1
        
        # 2. Sharpness score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        sharpness_score = min(sharpness / 1000, 1.0)  # Normalize
        
        # 3. Non-blank area ratio
        # Binary threshold
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        non_zero_ratio = np.count_nonzero(binary) / binary.size
        
        # Penalize if too blank or too dark
        if non_zero_ratio < 0.05 or non_zero_ratio > 0.95:
            area_score = 0.3
        else:
            area_score = 0.8
        
        # 4. Size score (prefer 20x30 or close)
        target_size = 20 * 30
        actual_size = gray.shape[0] * gray.shape[1]
        size_score = 1.0 / (1.0 + abs(actual_size - target_size) / target_size)
        
        # Combine scores (weighted)
        quality = (
            contrast_score * 0.3 +
            sharpness_score * 0.3 +
            area_score * 0.2 +
            size_score * 0.2
        )
        
        return quality
    
    def filter_best_templates(self):
        """
        Filter best template cho mỗi ký tự
        """
        print("\n" + "="*70)
        print("🎯 FILTER BEST TEMPLATES FROM AUTO-LABELED CHARACTERS")
        print("="*70)
        
        classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        results = {}
        
        print(f"\n🔄 Processing {len(classes)} character classes...\n")
        
        for cls in tqdm(classes, desc="Filtering"):
            class_dir = os.path.join(self.input_base, cls)
            
            if not os.path.exists(class_dir):
                results[cls] = {'status': 'not_found', 'count': 0}
                continue
            
            # Get all images of this class
            img_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            
            if len(img_files) == 0:
                results[cls] = {'status': 'no_images', 'count': 0}
                continue
            
            # Calculate quality scores
            best_img = None
            best_score = -1
            best_file = None
            
            for img_file in img_files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                quality = self.calculate_image_quality(img)
                
                if quality > best_score:
                    best_score = quality
                    best_img = img
                    best_file = img_file
            
            # Save best template
            if best_img is not None:
                output_path = os.path.join(self.output_dir, f"{cls}.png")
                cv2.imwrite(output_path, best_img)
                results[cls] = {
                    'status': 'saved',
                    'count': len(img_files),
                    'quality_score': best_score,
                    'source_file': best_file
                }
            else:
                results[cls] = {'status': 'failed', 'count': len(img_files)}
        
        # Print statistics
        self._print_statistics(results, classes)
    
    def _print_statistics(self, results, classes):
        """Print filtering statistics"""
        print("\n" + "="*70)
        print("📊 FILTERING STATISTICS")
        print("="*70)
        
        saved = sum(1 for r in results.values() if r['status'] == 'saved')
        failed = sum(1 for r in results.values() if r['status'] in ['not_found', 'no_images', 'failed'])
        
        print(f"\n✅ Successfully saved: {saved}/{len(classes)}")
        print(f"❌ Failed: {failed}/{len(classes)}")
        
        # Show quality scores
        print(f"\n📋 TEMPLATES BY QUALITY SCORE:")
        quality_list = []
        for cls in classes:
            if results[cls]['status'] == 'saved':
                quality_list.append((cls, results[cls]['quality_score']))
        
        quality_list.sort(key=lambda x: x[1], reverse=True)
        for cls, score in quality_list:
            bar = "█" * int(score * 20)
            print(f"   {cls}: {score:.3f} {bar}")
        
        print(f"\n📁 Input directory: {self.input_base}/")
        print(f"💾 Output directory: {self.output_dir}/")
        print("="*70)
    
    def interactive_mode(self):
        """
        Interactive mode: let user select best template for each class
        """
        print("\n" + "="*70)
        print("🎯 INTERACTIVE MODE - SELECT BEST TEMPLATES")
        print("="*70)
        print("\nNote: This mode is for future enhancement")
        print("Currently, auto filtering is recommended")

def main():
    # Check if input directory exists
    input_dir = "datasets/kaggle_foreign/characters_auto_labeled"
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        print(f"Please run: python scripts/auto_extract_and_label_kaggle.py")
        return
    
    # Create filter
    filter_obj = FilterBestTemplates()
    
    # Filter best templates
    filter_obj.filter_best_templates()

if __name__ == "__main__":
    main()
