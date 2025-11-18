"""
Automatically extract and label characters from Kaggle dataset
Uses EasyOCR to recognize plate text, then saves characters by class

Input: 473 images from Kaggle + annotations.csv
Output: characters_auto_labeled/{CLASS}/{image}.png
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import easyocr
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer

class AutoLabelKaggle:
    def __init__(self):
        self.recognizer = CharacterRecognizer()
        self.output_base = "datasets/kaggle_foreign/characters_auto_labeled"
        
        # Initialize EasyOCR reader (English + numbers)
        print("🔄 Initializing EasyOCR reader...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR initialized")
        except Exception as e:
            print(f"⚠️  EasyOCR error: {e}")
            self.reader = None
        
        # Create output directories for all classes
        self._create_class_directories()
    
    def _create_class_directories(self):
        """Create output directories for all character classes"""
        # Classes: A-Z (except I, O, Q for some regions) + 0-9
        classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for cls in classes:
            class_dir = os.path.join(self.output_base, cls)
            os.makedirs(class_dir, exist_ok=True)
        
        print(f"✅ Created {len(classes)} output directories")
    
    def extract_plate_text_easyocr(self, img):
        """Extract plate text using EasyOCR"""
        if self.reader is None:
            return ""
        
        try:
            results = self.reader.readtext(img, detail=0)
            text = ''.join(results).upper()
            # Clean: keep only alphanumeric
            text = ''.join(c for c in text if c.isalnum())
            return text
        except:
            return ""
    
    def segment_and_label_characters(self, img, plate_text, img_name):
        """
        Segment characters from plate image and label them
        
        Args:
            img: Plate image
            plate_text: OCR-recognized text (labels)
            img_name: Original image filename
        
        Returns:
            Dict with statistics
        """
        # Segment characters
        char_images = self.recognizer.segment_characters(img)
        
        if len(char_images) == 0:
            return {
                'filename': img_name,
                'status': 'no_detection',
                'chars_detected': 0,
                'chars_labeled': 0,
                'ocr_text': plate_text
            }
        
        # Label characters based on OCR result
        labeled_count = 0
        
        if len(plate_text) > 0:
            # Match detected characters with OCR text
            num_chars_to_label = min(len(char_images), len(plate_text))
            
            for i in range(num_chars_to_label):
                char_img = char_images[i]
                label = plate_text[i]  # Character class
                
                # Validate character
                if not (label.isalnum()):
                    continue
                
                # Save character with label
                class_dir = os.path.join(self.output_base, label)
                counter = len(os.listdir(class_dir))
                filename = f"{label}_{counter:04d}_{img_name[:-4]}_pos{i}.png"
                filepath = os.path.join(class_dir, filename)
                
                try:
                    cv2.imwrite(filepath, char_img)
                    labeled_count += 1
                except:
                    pass
        
        return {
            'filename': img_name,
            'status': 'labeled' if labeled_count > 0 else 'partial',
            'chars_detected': len(char_images),
            'chars_labeled': labeled_count,
            'ocr_text': plate_text
        }
    
    def process_dataset(self, image_dir, annotations_csv, max_images=None):
        """
        Process all images from Kaggle dataset
        
        Args:
            image_dir: Path to images folder
            annotations_csv: Path to annotations.csv
            max_images: Limit number of images to process (None = all)
        """
        print("\n" + "="*70)
        print("🚗 AUTO-EXTRACT & LABEL KAGGLE DATASET")
        print("="*70)
        
        # Load annotations
        print(f"\n📋 Loading annotations from {annotations_csv}...")
        if not os.path.exists(annotations_csv):
            print(f"❌ Annotations file not found: {annotations_csv}")
            return
        
        df = pd.read_csv(annotations_csv)
        print(f"✅ Loaded {len(df)} annotations")
        
        if max_images:
            df = df.head(max_images)
            print(f"⚠️  Limited to first {max_images} images")
        
        # Statistics
        stats = {
            'processed': 0,
            'success': 0,
            'partial': 0,
            'failed': 0,
            'total_chars_detected': 0,
            'total_chars_labeled': 0
        }
        
        results = []
        
        # Process each image
        print(f"\n🔄 Processing {len(df)} images...\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            # Get image filename - handle both 'filename' and 'image_path' columns
            if 'filename' in row.index:
                img_name = row['filename']
                img_path = os.path.join(image_dir, img_name)
            elif 'image_path' in row.index:
                # Extract filename from full path
                img_path = row['image_path']
                img_name = os.path.basename(img_path)
            else:
                print(f"⚠️  No 'filename' or 'image_path' column found")
                stats['failed'] += 1
                continue
            
            # Check if image exists
            if not os.path.exists(img_path):
                stats['failed'] += 1
                continue
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                stats['failed'] += 1
                continue
            
            stats['processed'] += 1
            
            try:
                # Extract plate text using EasyOCR
                plate_text = self.extract_plate_text_easyocr(img)
                
                # Segment and label characters
                result = self.segment_and_label_characters(img, plate_text, img_name)
                results.append(result)
                
                stats['total_chars_detected'] += result['chars_detected']
                stats['total_chars_labeled'] += result['chars_labeled']
                
                if result['status'] == 'labeled':
                    stats['success'] += 1
                else:
                    stats['partial'] += 1
                
            except Exception as e:
                stats['failed'] += 1
        
        # Print statistics
        self._print_statistics(stats, results)
        
        # Save detailed results
        self._save_results(results)
    
    def _print_statistics(self, stats, results):
        """Print processing statistics"""
        print("\n" + "="*70)
        print("📊 PROCESSING STATISTICS")
        print("="*70)
        
        print(f"\n✅ Processed: {stats['processed']}/{stats['processed'] + stats['failed']}")
        print(f"   - Success (labeled): {stats['success']}")
        print(f"   - Partial (detected): {stats['partial']}")
        print(f"   - Failed: {stats['failed']}")
        
        print(f"\n📊 CHARACTER STATISTICS:")
        print(f"   - Total detected: {stats['total_chars_detected']}")
        print(f"   - Total labeled: {stats['total_chars_labeled']}")
        
        if stats['total_chars_detected'] > 0:
            accuracy = (stats['total_chars_labeled'] / stats['total_chars_detected']) * 100
            print(f"   - Labeling accuracy: {accuracy:.1f}%")
        
        # Count saved characters per class
        print(f"\n📁 SAVED CHARACTERS BY CLASS:")
        class_counts = {}
        for cls in list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            class_dir = os.path.join(self.output_base, cls)
            count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            if count > 0:
                class_counts[cls] = count
        
        # Sort by count
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {cls}: {count:4d} characters")
        
        total_saved = sum(class_counts.values())
        print(f"\n   TOTAL SAVED: {total_saved} characters")
        
        print("\n" + "="*70)
        print(f"💾 Saved to: {self.output_base}/")
        print("="*70)
    
    def _save_results(self, results):
        """Save detailed results to CSV"""
        df = pd.DataFrame(results)
        output_csv = "auto_extract_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Detailed results saved to: {output_csv}")
        
        # Print summary
        print("\n📋 SAMPLE RESULTS:")
        print(df.head(10).to_string(index=False))

def main():
    # Paths
    image_dir = "datasets/kaggle_foreign/images"
    annotations_csv = "datasets/kaggle_foreign/annotations.csv"
    
    # Check if directories exist
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(annotations_csv):
        print(f"❌ Annotations file not found: {annotations_csv}")
        return
    
    # Create extractor
    extractor = AutoLabelKaggle()
    
    # Process dataset
    # Set max_images=None to process all 473 images
    extractor.process_dataset(
        image_dir,
        annotations_csv,
        max_images=None  # Process all
    )

if __name__ == "__main__":
    main()
