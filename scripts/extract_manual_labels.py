"""
Extract and label characters từ test set
Cắt ký tự từ 17 ảnh test và lưu vào characters_manual_labeled/{CLASS}/

Input: 17 test images + ground truth text từ test_annotations.csv
Output: characters_manual_labeled/{CLASS}/{image}.png
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer

class ExtractManualLabels:
    def __init__(self):
        self.recognizer = CharacterRecognizer()
        self.output_base = "datasets/kaggle_foreign/characters_manual_labeled"
        self._create_class_directories()
    
    def _create_class_directories(self):
        """Create output directories for all character classes"""
        classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for cls in classes:
            class_dir = os.path.join(self.output_base, cls)
            os.makedirs(class_dir, exist_ok=True)
        
        print(f"✅ Created {len(classes)} output directories")
    
    def extract_and_label_characters(self, img, ground_truth, img_name):
        """
        Segment characters từ plate image và label theo ground truth
        
        Args:
            img: Plate image
            ground_truth: Text chứ số đúng (từ annotations)
            img_name: Tên file ảnh
        
        Returns:
            Dict với statistics
        """
        # Segment characters
        char_images = self.recognizer.segment_characters(img)
        
        if len(char_images) == 0:
            return {
                'filename': img_name,
                'status': 'no_detection',
                'chars_detected': 0,
                'chars_labeled': 0,
                'ground_truth': ground_truth
            }
        
        # Label characters dựa trên ground truth
        labeled_count = 0
        
        if len(ground_truth) > 0:
            # Match detected characters với ground truth
            num_chars_to_label = min(len(char_images), len(ground_truth))
            
            for i in range(num_chars_to_label):
                char_img = char_images[i]
                label = ground_truth[i]
                
                # Validate character
                if not label.isalnum():
                    continue
                
                # Save character with label
                class_dir = os.path.join(self.output_base, label)
                counter = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
                filename = f"{label}_{counter:03d}_{img_name[:-4]}_pos{i}.png"
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
            'ground_truth': ground_truth,
            'mismatch': len(ground_truth) - labeled_count
        }
    
    def process_test_set(self, test_dir, annotations_csv):
        """
        Process test set từ test_annotations.csv
        
        Args:
            test_dir: Path to test images folder
            annotations_csv: Path to test_annotations.csv
        """
        print("\n" + "="*70)
        print("🏷️  EXTRACT & LABEL FROM TEST SET")
        print("="*70)
        
        # Load annotations
        print(f"\n📋 Loading test annotations from {annotations_csv}...")
        if not os.path.exists(annotations_csv):
            print(f"❌ Annotations file not found: {annotations_csv}")
            return
        
        df = pd.read_csv(annotations_csv)
        print(f"✅ Loaded {len(df)} test images")
        
        # Statistics
        stats = {
            'processed': 0,
            'success': 0,
            'partial': 0,
            'failed': 0,
            'total_chars_detected': 0,
            'total_chars_labeled': 0,
            'total_mismatches': 0
        }
        
        results = []
        
        # Process each image
        print(f"\n🔄 Processing {len(df)} test images...\n")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            # Get image filename
            if 'filename' in row.index:
                img_name = row['filename']
                img_path = os.path.join(test_dir, img_name)
            elif 'image_path' in row.index:
                img_path = row['image_path']
                img_name = os.path.basename(img_path)
            else:
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
                # Get ground truth text
                ground_truth = str(row['plate_text']).upper()
                ground_truth = ''.join(c for c in ground_truth if c.isalnum())
                
                # Extract and label characters
                result = self.extract_and_label_characters(img, ground_truth, img_name)
                results.append(result)
                
                stats['total_chars_detected'] += result['chars_detected']
                stats['total_chars_labeled'] += result['chars_labeled']
                stats['total_mismatches'] += result['mismatch']
                
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
        print("📊 EXTRACTION STATISTICS")
        print("="*70)
        
        print(f"\n✅ Processed: {stats['processed']}/{stats['processed'] + stats['failed']}")
        print(f"   - Success (labeled): {stats['success']}")
        print(f"   - Partial (detected): {stats['partial']}")
        print(f"   - Failed: {stats['failed']}")
        
        print(f"\n📊 CHARACTER STATISTICS:")
        print(f"   - Total detected: {stats['total_chars_detected']}")
        print(f"   - Total labeled: {stats['total_chars_labeled']}")
        print(f"   - Total mismatches: {stats['total_mismatches']}")
        
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
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls}: {count:3d} characters")
        
        total_saved = sum(class_counts.values())
        print(f"\n   TOTAL SAVED: {total_saved} characters")
        
        print("\n" + "="*70)
        print(f"💾 Saved to: {self.output_base}/")
        print("="*70)
    
    def _save_results(self, results):
        """Save detailed results to CSV"""
        df = pd.DataFrame(results)
        output_csv = "extract_manual_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Detailed results saved to: {output_csv}")

def main():
    # Paths
    test_dir = "datasets/kaggle_foreign/test"
    test_annotations = "datasets/kaggle_foreign/test_annotations.csv"
    
    # Check if directories exist
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    if not os.path.exists(test_annotations):
        print(f"❌ Test annotations file not found: {test_annotations}")
        return
    
    # Create extractor
    extractor = ExtractManualLabels()
    
    # Process test set
    extractor.process_test_set(test_dir, test_annotations)

if __name__ == "__main__":
    main()
