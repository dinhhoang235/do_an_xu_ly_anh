"""
Comprehensive Comparison: Traditional CV + KNN vs YOLO v8 + CNN
Testing on LP-characters Dataset

Traditional Approach:
- Manual feature extraction (HOG)
- K-Nearest Neighbors (KNN) classifier

YOLO v8 + CNN Approach:
- YOLO v8 for character detection
- CNN for character classification
"""

import cv2
import sys
import os
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torch
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from skimage.feature import hog
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import Preprocessor
from src.skew_corrector import SkewCorrector
from src.plate_detector import PlateDetector
from src.yolo_plate_detector import YOLOPlateDetector
from src.cnn_recognizer import CharacterDataset, SimpleCNN, CNNRecognizer


# ============================================================================
# PART 1: TRADITIONAL APPROACH - HOG + KNN
# ============================================================================

class TraditionalKNNRecognizer:
    """Traditional approach using HOG features and KNN"""
    
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.label_encoder = LabelEncoder()
        self.preprocessor = Preprocessor()
        self.skew_corrector = SkewCorrector()
        self.is_trained = False
        
    def extract_hog_features(self, char_img, size=(32, 32)):
        """Extract HOG features from character image"""
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        
        resized = cv2.resize(char_img, size)
        # Normalize
        resized = resized.astype(np.float32) / 255.0
        
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        
        return features.astype(np.float32)
    
    def train(self, dataset_path):
        """Train KNN model on character dataset"""
        print("\n🔄 [TRADITIONAL] Training KNN model...")
        
        X = []
        y = []
        char_count = {}
        
        dataset_path = Path(dataset_path)
        
        for char_dir in tqdm(sorted(dataset_path.iterdir()), desc="Loading characters"):
            if not char_dir.is_dir():
                continue
            
            char_label = char_dir.name
            char_images = list(char_dir.glob("*.png")) + list(char_dir.glob("*.jpg"))
            
            if len(char_images) == 0:
                continue
            
            char_count[char_label] = len(char_images)
            
            for img_path in char_images:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                features = self.extract_hog_features(img)
                X.append(features)
                y.append(char_label)
        
        if len(X) == 0:
            raise ValueError("No training data found!")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        print(f"   📊 Dataset size: {len(X)} samples")
        print(f"   📝 Classes: {len(np.unique(y))}")
        
        # Fit label encoder
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Train KNN
        self.model.fit(X, y_encoded)
        self.is_trained = True
        
        print(f"   ✅ KNN training completed")
        
        return char_count
    
    def predict(self, char_img):
        """Predict character label"""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        features = self.extract_hog_features(char_img)
        pred_encoded = self.model.predict([features])[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
        
        return pred_label
    
    def predict_batch(self, char_images):
        """Predict batch of characters"""
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        features_list = [self.extract_hog_features(img) for img in char_images]
        features = np.array(features_list, dtype=np.float32)
        
        pred_encoded = self.model.predict(features)
        pred_labels = self.label_encoder.inverse_transform(pred_encoded)
        
        return pred_labels

# ============================================================================
# PART 2: TESTING & EVALUATION
# ============================================================================

def parse_xml_characters(xml_path):
    """Parse XML annotations for character bounding boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    characters = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        characters.append((name, xmin, ymin, xmax, ymax))
    
    return characters


def test_on_lp_characters():
    """Test both approaches on LP-characters dataset"""
    
    print("\n" + "="*80)
    print("COMPARISON: TRADITIONAL CV+KNN vs YOLO v8+CNN")
    print("="*80)
    
    # Paths
    dataset_path = Path("datasets/LP-characters")
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "annotations"
    characters_path = dataset_path / "characters_organized"
    
    if not images_path.exists():
        print("❌ Dataset path not found!")
        return
    
    # Load ground truth
    gt_csv = dataset_path / "annotations.csv"
    if not gt_csv.exists():
        print("❌ Ground truth CSV not found!")
        return
    
    gt_df = pd.read_csv(gt_csv)
    print(f"\n📋 Loaded {len(gt_df)} ground truth entries")
    
    # Initialize models & components
    print("\n" + "-"*80)
    print("INITIALIZING MODELS & COMPONENTS")
    print("-"*80)
    
    # Initialize components
    preprocessor = Preprocessor()
    skew_corrector = SkewCorrector()
    
    # Try loading YOLO detector
    try:
        yolo_detector = YOLOPlateDetector(model_path="models/yolov8_plate_detector.pt")
        print("✅ YOLO plate detector loaded")
    except Exception as e:
        print(f"⚠️  YOLO detector not available: {e}")
        yolo_detector = None
    
    # Plate detector (traditional CV)
    plate_detector = PlateDetector()
    print("✅ Traditional plate detector loaded")
    
    # Traditional KNN - Load pre-trained model
    try:
        with open("models/knn_character_recognizer_lp_dataset.pkl", 'rb') as f:
            knn_data = pickle.load(f)
        knn_recognizer = knn_data['model']
        print("✅ Traditional KNN model loaded from pickle")
    except FileNotFoundError:
        print("⚠️  KNN pickle model not found, training new KNN model...")
        knn_recognizer = TraditionalKNNRecognizer(n_neighbors=5)
        knn_recognizer.train(characters_path)
    
    # CNN
    cnn_recognizer = CNNRecognizer()
    cnn_recognizer.train(characters_path, epochs=20, batch_size=32)
    
    # Test on dataset
    print("\n" + "-"*80)
    print("TESTING ON LP-CHARACTERS DATASET")
    print("-"*80)
    
    results = {
        'traditional': [],
        'cnn': []
    }
    
    metrics = {
        'traditional': {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'time': 0, 'correct': 0, 'total': 0
        },
        'cnn': {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'time': 0, 'correct': 0, 'total': 0
        }
    }
    
    all_gt_texts = []
    all_pred_traditional = []
    all_pred_cnn = []
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Test each image
    for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Testing"):
        image_name = row['image']
        gt_text = row['plate_text']
        
        image_path = images_path / image_name
        if not image_path.exists():
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        # Create image-specific results folder
        image_dir = results_dir / Path(image_name).stem
        image_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(image_dir / "original.jpg"), image)
        
        # Get XML annotations
        xml_path = annotations_path / f"{image_name.split('.')[0]}.xml"
        if not xml_path.exists():
            continue
        
        try:
            characters_data = parse_xml_characters(xml_path)
        except:
            continue
        
        if not characters_data:
            continue
        
        # Extract character images from full image
        char_images = []
        char_labels = []
        
        for char_label, xmin, ymin, xmax, ymax in characters_data:
            char_img = image[ymin:ymax, xmin:xmax]
            if char_img.size > 0:
                char_images.append(char_img)
                char_labels.append(char_label)
        
        if not char_images:
            continue
        
        # Preprocess and prepare visualization
        preprocessed_img = preprocessor.preprocess(image)
        skew_corrected_img, angle = skew_corrector.correct_skew(preprocessed_img)
        
        # Draw character segmentation boxes
        segmented_img = image.copy()
        for char_label, xmin, ymin, xmax, ymax in characters_data:
            cv2.rectangle(segmented_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
        # Create subdirectories for traditional and CNN
        trad_dir = image_dir / "traditional_knn"
        trad_dir.mkdir(exist_ok=True)
        
        cnn_dir = image_dir / "yolo_cnn"
        cnn_dir.mkdir(exist_ok=True)
        
        # Save preprocessing steps for both
        cv2.imwrite(str(trad_dir / "preprocessed.jpg"), preprocessed_img)
        cv2.imwrite(str(trad_dir / "skew_corrected.jpg"), skew_corrected_img)
        cv2.imwrite(str(trad_dir / "segmented.jpg"), segmented_img)
        
        cv2.imwrite(str(cnn_dir / "preprocessed.jpg"), preprocessed_img)
        cv2.imwrite(str(cnn_dir / "skew_corrected.jpg"), skew_corrected_img)
        cv2.imwrite(str(cnn_dir / "segmented.jpg"), segmented_img)
        
        # Test Traditional KNN (use original character images)
        t_start = time.time()
        # Extract HOG features manually for loaded model
        from skimage.feature import hog
        
        features_list = []
        for char_img in char_images:
            if len(char_img.shape) == 3:
                char_img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            else:
                char_img_gray = char_img
            
            resized = cv2.resize(char_img_gray, (32, 32))
            resized = resized.astype(np.float32) / 255.0
            features = hog(resized, orientations=9, pixels_per_cell=(4, 4), 
                          cells_per_block=(2, 2), block_norm='L2-Hys')
            features_list.append(features.astype(np.float32))
        
        features_array = np.array(features_list)
        
        # Check if it's old pickle model or new wrapper
        if hasattr(knn_recognizer, 'predict_batch'):
            pred_trad = knn_recognizer.predict_batch(char_images)
        else:
            # Old pickle model - predict directly
            pred_trad = knn_recognizer.predict(features_array)
        
        t_trad = time.time() - t_start
        
        pred_trad_text = ''.join(pred_trad)
        
        # Save Traditional KNN results
        trad_result_img = image.copy()
        cv2.putText(trad_result_img, f"{pred_trad_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(trad_dir / "recognized.jpg"), trad_result_img)
        
        results['traditional'].append({
            'image': image_name,
            'ground_truth': gt_text,
            'prediction': pred_trad_text,
            'correct': pred_trad_text == gt_text,
            'time': t_trad,
            'num_chars': len(char_images)
        })
        
        # Test CNN (use original character images)
        t_start = time.time()
        pred_cnn = cnn_recognizer.predict_batch(char_images)
        t_cnn = time.time() - t_start
        
        pred_cnn_text = ''.join(pred_cnn)
        
        # Save CNN results
        cnn_result_img = image.copy()
        cv2.putText(cnn_result_img, f"{pred_cnn_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(cnn_dir / "recognized.jpg"), cnn_result_img)
        
        results['cnn'].append({
            'image': image_name,
            'ground_truth': gt_text,
            'prediction': pred_cnn_text,
            'correct': pred_cnn_text == gt_text,
            'time': t_cnn,
            'num_chars': len(char_images)
        })
        
        # Accumulate for detailed metrics
        all_gt_texts.append(gt_text)
        all_pred_traditional.append(pred_trad_text)
        all_pred_cnn.append(pred_cnn_text)
        
        # Update metrics
        if pred_trad_text == gt_text:
            metrics['traditional']['correct'] += 1
        metrics['traditional']['total'] += 1
        metrics['traditional']['time'] += t_trad
        
        if pred_cnn_text == gt_text:
            metrics['cnn']['correct'] += 1
        metrics['cnn']['total'] += 1
        metrics['cnn']['time'] += t_cnn
    
    # Calculate final metrics
    for method in ['traditional', 'cnn']:
        total = metrics[method]['total']
        if total > 0:
            accuracy = metrics[method]['correct'] / total
            metrics[method]['accuracy'] = accuracy
            metrics[method]['time'] = metrics[method]['time'] / total
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Traditional CV + KNN:")
    print(f"   ✓ Accuracy: {metrics['traditional']['accuracy']:.2%}")
    print(f"   ✓ Correct: {metrics['traditional']['correct']}/{metrics['traditional']['total']}")
    print(f"   ✓ Avg Time per Image: {metrics['traditional']['time']*1000:.2f}ms")
    
    print(f"\n📊 YOLO v8 + CNN:")
    print(f"   ✓ Accuracy: {metrics['cnn']['accuracy']:.2%}")
    print(f"   ✓ Correct: {metrics['cnn']['correct']}/{metrics['cnn']['total']}")
    print(f"   ✓ Avg Time per Image: {metrics['cnn']['time']*1000:.2f}ms")
    
    # Calculate improvement
    acc_diff = (metrics['cnn']['accuracy'] - metrics['traditional']['accuracy']) * 100
    print(f"\n📈 Improvement (CNN vs Traditional): {acc_diff:+.2f}%")
    
    # Save detailed results
    print(f"\n📊 Results stats:")
    print(f"   Traditional results: {len(results['traditional'])}")
    print(f"   CNN results: {len(results['cnn'])}")
    
    rows_list = []
    
    # Traditional và CNN có thể có số lượng predictions khác nhau
    # Sẽ loop qua tất cả images được test
    for i in range(len(results['traditional'])):
        trad_result = results['traditional'][i]
        
        # Find corresponding CNN result
        cnn_result = None
        for cnn_res in results['cnn']:
            if cnn_res['image'] == trad_result['image']:
                cnn_result = cnn_res
                break
        
        if cnn_result:
            row = {
                'image': trad_result['image'],
                'ground_truth': trad_result['ground_truth'],
                'traditional_pred': trad_result['prediction'],
                'traditional_correct': trad_result['correct'],
                'traditional_time_ms': trad_result['time'] * 1000,
                'cnn_pred': cnn_result['prediction'],
                'cnn_correct': cnn_result['correct'],
                'cnn_time_ms': cnn_result['time'] * 1000,
            }
            rows_list.append(row)
    
    print(f"   Matched rows: {len(rows_list)}")
    
    if rows_list:
        results_df = pd.DataFrame(rows_list)
        results_df.to_csv('comparison_results.csv', index=False)
        print(f"✅ Results saved: {len(rows_list)} rows")
    else:
        print("⚠️  No results to save!")
    print(f"\n💾 Detailed results saved to: comparison_results.csv")
    
    # Save summary metrics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'LP-characters',
        'total_images_tested': metrics['traditional']['total'],
        'traditional_cv_knn': {
            'accuracy': float(metrics['traditional']['accuracy']),
            'correct_predictions': int(metrics['traditional']['correct']),
            'total_predictions': int(metrics['traditional']['total']),
            'avg_time_ms': float(metrics['traditional']['time'] * 1000),
        },
        'yolo_cnn': {
            'accuracy': float(metrics['cnn']['accuracy']),
            'correct_predictions': int(metrics['cnn']['correct']),
            'total_predictions': int(metrics['cnn']['total']),
            'avg_time_ms': float(metrics['cnn']['time'] * 1000),
        },
        'improvement': {
            'accuracy_diff_percent': float(acc_diff),
            'winner': 'CNN' if metrics['cnn']['accuracy'] > metrics['traditional']['accuracy'] else 'Traditional CV+KNN'
        }
    }
    
    with open('comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary metrics saved to: comparison_summary.json")
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETED")
    print("="*80)


if __name__ == "__main__":
    test_on_lp_characters()
