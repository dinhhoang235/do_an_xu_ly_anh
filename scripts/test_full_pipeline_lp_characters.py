"""
Test Full Pipeline on LP-characters Dataset
"""

import cv2
import sys
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import xml.etree.ElementTree as ET

def parse_xml_characters(xml_path):
    """Parse XML to get character bounding boxes and labels"""
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

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yolo_plate_detector import YOLOPlateDetector
from src.character_recognizer import CharacterRecognizer
from src.preprocessor import Preprocessor
from src.skew_corrector import SkewCorrector

def extract_features(char_img):
    """Extract HOG features for KNN"""
    from skimage.feature import hog
    # Convert to grayscale if needed
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(char_img, (32, 32))
    features = hog(resized, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys')
    return features.astype(np.float32)

def test_full_pipeline_on_lp_characters():
    print("🧪 TEST FULL PIPELINE ON LP-CHARACTERS DATASET")
    print("=" * 70)

    # Load ground truth
    gt_path = Path("datasets/LP-characters/annotations.csv")
    if not gt_path.exists():
        print("❌ Ground truth not found")
        return

    gt_df = pd.read_csv(gt_path)
    print(f"📋 Loaded {len(gt_df)} ground truth entries")

    # Load models
    yolo_detector = YOLOPlateDetector(model_path="models/yolov8_plate_detector.pt")
    with open("models/knn_character_recognizer_lp_dataset.pkl", 'rb') as f:
        knn_data = pickle.load(f)
    knn_model = knn_data['model']

    # Initialize components
    preprocessor = Preprocessor()
    skew_corrector = SkewCorrector()
    character_recognizer = CharacterRecognizer()

    results = []
    correct = 0
    total = 0

    for idx, row in gt_df.iterrows():
        image_name = row['image']
        gt_text = row['plate_text']
        bbox_str = row['bbox']
        image_path = Path("datasets/LP-characters/images") / image_name

        if not image_path.exists():
            continue

        print(f"\n🔍 Testing: {image_name} (GT: {gt_text})")

        # Load image
        image = cv2.imread(str(image_path))

        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        image_dir = results_dir / Path(image_name).stem
        image_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(image_dir / "original.jpg"), image)

        # Parse bbox
        bbox = [int(x) for x in bbox_str.split(',')]
        x1, y1, x2, y2 = bbox
        plate_img = image[y1:y2, x1:x2]
        cv2.imwrite(str(image_dir / "detected_plate.jpg"), plate_img)

        # Preprocess
        preprocessed = preprocessor.preprocess(plate_img)
        cv2.imwrite(str(image_dir / "preprocessed.jpg"), preprocessed)

        # Skew correct
        skew_corrected, angle = skew_corrector.correct_skew(preprocessed)
        cv2.imwrite(str(image_dir / "skew_corrected.jpg"), skew_corrected)

        # Get character annotations from XML
        xml_path = Path("datasets/LP-characters/annotations") / f"{image_name.split('.')[0]}.xml"
        if not xml_path.exists():
            print("  ❌ XML annotation not found")
            results.append({'image': image_name, 'gt': gt_text, 'pred': '', 'correct': False})
            continue
        
        characters_data = parse_xml_characters(xml_path)
        
        if not characters_data:
            print("  ❌ No characters in XML")
            results.append({'image': image_name, 'gt': gt_text, 'pred': '', 'correct': False})
            continue

        # Create segmented image with blue rectangles
        segmented_img = plate_img.copy()
        for char_label, xmin, ymin, xmax, ymax in characters_data:
            adj_xmin = xmin - x1
            adj_ymin = ymin - y1
            adj_xmax = xmax - x1
            adj_ymax = ymax - y1
            cv2.rectangle(segmented_img, (adj_xmin, adj_ymin), (adj_xmax, adj_ymax), (255, 0, 0), 2)
        cv2.imwrite(str(image_dir / "segmented.jpg"), segmented_img)
        
        # Extract character images using XML bboxes (from full image)
        char_images = []
        for char_label, xmin, ymin, xmax, ymax in characters_data:
            char_img = image[ymin:ymax, xmin:xmax]
            if char_img.size > 0:
                char_images.append(char_img)

        features_list = [extract_features(char) for char in char_images]
        features_array = np.array(features_list)
        predictions = knn_model.predict(features_array)
        pred_text = ''.join(predictions)

        # Post-process
        import re
        pred_text = re.sub(r'[^A-Z0-9]', '', pred_text.upper())
        if not (3 <= len(pred_text) <= 10):
            pred_text = ''

        is_correct = pred_text == gt_text
        if is_correct:
            correct += 1
        total += 1

        # Save recognized image
        final_img = plate_img.copy()
        cv2.putText(final_img, f"{pred_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imwrite(str(image_dir / "recognized.jpg"), final_img)

        print(f"  ✅ Predicted: {pred_text} | Correct: {is_correct}")

        results.append({
            'image': image_name,
            'gt': gt_text,
            'pred': pred_text,
            'correct': is_correct
        })

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n📊 RESULTS: {correct}/{total} correct ({accuracy:.1f}%)")

    print("💾 Results saved to results/ folder")

if __name__ == "__main__":
    test_full_pipeline_on_lp_characters()