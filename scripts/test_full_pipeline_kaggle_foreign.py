"""
Test Full Pipeline on Kaggle Foreign Test Dataset
"""

import cv2
import sys
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

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

def test_full_pipeline_on_kaggle_foreign():
    print("🧪 TEST FULL PIPELINE ON KAGGLE FOREIGN TEST DATASET")
    print("=" * 70)

    # Load ground truth
    gt_path = Path("datasets/kaggle_foreign/test_annotations.csv")
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
        image_name = row['filename']
        gt_text = row['plate_text']
        bbox_str = f"{row['xmin']},{row['ymin']},{row['xmax']},{row['ymax']}"
        image_path = Path("datasets/kaggle_foreign/test") / image_name

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

        # Segment characters
        char_images = character_recognizer.segment_characters(skew_corrected)
        
        if not char_images:
            print("  ❌ No characters segmented")
            results.append({'image': image_name, 'gt': gt_text, 'pred': '', 'correct': False})
            continue

        # Create segmented image (copy of skew_corrected, since no XML for positions)
        segmented_img = skew_corrected.copy()
        cv2.imwrite(str(image_dir / "segmented.jpg"), segmented_img)

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
    test_full_pipeline_on_kaggle_foreign()