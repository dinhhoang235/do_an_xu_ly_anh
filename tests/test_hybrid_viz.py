"""
Test Hybrid Model with Visualization - Shows images and detailed results
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer

def load_hybrid_model():
    """Load the best hybrid model"""
    with open('models/knn_character_recognizer_hybrid.pkl', 'rb') as f:
        return pickle.load(f)['model']

def extract_features(char_img):
    """Extract features from character image"""
    resized = cv2.resize(char_img, (20, 30))
    _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
    features = binary.flatten().astype(np.float32) / 255.0
    features = np.clip(features, 0, 1)
    return features

def calculate_accuracy(predicted, ground_truth):
    """Calculate character-level accuracy"""
    if len(ground_truth) == 0:
        return 0.0
    
    correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return (correct / len(ground_truth)) * 100

def visualize_result(img_path, ground_truth, bbox, recognizer, model):
    """Visualize segmentation and recognition results"""
    
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Crop plate using bounding box
    x1, y1, x2, y2 = bbox
    plate_img = img[y1:y2, x1:x2]
    if plate_img.size == 0:
        return None
    
    # Segment characters
    char_images = recognizer.segment_characters(plate_img)
    
    if len(char_images) == 0:
        return None
    
    # Extract features and recognize
    features_list = [extract_features(char) for char in char_images]
    features_array = np.array(features_list)
    predictions = model.predict(features_array)
    predicted = ''.join(predictions)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predicted, ground_truth)
    
    # Create visualization
    display_img = plate_img.copy()
    h, w = display_img.shape[:2]
    
    # Add title with results
    title = f"GT: {ground_truth} | Pred: {predicted} | Acc: {accuracy:.1f}%"
    cv2.putText(display_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw detected characters with their predictions
    char_width = w // len(char_images) if len(char_images) > 0 else w
    for i, (char_img, pred) in enumerate(zip(char_images, predictions)):
        x = i * char_width
        cv2.rectangle(display_img, (x, 50), (x + char_width, h), (0, 255, 0), 2)
        cv2.putText(display_img, pred, (x + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show
    cv2.imshow(f"{Path(img_path).name}", display_img)
    cv2.waitKey(1500)  # Show for 1.5 seconds
    cv2.destroyAllWindows()
    
    return {
        'filename': Path(img_path).name,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'accuracy': accuracy,
        'char_count': len(predicted)
    }

def main():
    print("=" * 80)
    print("🚗 HYBRID MODEL TEST WITH VISUALIZATION")
    print("=" * 80)
    
    # Load model
    print("\n🔄 Loading hybrid model...")
    model = load_hybrid_model()
    print("✅ Model loaded")
    
    # Create recognizer
    recognizer = CharacterRecognizer()
    
    # Load annotations
    print("📋 Loading annotations...")
    df = pd.read_csv('datasets/kaggle_foreign/test_annotations.csv')
    print(f"✅ Loaded {len(df)} annotations")
    
    # Test on all images
    print(f"\n📊 Testing {len(df)} images:\n")
    
    results = []
    detected_count = 0
    
    for idx, row in df.iterrows():
        filename = row['filename']
        ground_truth = row['plate_text']
        img_path = f"datasets/kaggle_foreign/test/{filename}"
        
        if not Path(img_path).exists():
            continue
        
        # Get bounding box
        try:
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            bbox = (x1, y1, x2, y2)
        except:
            continue
        
        result = visualize_result(img_path, ground_truth, bbox, recognizer, model)
        
        if result:
            results.append(result)
            detected_count += 1
            status = "✅" if result['accuracy'] == 100 else "⚠️ "
            print(f"{status} {filename:20s} GT: {ground_truth:15s} Pred: {result['predicted']:15s} Acc: {result['accuracy']:6.1f}%")
        else:
            print(f"❌ {filename:20s} No characters detected")
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    if results:
        total_acc = sum(r['accuracy'] for r in results) / len(results)
        perfect = sum(1 for r in results if r['accuracy'] == 100)
        partial = sum(1 for r in results if r['accuracy'] > 0)
        
        print(f"\n✅ Images with detected characters: {detected_count}/{len(df)}")
        print(f"\n📊 HYBRID MODEL PERFORMANCE:")
        print(f"   Average accuracy: {total_acc:.2f}%")
        print(f"   Perfect (100%):   {perfect}/{len(results)} images")
        print(f"   Partial (>0%):    {partial}/{len(results)} images")
        print(f"\n📌 Best predictions:")
        best = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:3]
        for i, r in enumerate(best, 1):
            print(f"   {i}. {r['filename']:20s} {r['accuracy']:6.1f}% - {r['ground_truth']} → {r['predicted']}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
