"""
Test 3 models: Augmented vs Templates vs Hybrid
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pickle
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer

def load_model(model_path):
    """Load model từ file"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            return data.get('model', data)
        return data

def test_all_models():
    """Test cả 3 models"""
    print("=" * 70)
    print("🚗 COMPARE 3 KNN MODELS")
    print("   1. Augmented (3100 images)")
    print("   2. Templates (31 templates)")
    print("   3. Hybrid (31 templates + 46 manual labeled)")
    print("=" * 70)
    
    # Load annotations
    test_csv = "datasets/kaggle_foreign/test_annotations.csv"
    df = pd.read_csv(test_csv)
    
    print(f"\n📊 Test set: {len(df)} images")
    
    # Load recognizer
    recognizer = CharacterRecognizer()
    
    # Load 3 models
    print("\n🔄 Loading models...")
    models = {}
    
    try:
        models['augmented'] = load_model("models/knn_character_recognizer.pkl")
        print("✅ Augmented model loaded")
    except:
        print("❌ Augmented model failed")
    
    try:
        models['templates'] = load_model("models/knn_character_recognizer_templates.pkl")
        print("✅ Templates model loaded")
    except:
        print("❌ Templates model failed")
    
    try:
        models['hybrid'] = load_model("models/knn_character_recognizer_hybrid.pkl")
        print("✅ Hybrid model loaded")
    except:
        print("❌ Hybrid model failed")
    
    if len(models) == 0:
        print("❌ No models loaded")
        return
    
    # Test each image
    results = []
    total_processing_time = 0
    image_count = 0
    
    print("\n🔄 Testing...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        image_name = row['filename']
        ground_truth = row['plate_text']
        
        img_path = f"datasets/kaggle_foreign/test/{image_name}"
        if not Path(img_path).exists():
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        try:
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            
            plate_img = img[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            char_images = recognizer.segment_characters(plate_img)
            
            if len(char_images) == 0:
                continue
            
            # Extract features
            features_list = []
            for char_img in char_images:
                resized = cv2.resize(char_img, (20, 30))
                _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
                features = binary.flatten().astype(np.float32) / 255.0
                features = np.clip(features, 0, 1)
                features_list.append(features)
            
            features_array = np.array(features_list)
            
            # Test each model
            model_results = {'image': image_name, 'ground_truth': ground_truth, 'detected': len(char_images)}
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(features_array)
                    result = ''.join(pred)
                    accuracy = sum(1 for i, c in enumerate(ground_truth) if i < len(pred) and pred[i] == c) / max(len(ground_truth), len(pred))
                    model_results[f'{model_name}_result'] = result
                    model_results[f'{model_name}_acc'] = accuracy
                except:
                    model_results[f'{model_name}_result'] = ''
                    model_results[f'{model_name}_acc'] = 0.0
            
            # Kết thúc đo thời gian
            end_time = time.time()
            processing_time = end_time - start_time
            model_results['processing_time'] = processing_time
            
            total_processing_time += processing_time
            image_count += 1
            
            results.append(model_results)
        
        except:
            continue
    
    # Tính FPS
    avg_processing_time = total_processing_time / image_count if image_count > 0 else 0
    fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{len(results_df)} images with detected characters:\n")
    
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['image']}")
        print(f"   Ground truth: {row['ground_truth']}")
        
        if 'augmented_result' in row and row['augmented_result']:
            print(f"   Augmented:    {row['augmented_result']:12} | {row['augmented_acc']*100:5.1f}%")
        if 'templates_result' in row and row['templates_result']:
            print(f"   Templates:    {row['templates_result']:12} | {row['templates_acc']*100:5.1f}%")
        if 'hybrid_result' in row and row['hybrid_result']:
            print(f"   Hybrid:       {row['hybrid_result']:12} | {row['hybrid_acc']*100:5.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 SUMMARY")
    print("=" * 70)
    
    for model_name in ['augmented', 'templates', 'hybrid']:
        if f'{model_name}_acc' in results_df.columns:
            avg_acc = results_df[f'{model_name}_acc'].mean()
            perfect = (results_df[f'{model_name}_acc'] == 1.0).sum()
            partial = (results_df[f'{model_name}_acc'] > 0).sum()
            
            print(f"\n{model_name.upper()}:")
            print(f"   Avg accuracy: {avg_acc*100:.2f}%")
            print(f"   Perfect (100%): {perfect}/{len(results_df)}")
            print(f"   Partial (>0%): {partial}/{len(results_df)}")
    
    # Performance metrics
    print("\n" + "=" * 70)
    print("⚡ PERFORMANCE METRICS")
    print("=" * 70)
    print(f"\nAverage processing time: {avg_processing_time*1000:.2f} ms")
    print(f"FPS (Frames Per Second): {fps:.2f} FPS")
    print(f"Total images processed: {image_count}")
    print(f"Total processing time: {total_processing_time:.2f}s")
    
    # Compare
    print("\n" + "=" * 70)
    print("🏆 COMPARISON")
    print("=" * 70)
    
    comparisons = []
    for model_name in ['augmented', 'templates', 'hybrid']:
        if f'{model_name}_acc' in results_df.columns:
            avg_acc = results_df[f'{model_name}_acc'].mean()
            comparisons.append((model_name, avg_acc))
    
    comparisons.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nRanking by average accuracy:")
    for i, (name, acc) in enumerate(comparisons, 1):
        print(f"  {i}. {name:12} : {acc*100:.2f}%")

if __name__ == "__main__":
    test_all_models()
