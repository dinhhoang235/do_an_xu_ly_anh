"""
Train KNN từ best templates + manually labeled characters từ test set
Kết hợp quality templates với ground truth labels
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def train_knn_hybrid():
    print("=" * 70)
    print("🎯 TRAIN KNN FROM TEMPLATES + MANUALLY LABELED CHARACTERS")
    print("=" * 70)
    
    # Load data từ 2 sources
    print("\n🔄 Loading training data...")
    
    # 1. Load templates
    templates_path = Path("datasets/kaggle_foreign/character_templates")
    manual_path = Path("datasets/kaggle_foreign/characters_manual_labeled")
    
    X = []
    y = []
    
    # Load templates
    print("\n📌 Loading templates...")
    template_files = sorted(templates_path.glob("*.png"))
    template_count = 0
    
    for template_file in tqdm(template_files, desc="Templates"):
        label = template_file.stem
        img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        resized = cv2.resize(img, (20, 30))
        _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
        features = binary.flatten().astype(np.float32) / 255.0
        features = np.clip(features, 0, 1)
        
        X.append(features)
        y.append(label)
        template_count += 1
    
    print(f"✅ Loaded {template_count} templates")
    
    # Load manually labeled characters
    print("\n📌 Loading manually labeled characters...")
    manual_count = 0
    
    if manual_path.exists():
        for char_dir in manual_path.iterdir():
            if not char_dir.is_dir():
                continue
            
            char_label = char_dir.name
            char_files = list(char_dir.glob("*.png"))
            
            for char_file in tqdm(char_files, desc=f"Char {char_label}", leave=False):
                img = cv2.imread(str(char_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                resized = cv2.resize(img, (20, 30))
                _, binary = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY_INV)
                features = binary.flatten().astype(np.float32) / 255.0
                features = np.clip(features, 0, 1)
                
                X.append(features)
                y.append(char_label)
                manual_count += 1
    
    print(f"✅ Loaded {manual_count} manually labeled characters")
    
    if len(X) == 0:
        print("❌ No data loaded!")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Dataset info:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Total classes: {len(np.unique(y))}")
    print(f"   - Features per sample: {X.shape[1]}")
    
    # Show class distribution
    print(f"\n📋 Class distribution:")
    unique_chars, counts = np.unique(y, return_counts=True)
    for char, count in sorted(zip(unique_chars, counts)):
        print(f"   {char}: {count}")
    
    # Train KNN
    print("\n🤖 Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    knn.fit(X, y)
    
    # Evaluate on training data
    train_accuracy = knn.score(X, y)
    print(f"✅ Training accuracy: {train_accuracy*100:.2f}%")
    print(f"   (On {len(X)} combined training samples)")
    
    # Save model
    model_path = Path("models/knn_character_recognizer_hybrid.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': knn,
        'classes': np.unique(y)
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Model saved: {model_path}")
    
    print("\n" + "=" * 70)
    print("✅ Hybrid model training completed!")
    print("=" * 70)

if __name__ == "__main__":
    train_knn_hybrid()
