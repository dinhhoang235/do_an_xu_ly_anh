"""
Train KNN từ characters đã được extract và organize từ LP-characters dataset
Sử dụng real data từ YOLO detection + character segmentation với XML labels
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
from skimage.feature import hog
from sklearn.svm import SVC

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def augment_image(img, augmentations=2):
    """
    Create augmented versions of an image
    Returns list of images: [original, aug1, aug2, ...]
    """
    augmented = [img]  # include original

    for _ in range(augmentations):
        aug_img = img.copy()

        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        h, w = aug_img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=255)

        # Random noise
        noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
        aug_img = cv2.add(aug_img, noise)

        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.randint(-10, 10)    # brightness
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        augmented.append(aug_img)

    return augmented

def train_knn_from_lp_dataset():
    print("=" * 70)
    print("🎯 TRAIN KNN FROM LP-CHARACTERS DATASET")
    print("=" * 70)

    # Load data từ organized LP-characters
    print("\n🔄 Loading training data...")

    organized_path = Path("/Users/hoang/Documents/code/license_plate_system/datasets/LP-characters/characters_organized")

    if not organized_path.exists():
        print(f"❌ Organized LP-characters not found: {organized_path}")
        print("   Run: python scripts/extract_characters_from_lp_dataset.py first")
        return

    X = []
    y = []

    # Load organized characters
    print("\n📌 Loading organized LP-characters...")
    total_samples = 0

    for char_dir in organized_path.iterdir():
        if not char_dir.is_dir():
            continue

        char_label = char_dir.name
        char_files = list(char_dir.glob("*.png"))

        if len(char_files) == 0:
            continue

        print(f"   {char_label}: {len(char_files)} samples")

        for char_file in char_files:
            img = cv2.imread(str(char_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Augment image (original + 2 augmented versions)
            augmented_images = augment_image(img, augmentations=2)

            for aug_img in augmented_images:
                # Resize to standard size for HOG
                resized = cv2.resize(aug_img, (32, 32))

                # Extract HOG features
                features = hog(resized, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm='L2-Hys')
                features = features.astype(np.float32)

                X.append(features)
                y.append(char_label)
                total_samples += 1

    if total_samples == 0:
        print("❌ No character data found!")
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

    # Split data for validation
    print("\n🔄 Splitting data for validation...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails due to small classes, split without stratification
        print("   ⚠️  Some classes have too few samples, splitting without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Train KNN with different k values
    print("\n🤖 Training KNN models...")

    best_accuracy = 0
    best_k = 1
    best_model = None

    k_values = [1, 3, 5, 7, 9]

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)

        # Evaluate on test set
        test_accuracy = knn.score(X_test, y_test)
        print(f"   k={k}: Test accuracy = {test_accuracy*100:.2f}%")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_k = k
            best_model = knn

    print(f"\n🏆 Best model: k={best_k}, Accuracy = {best_accuracy*100:.2f}%")

    # Try SVM for comparison
    print("\n🤖 Training SVM for comparison...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    print(f"   SVM accuracy = {svm_accuracy*100:.2f}%")

    if svm_accuracy > best_accuracy:
        best_accuracy = svm_accuracy
        best_k = 'SVM'
        best_model = svm
        print("   🎉 SVM performs better!")

    print(f"\n🏆 Final best model: {best_k}, Accuracy = {best_accuracy*100:.2f}%")

    # Save best model
    model_path = Path("models/knn_character_recognizer_lp_dataset.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': best_model,
        'classes': np.unique(y),
        'k': best_k,
        'accuracy': best_accuracy,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n💾 Model saved: {model_path}")

    # Compare with old model if exists
    old_model_path = Path("models/knn_character_recognizer_dataset.pkl")
    if old_model_path.exists():
        print("\n🔄 Comparing with old model...")
        try:
            with open(old_model_path, 'rb') as f:
                old_model_data = pickle.load(f)

            old_accuracy = old_model_data.get('accuracy', 'unknown')
            improvement = best_accuracy - old_accuracy if isinstance(old_accuracy, (int, float)) else "unknown"

            print(f"   Old model accuracy: {old_accuracy}")
            print(f"   New model accuracy: {best_accuracy*100:.2f}%")
            if isinstance(improvement, (int, float)):
                print(f"   Improvement: {improvement*100:+.2f}%")
        except:
            print("   Could not load old model for comparison")

    print("\n" + "=" * 70)
    print("✅ LP-characters KNN training completed!")
    print(f"   Model: {best_k}, Accuracy: {best_accuracy*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    train_knn_from_lp_dataset()