# 🚗 License Plate Recognition System

Hệ thống nhận dạng biển số xe so sánh **Traditional CV+KNN** vs **YOLO v8+CNN** trên dataset LP-characters.

---

## 📊 Key Results

| Approach | Accuracy | Speed | Correct |
|----------|----------|-------|---------|
| **Traditional CV+KNN** | 82.30% | 52.78ms | 172/209 |
| **YOLO v8+CNN** | 89.00% | 2.76ms | 186/209 |
| **Winner** | CNN +6.70% | CNN 19.1x faster | CNN |

---

## 🚀 Quick Start

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Compare both approaches
python scripts/compare_traditional_vs_yolo.py

# 3. Generate reports
python scripts/generate_comparison_report.py
python scripts/visualize_comparison.py
```

Results sẽ save tại:
- `comparison_results.csv` - Chi tiết từng image
- `comparison_summary.json` - Metrics tổng hợp  
- `COMPARISON_REPORT.txt/md` - Report đầy đủ
- `comparison_visualization.png` - Charts
- `results/[image_name]/` - Processing steps (original, preprocessed, segmented, recognized)

---

## 🏗️ Architecture

### Traditional Approach
```
Input Image
    ↓
Preprocessing (grayscale, blur)
    ↓
Character Segmentation
    ↓
HOG Feature Extraction
    ↓
KNN Classification
    ↓
Output
```

### Deep Learning Approach  
```
Input Image
    ↓
Preprocessing
    ↓
Character Segmentation
    ↓
CNN Feature Learning
    ↓
Character Recognition
    ↓
Output
```

---

## 📁 Project Structure

```
src/
├── preprocessor.py          # Image preprocessing
├── plate_detector.py        # Plate detection (contour-based)
├── skew_corrector.py        # Angle correction
├── character_recognizer.py  # Character segmentation + KNN
├── yolo_plate_detector.py   # YOLO v8 plate detector
└── cnn_recognizer.py        # CNN model (SimpleCNN + training)

scripts/
├── compare_traditional_vs_yolo.py    # Main comparison script
├── generate_comparison_report.py      # Report generation
├── visualize_comparison.py            # Visualization & analysis
├── train_knn_from_lp_dataset.py      # KNN training
├── extract_characters_from_lp_dataset.py  # Character extraction
├── prepare_lp_dataset_for_yolo.py    # YOLO dataset prep
└── train_yolov8_fast.py              # YOLO training

datasets/
├── LP-characters/           # Main dataset
│   ├── images/
│   ├── annotations/
│   └── characters_organized/
└── lp_characters_yolo/      # YOLO format

models/
├── knn_character_recognizer_lp_dataset.pkl  # Trained KNN
├── yolov8_plate_detector.pt                 # YOLO model
└── yolov8_yolov8n_mac/                      # YOLO training output

results/                      # Output folder
├── [image_name]/
│   ├── original.jpg
│   ├── traditional_knn/
│   │   ├── preprocessed.jpg
│   │   ├── skew_corrected.jpg
│   │   ├── segmented.jpg
│   │   └── recognized.jpg
│   └── yolo_cnn/
│       ├── preprocessed.jpg
│       ├── skew_corrected.jpg
│       ├── segmented.jpg
│       └── recognized.jpg
```

---

## 📚 Core Components

| Component | Type | Purpose |
|-----------|------|---------|
| **Preprocessor** | OpenCV | Grayscale, blur, denoise |
| **PlateDetector** | CV | Contour-based detection |
| **SkewCorrector** | CV | Angle correction (moments/hough/contour) |
| **YOLOPlateDetector** | DL | YOLO v8 detection |
| **CharacterRecognizer** | CV+ML | Segmentation + HOG + KNN |
| **CNNRecognizer** | DL | SimpleCNN (3-layer conv + 2-layer dense) |

---

## 🎯 Key Scripts

| Script | Purpose |
|--------|---------|
| `compare_traditional_vs_yolo.py` | Main comparison on 209 test images |
| `generate_comparison_report.py` | Generate TXT/MD reports |
| `visualize_comparison.py` | Create comparison charts |
| `train_knn_from_lp_dataset.py` | Train KNN with augmentation |
| `extract_characters_from_lp_dataset.py` | Extract characters from XML |
| `prepare_lp_dataset_for_yolo.py` | Convert to YOLO format |
| `train_yolov8_fast.py` | Train YOLO (Mac optimized) |

---

## 📊 Dataset Info

**LP-characters Dataset:**
- 209 test images with character-level annotations
- 36 character classes (0-9, A-Z)
- XML format with bounding boxes
- Clean, well-organized data

**Training Data:**
- Traditional KNN: 2,026+ character samples with augmentation
- CNN: Same 2,026+ samples, 80/20 train/val split
- 20 epochs training, Adam optimizer

---

## 🔧 Advanced Usage

### Train Custom Models

```bash
# Extract characters from dataset
python scripts/extract_characters_from_lp_dataset.py

# Train KNN with augmentation
python scripts/train_knn_from_lp_dataset.py

# Prepare YOLO dataset
python scripts/prepare_lp_dataset_for_yolo.py

# Train YOLO (Mac M4 optimized)
python scripts/train_yolov8_fast.py
```

### Analyze Results

```bash
# View detailed comparison
python scripts/visualize_comparison.py

# Check specific images in results/[name]/
ls results/0000/traditional_knn/
ls results/0000/yolo_cnn/
```

---

## 💡 Key Insights

✅ **CNN Advantages:**
- 6.70% higher accuracy (89.00% vs 82.30%)
- 19.1x faster inference (2.76ms vs 52.78ms)
- Better feature learning with deep neural networks

✅ **Traditional Advantages:**
- Lightweight, no GPU needed
- Explainable (HOG features visible)
- Good baseline comparison

✅ **Recommendations:**
- **Production**: Use CNN for best accuracy + speed
- **Resource-limited**: Use Traditional CV+KNN
- **Comparison**: Both approaches valuable for benchmarking

---

## 📈 Performance Metrics

Both methods trained on **2,026 character samples** from LP-characters:

- **Character-level accuracy**: CNN 89.00%, Traditional 82.30%
- **Processing time**: CNN 2.76ms/image, Traditional 52.78ms/image
- **GPU**: Optional (CNN faster on GPU/MPS)
- **Inference**: Both real-time capable

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.10+
- scikit-learn 1.5+
- YOLO v8 (ultralytics)
- See `requirements.txt` for full list

---

## 📝 Output Files

After running comparison:

```
comparison_results.csv           # Per-image results
comparison_summary.json          # Aggregate metrics
COMPARISON_REPORT.txt           # Detailed text report
COMPARISON_REPORT.md            # Markdown version
comparison_visualization.png    # Charts (accuracy, speed, etc)
results/                        # Folder with images
├── 0000/original.jpg
├── 0000/traditional_knn/{4 images}
├── 0000/yolo_cnn/{4 images}
├── 0001/...
└── ...
```

---

**Status**: ✅ Ready to use | **Last Update**: 2025-12-08
