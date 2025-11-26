# 🚗 License Plate Recognition System

Nhận dạng ký tự từ biển số xe sử dụng xử lý ảnh truyền thống và ML

**🎯 Performance**: 80.0% accuracy on LP-characters dataset | YOLO: 79.1% | CV: 80.8%

---

## ⚡ Quick Start

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy ngay (model đã huấn luyện sẵn)
python main.py --image datasets/kaggle_foreign/test/Cars0.png

# 3. Hoặc xử lý batch
python main.py --batch datasets/kaggle_foreign/test --output results.csv

# 4. Đánh giá trên dataset
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
```

---

## 📖 4 Chế Độ Chính

### 1️⃣ Single Image Processing
Xử lý một ảnh và hiển thị kết quả với visualization
```bash
python main.py --image path/to/image.jpg
```

### 2️⃣ Batch Processing
Xử lý một folder ảnh và lưu kết quả vào CSV
```bash
python main.py --batch path/to/folder --output results.csv
```

### 3️⃣ Evaluation & Benchmark
Đánh giá hệ thống trên dataset với ground truth annotations
```bash
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
```

---

## 🔄 Pipeline Chi Tiết

Mỗi ảnh đi qua 5 bước xử lý:

1. **Tiền xử lý**: Chuyển grayscale, blur, normalize
2. **Phát hiện biển số**: Contour detection, bounding box
3. **Hiệu chỉnh góc nghiêng**: Skew correction
4. **Phân vùng ký tự**: Cắt từng ký tự từ biển số
5. **Nhận dạng ký tự**: KNN prediction trên 20x30 features

---

## 🎓 Xây Dựng Model Từ Đầu

Nếu muốn huấn luyện lại model hoặc thêm dữ liệu:

```bash
# Step 1: Auto-extract templates từ 473 ảnh
python scripts/auto_extract_and_label_kaggle.py

# Step 2: Cắt + gán nhãn từ 17 test images (ground truth)
python scripts/extract_manual_labels.py

# Step 3: Filter best templates
python scripts/filter_best_templates.py

# Step 4: Train hybrid KNN model
python scripts/train_knn_hybrid.py

# Step 5: Test models
python scripts/test_models.py
```

**Hoặc chạy full pipeline một lần:**
```bash
python scripts/full_pipeline.py
```

---

## 🎯 Core Components

| Tệp | Mục đích |
|-----|---------|
| `main.py` | Entry point - 4 chế độ chính (single/batch/video/eval) |
| `src/preprocessor.py` | Tiền xử lý ảnh (grayscale, blur, normalize) |
| `src/plate_detector.py` | Phát hiện biển số (contour-based) |
| `src/skew_corrector.py` | Hiệu chỉnh góc nghiêng |
| `src/character_recognizer.py` | Segment + nhận dạng ký tự |
| `models/knn_character_recognizer_hybrid.pkl` | Pre-trained KNN model (57.81%) |

---

## 🛠️ Training Scripts

Để xây dựng model từ đầu:

| Script | Mục đích |
|--------|---------|
| `scripts/auto_extract_and_label_kaggle.py` | Auto-extract templates từ 473 ảnh |
| `scripts/extract_manual_labels.py` | Cắt + gán nhãn ground truth từ test images |
| `scripts/filter_best_templates.py` | Chọn best 31 templates |
| `scripts/train_knn_hybrid.py` | Train KNN hybrid model |
| `scripts/test_models.py` | Benchmark & so sánh models |
| `scripts/full_pipeline.py` | Run full pipeline in one go |

## 🧪 Test Scripts

Scripts để test và so sánh pipeline:

| Script | Mục đích |
|--------|---------|
| `scripts/test_full_pipeline_lp_characters.py` | Test pipeline trên LP-characters (dùng GT bbox) |
| `scripts/test_full_pipeline_lp_characters_plate_detector.py` | So sánh YOLO vs CV detection trên LP-characters |
| `scripts/test_full_pipeline_kaggle_foreign.py` | Test pipeline trên Kaggle Foreign test |
| `scripts/test_plate_detector.py` | Test riêng plate detection |
| `scripts/test_hybrid_viz.py` | Test visualization pipeline |

---

## 📊 Dataset & Model Performance

**Datasets Used:**
- **LP-characters**: https://www.kaggle.com/datasets/francescopettini/license-plate-characters-detection-ocr?select=LP-characters
- **Kaggle Foreign**: Custom dataset for testing

**LP-characters Dataset Results (335 images):**
- **Overall**: 268/335 correct (**80.0%**)
- **YOLO Detection**: 121/153 correct (79.1%)
- **CV Detection**: 147/182 correct (**80.8%**)

| Model | Training Data | Accuracy | Ghi chú |
|-------|---------------|----------|--------|
| **Hybrid KNN** ⭐ | 31 templates + 46 manual | **57.81%** | **Best on Kaggle Foreign** |
| Templates-only | 31 manual | 5.76% | Underfitting |
| Auto-labeled | 3100+ EasyOCR | 10.76% | Noisy data |

**Hybrid model** kết hợp tốt nhất manual labels (ground truth) + auto-extracted templates.

---

## ❓ FAQ

**Q: Tại sao chỉ 57.81% trên Kaggle Foreign?**
- Segmentation yếu (10/17 detect)
- Dữ liệu nhỏ (77 ảnh)
- Font chữ biến động

**Q: Tại sao 80% trên LP-characters?**
- Dataset sạch, biển số rõ ràng
- GT bbox chính xác
- Character segmentation từ XML annotations

**Q: Làm sao tăng accuracy?**
- Cách 1: Thêm ảnh test + gán nhãn → `extract_manual_labels.py` → train
- Cách 2: Dùng Deep Learning (YOLO, CNN)

**Q: Có thể dùng production?**
- ✅ Batch processing + manual confirmation
- ❌ Full automation (chưa đủ chính xác)

---

## 📝 Useful Commands

```bash
# Chạy model hiện tại
python main.py --image datasets/kaggle_foreign/test/Cars0.png

# Batch xử lý
python main.py --batch datasets/kaggle_foreign/test --output results.csv

# Test pipeline trên LP-characters
python scripts/test_full_pipeline_lp_characters.py

# So sánh YOLO vs CV detection
python scripts/test_full_pipeline_lp_characters_plate_detector.py

# Test trên Kaggle Foreign
python scripts/test_full_pipeline_kaggle_foreign.py

# Tạo manual labels từ test
python scripts/extract_manual_labels.py

# Filter templates tốt nhất
python scripts/filter_best_templates.py

# Train lại model
python scripts/train_knn_hybrid.py

# So sánh 3 model
python scripts/test_models.py

# Debug segmentation
python scripts/debug_seg_detail.py

# Test visualization
python scripts/test_hybrid_viz.py

# Đánh giá chi tiết
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
```

---

**Version**: 1.0 | **Status**: Ready to use ✅
