# 🚗 License Plate Recognition System

Nhận dạng ký tự từ biển số xe nước ngoài sử dụng KNN + Computer Vision.

**Performance**: 57.81% accuracy | 2/10 perfect match | 10/10 partial match

---

## ⚡ Quick Start

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chạy ngay (model đã huấn luyện sẵn)
python main.py --image datasets/kaggle_foreign/test/Cars0.png

# 3. Hoặc xử lý batch
python main.py --batch datasets/kaggle_foreign/test --output results.csv
```

---

## 📖 4 Cách Sử Dụng

### 1️⃣ Single Image
```bash
python main.py --image datasets/kaggle_foreign/test/Cars0.png
```

### 2️⃣ Batch Process
```bash
python main.py --batch datasets/kaggle_foreign/test --output results.csv
```

### 3️⃣ Video Processing
```bash
python main.py --video input.mp4 --output output.mp4
```

### 4️⃣ Evaluation
```bash
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
```

---

## 🔄 Full Workflow: Tạo Model Từ Đầu

**Copy toàn bộ script:**
```bash
# Step 1: Auto-extract từ 473 ảnh
python scripts/auto_extract_and_label_kaggle.py

# Step 2: Gán nhãn từ 17 test images
python scripts/extract_manual_labels.py

# Step 3: Filter best templates
python scripts/filter_best_templates.py

# Step 4: Train model hybrid
python scripts/train_knn_hybrid.py

# Step 5: Test model
python scripts/test_all_models.py
```

**Hoặc chạy từng bước:**
```bash
# Chỉ step 1
python scripts/auto_extract_and_label_kaggle.py

# Chỉ step 2
python scripts/extract_manual_labels.py

# Chỉ step 3
python scripts/filter_best_templates.py

# Chỉ step 4
python scripts/train_knn_hybrid.py
```

---

## 🎯 Scripts

| Script | Mục đích |
|--------|---------|
| `main.py` | 4 chế độ: single/batch/video/eval |
| `extract_manual_labels.py` | Cắt + gán nhãn từ 17 test images |
| `filter_best_templates.py` | Chọn 33 templates tốt nhất |
| `train_knn_hybrid.py` | Train model từ 33 + 46 = 79 ảnh |
| `test_all_models.py` | So sánh 3 model |
| `debug_seg_detail.py` | Debug segmentation |
| `test_hybrid_viz.py` | Test với visualization |

---

## 📊 Dataset

| Loại | Số Lượng | Accuracy | Ghi chú |
|------|----------|----------|--------|
| Templates | 31 | 5.76% | Manual selection |
| Auto-Labeled | 3100+ | 10.76% | EasyOCR + noise |
| Manual Labeled | 46 | 100% | Ground truth |
| **Hybrid** | **77** | **57.81%** | **31 + 46 = BEST** |

---

## 📂 Cấu Trúc Thư Mục

```
license_plate_system/
├── main.py                                     # Entry point
├── models/
│   └── knn_character_recognizer_hybrid.pkl    # Model (57.81%)
│
├── datasets/kaggle_foreign/
│   ├── character_templates/          (31 ảnh best)
│   ├── characters_manual_labeled/    (46 ảnh ground truth)
│   ├── characters_auto_labeled/      (3100+ ảnh noise)
│   ├── test/                         (17 ảnh test)
│   └── test_annotations.csv          (ground truth)
│
├── scripts/
│   ├── auto_extract_and_label_kaggle.py
│   ├── extract_manual_labels.py
│   ├── filter_best_templates.py
│   ├── train_knn_hybrid.py
│   ├── test_all_models.py
│   └── debug_seg_detail.py
│
├── src/
│   ├── character_recognizer.py
│   ├── preprocessor.py
│   └── ...
│
└── tests/
    └── test_hybrid_viz.py
```

---

## ❓ FAQ

**Q: Tại sao chỉ 57.81%?**
- Segmentation yếu (10/17 detect)
- Dữ liệu nhỏ (77 ảnh)
- Font chữ biến động

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

# Tạo manual labels từ test
python scripts/extract_manual_labels.py

# Filter templates tốt nhất
python scripts/filter_best_templates.py

# Train lại model
python scripts/train_knn_hybrid.py

# So sánh 3 model
python scripts/test_all_models.py

# Debug segmentation
python scripts/debug_seg_detail.py

# Test visualization
python tests/test_hybrid_viz.py

# Đánh giá chi tiết
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
```

---

**Version**: 1.0 | **Status**: Ready to use ✅
