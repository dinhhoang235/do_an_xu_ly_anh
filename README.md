# 🚗 License Plate Recognition System

Nhận dạng ký tự từ biển số xe nước ngoài sử dụng KNN + Computer Vision.

**Performance**: 57.81% accuracy | 2/10 perfect match | 10/10 partial match

---

## ⚡ Quick Start

```bash
# 1. Cài đặt
pip install -r requirements.txt

# 2. Chạy ngay (model đã huấn luyện sẵn)
python main.py --image datasets/kaggle_foreign/test/Cars0.png
```

---

## 📖 4 Cách Sử Dụng

### 1️⃣ Single Image
```bash
python main.py --image path/to/image.jpg
# Output: Kết quả nhận dạng + thời gian xử lý
```

### 2️⃣ Batch Process
```bash
python main.py --batch datasets/kaggle_foreign/test --output results.csv
# Output: CSV file với kết quả cho 17 ảnh
```

### 3️⃣ Video Processing
```bash
python main.py --video input.mp4 --output output.mp4
# Output: Video với bounding box + text
```

### 4️⃣ Evaluation
```bash
python main.py --eval datasets/kaggle_foreign/test --annotations datasets/kaggle_foreign/test_annotations.csv
# Output: Chi tiết accuracy từng ảnh
```

---

## 🔄 Workflow: Tạo Model Từ Đầu

```bash
# Step 1: Auto-extract từ 473 ảnh (3100+ ký tự)
python scripts/auto_extract_and_label_kaggle.py

# Step 2: Gán nhãn từ 17 test images (46 ký tự)
python scripts/extract_manual_labels.py

# Step 3: Filter best templates (31 ảnh)
python scripts/filter_best_templates.py

# Step 4: Train model hybrid (77 ảnh = 31 + 46)
python scripts/train_knn_hybrid.py
# → Model: 57.81% accuracy ✅
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

| Loại | Số Lượng | Accuracy | Ghi chú |
|------|----------|----------|--------|
| Templates | 33 | 5.76% | Manual curation |
| Auto-Labeled | 3100+ | 10.76% | EasyOCR + noise |
| Manual Labeled | 46 | 100% | Ground truth |
| **Hybrid** | **79** | **57.81%** | **33 + 46 = BEST** |

---

## 📂 Cấu Trúc

```
license_plate_system/
├── main.py
├── models/knn_character_recognizer_hybrid.pkl  (Model đã train)
├── datasets/kaggle_foreign/
│   ├── character_templates/           (33 ảnh)
│   ├── characters_manual_labeled/     (46 ảnh)
│   ├── characters_auto_labeled/       (3100+ ảnh)
│   └── test/                          (17 ảnh)
├── scripts/
│   ├── extract_manual_labels.py
│   ├── filter_best_templates.py
│   ├── train_knn_hybrid.py
│   └── ...
└── src/
    ├── character_recognizer.py
    ├── preprocessor.py
    └── ...
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

## 🚀 Improvement Roadmap

| Ngắn hạn | Dài hạn |
|---------|--------|
| Thêm 50 ảnh → 65% | Deep Learning → 80% |
| Cải segmentation | REST API |
| | Mobile app |

---

## 📝 License

MIT License - 2024

**Version**: 1.0 | **Status**: Ready to use ✅
