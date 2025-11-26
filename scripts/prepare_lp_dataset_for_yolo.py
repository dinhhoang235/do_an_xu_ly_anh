"""
Prepare LP-characters dataset for YOLO training
Convert CSV bbox to YOLO format and create data.yaml
"""

import os
import sys
from pathlib import Path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def create_yolo_dataset():
    print("🔄 Preparing LP-characters dataset for YOLO training")
    print("=" * 60)

    base_path = Path(__file__).parent.parent
    dataset_path = base_path / "datasets" / "LP-characters"
    output_path = base_path / "datasets" / "lp_characters_yolo"

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_csv = dataset_path / "annotations.csv"
    if not gt_csv.exists():
        print(f"❌ Ground truth CSV not found: {gt_csv}")
        return

    df = pd.read_csv(gt_csv)
    print(f"📊 Loaded {len(df)} samples")

    # Convert bbox format and create YOLO annotations
    valid_samples = []

    for idx, row in df.iterrows():
        image_name = row['image']
        bbox_str = row['bbox']
        ground_truth = row['ground_truth']

        # Parse bbox: "x,y,width,height"
        try:
            bbox_parts = bbox_str.strip('"').split(',')
            x, y, w, h = map(int, bbox_parts)
        except:
            print(f"⚠️  Invalid bbox for {image_name}: {bbox_str}")
            continue

        # Load image to get dimensions
        img_path = dataset_path / "images" / image_name
        if not img_path.exists():
            continue

        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Convert to YOLO format (normalized center x, center y, width, height)
        center_x = (x + w/2) / img_w
        center_y = (y + h/2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        # YOLO format: class_id center_x center_y width height
        yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"

        valid_samples.append({
            'image': image_name,
            'label': yolo_line,
            'img_path': img_path
        })

    print(f"✅ Valid samples: {len(valid_samples)}")

    if len(valid_samples) == 0:
        print("❌ No valid samples found")
        return

    # Split data
    train_val, test = train_test_split(valid_samples, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)

    print(f"📊 Split: Train {len(train)}, Val {len(val)}, Test {len(test)}")

    # Copy files and create labels
    def copy_split(split_name, samples):
        for sample in samples:
            # Copy image
            src_img = sample['img_path']
            dst_img = output_path / split_name / 'images' / sample['image']
            shutil.copy(src_img, dst_img)

            # Create label file
            label_file = output_path / split_name / 'labels' / sample['image'].replace('.png', '.txt')
            with open(label_file, 'w') as f:
                f.write(sample['label'] + '\n')

    copy_split('train', train)
    copy_split('val', val)
    copy_split('test', test)

    # Create data.yaml
    data_yaml = base_path / "datasets" / "data.yaml"
    yaml_content = f"""
# LP-characters dataset for YOLO training
path: {output_path}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: license_plate

# Number of classes
nc: 1
"""

    with open(data_yaml, 'w') as f:
        f.write(yaml_content.strip())

    print(f"💾 Created data.yaml: {data_yaml}")
    print(f"📁 Dataset prepared at: {output_path}")

    print("\n" + "=" * 60)
    print("✅ Dataset preparation completed!")
    print("=" * 60)
    print(f"   Train: {len(train)} images")
    print(f"   Val: {len(val)} images")
    print(f"   Test: {len(test)} images")
    print(f"   Total: {len(valid_samples)} images")
    print(f"\n🚀 Ready for YOLO training!")

if __name__ == "__main__":
    create_yolo_dataset()