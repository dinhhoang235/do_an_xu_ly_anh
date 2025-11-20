import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_recognizer import CharacterRecognizer
from src.preprocessor import Preprocessor

def debug_segmentation_with_visualization():
    print("=" * 70)
    print("🔍 SEGMENTATION DEBUG WITH PREPROCESSING VISUALIZATION")
    print("=" * 70)
    
    test_csv = "datasets/kaggle_foreign/test_annotations.csv"
    df = pd.read_csv(test_csv)
    
    recognizer = CharacterRecognizer()
    preprocessor = Preprocessor()
    
    # Create output directory for saving preprocessing results
    Path("debug_output").mkdir(exist_ok=True)
    
    print(f"\n{'Image':<20} {'Ground Truth':<15} {'Detected':<15} {'Match?':<10}")
    print("-" * 60)
    
    correct = 0
    
    for idx, row in df.iterrows():
        image_name = row['filename']
        ground_truth = row['plate_text']
        
        img_path = f"datasets/kaggle_foreign/test/{image_name}"
        if not Path(img_path).exists():
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        try:
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            
            plate_img = img[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # Preprocess - chi tiết từng bước
            img_grayscale = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            img_median = cv2.medianBlur(img_grayscale, preprocessor.MEDIAN_BLUR_SIZE)
            img_blurred = cv2.GaussianBlur(img_median, preprocessor.GAUSSIAN_BLUR_SIZE, 0)
            
            # Segment characters
            char_images = recognizer.segment_characters(plate_img)
            detected_count = len(char_images)
            
            match = "✅" if detected_count == len(ground_truth) else "❌"
            if detected_count == len(ground_truth):
                correct += 1
            
            print(f"{image_name:<20} {ground_truth:<15} {detected_count:<15} {match:<10}")
            
            # Visualization - tất cả bước tiền xử lý
            fig_width = 400
            height = int(fig_width * plate_img.shape[0] / plate_img.shape[1])
            
            # Hiển thị từng bước
            cv2.imshow(f"01_Original - {image_name}", cv2.resize(plate_img, (fig_width, height)))
            cv2.imshow(f"02_Grayscale - {image_name}", cv2.resize(img_grayscale, (fig_width, height)))
            cv2.imshow(f"03_Median - {image_name}", cv2.resize(img_median, (fig_width, height)))
            cv2.imshow(f"04_Blurred - {image_name}", cv2.resize(img_blurred, (fig_width, height)))
            
            # Save preprocessing steps
            output_dir = Path("debug_output") / image_name.replace(".png", "").replace(".jpg", "")
            output_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(output_dir / "01_original.png"), plate_img)
            cv2.imwrite(str(output_dir / "02_grayscale.png"), img_grayscale)
            cv2.imwrite(str(output_dir / "03_median.png"), img_median)
            cv2.imwrite(str(output_dir / "04_blurred.png"), img_blurred)
            
            # Display segmented characters
            for i, char_img in enumerate(char_images):
                char_resized = cv2.resize(char_img, (80, 120))
                cv2.imshow(f"Char {i} - {image_name}", char_resized)
                cv2.imwrite(str(output_dir / f"char_{i:02d}.png"), char_img)
            
            print(f"  👉 Ảnh đã lưu vào: debug_output/{image_name.replace('.png', '').replace('.jpg', '')}/")
            print(f"  👉 Nhấn phím để tiếp tục hoặc 'q' để thoát...")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                break
        
        except Exception as e:
            print(f"{image_name:<20} {ground_truth:<15} ERROR: {str(e)[:20]:<15}")
    
    print("-" * 60)
    print(f"Correct character count: {correct}/{len(df)} = {correct/len(df)*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    debug_segmentation_with_visualization()
