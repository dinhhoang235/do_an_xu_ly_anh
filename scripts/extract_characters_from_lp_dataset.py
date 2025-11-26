"""
Extract Characters from LP-characters Dataset using XML Annotations
Create labeled training data for KNN character recognition
"""

import cv2
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file to get character bounding boxes and labels
    
    Returns:
        list of tuples: (char, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    characters = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        characters.append((name, xmin, ymin, xmax, ymax))
    
    return characters

def extract_characters_from_lp_dataset(dataset_path, output_path):
    """
    Extract characters from LP-characters dataset using XML annotations
    
    Args:
        dataset_path: Path to LP-characters folder
        output_path: Path to save organized characters
    """
    print("🚀 STARTING CHARACTER EXTRACTION FROM LP-CHARACTERS")
    print("=" * 70)
    
    dataset_path = Path(dataset_path)
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "annotations"
    output_path = Path(output_path)
    
    if not images_path.exists():
        print(f"❌ Images folder not found: {images_path}")
        return
    
    if not annotations_path.exists():
        print(f"❌ Annotations folder not found: {annotations_path}")
        return
    
    # Create character folders
    characters = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9, A-Z
    for char in characters:
        (output_path / char).mkdir(parents=True, exist_ok=True)
    
    # Get XML files
    xml_files = list(annotations_path.glob("*.xml"))
    print(f"📊 Processing {len(xml_files)} XML annotations")
    
    total_characters = 0
    char_count = {char: 0 for char in characters}
    
    for xml_path in tqdm(xml_files, desc="Processing annotations"):
        # Get corresponding image
        img_filename = xml_path.stem + ".png"
        img_path = images_path / img_filename
        
        if not img_path.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Parse XML
        characters_data = parse_xml_annotation(xml_path)
        
        # Extract each character
        for char, xmin, ymin, xmax, ymax in characters_data:
            # Crop character
            char_img = image[ymin:ymax, xmin:xmax]
            
            if char_img.size == 0:
                continue
            
            # Resize to standard size (32x32 for consistency)
            char_img = cv2.resize(char_img, (32, 32))
            
            # Save to appropriate folder
            if char in char_count:
                char_count[char] += 1
                char_filename = f"{xml_path.stem}_{char_count[char]}.png"
                char_path = output_path / char / char_filename
                cv2.imwrite(str(char_path), char_img)
                total_characters += 1
    
    print("\n" + "=" * 70)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"✅ Processed annotations: {len(xml_files)}")
    print(f"🔤 Characters extracted: {total_characters}")
    print(f"💾 Saved to: {output_path}")
    
    print("\n📊 Characters per type:")
    for char, count in sorted(char_count.items()):
        if count > 0:
            print(f"   {char}: {count}")
    
    if total_characters > 0:
        print(f"📈 Average per character: {total_characters/len([c for c in char_count.values() if c > 0]):.1f}")

def main():
    print("🔤 CHARACTER EXTRACTION FROM LP-CHARACTERS DATASET")
    print("=" * 70)
    
    base_path = Path(__file__).parent.parent
    
    # Paths
    dataset_path = Path("/Users/hoang/Documents/code/license_plate_system/datasets/LP-characters")
    output_path = dataset_path / "characters_organized"
    
    if not dataset_path.exists():
        print(f"❌ LP-characters dataset not found: {dataset_path}")
        print("   Download from Kaggle and place in project root")
        return
    
    # Extract characters
    extract_characters_from_lp_dataset(str(dataset_path), str(output_path))
    
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print("1. 📊 Check extracted characters in datasets/characters_organized/")
    print("2. 🏃 Run KNN training: python scripts/train_knn_from_dataset.py")
    print("3. 📈 Expect much better accuracy with labeled data!")
    print()
    print("🚀 Ready for superior character recognition!")

if __name__ == '__main__':
    main()