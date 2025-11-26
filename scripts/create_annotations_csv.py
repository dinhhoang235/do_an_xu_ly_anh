"""
Create annotations.csv from XML annotation files in LP-characters dataset
Extract plate text and bbox information from XML files
"""

import os
import sys
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

def extract_plate_info_from_xml(xml_path):
    """
    Extract plate text and bbox from XML annotation file
    Returns: (plate_text, bbox_width, bbox_height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image size
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # Extract all characters
        characters = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            characters.append(name)

        # Join characters to form plate text
        plate_text = ''.join(characters)

        return plate_text, img_width, img_height

    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, None, None

def create_annotations_csv(dataset_path, output_csv):
    """
    Create annotations.csv from XML files
    """
    print("🔄 Creating annotations.csv from XML annotations")
    print("=" * 60)

    dataset_path = Path(dataset_path)
    annotations_dir = dataset_path / "annotations"
    images_dir = dataset_path / "images"

    if not annotations_dir.exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    # Get all XML files
    xml_files = list(annotations_dir.glob("*.xml"))
    print(f"📊 Found {len(xml_files)} XML annotation files")

    data = []

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        # Get image name (without extension)
        image_stem = xml_file.stem

        # Check if corresponding image exists
        image_path = images_dir / f"{image_stem}.png"
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue

        # Extract plate info from XML
        plate_text, img_width, img_height = extract_plate_info_from_xml(xml_file)

        if plate_text is None:
            continue

        # Create bbox string (full image for now, since XML doesn't have plate bbox)
        bbox = f"0,0,{img_width},{img_height}"

        # Calculate length
        length = len(plate_text)

        data.append({
            'image': f"{image_stem}.png",
            'bbox': f'"{bbox}"',
            'plate_text': plate_text,
            'length': length
        })

    if not data:
        print("❌ No data extracted from XML files")
        return

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df = df.sort_values('image')  # Sort by image name

    df.to_csv(output_csv, index=False)

    print(f"\n✅ Created annotations.csv: {output_csv}")
    print(f"📊 Total entries: {len(df)}")
    print(f"📋 Sample entries:")
    print(df.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("🎯 Next steps:")
    print("  • Check annotations.csv for accuracy")
    print("  • Run: python scripts/test_full_pipeline_lp_characters.py")
    print("=" * 60)

def main():
    print("📝 CREATE ANNOTATIONS.CSV FROM XML FILES")
    print("=" * 60)

    base_path = Path(__file__).parent.parent
    dataset_path = base_path / "datasets" / "LP-characters"
    output_csv = dataset_path / "annotations.csv"

    # Check if annotations.csv already exists
    if output_csv.exists():
        response = input(f"⚠️  {output_csv} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operation cancelled")
            return

    create_annotations_csv(str(dataset_path), str(output_csv))

if __name__ == "__main__":
    main()