"""
YOLOv8-based License Plate Detector
Replace contour-based detection with deep learning
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

class YOLOPlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8 plate detector
        
        Args:
            model_path: Path to trained YOLOv8 model (best.pt)
                       If None, will look for latest trained model
        """
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if model_path is None:
            # Auto-find latest model
            model_dir = Path(__file__).parent.parent / "models"
            best_models = list(model_dir.glob("*/weights/best.pt"))
            
            if best_models:
                model_path = sorted(best_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                print(f"📦 Found model: {model_path}")
            else:
                raise FileNotFoundError(
                    "No trained YOLOv8 model found. "
                    "Run: python scripts/train_yolov8.py"
                )
        
        # Load model
        try:
            self.model = YOLO(str(model_path))
            print(f"✅ Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect_plates(self, image, conf_threshold=0.5, visualize=False):
        """
        Detect license plates in image
        
        Args:
            image: Input image (BGR)
            conf_threshold: Confidence threshold for detections
            visualize: Whether to show detection visualization
        
        Returns:
            List of detected plates as (x, y, width, height) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=0.6,
            device=self.device,
            verbose=False
        )
        
        # Parse detections
        plates = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert to (x, y, width, height)
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Ensure coordinates are within image bounds
                    h, w = image.shape[:2]
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    plates.append((x, y, width, height))
        
        # Visualization
        if visualize and len(plates) > 0:
            result.plot(conf=conf_threshold)
            result.show()
        
        return plates
    
    def crop_plates(self, image, conf_threshold=0.5):
        """
        Detect and crop license plates from image
        
        Args:
            image: Input image (BGR)
            conf_threshold: Confidence threshold for detections
        
        Returns:
            List of cropped plate images
        """
        plates = self.detect_plates(image, conf_threshold)
        cropped_plates = []
        
        for (x, y, width, height) in plates:
            # Crop plate from image
            plate_crop = image[y:y+height, x:x+width]
            if plate_crop.size > 0:
                cropped_plates.append(plate_crop)
        
        return cropped_plates
    
    def detect_plates_batch(self, images, conf_threshold=0.5):
        """
        Detect plates in batch of images
        
        Args:
            images: List of images
            conf_threshold: Confidence threshold
        
        Returns:
            List of detection results
        """
        results_list = []
        
        for image in images:
            try:
                plates = self.detect_plates(image, conf_threshold)
                results_list.append({
                    'plates': plates,
                    'count': len(plates),
                    'error': None
                })
            except Exception as e:
                results_list.append({
                    'plates': [],
                    'count': 0,
                    'error': str(e)
                })
        
        return results_list
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        return {
            'model': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'device': self.device,
            'task': 'detect',
            'imgsz': self.model.imgsz if hasattr(self.model, 'imgsz') else 'N/A'
        }
