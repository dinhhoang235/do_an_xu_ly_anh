"""
YOLOv8 License Plate Detection - Fast Training Script (Optimized)
Train nhanh với settings tối ưu cho speed
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print("🚀 YOLOv8 License Plate Detection - Mac M4 FAST Training")
    print("=" * 70)
    
    # Detect device
    import platform
    if torch.backends.mps.is_available():
        device = 'mps'  # Metal Performance Shaders (Apple Silicon)
        print(f"📊 Device: Mac M4 (MPS - Metal Performance Shaders)")
        print(f"   Processor: {platform.processor()}")
    else:
        device = 'cpu'
        print(f"📊 Device: CPU (MPS not available)")
    
    # Get Mac specs
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        total_mem_gb = int(result.stdout.strip()) / 1e9
        print(f"   Total Memory: {total_mem_gb:.1f} GB")
    except:
        pass
    
    # Paths
    base_path = Path(__file__).parent.parent
    dataset_yaml = base_path / "datasets" / "data.yaml"
    models_dir = base_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Dataset: {dataset_yaml}")
    if not dataset_yaml.exists():
        print(f"❌ Not found: {dataset_yaml}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("⚙️  MAC M4 TRAINING CONFIG")
    print("=" * 70)
    
    # Mac M4 optimizations - EXTREME SPEED MODE
    # Trade accuracy for speed - quick baseline
    model_name = 'yolov8n'  # Nano (best for Mac)
    batch = 16  # Small batch for raw speed
    epochs = 5  # Ultra minimal - just 5 epochs
    
    print(f"📦 Model: {model_name} (Nano - EXTREME SPEED)")
    print(f"📊 Batch: {batch} (small for maximum speed)")
    print(f"⏱️  Epochs: {epochs} (ultra minimal)")
    print(f"⏱️  Estimated time: 3-5 minutes (Mac M4)")
    print(f"\n💡 Tips:")
    print(f"   • Close other apps to free memory")
    print(f"   • Connect power adapter for better performance")
    print(f"   • Using aggressive optimizations for fastest training")
    
    print("\n" + "=" * 70)
    print("📥 Loading Model")
    print("=" * 70)
    
    try:
        model = YOLO(f'{model_name}.pt')
        print(f"✅ Model loaded: {model_name}.pt")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🏋️  TRAINING (FAST MODE)")
    print("=" * 70)
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=256,  # Ultra small - 256x256 (33% faster than 320)
            batch=batch,
            patience=1,  # Stop immediately if no improvement
            device=device,
            project=str(models_dir),
            name=f'yolov8_{model_name}_mac',
            exist_ok=False,
            verbose=False,  # Less output overhead
            save=True,
            amp=False,
            workers=0,  # No workers - Mac M4 with loader in main thread is faster
            cache=False,  # No cache - reduce memory overhead
            mosaic=0.0,  # Disable all augmentation
            flipud=0.0,
            fliplr=0.0,
            degrees=0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            optimizer='SGD',
            lr0=0.05,  # Aggressive lr for fast convergence
            lrf=0.05,  # Keep high throughout
            momentum=0.937,
            weight_decay=0,  # Disable weight decay for speed
            warmup_epochs=0,  # No warmup - go straight to training
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            conf=0.5,
            iou=0.6,
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        
        best_model = models_dir / f'yolov8_{model_name}_mac' / 'weights' / 'best.pt'
        if best_model.exists():
            final_model = models_dir / 'yolov8_plate_detector.pt'
            import shutil
            shutil.copy(best_model, final_model)
            print(f"\n🏆 Best model: {final_model}")
            print(f"   Size: {final_model.stat().st_size / 1e6:.1f} MB")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✨ NEXT STEPS")
    print("=" * 70)
    print("""
Test the model:
  python scripts/test_yolov8.py

Use in main pipeline:
  python main.py --image test.jpg
    """)

if __name__ == '__main__':
    main()
