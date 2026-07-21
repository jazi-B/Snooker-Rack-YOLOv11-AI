import os
import argparse
import shutil
from ultralytics import YOLO

def train_snooker_rack_model(
    data_yaml="data.yaml",
    epochs=25,
    batch_size=8,
    imgsz=640,
    model_variant="yolo12n.pt",
    output_name="snooker_rack_yolov12_production"
):
    """
    Train Multi-Perspective Production YOLOv12 Model for Snooker Rack Detection.
    Optimized for BOTH Overhead CCTV Cameras AND Low-Angle Broadcast / Mobile Photos.
    """
    print("==================================================")
    print(f"[*] TRAINING MULTI-PERSPECTIVE SNOOKER RACK MODEL ({model_variant})")
    print("==================================================")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}.")
        
    model = YOLO(model_variant)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=output_name,
        project="runs/detect",
        exist_ok=True,
        plots=True,
        save=True,
        # Multi-Perspective & Background Noise Suppression Hyperparameters
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,         # Wide rotation tolerance
        translate=0.15,
        scale=0.4,            # Scale jitter for distant broadcast shots
        shear=5.0,
        perspective=0.001,    # Low-angle perspective distortion
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,           # Complex background & occlusion composite
        mixup=0.15,           # Multi-table texture blending
        patience=20,
        verbose=True
    )
    
    # Copy best weights to models/ folder
    os.makedirs("models", exist_ok=True)
    best_weights = f"runs/detect/{output_name}/weights/best.pt"
    target_weights = "models/snooker_rack_yolov12.pt"

    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        shutil.copy(best_weights, "models/snooker_rack_yolov11.pt")
        print("\n[+] Multi-Perspective Training Complete!")
        print(f"[+] Trained YOLOv12 model saved to: {target_weights}")
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Perspective Snooker Rack YOLOv12 Model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--variant", type=str, default="yolo12n.pt", help="YOLOv12 variant (yolo12n.pt, yolo12s.pt)")
    args = parser.parse_args()
    
    train_snooker_rack_model(epochs=args.epochs, batch_size=args.batch, model_variant=args.variant)
