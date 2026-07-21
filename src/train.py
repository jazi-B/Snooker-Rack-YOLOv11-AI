import os
import argparse
from ultralytics import YOLO

def train_snooker_rack_model(
    data_yaml="data.yaml",
    epochs=100,
    batch_size=8,
    imgsz=640,
    model_variant="yolo11s.pt",
    output_name="snooker_rack_model"
):
    """
    Train Custom YOLOv11 Model for Snooker Rack Detection.
    """
    print("==================================================")
    print(f"[*] TRAINING SNOOKER RACK MODEL ({model_variant})")
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
        # Augmentation hyperparameters optimized for CCTV overhead camera
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        shear=2.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        patience=20,
        verbose=True
    )
    
    # Copy best weights to models/ folder
    os.makedirs("models", exist_ok=True)
    best_weights = f"runs/detect/{output_name}/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, "models/snooker_rack_yolov11.pt")
        print("\n[+] Model Training Complete!")
        print("[+] Trained model saved to: models/snooker_rack_yolov11.pt")
        
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snooker Rack Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    
    train_snooker_rack_model(epochs=args.epochs, batch_size=args.batch)
