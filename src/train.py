import os
import torch
from ultralytics import YOLO

def main():
    # Auto-detect GPU or CPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[*] Active Training Device: {device} (CUDA Available: {torch.cuda.is_available()})")
    
    # Load pre-trained YOLOv12 Nano weights
    model_name = "yolo12n.pt"
    print(f"[*] Initializing model with {model_name}...")
    model = YOLO(model_name)
    
    # Run YOLO training
    print("[*] Starting training on data.yaml...")
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="snooker_rack_yolov12",
        project="runs/detect",
        device=device,
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20.0,
        translate=0.1,
        scale=0.2,
        mosaic=0.8,
        patience=25
    )
    print("[+] Training completed successfully!")

if __name__ == "__main__":
    main()
