import os
from ultralytics import YOLO

def main():
    # Path to the best weights
    weights_path = "runs/detect/snooker_rack_yolov12/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"[!] Trained model weights not found at: {weights_path}")
        print("Please run src/train.py first or make sure training has completed successfully.")
        return
        
    print(f"[*] Loading model for evaluation: {weights_path}...")
    model = YOLO(weights_path)
    
    print("[*] Running validation...")
    metrics = model.val(data="data.yaml", split="val")
    
    print("\n" + "="*50)
    print("           SNOOKER RACK MODEL METRICS")
    print("="*50)
    print(f"Precision (P):   {metrics.box.mp * 100:.2f}%")
    print(f"Recall (R):      {metrics.box.mr * 100:.2f}%")
    print(f"mAP @ 50:        {metrics.box.map50 * 100:.2f}%")
    print(f"mAP @ 50-95:     {metrics.box.map * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
