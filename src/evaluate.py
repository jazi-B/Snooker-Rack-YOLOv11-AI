import os
from ultralytics import YOLO

def evaluate_model(weights_path="models/snooker_rack_yolov11.pt", data_yaml="data.yaml"):
    """
    Validation and testing evaluation script for trained YOLOv11 model.
    Prints mAP50 and mAP50-95 metrics.
    """
    if not os.path.exists(weights_path):
        # Fallback search
        fallback_paths = [
            "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt",
            "runs/detect/snooker_rack_pilot/weights/best.pt"
        ]
        for f in fallback_paths:
            if os.path.exists(f):
                weights_path = f
                break

    if not os.path.exists(weights_path):
        print(f"[!] Weights file not found at: {weights_path}")
        return

    print(f"[*] Evaluating model weights: {weights_path}")
    model = YOLO(weights_path)
    
    # Run validation
    metrics = model.val(data=data_yaml, split="val")
    
    print("\n==================================================")
    print("[*] EVALUATION METRICS REPORT")
    print("==================================================")
    print(f"Precision (P):    {metrics.box.mp:.4f}")
    print(f"Recall (R):       {metrics.box.mr:.4f}")
    print(f"mAP@50:           {metrics.box.map50:.4f}")
    print(f"mAP@50-95:        {metrics.box.map:.4f}")
    print("==================================================")

if __name__ == "__main__":
    evaluate_model()
