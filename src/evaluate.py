import os
from ultralytics import YOLO

def evaluate_model(weights_path=None, data_yaml="data.yaml"):
    """
    Validation and testing evaluation script for trained YOLOv12 model.
    Prints Precision, Recall, mAP50, and mAP50-95 metrics.
    """
    if not weights_path:
        possible_weights = [
            "models/snooker_rack_yolov12.pt",
            "models/snooker_rack_yolov11.pt",
            "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt"
        ]
        for w in possible_weights:
            if os.path.exists(w):
                weights_path = w
                break

    if not weights_path or not os.path.exists(weights_path):
        print(f"[!] Weights file not found at: {weights_path}")
        return

    print(f"[*] Evaluating model weights: {weights_path}")
    model = YOLO(weights_path)
    
    # Run validation
    metrics = model.val(data=data_yaml, split="val")
    
    print("\n==================================================")
    print("[*] OFFICIAL EVALUATION METRICS REPORT")
    print("==================================================")
    print(f"Precision (P):    {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)")
    print(f"Recall (R):       {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)")
    print(f"mAP@50:           {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
    print(f"mAP@50-95:        {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
    print("==================================================")

if __name__ == "__main__":
    evaluate_model()
