import os
import yaml
from ultralytics import YOLO

def main():
    weights_path = "runs/detect/snooker_rack_yolov12/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo12n.pt"  # fallback
        print(f"[*] Custom weights not found, using fallback: {weights_path}")
        
    print(f"[*] Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Check if golden test set is empty
    images_dir = "golden_test_set/images"
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print("[!] Golden Test Set is empty. Please add images to golden_test_set/images/ and labels to golden_test_set/labels/ to audit the model.")
        return
        
    # Create temporary YAML file for golden set validation
    golden_yaml = {
        'path': os.path.abspath('.'),
        'val': 'golden_test_set/images',
        'nc': 1,
        'names': {0: 'snooker_rack'}
    }
    
    temp_yaml_path = 'golden_temp.yaml'
    with open(temp_yaml_path, 'w') as f:
        yaml.safe_dump(golden_yaml, f)
        
    print("[*] Auditing model on Golden Test Set...")
    try:
        # Run validation
        metrics = model.val(
            data=temp_yaml_path,
            split='val',
            project='runs/detect',
            name='golden_audit',
            exist_ok=True,
            plots=True
        )
        
        print("\n" + "="*50)
        print("           GOLDEN TEST SET AUDIT RESULTS")
        print("="*50)
        try:
            print(f"Precision (P):    {metrics.box.mp*100:.2f}%")
            print(f"Recall (R):       {metrics.box.mr*100:.2f}%")
            print(f"mAP@50:           {metrics.box.map50*100:.2f}%")
            print(f"mAP@50-95:        {metrics.box.map*100:.2f}%")
        except Exception:
            print("[-] No positive instances found in Golden Test Set labels to compute metrics.")
        print("="*50)
        print("[*] Audit metrics and plots saved to: runs/detect/golden_audit/")
        
    finally:
        # Clean up temporary YAML
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

if __name__ == '__main__':
    main()
