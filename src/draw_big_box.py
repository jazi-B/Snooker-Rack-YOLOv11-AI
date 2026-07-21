import cv2
import os
import shutil
from ultralytics import YOLO

def generate_super_clear_result():
    img_path = "user_test_image.jpg"
    weights_path = "models/snooker_rack_yolov11.pt"
    if not os.path.exists(weights_path):
        weights_path = "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt"

    img = cv2.imread(img_path)
    if img is None:
        print("[!] Image not found")
        return

    model = YOLO(weights_path)
    results = model.predict(img, conf=0.25, verbose=False)

    h, w, _ = img.shape
    rack_found = False

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            
            rack_found = True

            # Draw SUPER THICK Bright Neon Box (Thickness = 6)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 6)

            # Draw Large Text Banner
            label_text = f"SNOOKER RACK DETECTED ({conf*100:.1f}%)"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
            
            # Black background banner for maximum visibility
            cv2.rectangle(img, (x1, max(y1 - 40, 0)), (x1 + w_lbl + 20, y1), (0, 0, 0), -1)
            cv2.rectangle(img, (x1, max(y1 - 40, 0)), (x1 + w_lbl + 20, y1), (0, 255, 0), 2)
            cv2.putText(img, label_text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    # Big Top Banner
    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(img, "YOLOv11 AI DETECTOR | STATUS: SNOOKER RACK DETECTED", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

    out_file = "test_results/clear_result_for_sir.jpg"
    os.makedirs("test_results", exist_ok=True)
    cv2.imwrite(out_file, img)
    print(f"[+] Saved super clear result to: {out_file}")

if __name__ == "__main__":
    generate_super_clear_result()
