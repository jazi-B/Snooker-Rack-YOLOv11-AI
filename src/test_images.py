import os
import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

def test_model_on_images(
    weights_path=None,
    val_img_dir="dataset/images/val",
    output_dir="test_results",
    num_samples=10,
    conf_thresh=0.3
):
    """
    Runs inference on validation/test images when NO live CCTV camera is available.
    Saves visual prediction images with bounding boxes & HUD into test_results/ folder!
    """
    print("==================================================")
    print("🖼️ TESTING MODEL ON TEST IMAGES (NO CAMERA NEEDED)")
    print("==================================================")

    # Find trained weights automatically
    if not weights_path:
        possible_weights = [
            "models/snooker_rack_yolov11.pt",
            "runs/detect/snooker_rack_full/weights/best.pt",
            "runs/detect/snooker_rack_pilot/weights/best.pt",
            "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt"
        ]
        for w in possible_weights:
            if os.path.exists(w):
                weights_path = w
                break
                
    if not weights_path or not os.path.exists(weights_path):
        print("[!] No trained weights file found.")
        return

    print(f"[i] Loading weights: {weights_path}")
    model = YOLO(weights_path)

    val_path = Path(val_img_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(val_path.glob("*.jpg")) + list(val_path.glob("*.png")) + list(val_path.glob("*.jpeg")))
    if not img_files:
        print(f"[!] No test images found in {val_img_dir}")
        return

    print(f"[i] Running inference on {min(num_samples, len(img_files))} test images...")

    processed = 0
    for img_file in img_files[:num_samples]:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        results = model.predict(img, conf=conf_thresh, verbose=False)

        h, w, _ = img.shape
        rack_found = False

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                b = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                
                rack_found = True

                # Draw Emerald Green bounding box around rack
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), 3)

                # Label tag
                lbl_text = f"SNOOKER RACK {conf:.2f}"
                (w_lbl, h_lbl), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, max(y1 - 25, 0)), (x1 + w_lbl + 10, y1), (0, 230, 115), -1)
                cv2.putText(img, lbl_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw HUD Banner
        cv2.rectangle(img, (0, 0), (w, 35), (20, 20, 20), -1)
        status_str = "STATUS: RACK DETECTED" if rack_found else "STATUS: NO RACK DETECTED"
        color_str = (0, 255, 0) if rack_found else (0, 165, 255)
        cv2.putText(img, f"Test Image: {img_file.name}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, status_str, (w - 320, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_str, 2)

        # Save result image
        save_target = out_path / f"result_{img_file.name}"
        cv2.imwrite(str(save_target), img)
        processed += 1
        print(f"[✓] Saved prediction visual: {save_target}")

    print("\n==================================================")
    print(f"🎉 Testing Complete! Saved {processed} annotated images to: {out_path.get_absolute_path() if hasattr(out_path, 'get_absolute_path') else os.path.abspath(out_path)}")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Snooker Rack Model on Images")
    parser.add_argument("--image", type=str, default=None, help="Path to a single custom image file")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights")
    args = parser.parse_args()

    if args.image:
        # Test specific image
        weights = args.weights or "models/snooker_rack_yolov11.pt"
        if not os.path.exists(weights):
            weights = "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt"
            
        model = YOLO(weights)
        img = cv2.imread(args.image)
        if img is not None:
            results = model.predict(img, conf=0.3, verbose=False)
            h, w, _ = img.shape
            rack_found = False
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    b = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    rack_found = True
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), 3)
                    lbl = f"SNOOKER RACK {conf:.2f}"
                    cv2.putText(img, lbl, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 115), 2)
            
            # HUD
            cv2.rectangle(img, (0, 0), (w, 35), (20, 20, 20), -1)
            status_str = "STATUS: RACK DETECTED" if rack_found else "STATUS: GAME IN PROGRESS / NO RACK"
            color_str = (0, 255, 0) if rack_found else (0, 165, 255)
            cv2.putText(img, f"Image: {os.path.basename(args.image)}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, status_str, (w - 400, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_str, 2)

            os.makedirs("test_results", exist_ok=True)
            out_file = f"test_results/result_{os.path.basename(args.image)}"
            cv2.imwrite(out_file, img)
            print(f"[✓] Saved visual prediction: {out_file}")
    else:
        test_model_on_images(weights_path=args.weights)
