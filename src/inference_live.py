import os
import cv2
import argparse
from ultralytics import YOLO

def draw_dynamic_hud(img, rack_detected, conf, box_coords=None):
    """
    Draws a clean, non-overlapping HUD overlay on the video frames.
    """
    h, w, _ = img.shape
    font_scale = min(0.7, max(0.4, w / 900.0))
    thickness = 2 if font_scale > 0.5 else 1
    
    # 1. Draw Emerald Bounding Box
    if rack_detected and box_coords:
        x1, y1, x2, y2 = box_coords
        box_thickness = max(2, int(min(w, h) / 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), box_thickness)
        
    # 2. Draw HUD Top Banner
    banner_h = max(36, int(h * 0.08))
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 18), -1)
    
    left_text = "YOLOv12 AI CCTV Engine"
    if rack_detected:
        right_text = f"STATUS: RACK SET ({conf*100:.0f}%)"
        right_color = (0, 230, 115)
    else:
        right_text = "STATUS: GAME IN PROGRESS / NO RACK"
        right_color = (0, 165, 255)
        
    (w_l, h_l), _ = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (w_r, h_r), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    y_pos = int(banner_h / 2.0 + h_l / 2.0)
    
    cv2.putText(img, left_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    right_x = max(w_l + 25, w - w_r - 15)
    cv2.putText(img, right_text, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, right_color, thickness)
    
    return img

def main():
    parser = argparse.ArgumentParser(description="Live Snooker Rack CCTV Inference")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0), RTSP URL, or video path")
    parser.add_argument("--weights", type=str, default="runs/detect/snooker_rack_yolov12/weights/best.pt", help="Path to weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    
    # Fallback to general YOLOv12 model if custom model doesn't exist yet
    weights_path = args.weights
    if not os.path.exists(weights_path):
        fallback = "yolo12n.pt"
        print(f"[!] Warning: Custom weights not found at: {weights_path}")
        print(f"[*] Falling back to pre-trained {fallback}...")
        weights_path = fallback
        
    print(f"[*] Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Determine if source is an integer camera index
    source = args.source
    if source.isdigit():
        source = int(source)
        
    print(f"[*] Initializing video feed from source: {source}...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[!] Error: Could not open source: {source}")
        return
        
    cv2.namedWindow("Snooker CCTV Engine - Press 'q' to Quit", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame, or video ended.")
            break
            
        # Run YOLO inference
        results = model.predict(frame, conf=args.conf, verbose=False)[0]
        
        rack_detected = False
        max_conf = 0.0
        box_coords = None
        
        # Check if we have detections
        if len(results.boxes) > 0:
            # Get the detection with the highest confidence
            highest_conf_idx = results.boxes.conf.argmax().item()
            conf = results.boxes.conf[highest_conf_idx].item()
            
            # Since it's class 0 (snooker_rack), we verify class index
            cls = int(results.boxes.cls[highest_conf_idx].item())
            if cls == 0 and conf >= args.conf:
                rack_detected = True
                max_conf = conf
                # Extract coordinates
                xyxy = results.boxes.xyxy[highest_conf_idx].cpu().numpy()
                box_coords = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                
        # Draw HUD on the frame
        frame = draw_dynamic_hud(frame, rack_detected, max_conf, box_coords)
        
        cv2.imshow("Snooker CCTV Engine - Press 'q' to Quit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[+] Video stream stopped.")

if __name__ == "__main__":
    main()
