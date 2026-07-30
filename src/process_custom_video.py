import os
import cv2
import sys
import time
import subprocess
from ultralytics import YOLO

# Load weights (custom weights first, fallback to pre-trained yolov12 nano)
WEIGHTS_PATH = "runs/detect/snooker_rack_yolov12/weights/best.pt"
if not os.path.exists(WEIGHTS_PATH):
    fallback_paths = ["yolo12n.pt", "models/snooker_rack_yolov11.pt"]
    for f in fallback_paths:
        if os.path.exists(f):
            WEIGHTS_PATH = f
            break
        elif os.path.exists(os.path.join("models", f)):
            WEIGHTS_PATH = os.path.join("models", f)
            break
    else:
        WEIGHTS_PATH = "yolo12n.pt"

print(f"[*] Loading model weights from: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)

def draw_dynamic_hud(img, tables_results):
    h, w, _ = img.shape
    font_scale = min(0.7, max(0.4, w / 900.0))
    thickness = 2 if font_scale > 0.5 else 1
    
    # HUD Banner
    banner_h = max(36, int(h * 0.08))
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 18), -1)

    left_text = "YOLOv12 AI Multi-Table Engine"
    cv2.putText(img, left_text, (12, int(banner_h / 2.0 + 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Compile table statuses
    status_parts = []
    for t in tables_results:
        status_str = "RACK DETECTED" if t['rack_detected'] else "NO RACK / GAME IN PROGRESS"
        status_parts.append(f"{t['table_id']}: {status_str}")
        
    right_text = " | ".join(status_parts)
    (w_r, h_r), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    right_x = max(w - w_r - 15, w // 2)
    
    # Color coding: Green if rack is detected, Orange if no rack / game in progress
    any_rack = any(t['rack_detected'] for t in tables_results)
    text_color = (0, 230, 115) if any_rack else (0, 165, 255) # Green vs Orange (BGR)
    
    cv2.putText(img, right_text, (int(right_x), int(banner_h / 2.0 + 5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return img

def main():
    input_path = r"c:\Users\m_jaz\Desktop\Test_for_snooker\VID-20260724-WA0000(1).mp4"
    output_path = r"c:\Users\m_jaz\Desktop\Test_for_snooker\processed_VID-20260724-WA0000(1).mp4"
    temp_output_path = r"c:\Users\m_jaz\Desktop\Test_for_snooker\temp_processed.mp4"
    
    if not os.path.exists(input_path):
        print(f"Error: Input video not found at: {input_path}")
        return
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or fps is None:
        fps = 30.0
        
    print(f"[*] Video Properties: {width}x{height} @ {fps} fps, Total Frames: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Could not create output video writer")
        return
        
    frame_idx = 0
    start_time = time.time()
    
    # Cache variables for sub-sampled YOLO inference
    rack_detected = False
    max_conf = 0.0
    box_coords = None
    
    yolo_interval = 15 # Predict only once every 15 frames (0.5s at 30 FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        processed_frame = frame.copy()
        
        # Only run YOLO every 'yolo_interval' frames
        if (frame_idx - 1) % yolo_interval == 0:
            # Run YOLO inference with downscaling (imgsz=416) for extreme CPU speed
            results = model.predict(frame, conf=0.11, imgsz=416, verbose=False)[0]
            
            rack_detected = False
            max_conf = 0.0
            box_coords = None
            
            if len(results.boxes) > 0:
                highest_conf_idx = results.boxes.conf.argmax().item()
                conf = results.boxes.conf[highest_conf_idx].item()
                cls = int(results.boxes.cls[highest_conf_idx].item())
                
                if cls == 0:
                    rack_detected = True
                    max_conf = conf
                    xyxy = results.boxes.xyxy[highest_conf_idx].cpu().numpy()
                    box_coords = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                
        tables_results = [{
            'table_id': 'Table_1',
            'rack_detected': rack_detected,
            'conf': float(max_conf),
            'box_coords': box_coords
        }]
        
        if rack_detected and box_coords:
            box_thickness = max(2, int(min(width, height) / 180))
            cv2.rectangle(processed_frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 230, 115), box_thickness)
            
        processed_frame = draw_dynamic_hud(processed_frame, tables_results)
        out.write(processed_frame)
        
        if frame_idx % 30 == 0:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[*] Processed {frame_idx}/{total_frames} frames ({pct:.1f}%) | Speed: {fps_proc:.1f} FPS")
            
    cap.release()
    out.release()
    
    print("[*] Transcoding output video to H.264 using FFmpeg for native mobile/desktop compatibility...")
    try:
        cmd = f'ffmpeg -y -nostdin -i "{temp_output_path}" -c:v libx264 -pix_fmt yuv420p "{output_path}"'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[+] Successfully saved processed video to: {output_path}")
        # Clean up temp file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except Exception as e:
        print(f"[!] Warning: FFmpeg transcoding failed ({e}), keeping original output.")
        if os.path.exists(temp_output_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output_path, output_path)
            print(f"[+] Output saved (raw format) to: {output_path}")

if __name__ == "__main__":
    main()
