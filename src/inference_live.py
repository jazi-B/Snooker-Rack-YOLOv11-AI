import cv2
import time
import os
import argparse
from ultralytics import YOLO

def run_live_inference(
    weights_path="models/snooker_rack_yolov11.pt",
    source="0",  # "0" for webcam, "rtsp://..." for CCTV RTSP stream, or path to video file
    conf_thresh=0.4,
    use_roi=False,
    roi_coords=None  # (x1, y1, x2, y2)
):
    """
    Real-Time CCTV Stream Inference Engine for Snooker Rack Detection.
    """
    if not os.path.exists(weights_path):
        fallback_paths = [
            "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt",
            "runs/detect/snooker_rack_pilot/weights/best.pt"
        ]
        for f in fallback_paths:
            if os.path.exists(f):
                weights_path = f
                break

    print("==================================================")
    print("[*] STARTING SNOOKER RACK LIVE CCTV INFERENCE")
    print(f"Source: {source} | Weights: {weights_path} | Conf: {conf_thresh}")
    print("==================================================")

    # Load model
    model = YOLO(weights_path)
    
    # Handle video source input (int for camera index, string for file/RTSP)
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"[!] Unable to open video source: {source}")
        return

    fps_counter = 0
    start_time = time.time()
    current_fps = 0.0

    window_name = "Overhead CCTV - Snooker Rack Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Stream ended or frame not received.")
            break

        h, w, _ = frame.shape

        # Region of interest processing
        if use_roi and roi_coords:
            rx1, ry1, rx2, ry2 = roi_coords
            roi_frame = frame[ry1:ry2, rx1:rx2]
            results = model.predict(roi_frame, conf=conf_thresh, verbose=False)
        else:
            results = model.predict(frame, conf=conf_thresh, verbose=False)

        rack_detected = False
        rack_box = None
        max_conf = 0.0

        # Parse detection outputs
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence > max_conf:
                    max_conf = confidence
                    rack_detected = True
                    # Bounding box coordinates
                    b = box.xyxy[0].cpu().numpy()
                    if use_roi and roi_coords:
                        rack_box = (int(b[0]) + roi_coords[0], int(b[1]) + roi_coords[1],
                                    int(b[2]) + roi_coords[0], int(b[3]) + roi_coords[1])
                    else:
                        rack_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # Calculate FPS
        fps_counter += 1
        if (time.time() - start_time) > 1.0:
            current_fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()

        # Draw Visual Overlays (HUD)
        if use_roi and roi_coords:
            cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (255, 255, 0), 1)

        if rack_detected and rack_box:
            x1, y1, x2, y2 = rack_box
            # Draw primary bounding box around snooker rack (Emerald Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 115), 3)
            
            # Label banner
            label = f"SNOOKER RACK: {max_conf:.2f}"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(y1 - 25, 0)), (x1 + w_lbl + 10, y1), (0, 230, 115), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Status Badge: RACK SET
            status_text = "STATUS: RACK SET (READY TO BREAK)"
            status_color = (0, 255, 0)
        else:
            status_text = "STATUS: GAME IN PROGRESS / NO RACK"
            status_color = (0, 165, 255)

        # Draw Top HUD Banner
        # Draw Top HUD Banner (Clean & Non-overlapping)
        banner_h = max(36, int(h * 0.08))
        font_scale = min(0.65, max(0.4, w / 900.0))
        thickness = 2 if font_scale > 0.5 else 1
        
        cv2.rectangle(frame, (0, 0), (w, banner_h), (15, 15, 18), -1)
        left_str = f"CCTV Stream | FPS: {current_fps:.1f}"
        
        (w_l, h_l), _ = cv2.getTextSize(left_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (w_r, h_r), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        y_pos = int(banner_h / 2.0 + h_l / 2.0)
        cv2.putText(frame, left_str, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        right_x = max(w_l + 25, w - w_r - 15)
        cv2.putText(frame, status_text, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC or q to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live CCTV Snooker Rack Detection")
    parser.add_argument("--weights", type=str, default="models/snooker_rack_yolov11.pt", help="Path to trained weights")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, path to mp4, or RTSP url)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    
    args = parser.parse_args()
    run_live_inference(weights_path=args.weights, source=args.source, conf_thresh=args.conf)
