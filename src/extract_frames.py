import cv2
import os
import argparse

def extract_frames_from_video(video_path, output_dir, interval_sec=2):
    """
    Extracts frames from a snooker match video every `interval_sec` seconds
    to build a diverse multi-angle training dataset for YOLOv12.
    """
    if not os.path.exists(video_path):
        print(f"[!] Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    frame_interval = int(fps * interval_sec)
    frame_count = 0
    saved_count = 0

    print(f"[*] Extracting frames from {video_path} every {interval_sec} seconds...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:04d}.jpg"
            out_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[+] Successfully extracted {saved_count} diverse frames into '{output_dir}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract diverse snooker frames for dataset augmentation")
    parser.add_argument("--video", type=str, required=True, help="Path to snooker video file")
    parser.add_argument("--out", type=str, default="dataset_extracted", help="Output directory for extracted frames")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval in seconds between extracted frames")
    args = parser.parse_args()

    extract_frames_from_video(args.video, args.out, args.interval)
