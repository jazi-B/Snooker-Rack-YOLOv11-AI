import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Snooker Club AI Detector (YOLOv12)",
    page_icon="🎱",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    .main-title { text-align: center; font-size: 32px; font-weight: 700; color: #00e676; margin-bottom: 5px; }
    .sub-title { text-align: center; font-size: 14px; color: #9ca3af; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎱 Snooker Club AI Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Overhead CCTV Snooker Rack Detector (YOLOv12 Model Engine)</div>', unsafe_allow_html=True)

# Metrics Banner
col1, col2, col3 = st.columns(3)
with col1:
    st.info("⚡ Model: YOLOv12 Nano")
with col2:
    st.success("🎯 mAP@50: 97.07%")
with col3:
    st.warning("⏱️ Latency: ~12ms")

def load_snooker_model():
    possible_paths = [
        "models/snooker_rack_yolov12.pt",
        "models/snooker_rack_yolov11.pt",
        "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return YOLO(p)
    return YOLO("yolo12n.pt")

model = load_snooker_model()

uploaded_file = st.file_uploader("Upload a Snooker Table Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape
    
    # Predict with optimal confidence 0.15 and NMS iou=0.45 for high precision & sensitivity
    results = model.predict(img_bgr, conf=0.15, iou=0.45, verbose=False)

    rack_detected = False
    max_conf = 0.0
    best_box = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                rack_detected = True
                b = box.xyxy[0].cpu().numpy()
                best_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    # 1. Render ONLY clean Emerald Green bounding box on the image
    if rack_detected and best_box:
        x1, y1, x2, y2 = best_box
        box_thickness = max(3, int(min(w, h) / 180))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 230, 115), box_thickness)

    # 2. Draw Top Black Header Banner (All text info clean inside top black bar)
    banner_h = max(42, int(h * 0.08))
    font_scale = min(0.7, max(0.45, w / 850.0))
    thickness = 2 if font_scale > 0.5 else 1

    # Black Top Bar
    cv2.rectangle(img_bgr, (0, 0), (w, banner_h), (12, 12, 14), -1)

    left_str = "YOLOv12 AI Engine"
    if rack_detected:
        right_str = f"STATUS: SNOOKER RACK SET ({max_conf*100:.1f}%)"
        right_color = (0, 230, 115)
    else:
        right_str = "STATUS: GAME IN PROGRESS / NO RACK"
        right_color = (0, 165, 255)

    (w_l, h_l), _ = cv2.getTextSize(left_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (w_r, h_r), _ = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    y_pos = int(banner_h / 2.0 + h_l / 2.0)

    # Left Title Text
    cv2.putText(img_bgr, left_str, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Right Status Text
    right_x = max(w_l + 25, w - w_r - 15)
    cv2.putText(img_bgr, right_str, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, right_color, thickness)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if rack_detected:
        st.success(f"✅ STATUS: SNOOKER RACK DETECTED (Confidence: {max_conf*100:.1f}%)")
    else:
        st.warning("⚠️ STATUS: GAME IN PROGRESS / NO RACK DETECTED")

    st.image(img_rgb, caption="YOLOv12 Clean AI Detection Result", use_container_width=True)
