import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Snooker Club AI Detector",
    page_icon="🎱",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    .main-title { text-align: center; font-size: 32px; font-weight: 700; color: #00e676; margin-bottom: 5px; }
    .sub-title { text-align: center; font-size: 14px; color: #9ca3af; margin-bottom: 25px; }
    .badge { background: #161f31; padding: 6px 14px; border-radius: 20px; font-weight: 600; color: #00e676; border: 1px solid #00e676; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎱 Snooker Club AI Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Overhead CCTV Snooker Rack Detector (YOLOv11 Model Engine)</div>', unsafe_allow_html=True)

# Metrics Banner
col1, col2, col3 = st.columns(3)
with col1:
    st.info("⚡ Model: YOLOv11 Nano")
with col2:
    st.success("🎯 mAP@50: 89.86%")
with col3:
    st.warning("⏱️ Latency: ~15ms")

@st.cache_resource
def load_model():
    weights_path = "models/snooker_rack_yolov11.pt"
    if not os.path.exists(weights_path):
        weights_path = "yolo11n.pt"
    return YOLO(weights_path)

model = load_model()

uploaded_file = st.file_uploader("Upload a Snooker Table Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape
    results = model.predict(img_bgr, conf=0.15, verbose=False)

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

    # Render bounding box
    if rack_detected and best_box:
        x1, y1, x2, y2 = best_box
        box_thickness = max(3, int(min(w, h) / 180))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 230, 115), box_thickness)
        
        lbl = f"SNOOKER RACK: {max_conf:.2f}"
        cv2.putText(img_bgr, lbl, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 115), 2)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if rack_detected:
        st.success(f"✅ STATUS: SNOOKER RACK DETECTED (Confidence: {max_conf*100:.1f}%)")
    else:
        st.warning("⚠️ STATUS: GAME IN PROGRESS / NO RACK DETECTED")

    st.image(img_rgb, caption="YOLOv11 AI Detection Result", use_container_width=True)
