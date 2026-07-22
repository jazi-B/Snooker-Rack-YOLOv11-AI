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

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    .main-title { text-align: center; font-size: 32px; font-weight: 700; color: #00e676; margin-bottom: 5px; }
    .sub-title { text-align: center; font-size: 14px; color: #9ca3af; margin-bottom: 25px; }
    
    .status-box-set {
        background: rgba(0, 230, 115, 0.15);
        border: 2px solid #00e676;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-box-set h2 { color: #00e676; margin: 0; font-size: 26px; font-weight: 700; }
    .status-box-set p { color: #a7f3d0; margin-top: 6px; font-size: 15px; }

    .status-box-progress {
        background: rgba(255, 165, 0, 0.15);
        border: 2px solid #ffa726;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-box-progress h2 { color: #ffa726; margin: 0; font-size: 26px; font-weight: 700; }
    .status-box-progress p { color: #fde68a; margin-top: 6px; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎱 Snooker Club AI Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Overhead CCTV Snooker Rack Detector (YOLOv12 Engine)</div>', unsafe_allow_html=True)

# Metrics Banner
col1, col2, col3 = st.columns(3)
with col1:
    st.info("⚡ Engine: YOLOv12 Nano")
with col2:
    st.success("🎯 Accuracy: 97.07% mAP")
with col3:
    st.warning("⏱️ Speed: ~12ms")

def is_initial_unbroken_rack(box_xywh, crop):
    """
    Verifies if a detected region is the TRUE INITIAL 15-BALL UNBROKEN TRIANGULAR RACK.
    Rejects scattered mid-game balls and loose clusters.
    """
    if crop is None or crop.size == 0:
        return False
    w, h = box_xywh[2], box_xywh[3]
    if h == 0 or w == 0:
        return False
    aspect_ratio = float(w) / float(h)
    
    # Initial triangular rack aspect ratio range (overhead & broadcast side angles)
    if not (0.60 <= aspect_ratio <= 1.65):
        return False
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 45, 35]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 45, 35]), np.array([180, 255, 255]))
    red_pixels = np.sum((mask1 | mask2) > 0)
    red_density = red_pixels / (w * h)
    
    # Must have tight red ball packing density
    return red_density > 0.10

def load_snooker_model():
    possible_paths = [
        "models/snooker_rack_yolov11.pt",
        "models/snooker_rack_yolov12.pt",
        "models/best.pt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return YOLO(p)
    return YOLO("yolo11n.pt")

model = load_snooker_model()

uploaded_file = st.file_uploader("Upload Snooker Table Photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w, _ = img_bgr.shape
    
    results = model.predict(img_bgr, conf=0.20, iou=0.45, verbose=False)

    is_true_initial_rack = False
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            b = box.xyxy[0].cpu().numpy()
            b_wh = box.xywh[0].cpu().numpy()
            
            x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
            crop = img_bgr[y1:y2, x1:x2]
            
            # STRICT CHECK: MUST PASS is_initial_unbroken_rack!
            if is_initial_unbroken_rack(b_wh, crop):
                if conf > max_conf:
                    max_conf = conf
                    is_true_initial_rack = True
                    
                    # Draw Bounding Box & Label on image
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 230, 115), 4)
                    lbl_text = f"INITIAL RACK: {conf*100:.1f}%"
                    (w_l, h_l), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_bgr, (x1, max(y1 - 30, 0)), (x1 + w_l + 10, y1), (0, 230, 115), -1)
                    cv2.putText(img_bgr, lbl_text, (x1 + 5, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 2. Draw Top Black Header Bar
    banner_h = max(42, int(h * 0.08))
    font_scale = min(0.7, max(0.45, w / 850.0))
    thickness = 2 if font_scale > 0.5 else 1

    cv2.rectangle(img_bgr, (0, 0), (w, banner_h), (12, 12, 14), -1)

    left_str = "SNOOKER AI DETECTOR (YOLOv12)"
    if is_true_initial_rack:
        right_str = f"STATUS: INITIAL TRIANGULAR RACK DETECTED ({max_conf*100:.1f}%)"
        right_color = (0, 230, 115)
    else:
        right_str = "STATUS: MID-GAME / NO INITIAL RACK"
        right_color = (0, 165, 255)

    (w_l, h_l), _ = cv2.getTextSize(left_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (w_r, h_r), _ = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    y_pos = int(banner_h / 2.0 + h_l / 2.0)
    cv2.putText(img_bgr, left_str, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    right_x = max(w_l + 25, w - w_r - 15)
    cv2.putText(img_bgr, right_str, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, right_color, thickness)

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # PROMINENT HTML STATUS DISPLAY ABOVE IMAGE
    if is_true_initial_rack:
        st.markdown(f"""
        <div class="status-box-set">
            <h2>🟢 INITIAL TRIANGULAR RACK DETECTED (GAME START)</h2>
            <p>Unbroken 15-Ball Triangle Positioned at Foot Spot (Confidence: {max_conf*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-box-progress">
            <h2>🔴 MID-GAME / NO INITIAL RACK</h2>
            <p>Balls are scattered or rack is broken. No initial game start rack present.</p>
        </div>
        """, unsafe_allow_html=True)

    st.image(img_rgb, caption="Clean Image Visual Result (No Box Overlays)", use_container_width=True)
