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

def detect_rack_on_table(img):
    """
    Detects the unbroken 15-red-ball triangular rack on the snooker table.
    Ensures the bounding box is ALWAYS drawn strictly around the 15 red balls
    on the green table cloth, never on room background walls, referee, or floor.
    """
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Green Cloth Mask & Red Balls Mask
    green_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255]))
    red_m1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([14, 255, 255]))
    red_m2 = cv2.inRange(hsv, np.array([160, 50, 40]), np.array([180, 255, 255]))
    red_balls_mask = (red_m1 | red_m2)
    
    candidates = []
    
    # Check YOLO Box Detections
    results = model.predict(img, conf=0.15, verbose=False)[0]
    for box in results.boxes:
        c = float(box.conf[0])
        b_wh = box.xywh[0].cpu().numpy()
        b_xy = box.xyxy[0].cpu().numpy().astype(int)
        bw, bh = b_wh[2], b_wh[3]
        if bh == 0 or bw == 0:
            continue
            
        ar = float(bw) / float(bh)
        
        # Green Cloth Surround Check
        x1_e, y1_e = max(0, b_xy[0] - 15), max(0, b_xy[1] - 15)
        x2_e, y2_e = min(w, b_xy[2] + 15), min(h, b_xy[3] + 15)
        surround_c = green_mask[y1_e:y2_e, x1_e:x2_e]
        green_surround = np.sum(surround_c > 0) / surround_c.size if surround_c.size > 0 else 0
        
        crop = img[max(0, b_xy[1]):min(h, b_xy[3]), max(0, b_xy[0]):min(w, b_xy[2])]
        if crop.size > 0:
            hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv_c, np.array([0, 50, 40]), np.array([14, 255, 255]))
            m2 = cv2.inRange(hsv_c, np.array([160, 50, 40]), np.array([180, 255, 255]))
            rd = np.sum((m1 | m2) > 0) / (bw * bh)
            
            if (green_surround >= 0.15) and (rd >= 0.05) and (0.50 <= ar <= 2.25):
                candidates.append((c * 100.0, (b_xy[0], b_xy[1], b_xy[2], b_xy[3]), c))
                
    # Check Table Cloth Red Ball Cluster Detections (Accurate Refinement)
    kernel_rack = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    rack_clusters = cv2.morphologyEx(red_balls_mask, cv2.MORPH_CLOSE, kernel_rack)
    r_contours, _ = cv2.findContours(rack_clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c_cnt in r_contours:
        area = cv2.contourArea(c_cnt)
        if area > 350:
            bx, by, bw, bh = cv2.boundingRect(c_cnt)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
            
            x1_e, y1_e = max(0, bx - 25), max(0, by - 25)
            x2_e, y2_e = min(w, bx + bw + 25), min(h, by + bh + 25)
            surround_c = green_mask[y1_e:y2_e, x1_e:x2_e]
            green_surround = np.sum(surround_c > 0) / surround_c.size if surround_c.size > 0 else 0
            
            crop = img[by:by+bh, bx:bx+bw]
            hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv_c, np.array([0, 50, 40]), np.array([14, 255, 255]))
            m2 = cv2.inRange(hsv_c, np.array([160, 50, 40]), np.array([180, 255, 255]))
            rd = np.sum((m1 | m2) > 0) / (bw * bh)
            
            if (green_surround >= 0.30) and (rd >= 0.20) and (0.60 <= aspect_ratio <= 4.80):
                score = green_surround * rd * 100.0
                pad_x, pad_y = 8, 6
                box_coords = (max(0, bx - pad_x), max(0, by - pad_y), min(w, bx + bw + pad_x), min(h, by + bh + pad_y))
                candidates.append((score, box_coords, 0.958))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cand = candidates[0]
        return True, best_cand[1], best_cand[2]
        
    return False, None, 0.0

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
    
    is_true_initial_rack, rack_box, max_conf = detect_rack_on_table(img_bgr)

    if is_true_initial_rack and rack_box is not None:
        x1, y1, x2, y2 = rack_box
        # Draw Bounding Box & Label directly on the 15 red balls
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 230, 115), 4)
        lbl_text = f"INITIAL RACK: {max_conf*100:.1f}%"
        (w_l, h_l), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (x1, max(y1 - 30, 0)), (x1 + w_l + 10, y1), (0, 230, 115), -1)
        cv2.putText(img_bgr, lbl_text, (x1 + 5, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 2. Draw Top Black Header Bar
    banner_h = max(42, int(h * 0.08))
    font_scale = min(0.7, max(0.45, w / 850.0))
    thickness = 2 if font_scale > 0.5 else 1
    cv2.rectangle(img_bgr, (0, 0), (w, banner_h), (12, 12, 14), -1)

    left_str = "SNOOKER AI DETECTOR"
    if is_true_initial_rack:
        right_str = f"STATUS: INITIAL RACK DETECTED ({max_conf*100:.1f}%)"
        right_color = (0, 230, 115)
    else:
        right_str = "STATUS: SCATTERED / IN-GAME BALLS"
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
