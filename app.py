import os
import cv2
import base64
import socket
import numpy as np
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load trained YOLOv11 Snooker Rack model
WEIGHTS_PATH = "models/snooker_rack_yolov11.pt"
if not os.path.exists(WEIGHTS_PATH):
    fallback_paths = [
        "runs/detect/runs/detect/snooker_rack_pilot/weights/best.pt",
        "runs/detect/snooker_rack_pilot/weights/best.pt"
    ]
    for f in fallback_paths:
        if os.path.exists(f):
            WEIGHTS_PATH = f
            break

print(f"[*] Loading model weights from: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def draw_dynamic_hud(img, rack_detected, max_conf, box_coords=None):
    """
    Renders clean, non-overlapping visual HUD overlay on any image resolution.
    Dynamically computes text scaling and positioning to prevent text overlap.
    """
    h, w, _ = img.shape
    
    # Dynamic font scaling based on image width
    font_scale = min(0.7, max(0.4, w / 900.0))
    thickness = 2 if font_scale > 0.5 else 1
    
    # 1. Draw Bounding Box if Rack Detected (Clean Emerald Green Box)
    if rack_detected and box_coords:
        x1, y1, x2, y2 = box_coords
        box_thickness = max(2, int(min(w, h) / 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), box_thickness)

    # 2. Draw Top HUD Banner (Clean & Non-overlapping)
    banner_h = max(36, int(h * 0.08))
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 18), -1)

    left_text = "YOLOv11 AI Engine"
    if rack_detected:
        right_text = f"STATUS: RACK SET ({max_conf*100:.0f}%)"
        right_color = (0, 230, 115)
    else:
        right_text = "STATUS: GAME IN PROGRESS / NO RACK"
        right_color = (0, 165, 255)

    (w_l, h_l), _ = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (w_r, h_r), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    y_pos = int(banner_h / 2.0 + h_l / 2.0)
    
    # Left Text
    cv2.putText(img, left_text, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Right Text (Positioned dynamically from right margin to avoid overlap)
    right_x = max(w_l + 25, w - w_r - 15)
    cv2.putText(img, right_text, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, right_color, thickness)

    return img

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snooker Club AI - Rack Detection System</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 31, 49, 0.75);
            --accent: #00e676;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); min-height: 100vh; padding: 20px; display: flex; flex-direction: column; align-items: center; }

        .header { text-align: center; margin-bottom: 25px; max-width: 800px; }
        .header h1 { font-size: 32px; font-weight: 700; background: linear-gradient(135deg, #ffffff, var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { color: var(--text-muted); font-size: 15px; margin-top: 6px; }

        .metrics-banner { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 15px; }
        .metric-badge { background: var(--card-bg); border: 1px solid var(--border-color); backdrop-filter: blur(10px); padding: 8px 18px; border-radius: 30px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }

        .main-container { width: 100%; max-width: 950px; background: var(--card-bg); border: 1px solid var(--border-color); backdrop-filter: blur(12px); border-radius: 16px; padding: 25px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }

        .dropzone { border: 2px dashed var(--accent); border-radius: 12px; padding: 35px 20px; text-align: center; background: rgba(0, 230, 118, 0.03); cursor: pointer; transition: all 0.2s; margin-bottom: 20px; }
        .dropzone:hover { background: rgba(0, 230, 118, 0.08); border-color: #00c853; }
        .dropzone p { color: var(--text-muted); margin-top: 8px; font-size: 14px; }
        input[type="file"] { display: none; }

        .result-view { text-align: center; margin-top: 20px; }
        .result-view img { max-width: 100%; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 10px 30px rgba(0,0,0,0.6); }

        .status-badge { display: inline-block; padding: 10px 22px; border-radius: 30px; font-weight: 700; font-size: 15px; margin-top: 15px; }
        .status-rack { background: rgba(0, 230, 118, 0.2); color: var(--accent); border: 1px solid var(--accent); }
        .status-norack { background: rgba(255, 165, 0, 0.2); color: #ffa726; border: 1px solid #ffa726; }

        .share-banner { margin-top: 25px; background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border-color); border-radius: 10px; padding: 12px 20px; text-align: center; font-size: 13px; color: var(--text-muted); }
        .share-banner code { color: var(--accent); font-family: monospace; font-size: 14px; }
    </style>
</head>
<body>

<div class="header">
    <h1>🎱 Snooker Club AI Detection System</h1>
    <p>Real-Time Overhead CCTV Snooker Rack Detector (YOLOv11 Model Engine)</p>
    
    <div class="metrics-banner">
        <div class="metric-badge"><div class="dot"></div> System Status: Live & Ready</div>
        <div class="metric-badge">Model: YOLOv11 (PyTorch)</div>
        <div class="metric-badge">Accuracy: 89.86% mAP@50</div>
    </div>
</div>

<div class="main-container">
    <div class="dropzone" onclick="document.getElementById('file-input').click()">
        <div style="font-size: 36px;">📁</div>
        <div style="font-weight: 600; font-size: 16px; margin-top: 8px;">Click to Upload Snooker Photo</div>
        <p>PNG, JPG, JPEG files supported</p>
    </div>
    <input type="file" id="file-input" accept="image/*" onchange="uploadImage(this.files[0])">

    <div id="result-container" class="result-view" style="display: none;">
        <div id="status-tag" class="status-badge"></div>
        <div style="margin-top: 15px;">
            <img id="predicted-image" src="" alt="YOLOv11 Detection Output">
        </div>
    </div>

    <div class="share-banner">
        🌐 Local Network Link: <code>http://{{ local_ip }}:5000</code>
    </div>
</div>

<script>
function uploadImage(file) {
    if (!file) return;
    let formData = new FormData();
    formData.append('file', file);

    document.getElementById('result-container').style.display = 'block';
    document.getElementById('status-tag').innerText = 'Processing AI Inference...';

    fetch('/predict_api', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            document.getElementById('predicted-image').src = data.image_b64;
            let tag = document.getElementById('status-tag');
            if (data.rack_detected) {
                tag.innerText = `✅ SNOOKER RACK DETECTED (Confidence: ${(data.max_conf * 100).toFixed(1)}%)`;
                tag.className = 'status-badge status-rack';
            } else {
                tag.innerText = '⚠️ STATUS: GAME IN PROGRESS / NO RACK DETECTED';
                tag.className = 'status-badge status-norack';
            }
        } else {
            alert('Error processing image: ' + data.error);
        }
    });
}
</script>

</body>
</html>
"""

@app.route('/')
def index():
    local_ip = get_local_ip()
    return render_template_string(HTML_TEMPLATE, local_ip=local_ip)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'success': False, 'error': 'Invalid image file'})

    # Predict with optimal confidence threshold 0.15 and NMS iou=0.45
    results = model.predict(img, conf=0.15, iou=0.45, verbose=False)

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

    # Draw dynamic non-overlapping HUD
    img = draw_dynamic_hud(img, rack_detected, max_conf, best_box)

    # Encode image to Base64
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'success': True,
        'rack_detected': rack_detected,
        'max_conf': max_conf,
        'image_b64': img_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    local_ip = get_local_ip()
    print("==================================================")
    print("🚀 SNOOKER AI WEB APP RUNNING")
    print(f"Local Access: http://localhost:{port}")
    print(f"Network Access: http://{local_ip}:{port}")
    print("==================================================")
    app.run(host='0.0.0.0', port=port, debug=False)
