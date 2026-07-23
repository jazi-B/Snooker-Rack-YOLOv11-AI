import os
import cv2
import base64
import socket
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Model loading logic (custom weights first, fallback to pre-trained yolov12 nano)
WEIGHTS_PATH = "runs/detect/snooker_rack_yolov12/weights/best.pt"
if not os.path.exists(WEIGHTS_PATH):
    fallback_paths = [
        "yolo12n.pt",
        "models/snooker_rack_yolov11.pt"
    ]
    for f in fallback_paths:
        if os.path.exists(f):
            WEIGHTS_PATH = f
            break
        elif os.path.exists(os.path.join("models", f)):
            WEIGHTS_PATH = os.path.join("models", f)
            break
    else:
        # If no local fallback is found, Ultralytics will automatically download yolo12n.pt
        WEIGHTS_PATH = "yolo12n.pt"

print(f"[*] App loading model weights from: {WEIGHTS_PATH}")
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

def draw_dynamic_hud(img, rack_detected, conf, box_coords=None):
    """
    Renders a clean HUD overlay on the processed image.
    """
    h, w, _ = img.shape
    font_scale = min(0.7, max(0.4, w / 900.0))
    thickness = 2 if font_scale > 0.5 else 1
    
    # 1. Bounding Box (Emerald Green)
    if rack_detected and box_coords:
        x1, y1, x2, y2 = box_coords
        box_thickness = max(2, int(min(w, h) / 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), box_thickness)

    # 2. HUD Banner
    banner_h = max(36, int(h * 0.08))
    cv2.rectangle(img, (0, 0), (w, banner_h), (15, 15, 18), -1)

    left_text = "YOLOv12 AI Engine"
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snooker Club AI - YOLOv12 Rack Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #080b11;
            --panel-bg: rgba(17, 24, 39, 0.7);
            --accent-green: #00e676;
            --accent-glow: rgba(0, 230, 118, 0.15);
            --accent-orange: #ffa726;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.08);
            --card-radius: 18px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background: linear-gradient(135deg, #05070c 0%, #0c121e 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 30px 15px;
            overflow-x: hidden;
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
            max-width: 800px;
        }

        .header h1 {
            font-size: 38px;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, var(--accent-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 16px;
            margin-top: 8px;
        }

        .metrics-grid {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .metric-badge {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            padding: 8px 18px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.6; }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 14px var(--accent-green); }
            100% { transform: scale(0.9); opacity: 0.6; }
        }

        .main-card {
            width: 100%;
            max-width: 900px;
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
            border-radius: var(--card-radius);
            padding: 30px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.4);
            display: flex;
            flex-direction: column;
            gap: 25px;
            transition: var(--transition);
        }

        .dropzone {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 45px 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.01);
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }

        .dropzone:hover {
            border-color: var(--accent-green);
            background: rgba(0, 230, 118, 0.02);
            box-shadow: 0 0 20px var(--accent-glow);
        }

        .dropzone .icon {
            font-size: 44px;
            transition: var(--transition);
        }

        .dropzone:hover .icon {
            transform: translateY(-5px);
        }

        .dropzone h3 {
            font-size: 18px;
            font-weight: 600;
        }

        .dropzone p {
            color: var(--text-muted);
            font-size: 13px;
        }

        input[type="file"] {
            display: none;
        }

        .result-panel {
            text-align: center;
            display: none;
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 24px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .status-rack {
            background: rgba(0, 230, 118, 0.12);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }

        .status-norack {
            background: rgba(255, 165, 0, 0.12);
            color: var(--accent-orange);
            border: 1px solid var(--accent-orange);
        }

        .image-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            background: #000;
            display: inline-block;
            max-width: 100%;
        }

        .image-container img {
            max-width: 100%;
            display: block;
            height: auto;
        }

        .connection-info {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .connection-info code {
            color: var(--accent-green);
            font-family: monospace;
            font-size: 14px;
            font-weight: 600;
        }

        .footer {
            margin-top: 40px;
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }
    </style>
</head>
<body>

<div class="header">
    <h1>🎱 Snooker Club AI Rack Monitor</h1>
    <p>Production-Grade CCTV Snooker Rack Status Engine (YOLOv12 Core)</p>
    
    <div class="metrics-grid">
        <div class="metric-badge"><div class="live-dot"></div> CCTV Status: Ready</div>
        <div class="metric-badge">Model Engine: YOLOv12n</div>
        <div class="metric-badge">Weights In Use: <code>{{ weights_name }}</code></div>
    </div>
</div>

<div class="main-card">
    <div class="dropzone" onclick="document.getElementById('file-input').click()" id="dropzone">
        <div class="icon">📁</div>
        <h3>Upload or Drag & Drop Snooker Table Photo</h3>
        <p>Supports JPG, JPEG, and PNG formats</p>
    </div>
    <input type="file" id="file-input" accept="image/*" onchange="processImage(this.files[0])">

    <div id="result-panel" class="result-panel">
        <div id="status-badge" class="status-badge"></div>
        <div>
            <div class="image-container">
                <img id="result-img" src="" alt="YOLOv12 Prediction Output">
            </div>
        </div>
    </div>

    <div class="connection-info">
        <span>🌐 CCTV Network Streaming Link:</span>
        <code>http://{{ local_ip }}:5000</code>
    </div>
</div>

<div class="footer">
    <p>Powered by Ultralytics YOLOv12 Neural Attention Architecture</p>
</div>

<script>
    // Handle Drag & Drop
    const dropzone = document.getElementById('dropzone');
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--accent-green)';
            dropzone.style.background = 'rgba(0, 230, 118, 0.04)';
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--border-color)';
            dropzone.style.background = 'rgba(255, 255, 255, 0.01)';
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processImage(files[0]);
        }
    });

    function processImage(file) {
        if (!file) return;
        
        const resultPanel = document.getElementById('result-panel');
        const statusBadge = document.getElementById('status-badge');
        const resultImg = document.getElementById('result-img');
        
        resultPanel.style.display = 'block';
        statusBadge.innerText = 'AI Inference In Progress...';
        statusBadge.className = 'status-badge status-norack';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/predict_api', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                resultImg.src = data.image_b64;
                if (data.rack_detected) {
                    statusBadge.innerText = `✅ Rack Set (Confidence: ${(data.conf * 100).toFixed(1)}%)`;
                    statusBadge.className = 'status-badge status-rack';
                } else {
                    statusBadge.innerText = '⚠️ Game In Progress / No Rack Detected';
                    statusBadge.className = 'status-badge status-norack';
                }
            } else {
                alert('Error running inference: ' + data.error);
                statusBadge.innerText = 'Error processing image';
            }
        })
        .catch(err => {
            console.error(err);
            alert('Server connection error.');
        });
    }
</script>

</body>
</html>
"""

@app.route('/')
def index():
    local_ip = get_local_ip()
    weights_name = os.path.basename(WEIGHTS_PATH)
    return render_template_string(HTML_TEMPLATE, local_ip=local_ip, weights_name=weights_name)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
        
    try:
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'success': False, 'error': 'Could not decode image'})
            
        # Run YOLO inference
        results = model.predict(img, conf=0.30, verbose=False)[0]
        
        rack_detected = False
        max_conf = 0.0
        box_coords = None
        
        if len(results.boxes) > 0:
            highest_conf_idx = results.boxes.conf.argmax().item()
            conf = results.boxes.conf[highest_conf_idx].item()
            cls = int(results.boxes.cls[highest_conf_idx].item())
            
            if cls == 0:  # Class 0: snooker_rack
                rack_detected = True
                max_conf = conf
                xyxy = results.boxes.xyxy[highest_conf_idx].cpu().numpy()
                box_coords = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                
        # Draw HUD on the image
        processed_img = draw_dynamic_hud(img.copy(), rack_detected, max_conf, box_coords)
        
        # Encode back to PNG base64
        _, buffer = cv2.imencode('.png', processed_img)
        img_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'rack_detected': rack_detected,
            'conf': float(max_conf),
            'image_b64': img_b64
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Start development server
    app.run(host='0.0.0.0', port=5000, debug=True)
