import os
import cv2
import time
import datetime
import json
import numpy as np
import threading
import asyncio
from collections import deque
from ultralytics import YOLO
from backend.app.core.config import settings, get_model_weights
from backend.app.db import database

# Thread safety lock for YOLO inference
yolo_lock = threading.Lock()

# Global model instance, loaded lazily
_model = None

def get_yolo_model():
    global _model
    with yolo_lock:
        if _model is None:
            weights = get_model_weights()
            print(f"[*] Video Engine loading YOLO model weights from: {weights}")
            _model = YOLO(weights)
        return _model

class TableProcessor(threading.Thread):
    def __init__(self, table_id: str, name: str, source: str):
        super().__init__(daemon=True)
        self.table_id = table_id
        self.name = name
        self.source = source
        self.running = False
        
        # Load ROI from database
        self.roi_polygon = []
        self.load_roi()
        
        # Frame buffer
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Temporal smoothing window
        self.history_window = deque(maxlen=30)
        self.rack_present = False
        self.game_logic_started = False
        self.last_transition_time = time.time()
        self.rack_detection_start_time = None
        
        # YOLO results to overlay
        self.last_box_coords = None
        self.last_conf = 0.0
        
        # Inference rate-limiting variables
        self.last_inference_time = 0.0
        self.last_rack_detected = False
        
        # Caching variables for image directory simulator
        self.current_img_path = None
        self.base_frame = None

    def load_roi(self):
        tables = database.get_all_tables()
        for t in tables:
            if t['id'] == self.table_id:
                self.roi_polygon = t['roi_polygon']
                break

    def set_roi(self, points):
        self.roi_polygon = points

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def run(self):
        self.running = True
        print(f"[*] Starting camera thread for {self.name} (Source: {self.source})")
        
        # Attempt to open video capture first
        is_simulated = False
        cap = None
        
        src_val = self.source
        try:
            if src_val.isdigit():
                cap_src = int(src_val)
                cap = cv2.VideoCapture(cap_src)
            else:
                cap = cv2.VideoCapture(src_val)
                
            if cap is not None and cap.isOpened():
                print(f"[+] Camera feed successfully connected for {self.name} using source: {src_val}")
            else:
                is_simulated = True
                print(f"[!] Could not open camera feed for {self.name} using source: {src_val}. Falling back to simulation.")
        except Exception as e:
            is_simulated = True
            print(f"[!] Exception opening camera feed for {self.name}: {e}. Falling back to simulation.")
            
        # Check if we have the test dataset on the desktop (only for simulation fallback)
        rack_imgs = []
        no_rack_imgs = []
        is_img_dir = False
        
        if is_simulated:
            rack_dir = r"c:\Users\m_jaz\Desktop\Test_for_snooker\Rack"
            no_rack_dir = r"c:\Users\m_jaz\Desktop\Test_for_snooker\No_Rack"
            if os.path.exists(rack_dir) and os.path.exists(no_rack_dir):
                try:
                    rack_imgs = [os.path.join(rack_dir, f) for f in os.listdir(rack_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    no_rack_imgs = [os.path.join(no_rack_dir, f) for f in os.listdir(no_rack_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    rack_imgs.sort()
                    no_rack_imgs.sort()
                    if rack_imgs and no_rack_imgs:
                        is_img_dir = True
                        print(f"[*] Simulation Fallback: Using directory test images for {self.name}.")
                except Exception as e:
                    print(f"[!] Error reading test directories: {e}")
            
        # Simulation lifecycle variables
        sim_frame_count = 0
        sim_rack_visible = True
        sim_rack_cycle_start = time.time()
        
        # Directory simulator index tracking with unique table offset
        import hashlib
        table_hash = int(hashlib.md5(self.table_id.encode()).hexdigest(), 16)
        rack_idx = table_hash
        no_rack_idx = table_hash
        
        model = None
        # Load YOLO model
        try:
            model = get_yolo_model()
        except Exception as e:
            print(f"[!] Error loading YOLO model: {e}")
            is_simulated = True
                
        while self.running:
            frame = None
            rack_detected = self.last_rack_detected
            max_conf = self.last_conf
            box_coords = self.last_box_coords
            
            current_time = time.time()
            # Only run YOLO inference once every 1.5 seconds to conserve GIL/CPU resource!
            run_yolo = (current_time - self.last_inference_time) >= 1.5
            
            if is_simulated and is_img_dir:
                # --- A. Test Image Directory Simulator ---
                sim_frame_count += 1
                
                # Alternate every 50 seconds: Rack state for 25s, No_Rack for 25s
                elapsed_cycle = time.time() - sim_rack_cycle_start
                if elapsed_cycle > 50:
                    sim_rack_cycle_start = time.time()
                    sim_rack_visible = True
                elif elapsed_cycle > 25:
                    sim_rack_visible = False
                    
                # Load corresponding image from sorted list
                if sim_rack_visible:
                    img_path = rack_imgs[rack_idx % len(rack_imgs)]
                    if sim_frame_count % 15 == 0:
                        rack_idx += 1
                else:
                    img_path = no_rack_imgs[no_rack_idx % len(no_rack_imgs)]
                    if sim_frame_count % 15 == 0:
                        no_rack_idx += 1
                        
                # Only load image and run YOLO if the path has changed or cache is empty
                if img_path != self.current_img_path or self.base_frame is None:
                    self.current_img_path = img_path
                    frame_read = cv2.imread(img_path)
                    if frame_read is None:
                        # Fallback to green canvas if image loading fails
                        self.base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        self.base_frame[:] = (40, 115, 50)
                    else:
                        self.base_frame = cv2.resize(frame_read, (640, 480))
                    
                    # Run YOLO inference immediately for the new image frame
                    rack_detected = False
                    max_conf = 0.0
                    box_coords = None
                    if model:
                        with yolo_lock:
                            results = model.predict(self.base_frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)[0]
                        if len(results.boxes) > 0:
                            highest_conf_idx = -1
                            best_conf = 0.0
                            for idx in range(len(results.boxes)):
                                cls = int(results.boxes.cls[idx].item())
                                c = results.boxes.conf[idx].item()
                                if cls == 0 and c >= settings.CONFIDENCE_THRESHOLD and c > best_conf:
                                    best_conf = c
                                    highest_conf_idx = idx
                                    
                            if highest_conf_idx != -1:
                                xyxy = results.boxes.xyxy[highest_conf_idx].cpu().numpy()
                                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                
                                # Scale-invariant size filtering (looser to accept smaller racks on rotated tables)
                                h_img, w_img, _ = self.base_frame.shape
                                rel_w = (box[2] - box[0]) / w_img
                                rel_h = (box[3] - box[1]) / h_img
                                is_valid_size = rel_w >= 0.03 and rel_h >= 0.01 and (rel_w * rel_h) >= 0.0005
                                
                                if is_valid_size:
                                    # Color verification check to filter out pink ball/green cloth false positives
                                    crop = self.base_frame[box[1]:box[3], box[0]:box[2]]
                                    hc, wc, _ = crop.shape
                                    is_red = False
                                    if hc > 0 and wc > 0:
                                        cy_c, cx_c = hc // 2, wc // 2
                                        dy, dx = max(1, int(hc * 0.1)), max(1, int(wc * 0.1))
                                        center_crop = crop[max(0, cy_c-dy):min(hc, cy_c+dy), max(0, cx_c-dx):min(wc, cx_c+dx)]
                                        if center_crop.size > 0:
                                            mean_bgr = cv2.mean(center_crop)[:3]
                                            r_b = mean_bgr[2] / max(1.0, mean_bgr[0])
                                            r_g = mean_bgr[2] / max(1.0, mean_bgr[1])
                                            is_red = r_b >= 1.2 and r_g >= 1.3
                                            
                                    if is_red:
                                        # Verify if inside ROI polygon
                                        cx = int((box[0] + box[2]) / 2.0)
                                        cy = int((box[1] + box[3]) / 2.0)
                                        inside_roi = True
                                        if self.roi_polygon and len(self.roi_polygon) >= 3:
                                            poly_pts = np.array([[int(p[0] * w_img), int(p[1] * h_img)] for p in self.roi_polygon], np.int32)
                                            dist = cv2.pointPolygonTest(poly_pts, (cx, cy), False)
                                            inside_roi = dist >= 0
                                            
                                        if inside_roi:
                                            rack_detected = True
                                            max_conf = best_conf
                                            box_coords = box
                    self.last_rack_detected = rack_detected
                    self.last_conf = max_conf
                    self.last_box_coords = box_coords
                else:
                    # In between, load cached detection values and state
                    rack_detected = self.last_rack_detected
                    max_conf = self.last_conf
                    box_coords = self.last_box_coords
                    # In between inferences, if we transition to simulated No_Rack state, override to False
                    if not sim_rack_visible:
                        rack_detected = False
                        self.last_rack_detected = False

                # Use a copy of cached base frame to overlay graphics
                frame = self.base_frame.copy()
                
                # Increase loop sleep (run loop at 10hz, but AI runs at 0.66hz)
                time.sleep(0.1)
                
            elif is_simulated:
                # --- B. Synthetic Green Cloth Table Simulator ---
                sim_frame_count += 1
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (40, 115, 50)
                
                # Draw table outline
                cv2.rectangle(frame, (0, 0), (640, 480), (35, 60, 95), 25)
                pockets = [(30, 30), (320, 20), (610, 30), (30, 450), (320, 460), (610, 450)]
                for p in pockets:
                    cv2.circle(frame, p, 18, (15, 15, 15), -1)
                cv2.line(frame, (160, 25), (160, 455), (255, 255, 255), 2)
                cv2.ellipse(frame, (160, 240), (60, 60), 0, 90, 270, (255, 255, 255), 2)
                
                elapsed_cycle = time.time() - sim_rack_cycle_start
                if elapsed_cycle > 50:
                    sim_rack_cycle_start = time.time()
                    sim_rack_visible = True
                elif elapsed_cycle > 25:
                    sim_rack_visible = False
                    
                if sim_rack_visible:
                    # Draw visual rack
                    box_coords = [425, 205, 475, 275]
                    rack_detected = True
                    max_conf = 0.89 + 0.05 * np.sin(sim_frame_count / 10.0)
                    
                    poly_pts = np.array([[430, 240], [470, 210], [470, 270]], np.int32)
                    cv2.fillPoly(frame, [poly_pts], (10, 10, 200))
                    cv2.polylines(frame, [poly_pts], True, (0, 200, 255), 2)
                    cv2.putText(frame, "RACK", (435, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                else:
                    rack_detected = False
                    dispersed_spots = [(480, 200), (380, 280), (510, 310), (280, 190), (450, 150), (410, 230)]
                    for spot in dispersed_spots:
                        cv2.circle(frame, spot, 6, (15, 15, 200), -1)
                    cv2.circle(frame, (200, 260), 6, (255, 255, 255), -1)
                    
                time.sleep(0.1)
                
            else:
                # --- C. Real Camera Stream Source ---
                ret, img = cap.read()
                if not ret:
                    if isinstance(self.source, str) and self.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print(f"[!] Camera feed disconnected for {self.name}, entering simulation fallback.")
                        is_simulated = True
                        continue
                        
                frame = img
                h, w, _ = frame.shape
                
                if run_yolo:
                    rack_detected = False
                    self.last_inference_time = current_time
                    if model:
                        with yolo_lock:
                            results = model.predict(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)[0]
                        if len(results.boxes) > 0:
                            highest_conf_idx = -1
                            best_conf = 0.0
                            for idx in range(len(results.boxes)):
                                cls = int(results.boxes.cls[idx].item())
                                c = results.boxes.conf[idx].item()
                                if cls == 0 and c >= settings.CONFIDENCE_THRESHOLD and c > best_conf:
                                    best_conf = c
                                    highest_conf_idx = idx
                                    
                            if highest_conf_idx != -1:
                                xyxy = results.boxes.xyxy[highest_conf_idx].cpu().numpy()
                                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                
                                # Scale-invariant size filtering (looser to accept smaller racks on rotated tables)
                                rel_w = (box[2] - box[0]) / w
                                rel_h = (box[3] - box[1]) / h
                                is_valid_size = rel_w >= 0.03 and rel_h >= 0.01 and (rel_w * rel_h) >= 0.0005
                                
                                if is_valid_size:
                                    # Color verification check to filter out pink ball/green cloth false positives
                                    crop = frame[box[1]:box[3], box[0]:box[2]]
                                    hc, wc, _ = crop.shape
                                    is_red = False
                                    if hc > 0 and wc > 0:
                                        cy_c, cx_c = hc // 2, wc // 2
                                        dy, dx = max(1, int(hc * 0.1)), max(1, int(wc * 0.1))
                                        center_crop = crop[max(0, cy_c-dy):min(hc, cy_c+dy), max(0, cx_c-dx):min(wc, cx_c+dx)]
                                        if center_crop.size > 0:
                                            mean_bgr = cv2.mean(center_crop)[:3]
                                            r_b = mean_bgr[2] / max(1.0, mean_bgr[0])
                                            r_g = mean_bgr[2] / max(1.0, mean_bgr[1])
                                            is_red = r_b >= 1.2 and r_g >= 1.3
                                            
                                    if is_red:
                                        cx = int((box[0] + box[2]) / 2.0)
                                        cy = int((box[1] + box[3]) / 2.0)
                                        
                                        inside_roi = True
                                        if self.roi_polygon and len(self.roi_polygon) >= 3:
                                            poly_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in self.roi_polygon], np.int32)
                                            dist = cv2.pointPolygonTest(poly_pts, (cx, cy), False)
                                            inside_roi = dist >= 0
                                            
                                        if inside_roi:
                                            rack_detected = True
                                            max_conf = best_conf
                                            box_coords = box
                    self.last_rack_detected = rack_detected
                    self.last_conf = max_conf
                    self.last_box_coords = box_coords
                
                time.sleep(0.01)
            
            # 3. Game State Machine & Temporal Smoothing under session constraint
            self.history_window.append(rack_detected)
            ratio = sum(self.history_window) / len(self.history_window) if self.history_window else 0.0
            smoothed_rack = ratio >= 0.70
            
            active_sess = database.get_active_session(self.table_id)
            if not active_sess:
                # No active session: reset logic states and wait
                if self.game_logic_started or self.rack_present:
                    print(f"[-] Table {self.table_id}: No active session. Resetting game logic state.")
                self.game_logic_started = False
                self.rack_present = False
                self.rack_detection_start_time = None
            else:
                # Active session: check start constraints
                if not self.game_logic_started:
                    # Logic starts ONLY when session is active AND a rack is detected (stable for 15 seconds with conf >= 50%)
                    if smoothed_rack and max_conf >= 0.50:
                        if self.rack_detection_start_time is None:
                            self.rack_detection_start_time = time.time()
                        elif time.time() - self.rack_detection_start_time >= 15.0:
                            self.game_logic_started = True
                            self.rack_present = True
                            self.last_transition_time = time.time()
                            self.rack_detection_start_time = None
                            print(f"[+] Table {self.table_id}: Active session and rack set (stable 15s, conf >= 50%). Game logic initialized.")
                    else:
                        self.rack_detection_start_time = None
                else:
                    # Game logic running: track transitions
                    if self.rack_present:
                        if not smoothed_rack:
                            # Rack broken/removed -> Increment game count
                            self.rack_present = False
                            cooldown_time = 15.0 # Cooldown to avoid transient double-triggers of rack breaking
                            if time.time() - self.last_transition_time >= cooldown_time:
                                self.last_transition_time = time.time()
                                self.increment_game_count()
                                print(f"[!] Table {self.table_id}: Rack broken. Incrementing game count.")
                            self.rack_detection_start_time = None
                    else:
                        # Rack is NOT present
                        if smoothed_rack and max_conf >= 0.50:
                            # Rack reset/re-formed: Only allowed after 5-minute cooldown (300 seconds) AND 15-second stability with conf >= 50%
                            if time.time() - self.last_transition_time >= 300.0:
                                if self.rack_detection_start_time is None:
                                    self.rack_detection_start_time = time.time()
                                elif time.time() - self.rack_detection_start_time >= 15.0:
                                    self.rack_present = True
                                    self.rack_detection_start_time = None
                                    print(f"[+] Table {self.table_id}: Rack set again (stable 15s, conf >= 50%).")
                            else:
                                self.rack_detection_start_time = None
                        else:
                            self.rack_detection_start_time = None
                    
            # 4. Drawing Overlays (Bounding boxes, ROI, HUD)
            h, w, _ = frame.shape
            
            # Draw ROI outline
            if self.roi_polygon and len(self.roi_polygon) >= 3:
                poly_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in self.roi_polygon], np.int32)
                # Color code ROI polygon: Green if rack present, Orange if in progress
                roi_color = (0, 230, 115) if self.rack_present else (0, 165, 255) # BGR
                cv2.polylines(frame, [poly_pts], True, roi_color, 2)
                # Add tiny label
                cv2.putText(frame, "ACTIVE DETECTION ROI", (poly_pts[0][0], poly_pts[0][1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1)
                            
            # Draw Bounding Box only if currently detected in this frame
            # This prevents stale or false-positive boxes (e.g. from individual balls) from lingering
            if rack_detected and box_coords:
                self.last_box_coords = box_coords
                self.last_conf = max_conf
                
                # Color code the box: Green if stable rack is set, Amber/Orange if transient/detecting
                box_color = (0, 230, 115) if self.rack_present else (0, 165, 255) # BGR
                cv2.rectangle(frame, (box_coords[0], box_coords[1]), 
                              (box_coords[2], box_coords[3]), box_color, 3)
                cv2.putText(frame, f"Rack {max_conf*100:.0f}%", 
                            (box_coords[0], box_coords[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            
            # Draw Dynamic HUD
            frame = self.draw_hud_banner(frame)
            
            # Update latest frame
            with self.frame_lock:
                self.latest_frame = frame
                
        if cap is not None:
            cap.release()
        print(f"[*] Camera thread stopped for {self.name}")

    def draw_hud_banner(self, img):
        h, w, _ = img.shape
        font_scale = min(0.65, max(0.4, w / 950.0))
        thickness = 2 if font_scale > 0.5 else 1
        
        # Banner Height relative to frame height
        banner_h = max(38, int(h * 0.09))
        cv2.rectangle(img, (0, 0), (w, banner_h), (20, 20, 25), -1)
        
        # Fetch active session detail
        active_sess = database.get_active_session(self.table_id)
        
        left_text = f"{self.name}"
        if active_sess:
            customer = active_sess['customer_name']
            games = active_sess['games_played']
            # Calculate elapsed minutes
            start = datetime.datetime.fromisoformat(active_sess['start_time'])
            elapsed = datetime.datetime.now() - start
            elapsed_mins = int(elapsed.total_seconds() / 60)
            
            left_text += f" - Active: {customer} ({elapsed_mins}m)"
            right_text = f"GAMES: {games} | "
        else:
            left_text += " - IDLE"
            right_text = ""
            
        # Display detection status based on play logic constraints
        if not active_sess:
            right_text += "STATUS: VACANT"
            right_color = (0, 165, 255) # Orange
        else:
            if not self.game_logic_started:
                right_text += "STATUS: WAITING FOR RACK"
                right_color = (0, 165, 255) # Orange
            else:
                if self.rack_present:
                    right_text += "STATUS: RACK SET"
                    right_color = (0, 230, 115) # Green
                else:
                    right_text += "STATUS: GAME IN PROGRESS"
                    right_color = (0, 165, 255) # Orange
            
        (w_l, h_l), _ = cv2.getTextSize(left_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (w_r, h_r), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        y_pos = int(banner_h / 2.0 + h_l / 2.0)
        
        # Left HUD Text
        cv2.putText(img, left_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Right HUD Text
        right_x = max(w_l + 25, w - w_r - 15)
        cv2.putText(img, right_text, (int(right_x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, right_color, thickness)
        
        return img

    def increment_game_count(self):
        active_sess = database.get_active_session(self.table_id)
        if active_sess:
            new_games = active_sess['games_played'] + 1
            print(f"[!] AI Trigger Table {self.table_id}: Auto-incrementing game count to {new_games}")
            database.update_session_games(active_sess['id'], new_games)

    def stop(self):
        self.running = False


class VideoEngineManager:
    def __init__(self):
        self.processors = {}
        self.lock = threading.Lock()

    def start_all_processors(self):
        with self.lock:
            # Shutdown any active processors
            self.stop_all_processors_unsafe()
            
            # Load tables
            tables = database.get_all_tables()
            for t in tables:
                p = TableProcessor(t['id'], t['name'], t['camera_source'])
                p.start()
                self.processors[t['id']] = p

    def stop_all_processors(self):
        with self.lock:
            self.stop_all_processors_unsafe()

    def stop_all_processors_unsafe(self):
        for pid, p in self.processors.items():
            p.stop()
        for pid, p in self.processors.items():
            p.join(timeout=1.5)
        self.processors.clear()

    def update_table_source(self, table_id: str, new_source: str):
        with self.lock:
            if table_id in self.processors:
                p = self.processors[table_id]
                p.stop()
                p.join(timeout=1.5)
                del self.processors[table_id]
                
            tables = database.get_all_tables()
            for t in tables:
                if t['id'] == table_id:
                    new_p = TableProcessor(table_id, t['name'], new_source)
                    new_p.start()
                    self.processors[table_id] = new_p
                    break

    def refresh_roi(self, table_id: str):
        with self.lock:
            if table_id in self.processors:
                self.processors[table_id].load_roi()

    def get_feed_frame(self, table_id: str):
        with self.lock:
            p = self.processors.get(table_id)
            if p:
                return p.get_latest_frame()
        return None

# Global Engine instance
video_engine = VideoEngineManager()
