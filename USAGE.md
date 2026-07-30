# Snooker Rack AI CCTV Dashboard - User & Developer Guide

This document describes how to setup, run, and utilize the Snooker Rack AI Overhead CCTV Session Manager.

---

## 🚀 Getting Started & Running on Localhost

To start the production server locally, make sure you have the required Python dependencies installed, then run the FastAPI application:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Backend Server:**
   ```bash
   python backend/app/main.py
   ```
3. **Access Dashboard:**
   Open your browser and navigate to: **[http://localhost:8000/](http://localhost:8000/)**

---

## 🎮 Core Workflow & Operations

The Snooker Rack AI system manages table occupancies, tracks game/frame counts automatically using overhead cameras, and calculates billing invoices upon exit.

### 📥 1. Session Entry (Start Session)
* **How-to:** Under the **Active Dashboard** tab, click **Start Gaming Session** on the target table card.
* **Details:** Enter the Customer/Member Name and an optional Custom Billing Rate. Click **Begin Session**.
* **AI Initialization:** Once the session starts, the system automatically initializes the overhead YOLO detection engine for that table.

### 🎱 2. Game Play & Rack Auto-Detection
* **Rack Setup (15s Stability + 82% Conf):**
  * When a rack of balls is arranged, it must be detected continuously for **15 seconds** with a confidence score **>= 82%** to transition the state to `RACK SET`.
  * **Gathering Guard:** Gathering balls, player movement, and hand placement on the table during set up will automatically reset the 15-second stability timer, preventing false game triggers.
* **Rack Break (Auto-Increment):**
  * When the rack is broken (detection falls below **60%** confidence), the system registers the break, transitions to `GAME IN PROGRESS`, and increments the game count by `1`.
* **5-Minute Reset Time Lock:**
  * To prevent mid-game red ball clusters from triggering false game increments, a **5-minute cooldown** is enforced. The table cannot transition back to the `RACK SET` state until 5 minutes have elapsed since the last break.

### 📤 3. Session Exit (End Session & Billing)
* **How-to:** Click **End Session** on the active table card.
* **Billing Invoice Receipt:** A modal receipt is generated automatically showing:
  * Customer Name, Table ID, Start & End times.
  * Total duration in minutes.
  * Total games played (auto-calculated by the AI).
  * Rate per game and **Total Amount Due**.
* **Finalization:** Click **Print Receipt** (if needed) and click **Close Session** to release the table.

---

## 🛠️ Calibration & Settings

### ROI Calibration
1. Go to the **ROI & System Settings** tab.
2. Select the target table from the dropdown to load the overhead video feed.
3. Click points on the image overlay to draw a custom detection polygon strictly around the table's rack spot.
4. Click **Save ROI Polygon** to update the settings. YOLO detections outside this polygon will be ignored.

### Offline Video Processing (Fast CPU Test)
To run YOLO inference on a local video file at high speed on CPU (running 15x-30x faster using 15-frame interval skipping and imgsz=416 downscaling):
```bash
python src/process_custom_video.py
```
Outputs are transcoded and saved automatically for compatibility.
