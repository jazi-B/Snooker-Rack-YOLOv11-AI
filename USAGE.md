# Snooker Rack AI: Production CCTV Overhead Session Manager
## A-to-Z User & Administrator Manual (UI Step-by-Step)

This document serves as the master guide for cashier operators, club managers, and system administrators to operate the Snooker Rack AI system from the Web Dashboard.

---

## 🚀 1. Accessing and Starting the App

To run the system locally, execute the following from the project root directory:

1. **Start the FastAPI application:**
   ```bash
   python -m backend.app.main
   ```
2. **Open the Web Dashboard:**
   Open any web browser and go to: **[http://localhost:8000/](http://localhost:8000/)**

---

## 🎮 2. Cashier Operations (Daily Workflow)

All daily cashier and table management processes are handled from the **Active Dashboard** tab.

```
+-------------------------------------------------------------+
|                     ACTIVE DASHBOARD                        |
+-------------------------------------------------------------+
| [ Table 1 ]                    | [ Table 2 ]                |
| Status: VACANT                 | Status: WAITING FOR RACK   |
|                                | Active Time: 12 mins       |
|                                | Frames: 0 | Bill: Rs. 0.00 |
| [ Start New Session ]          | [ End Session & Bill ]     |
+-------------------------------------------------------------+
```

### 🟢 Process A: Starting a New Session
1. Locate the card of the table you want to open. Verify its status badge reads **VACANT** (Grey).
2. Click the green **Start New Session** button at the bottom of the card.
3. In the pop-up modal:
   - **Customer Name:** Enter the name of the player or booking member (e.g. `Jazib`).
   - **Custom Rate (Optional):** Enter a custom rate per frame (Rs.) to override the system default for this session.
4. Click **Begin Session**.
5. **UI Transition:** 
   - The status badge changes to **WAITING FOR RACK** (Amber).
   - The active elapsed duration timer (e.g. `1m`) starts ticking.

### 🎱 Process B: Automatic Game Plays & Rack Tracking
The system handles detection automatically using the overhead CCTV camera.
1. **Rack Setup:** When players arrange the red balls inside the triangle, the YOLO model spots it. 
   - On the **Live CCTV Feeds** tab, an **Orange/Amber** bounding box wraps around the balls.
   - Once the rack remains stationary inside the table's active region (ROI) for **15 continuous seconds** at **>= 60% confidence**, the box outline turns **Green**, and the card status badge changes to **RACK SET** (Green).
2. **Rack Break (Auto-Count):** As soon as a player strikes the cue ball to break the rack:
   - The YOLO detection falls below the threshold.
   - The system registers the break, changes the table status badge to **GAME IN PROGRESS** (Blue), and instantly increments the **Frames Played** count by **1** (e.g. `Frames: 1`).
   - The session bill is updated in real-time.
3. **In-Game Lockout (5-Min Cooldown):** To prevent clusters of red balls near pockets or rails from being mistaken for a new rack mid-game, a **5-minute cooldown** is armed immediately on break. The table will ignore any rack-like ball configurations until the 5-minute lockout has elapsed.

### ✏️ Process C: Manual Override (Dispute Resolution)
If a player disputes a frame count, or a camera glitch occurs, cashiers can override the counts:
1. On the active table card, locate the **Frames Played** controller.
2. Click the minus button `[-]` to reduce the count by one, or the plus button `[+]` to add one.
3. The live bill amount updates instantly on the card.

### 🔴 Process D: Ending a Session & Invoice Printing
1. Click the red **End Session & Bill** button on the active table card.
2. The **Receipt Modal** pops up, showing:
   - Customer Name & Table ID.
   - Start Time, End Time, and Total Active Duration.
   - Auto-calculated Frames Played, Game Rate (Rs.), and **Total Bill (Rs.)**.
3. Review the details with the customer.
4. Click **Confirm & Print Invoice**.
5. **UI Transition:**
   - The invoice prints (or saves a log).
   - The table card returns to the **VACANT** status badge.
   - The video engine thread resets to idle state, waiting for the next session.

---

## 🛠️ 3. Administrative Guides

All administrative configurations, calibration, and settings are located under the **ROI & System Settings** tab.

### ➕ Process E: Onboarding a New Table
1. Navigate to the **ROI & System Settings** tab and click **Add New Table**.
2. In the modal form, fill out:
   - **Unique Table ID:** Enter a string identifier with no spaces (e.g. `Table_4`).
   - **Table Display Name:** Enter the user-facing name (e.g. `Table 4`).
   - **CCTV Video Source:** Enter the input source:
     - USB webcam index (e.g. `0` for primary camera, `1` for second camera).
     - Network CCTV IP address RTSP URL (e.g. `rtsp://username:password@ip_address:554/stream`).
     - Local video file path for testing (e.g. `Video/test_match.mp4`).
3. Click **Create Table**.
4. **Result:** The table card immediately registers in all tabs.

### 📐 Process F: Calibrating the Region of Interest (ROI)
The ROI polygon restricts the AI model to scan only the playing bed, preventing false counts from players walking around the table.
1. Navigate to **ROI & System Settings**.
2. Select your target table from the dropdown. The live camera view loads on the canvas.
3. Click **Draw ROI** to activate drawing mode.
4. Click on the 4 corners of the table bed (clockwise: top-left, top-right, bottom-right, bottom-left) to draw the boundary polygon.
5. Click **Save ROI Polygon**. The AI will now ignore all activity outside this line.

### 🗑️ Process G: Deleting a Table
1. Ensure the table has no active session and its status reads **VACANT**.
2. Locate the vacant table card on the **Active Dashboard**.
3. Click the red **Trash Can** icon next to the status badge.
4. Confirm by clicking **OK** in the pop-up warning dialog.
5. **Result:** The table is removed, database records are archived, and the background CCTV video feed processor is killed.

### 💵 Process H: Editing Default Billing Rates
1. Navigate to the **ROI & System Settings** tab.
2. Under the **Billing Configurations** section, enter the price in the **Rate Per Frame / Game (Rs.)** field (e.g. `120`).
3. Click **Save Rate Configuration**. All future starting sessions will use this updated rate.

---

## 📊 4. Business Intelligence & Analytics

The **Business & Performance Analytics** tab provides dynamic, SQL-backed reports directly from the database.

```
+--------------------------------------------------------------+
|                BUSINESS & PERFORMANCE ANALYTICS              |
+--------------------------------------------------------------+
| [ Total Revenue Today ]       | [ Total Frames Played ]      |
| Rs. 24,500.00                 | 145 Games                    |
+--------------------------------------------------------------+
| [ Avg Session Duration ]      | [ Real-Time Occupancy ]      |
| 48 Mins                       | 66.7%                        |
+--------------------------------------------------------------+
```

### 📈 Process I: Querying Metrics & Loading Charts
1. Go to the **Business & Performance Analytics** tab.
2. Select your timeframe from the **Time Range** dropdown:
   - **Today:** Displays metrics only for sessions completed today.
   - **Last 7 Days:** Shows weekly statistics.
   - **All Time:** Pulls all historical records.
3. Click **Refresh Metrics** (circular arrows button).
4. **SQL-Backed Statistics Update:**
   - **Total Revenue Today (Rs.):** Sums up finalized invoice totals.
   - **Total Games Played:** Sums total frames across all tables.
   - **Average Session Duration:** Calculates average active minutes.
   - **Tables Occupancy Rate:** Real-time percentage of currently occupied tables.
5. **Interactive Charts:**
   - **Hourly Peak Occupancy (Line Chart):** Hover over data points to check table occupancy percentage curves by hour.
   - **Revenue Contribution Share (Doughnut Chart):** Hover over table segments to view exact billing totals (Rs.) and percentage share for each table.
