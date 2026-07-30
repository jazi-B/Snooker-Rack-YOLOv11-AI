import os
import asyncio
import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import cv2

from backend.app.core.config import settings
from backend.app.db import database
from backend.app.services.video_engine import video_engine

# FastAPI Initialization
app = FastAPI(title=settings.PROJECT_NAME)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup & Shutdown events
@app.on_event("startup")
def startup_event():
    # Initialize DB tables
    database.init_db()
    # Start background threads for video feeds
    video_engine.start_all_processors()
    print("[*] FastAPI Application successfully started and DB/Video Engines initialized.")

@app.on_event("shutdown")
def shutdown_event():
    # Gracefully stop video threads
    video_engine.stop_all_processors()
    print("[*] Application shutdown: camera processors cleaned up.")

# --- REQUEST SCHEMAS ---

class TableCreate(BaseModel):
    id: str
    name: str
    camera_source: str

class RoiUpdate(BaseModel):
    points: List[List[float]]

class SessionStart(BaseModel):
    table_id: str
    customer_name: str
    rate: Optional[float] = None

class SessionAdjust(BaseModel):
    games_played: int

class SettingsUpdate(BaseModel):
    game_rate: float

# --- API ROUTES ---

# 1. TABLES ENDPOINTS

@app.get("/api/tables")
def get_tables():
    tables = database.get_all_tables()
    for t in tables:
        t['active_session'] = database.get_active_session(t['id'])
        # Add real-time video engine processor flags
        processor = video_engine.processors.get(t['id'])
        if processor:
            t['rack_present'] = processor.rack_present
            t['game_logic_started'] = processor.game_logic_started
        else:
            t['rack_present'] = False
            t['game_logic_started'] = False
    return tables

@app.post("/api/tables")
def create_table(data: TableCreate):
    # Ensure ID has proper format
    table_id = data.id.strip().replace(" ", "_")
    success = database.add_table(table_id, data.name, data.camera_source)
    if not success:
        raise HTTPException(status_code=400, detail="Table ID already exists.")
    
    # Start video engine processor for new table
    video_engine.update_table_source(table_id, data.camera_source)
    return {"status": "success", "detail": f"Table '{data.name}' added successfully."}

@app.delete("/api/tables/{table_id}")
def delete_table(table_id: str):
    database.delete_table(table_id)
    # Stop processor
    video_engine.update_table_source(table_id, "")
    return {"status": "success"}

@app.post("/api/tables/{table_id}/roi")
def update_roi(table_id: str, data: RoiUpdate):
    database.update_table_roi(table_id, data.points)
    # Notify video engine processor to reload ROI
    video_engine.refresh_roi(table_id)
    return {"status": "success", "detail": "ROI updated successfully."}

# 2. SESSIONS ENDPOINTS

@app.post("/api/sessions/start")
def start_table_session(data: SessionStart):
    rate = data.rate
    if rate is None:
        try:
            rate = float(database.get_setting("game_rate", "10.0"))
        except ValueError:
            rate = 10.0
            
    session = database.start_session(data.table_id, data.customer_name, rate)
    return {"status": "success", "session": session}

@app.post("/api/sessions/{session_id}/adjust")
def adjust_session_games(session_id: int, data: SessionAdjust):
    session = database.update_session_games(session_id, data.games_played)
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found.")
    return {"status": "success", "session": session}

@app.post("/api/sessions/{session_id}/end")
def end_table_session(session_id: int):
    session = database.end_session(session_id)
    return {"status": "success", "session": session}

@app.get("/api/sessions/history")
def get_session_history():
    return database.get_completed_sessions(limit=50)

# 3. SETTINGS ENDPOINTS

@app.get("/api/settings")
def get_system_settings():
    game_rate = database.get_setting("game_rate", "10.0")
    return {"game_rate": float(game_rate)}

@app.post("/api/settings")
def update_system_settings(data: SettingsUpdate):
    database.set_setting("game_rate", str(data.game_rate))
    return {"status": "success", "detail": "Settings updated."}



# 3b. ANALYTICS ENDPOINTS

@app.get("/api/analytics")
def get_analytics(range: str = "7days"):
    import builtins
    conn = database.get_db_connection()
    cursor = conn.cursor()
    
    # Calculate date filtering based on range
    # range can be: "today", "7days", "all"
    today_str = datetime.date.today().isoformat()
    
    date_filter = ""
    params = []
    
    if range == "today":
        date_filter = "AND end_time LIKE ?"
        params.append(f"{today_str}%")
    elif range == "7days":
        # Calculate date 7 days ago
        seven_days_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        date_filter = "AND end_time >= ?"
        params.append(seven_days_ago)
    # for "all", date_filter is empty (no restriction)

    # 1. Total Revenue
    # Completed sessions in range
    cursor.execute(f"SELECT SUM(total_bill) FROM sessions WHERE status = 'completed' {date_filter}", params)
    completed_revenue = cursor.fetchone()[0] or 0.0
    
    # Active sessions (regardless of range, they are currently running and generating revenue now)
    cursor.execute("SELECT SUM(games_played * game_rate) FROM sessions WHERE status = 'active'")
    active_revenue = cursor.fetchone()[0] or 0.0
    
    total_revenue = completed_revenue + active_revenue
    
    # 2. Total Games
    cursor.execute(f"SELECT SUM(games_played) FROM sessions WHERE status = 'completed' {date_filter}", params)
    completed_games = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT SUM(games_played) FROM sessions WHERE status = 'active'")
    active_games = cursor.fetchone()[0] or 0
    
    total_games = completed_games + active_games
    
    # 3. Average Session Duration (in minutes)
    cursor.execute(f"SELECT start_time, end_time FROM sessions WHERE status = 'completed' {date_filter}", params)
    completed_rows = cursor.fetchall()
    
    durations = []
    for row in completed_rows:
        try:
            start = datetime.datetime.fromisoformat(row['start_time'])
            end = datetime.datetime.fromisoformat(row['end_time'])
            durations.append((end - start).total_seconds() / 60.0)
        except Exception:
            pass
            
    # Include currently active sessions' duration so far if range is today/all, or always since they are active
    cursor.execute("SELECT start_time FROM sessions WHERE status = 'active'")
    active_rows = cursor.fetchall()
    for row in active_rows:
        try:
            start = datetime.datetime.fromisoformat(row['start_time'])
            end = datetime.datetime.now()
            durations.append((end - start).total_seconds() / 60.0)
        except Exception:
            pass
            
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    
    # 4. Table Occupancy Rate (percentage of tables currently active)
    cursor.execute("SELECT COUNT(*) FROM tables WHERE is_active = 1")
    total_tables = cursor.fetchone()[0] or 1
    
    cursor.execute("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
    active_tables = cursor.fetchone()[0] or 0
    
    occupancy_rate = (active_tables / total_tables) * 100.0 if total_tables else 0.0
    
    # 5. Revenue Contribution Share by Table (based on filtered sessions)
    cursor.execute("SELECT id, name FROM tables WHERE is_active = 1")
    tables_list = cursor.fetchall()
    
    revenue_share = {}
    for t in tables_list:
        table_id = t['id']
        table_name = t['name']
        
        # Completed revenue for this table
        cursor.execute(f"SELECT SUM(total_bill) FROM sessions WHERE table_id = ? AND status = 'completed' {date_filter}", [table_id] + params)
        comp_rev = cursor.fetchone()[0] or 0.0
        
        # Active revenue for this table
        cursor.execute("SELECT SUM(games_played * game_rate) FROM sessions WHERE table_id = ? AND status = 'active'", (table_id,))
        act_rev = cursor.fetchone()[0] or 0.0
        
        revenue_share[table_name] = comp_rev + act_rev
        
    # 6. Hourly Occupancy Rate
    # For line chart: occupancy rate of snooker tables across different hours of the day (0-23)
    hourly_occupancy = []
    
    for h in builtins.range(24):
        h_str = f"{h:02d}"
        
        if range == "today":
            # For today, check actual overlap today
            hour_start_str = f"{today_str}T{h_str}:00:00"
            hour_end_str = f"{today_str}T{h_str}:59:59"
            cursor.execute("""
                SELECT COUNT(DISTINCT table_id) FROM sessions 
                WHERE (start_time <= ?) AND (end_time IS NULL OR end_time >= ?)
            """, (hour_end_str, hour_start_str))
            occupied_count = cursor.fetchone()[0] or 0
            rate = (occupied_count / total_tables) * 100.0 if total_tables else 0.0
        else:
            # For 7days or all, check average occupancy across all recorded days in that period
            cursor.execute(f"""
                SELECT COUNT(*) FROM sessions 
                WHERE status = 'completed' {date_filter} 
                AND strftime('%H', start_time) <= ? 
                AND strftime('%H', end_time) >= ?
            """, params + [h_str, h_str])
            comp_count = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT COUNT(*) FROM sessions 
                WHERE status = 'active'
                AND strftime('%H', start_time) <= ?
            """, [h_str])
            act_count = cursor.fetchone()[0] or 0
            
            total_active_at_hour = comp_count + act_count
            
            # Find number of unique days in range to average
            cursor.execute(f"SELECT COUNT(DISTINCT date(start_time)) FROM sessions WHERE 1=1 {date_filter}", params)
            unique_days = cursor.fetchone()[0] or 1
            
            rate = (total_active_at_hour / (total_tables * unique_days)) * 100.0 if total_tables else 0.0
            
        hourly_occupancy.append(min(100.0, round(rate, 1)))
        
    conn.close()
    
    return {
        "revenue_today": round(total_revenue, 2),
        "games_today": int(total_games),
        "avg_duration_mins": round(avg_duration, 1),
        "occupancy_rate": round(occupancy_rate, 1),
        "revenue_share": revenue_share,
        "hourly_occupancy": hourly_occupancy
    }

# 4. CAMERA MJPEG STREAMING ENDPOINT

async def mjpeg_frame_generator(table_id: str):
    while True:
        frame = video_engine.get_feed_frame(table_id)
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                # Yield frame bytes in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        # Rate-limiting streaming to ~30 FPS
        await asyncio.sleep(0.033)

@app.get("/api/tables/{table_id}/feed")
def get_table_video_feed(table_id: str):
    # Check if table exists
    tables = database.get_all_tables()
    table_exists = any(t['id'] == table_id for t in tables)
    if not table_exists:
        raise HTTPException(status_code=404, detail="Table not found.")
        
    return StreamingResponse(
        mjpeg_frame_generator(table_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# --- SERVE FRONTEND STATIC FILES ---

# Ensure frontend directory exists
os.makedirs("frontend", exist_ok=True)

# Mount the static files directory at "/" root.
# NOTE: Mount this last, otherwise static file routing overrides API endpoints.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
