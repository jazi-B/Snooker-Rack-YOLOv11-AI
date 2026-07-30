import sqlite3
import json
import datetime
from pathlib import Path

DB_FILE = "snooker_dashboard.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tables (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        camera_source TEXT NOT NULL,
        roi_polygon TEXT,
        is_active INTEGER DEFAULT 1
    )
    """)
    
    # Create sessions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id TEXT NOT NULL,
        customer_name TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        games_played INTEGER DEFAULT 0,
        game_rate REAL DEFAULT 10.0,
        total_bill REAL DEFAULT 0.0,
        status TEXT DEFAULT 'active',
        FOREIGN KEY (table_id) REFERENCES tables (id)
    )
    """)
    
    # Create settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)
    
    # Check if we should insert default tables
    cursor.execute("SELECT COUNT(*) FROM tables")
    if cursor.fetchone()[0] == 0:
        # Default tables: Table 1, Table 2, Table 3.
        # We can use "0" for the webcam stream as default source, or a placeholder RTSP source
        cursor.execute("INSERT INTO tables (id, name, camera_source) VALUES (?, ?, ?)", ("Table_1", "Table 1", "0"))
        cursor.execute("INSERT INTO tables (id, name, camera_source) VALUES (?, ?, ?)", ("Table_2", "Table 2", "0"))
        cursor.execute("INSERT INTO tables (id, name, camera_source) VALUES (?, ?, ?)", ("Table_3", "Table 3", "0"))
    
    # Check if we should insert default settings
    cursor.execute("SELECT COUNT(*) FROM settings WHERE key = 'game_rate'")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ("game_rate", "10.0"))
        
    conn.commit()
    conn.close()

# --- TABLES CRUD ---

def get_all_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tables WHERE is_active = 1")
    rows = cursor.fetchall()
    tables = [dict(row) for row in rows]
    for t in tables:
        if t['roi_polygon']:
            t['roi_polygon'] = json.loads(t['roi_polygon'])
        else:
            t['roi_polygon'] = []
    conn.close()
    return tables

def add_table(table_id: str, name: str, camera_source: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO tables (id, name, camera_source) VALUES (?, ?, ?)",
            (table_id, name, camera_source)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def delete_table(table_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE tables SET is_active = 0 WHERE id = ?", (table_id,))
    conn.commit()
    conn.close()
    return True

def update_table_roi(table_id: str, roi_points: list):
    conn = get_db_connection()
    cursor = conn.cursor()
    roi_json = json.dumps(roi_points)
    cursor.execute("UPDATE tables SET roi_polygon = ? WHERE id = ?", (roi_json, table_id))
    conn.commit()
    conn.close()
    return True

# --- SESSIONS CRUD ---

def get_active_session(table_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE table_id = ? AND status = 'active'", (table_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def start_session(table_id: str, customer_name: str, rate: float):
    # Ensure no active session already exists for this table
    active = get_active_session(table_id)
    if active:
        return active
    
    conn = get_db_connection()
    cursor = conn.cursor()
    start_time = datetime.datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO sessions (table_id, customer_name, start_time, game_rate, games_played, total_bill, status) VALUES (?, ?, ?, ?, 0, 0.0, 'active')",
        (table_id, customer_name, start_time, rate)
    )
    conn.commit()
    session_id = cursor.lastrowid
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def update_session_games(session_id: int, games_played: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Fetch current rate
    cursor.execute("SELECT game_rate FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    rate = row['game_rate']
    total_bill = games_played * rate
    
    cursor.execute(
        "UPDATE sessions SET games_played = ?, total_bill = ? WHERE id = ?",
        (games_played, total_bill, session_id)
    )
    conn.commit()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    updated_row = cursor.fetchone()
    conn.close()
    return dict(updated_row)

def end_session(session_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    end_time = datetime.datetime.now().isoformat()
    cursor.execute(
        "UPDATE sessions SET end_time = ?, status = 'completed' WHERE id = ?",
        (end_time, session_id)
    )
    conn.commit()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row)

def get_completed_sessions(limit: int = 50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE status = 'completed' ORDER BY end_time DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# --- SETTINGS CRUD ---

def get_setting(key: str, default: str = None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else default

def set_setting(key: str, value: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
    return True
