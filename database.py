import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

DB_PATH = Path("database.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS accidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT,
            time        TEXT,
            source      TEXT,
            label       TEXT,
            confidence  REAL,
            latitude    REAL,
            longitude   REAL,
            location_name TEXT,
            vehicles    TEXT
        )
    ''')
    # Add new columns if upgrading from old schema
    for col, typ in [("location_name", "TEXT"), ("vehicles", "TEXT")]:
        try:
            c.execute(f"ALTER TABLE accidents ADD COLUMN {col} {typ}")
        except:
            pass
    conn.commit()
    conn.close()

def log_accident(source, label="Accident Detected", confidence=None,
                 lat=None, lon=None, location_name=None, vehicles=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now()
    vehicles_str = ", ".join(vehicles) if isinstance(vehicles, list) else (vehicles or "")
    c.execute('''
        INSERT INTO accidents
            (date, time, source, label, confidence, latitude, longitude, location_name, vehicles)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
          source, label, confidence, lat, lon, location_name, vehicles_str))
    conn.commit()
    conn.close()

def get_accidents_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM accidents ORDER BY id DESC", conn,
                           parse_dates=["date"])
    conn.close()
    return df

def clear_accidents():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM accidents")
    conn.commit()
    conn.close()