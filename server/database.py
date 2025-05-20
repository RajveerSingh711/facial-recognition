import sqlite3
from datetime import datetime
import os

# Ensure the database directory exists
DB_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'emotions.db')

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        emotion TEXT NOT NULL,
        confidence REAL NOT NULL,
        image_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def store_prediction(emotion, confidence, image_path=None):
    """Store a prediction in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO predictions (timestamp, emotion, confidence, image_path)
    VALUES (?, ?, ?, ?)
    ''', (datetime.now(), emotion, confidence, image_path))
    
    conn.commit()
    conn.close()

def get_recent_predictions(limit=10, emotion_filter: str = None, sort_by: str = None):
    """Get predictions with optional filtering and sorting."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT timestamp, emotion, confidence, image_path FROM predictions"
    params = []
    
    if emotion_filter:
        query += " WHERE emotion = ?"
        params.append(emotion_filter)
    
    # Add sorting
    if sort_by == "timestamp_asc":
        query += " ORDER BY timestamp ASC"
    elif sort_by == "confidence_desc":
        query += " ORDER BY confidence DESC"
    elif sort_by == "confidence_asc":
        query += " ORDER BY confidence ASC"
    else: # Default sort by timestamp descending
        query += " ORDER BY timestamp DESC"
        
    query += " LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    
    results = cursor.fetchall()
    conn.close()
    
    return [{
        'timestamp': row[0],
        'emotion': row[1],
        'confidence': row[2],
        'image_path': row[3]
    } for row in results]

# Initialize the database when the module is imported
init_db() 