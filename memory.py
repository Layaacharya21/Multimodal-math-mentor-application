# memory.py
import sqlite3
import json
from datetime import datetime

DB_NAME = "math_mentor_memory.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS solutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            problem_text TEXT,
            parsed_json TEXT,
            solution TEXT,
            feedback TEXT,
            corrected_solution TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_solution(problem_text, parsed_json, solution, feedback=None, corrected=None):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO solutions 
        (timestamp, problem_text, parsed_json, solution, feedback, corrected_solution)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), problem_text, json.dumps(parsed_json), solution, feedback, corrected))
    conn.commit()
    conn.close()

def find_similar_solution(problem_text, threshold=3):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT solution, feedback FROM solutions 
        WHERE problem_text LIKE ? AND feedback = 'correct'
        ORDER BY timestamp DESC LIMIT ?
    ''', (f'%{problem_text[:50]}%', threshold))
    rows = c.fetchall()
    conn.close()
    return rows[0][0] if rows else None