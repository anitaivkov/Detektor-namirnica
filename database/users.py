import sqlite3
from contextlib import contextmanager

DB_PATH = "app.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

class UserDB:
    def __init__(self):
        with get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE
                )
            ''')

    def get_or_create_user(self, username):
        with get_db() as conn:
            cur = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur = conn.execute("INSERT INTO users (username) VALUES (?)", (username,))
            conn.commit()
            return cur.lastrowid
