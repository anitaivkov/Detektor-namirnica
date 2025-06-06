import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db(db_name):
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

class UserDB:
    def __init__(self):
        with get_db("users.db") as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def create_user(self, username):
        with get_db("users.db") as conn:
            cur = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if row:
                return row[0]  # korisnik već postoji, vrati user_id
            else:
                cur = conn.execute("INSERT INTO users (username) VALUES (?)", (username,))
                return cur.lastrowid  # vrati novi user_id


class FoodDB:
    def __init__(self):
        with get_db("foods.db") as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS foods (
                    food_id INTEGER PRIMARY KEY,
                    naziv TEXT UNIQUE,
                    kalorije INTEGER
                )
            ''')

class ListDB:
    def __init__(self):
        with get_db("lists.db") as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS shopping_lists (
                    list_id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    food_id INTEGER,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def save_list(self, user_id, items):
        with get_db("lists.db") as conn:
            for food, conf in items.items():
                food_id = self._get_food_id(food)
                conn.execute('''
                    INSERT INTO shopping_lists (user_id, food_id, confidence)
                    VALUES (?, ?, ?)
                ''', (user_id, food_id, conf))
    
    def _get_food_id(self, food_name):
        base_name = food_name.split('#')[0]  # npr. "pivo#2" postaje "pivo"
        with get_db("foods.db") as conn:
            cur = conn.execute("SELECT food_id FROM foods WHERE naziv = ?", (base_name,))
            row = cur.fetchone()
            return row[0] if row else None

