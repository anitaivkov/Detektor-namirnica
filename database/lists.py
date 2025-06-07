import sqlite3
from contextlib import contextmanager
from datetime import datetime
from database.food import FoodDB

@contextmanager
def get_db():
    conn = sqlite3.connect("app.db")
    try:
        yield conn
    finally:
        conn.close()

class ListDB:
    def __init__(self, food_db_instance):
        self.food_db = food_db_instance
        self._initialize_db()

    def _initialize_db(self):
        with get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS shopping_lists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    food_id INTEGER,
                    naziv TEXT,
                    count INTEGER,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    # Promijenjeno da prima conn objekt
    def _get_food_id(self, food_name, conn):
        # Sada prosljeđuje conn objekt metodi FoodDB-a
        return self.food_db.get_or_create_food_id(food_name, conn)

    def save_list(self, user_id, items):
        now = datetime.now()

        with get_db() as conn: # Jedna veza za cijelu save_list operaciju
            for food_name, data in items.items():
                # Sada prosljeđuje istu 'conn' vezu
                food_id = self._get_food_id(food_name, conn)
                if food_id is None:
                    print(f"Upozorenje: Nije pronađen food_id za '{food_name}'. Stavka neće biti spremljena.")
                    continue

                conn.execute(
                    "INSERT INTO shopping_lists (user_id, food_id, naziv, count, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, food_id, food_name, data["count"], data["confidence"], now.isoformat())
                )
            conn.commit() # Commit na kraju transakcije

    def get_all_lists(self, user_id):
        with get_db() as conn:
            cur = conn.execute('''
                SELECT l.timestamp, f.naziv, l.count, l.confidence
                FROM shopping_lists l
                JOIN foods f ON l.food_id = f.food_id
                WHERE l.user_id = ?
                ORDER BY l.timestamp DESC
            ''', (user_id,))
            return cur.fetchall()

    def get_list_items(self, user_id, naziv):
        with get_db() as conn:
            cur = conn.execute('''
                SELECT f.naziv, l.count, l.confidence, l.timestamp
                FROM shopping_lists l
                JOIN foods f ON l.food_id = f.food_id
                WHERE l.user_id = ? AND f.naziv = ?
                ORDER BY l.timestamp DESC
            ''', (user_id, naziv))
            return cur.fetchall()

    def get_unique_timestamps(self, user_id):
        with get_db() as conn:
            cur = conn.execute('''
                SELECT DISTINCT timestamp
                FROM shopping_lists
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id,))
            rows = cur.fetchall()
            return [datetime.fromisoformat(row[0]) for row in rows]

    def get_list_items_by_timestamp(self, user_id, timestamp):
        with get_db() as conn:
            cur = conn.execute('''
                SELECT f.naziv, l.count, l.confidence
                FROM shopping_lists l
                JOIN foods f ON l.food_id = f.food_id
                WHERE l.user_id = ? AND l.timestamp = ?
                ORDER BY f.naziv
            ''', (user_id, timestamp.isoformat()))
            return cur.fetchall()
