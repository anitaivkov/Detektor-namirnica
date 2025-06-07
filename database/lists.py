import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect("app.db")
    try:
        yield conn
    finally:
        conn.close()

class ListDB:
    def __init__(self):
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

    def _get_food_id(self, food_name):
        from database.food import FoodDB
        food_db = FoodDB()
        return food_db.get_or_create_food_id(food_name)

    def save_list(self, user_id, items):
        with get_db() as conn:
            for food, data in items.items():
                food_id = self._get_food_id(food)
                if food_id is None:
                    continue

                # Provjera postoji li već red za (user_id, food_id)
                cur = conn.execute('''
                    SELECT count, confidence FROM shopping_lists
                    WHERE user_id = ? AND food_id = ?
                ''', (user_id, food_id))
                row = cur.fetchone()

                if row:
                    new_count = row[0] + data["count"]
                    new_conf = max(row[1], data["confidence"])
                    conn.execute('''
                        UPDATE shopping_lists
                        SET count = ?, confidence = ?, timestamp = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND food_id = ?
                    ''', (new_count, new_conf, user_id, food_id))
                else:
                    conn.execute('''
                        INSERT INTO shopping_lists (user_id, food_id, naziv, count, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, food_id, food, data["count"], data["confidence"]))

            conn.commit()

    def get_lists(self, user_id):
        with get_db() as conn:
            cur = conn.execute('''
                SELECT l.count, l.confidence, l.timestamp, f.naziv
                FROM shopping_lists l
                JOIN foods f ON l.food_id = f.food_id
                WHERE l.user_id = ?
                ORDER BY l.timestamp DESC
            ''', (user_id,))
            return cur.fetchall()