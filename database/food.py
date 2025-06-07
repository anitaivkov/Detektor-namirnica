import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect("app.db")
    try:
        yield conn
    finally:
        conn.close()


class FoodDB:
    def __init__(self):
        self._initialize_table()

    def _initialize_table(self):
        with get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS foods (
                    food_id INTEGER PRIMARY KEY,
                    naziv TEXT UNIQUE,
                    kalorije INTEGER
                )
            ''')

    def add_food(self, naziv, kalorije=0):
        with get_db() as conn:
            try:
                conn.execute(
                    "INSERT INTO foods (naziv, kalorije) VALUES (?, ?)",
                    (naziv, kalorije)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # već postoji

    def get_all_foods(self):
        with get_db() as conn:
            cur = conn.execute("SELECT naziv FROM foods")
            return [row[0] for row in cur.fetchall()]

    def get_or_create_food_id(self, naziv):
        """Vrati ID hrane; ako ne postoji, dodaj je."""
        with get_db() as conn:
            # Prvo pokušaj pronaći
            cur = conn.execute("SELECT food_id FROM foods WHERE naziv = ?", (naziv,))
            row = cur.fetchone()
            if row:
                return row[0]

            # Ako ne postoji, ubaci novi unos
            cur = conn.execute("INSERT INTO foods (naziv, kalorije) VALUES (?, ?)", (naziv, 0))
            conn.commit()
            return cur.lastrowid
