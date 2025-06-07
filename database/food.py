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

    def add_food(self, naziv, kalorije=0, conn=None):
        if conn is None:
            with get_db() as conn:
                self._add_food_internal(naziv, kalorije, conn)
        else:
            self._add_food_internal(naziv, kalorije, conn)

    def _add_food_internal(self, naziv, kalorije, conn):
        try:
            conn.execute(
                "INSERT INTO foods (naziv, kalorije) VALUES (?, ?)",
                (naziv, kalorije)
            )
        except sqlite3.IntegrityError:
            pass # već postoji


    # metoda sada prihvaća 'conn'
    def get_or_create_food_id(self, naziv, conn=None):
        """Vrati ID hrane; ako ne postoji, dodaj je. Koristi proslijeđenu vezu ako postoji."""
        
        def _get_or_create(current_conn):
            # Prvo pokušaj pronaći
            cur = current_conn.execute("SELECT food_id FROM foods WHERE naziv = ?", (naziv,))
            row = cur.fetchone()
            if row:
                return row[0]

            # Ako ne postoji, ubaci novi unos
            current_conn.execute("INSERT INTO foods (naziv, kalorije) VALUES (?, ?)", (naziv, 0))
            # Ovdje NEMA commita, jer ako je conn proslijeđen, commit će se obaviti na višoj razini.
            # Ako je conn lokalno otvoren (slučaj 'with get_db()'), context manager će ga automatski commitati.
            return current_conn.execute("SELECT food_id FROM foods WHERE naziv = ?", (naziv,)).fetchone()[0]

        if conn is None:
            # Ako veza nije proslijeđena, otvori vlastitu vezu i obavi posao
            with get_db() as local_conn:
                return _get_or_create(local_conn)
        else:
            # Koristi proslijeđenu vezu
            return _get_or_create(conn)


    def get_all_foods(self, conn=None):
        if conn is None:
            with get_db() as conn:
                cur = conn.execute("SELECT naziv FROM foods")
                return [row[0] for row in cur.fetchall()]
        else:
            cur = conn.execute("SELECT naziv FROM foods")
            return [row[0] for row in cur.fetchall()]
