# is like the record-keeping department of your project—it helps the AI system and dashboard access stored information.
# 16
import sqlite3

DATABASE_FILE = "data/missing_persons.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_connection()

    # Missing persons table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS missing_persons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            last_seen_location TEXT,
            last_seen_date TEXT,
            description TEXT,
            image_path TEXT
        )
    """)

    # Possible match alerts table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            person_id TEXT,
            similarity REAL,
            observations INTEGER,
            camera_id TEXT,
            camera_location TEXT,
            captured_image_path TEXT,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified_at TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_tables()
    print("Database initialized successfully.")