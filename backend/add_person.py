# This file was used to add missing-person records to the database during development
# 17
from database import get_connection


def add_person():
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO missing_persons
        (
            id,
            name,
            age,
            gender,
            last_seen_location,
            last_seen_date,
            description,
            image_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "MP001",
            "Test Person",
            24,
            "Male",
            "Pune",
            "2026-09-30",
            "Test missing person record",
            "data/persons/MP001"
        )
    )

    connection.commit()
    connection.close()

    print("Missing person added successfully.")


if __name__ == "__main__":
    add_person()