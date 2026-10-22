# check whether person information was successfully stored and could be retrieved.
# 18
from database import get_connection


connection = get_connection()

persons = connection.execute(
    "SELECT * FROM missing_persons"
).fetchall()

for person in persons:
    print(dict(person))

connection.close()