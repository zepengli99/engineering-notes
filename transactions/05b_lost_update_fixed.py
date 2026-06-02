import psycopg2
import threading

# Lost update happens at the application layer:
#   read value into Python memory → compute new value → write back
# The gap between read and write is where another transaction sneaks in.
#
# Fix: keep the computation inside the database with an atomic UPDATE.
# The database does read-modify-write as a single indivisible operation.
# No gap, no lost update, no need for higher isolation level.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")

def t1():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    # cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    # No SELECT — let the database compute the new value atomically
    cur.execute("UPDATE accounts SET balance = balance + 100 WHERE name = 'Alice'")
    conn.commit()
    print("[T1] atomic UPDATE +100, committed")

    conn.close()


def t2():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    # cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("UPDATE accounts SET balance = balance + 200 WHERE name = 'Alice'")
    conn.commit()
    print("[T2] atomic UPDATE +200, committed")

    conn.close()


print("Fix: atomic UPDATE keeps read-compute-write inside the database\n")
th1 = threading.Thread(target=t1)
th2 = threading.Thread(target=t2)
th1.start()
th2.start()
th1.join()
th2.join()

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()
# cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
final = cur.fetchone()[0]
conn.close()

print(f"\nFinal Alice = {final}  (expected {1000 + 100 + 200})")
print("Both updates applied correctly. No lost update.")
