import psycopg2
import threading

# Phantom read: T1 runs the same WHERE query twice and gets a different number of rows
# because T2 inserted a new matching row in between.
#
# Unlike non-repeatable read (same row, different value),
# phantom read is about new rows appearing in a query range.
#
# Requires READ COMMITTED to trigger in PostgreSQL.
# PostgreSQL's REPEATABLE READ also prevents this (beyond SQL standard).

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")

t1_first_query_done = threading.Event()
t2_committed = threading.Event()


def t1():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    # cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT name FROM accounts WHERE balance > 500")
    rows1 = [r[0] for r in cur.fetchall()]
    print(f"[T1] query #1: {rows1}")

    t1_first_query_done.set()
    t2_committed.wait()

    cur.execute("SELECT name FROM accounts WHERE balance > 500")
    rows2 = [r[0] for r in cur.fetchall()]
    print(f"[T1] query #2: {rows2}" + ("  <-- phantom read" if rows2 != rows1 else ""))

    conn.commit()
    conn.close()


def t2():
    t1_first_query_done.wait()

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO accounts (name, balance) VALUES ('Charlie', 900)")
    conn.commit()
    print("[T2] inserted Charlie (balance=900) and committed")
    conn.close()

    t2_committed.set()


print("Phantom read at READ COMMITTED\n")
print("Same WHERE clause, same transaction, two different result sets.\n")

th1 = threading.Thread(target=t1)
th2 = threading.Thread(target=t2)
th1.start()
th2.start()
th1.join()
th2.join()
