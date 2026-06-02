import psycopg2
import threading

# Non-repeatable read: T1 reads the same row twice and gets different values.
# Happens at READ COMMITTED because each statement takes a fresh snapshot.
# Fix: REPEATABLE READ — one snapshot for the entire transaction.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")

t1_first_read_done = threading.Event()
t2_committed = threading.Event()


def t1():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    # cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
    b1 = cur.fetchone()[0]
    print(f"[T1] read #1: Alice = {b1}")

    t1_first_read_done.set()   # let T2 run
    t2_committed.wait()        # wait for T2 to commit

    cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
    b2 = cur.fetchone()[0]
    print(f"[T1] read #2: Alice = {b2}" + ("  <-- non-repeatable read" if b2 != b1 else ""))

    conn.commit()
    conn.close()


def t2():
    t1_first_read_done.wait()

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("UPDATE accounts SET balance = 800 WHERE name = 'Alice'")
    conn.commit()
    print("[T2] updated Alice to 800 and committed")
    conn.close()

    t2_committed.set()


print("READ COMMITTED — each statement takes a fresh snapshot\n")
th1 = threading.Thread(target=t1)
th2 = threading.Thread(target=t2)
th1.start()
th2.start()
th1.join()
th2.join()
