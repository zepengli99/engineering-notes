import psycopg2
import threading

# Lost update: T1 and T2 both read Alice's balance, compute new value, write back.
# T2 reads before T1 commits, so T2's write overwrites T1's update.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")

t1_read_done = threading.Event()
t2_read_done = threading.Event()
t1_committed = threading.Event()


def t1():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
    balance = cur.fetchone()[0]
    print(f"[T1] read Alice = {balance}")

    t1_read_done.set()
    t2_read_done.wait()  # both have read before either writes

    new_balance = balance + 100
    cur.execute("UPDATE accounts SET balance = %s WHERE name = 'Alice'", (new_balance,))
    conn.commit()
    print(f"[T1] wrote Alice = {new_balance} (+100), committed")
    t1_committed.set()

    conn.close()


def t2():
    t1_read_done.wait()

    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
    balance = cur.fetchone()[0]
    print(f"[T2] read Alice = {balance}  (same old value, T1 hasn't committed yet)")
    t2_read_done.set()

    t1_committed.wait()  # wait for T1 to commit first, then overwrite

    new_balance = balance + 200
    cur.execute("UPDATE accounts SET balance = %s WHERE name = 'Alice'", (new_balance,))
    conn.commit()
    print(f"[T2] wrote Alice = {new_balance} (+200 on top of original 1000), committed")

    conn.close()


print("Lost update: both transactions read 1000, compute independently, last write wins\n")
th1 = threading.Thread(target=t1)
th2 = threading.Thread(target=t2)
th1.start()
th2.start()
th1.join()
th2.join()

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT balance FROM accounts WHERE name = 'Alice'")
final = cur.fetchone()[0]
conn.close()

print(f"\nFinal Alice = {final}  (expected {1000 + 100 + 200} if both applied correctly)")
print("T1's +100 was lost.")
