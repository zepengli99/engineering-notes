import psycopg2
import threading

# Two fixes for write skew:
#
# Fix 1: SELECT FOR UPDATE — lock all rows being read.
#   T2's SELECT FOR UPDATE blocks until T1 commits.
#   T2 then reads the updated total and correctly rejects the withdrawal.
#
# Fix 2: SERIALIZABLE — PostgreSQL SSI detects the read-write dependency
#   and aborts one transaction. App must catch and retry.
#
# No Event coordination needed: FOR UPDATE makes T2 wait naturally.
# Run 06a_setup.py first.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")
WITHDRAW = 150
MIN_TOTAL = 200

# FIX = "for_update"  # or "serializable"
FIX = "serializable"


def show(label):
    conn = psycopg2.connect(**DB)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT name, balance FROM accounts ORDER BY name")
    rows = cur.fetchall()
    print(label)
    for name, bal in rows:
        print(f"  {name}: {bal}")
    print(f"  total: {sum(b for _, b in rows)}")
    conn.close()


def withdraw(name, fix):
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()

    if fix == "serializable":
        cur.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
    else:
        cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")

    try:
        if fix == "for_update":
            cur.execute("SELECT balance FROM accounts FOR UPDATE")
            total = sum(row[0] for row in cur.fetchall())
        else:
            cur.execute("SELECT SUM(balance) FROM accounts")
            total = cur.fetchone()[0]

        print(f"[{name}] total = {total}")

        if total - WITHDRAW >= MIN_TOTAL:
            cur.execute(
                "UPDATE accounts SET balance = balance - %s WHERE name = %s",
                (WITHDRAW, name)
            )
            conn.commit()
            print(f"[{name}] withdrew {WITHDRAW}, committed")
        else:
            conn.rollback()
            print(f"[{name}] total too low, rolled back")

    except psycopg2.errors.SerializationFailure:
        conn.rollback()
        print(f"[{name}] SERIALIZABLE conflict, aborted (app should retry)")

    conn.close()


print(f"Fix: {FIX}\n")
show("Initial state:")
print()

th1 = threading.Thread(target=withdraw, args=("Alice", FIX))
th2 = threading.Thread(target=withdraw, args=("Bob", FIX))
th1.start()
th2.start()
th1.join()
th2.join()

print()
show("Final state:")
