import psycopg2
import threading

# Write skew: T1 and T2 both read the same aggregate (total balance),
# each decides it's safe to withdraw, each writes to a DIFFERENT row.
# No row conflict → REPEATABLE READ doesn't detect it → both commit.
# The invariant (total >= 0) is violated.
#
# Run 06a_setup.py first.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")
WITHDRAW = 150   # each withdraws 150; individually stays positive (200-150=50)
MIN_TOTAL = 200  # business rule: total must stay >= 200 (DB has no constraint for this)


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


t1_read_done = threading.Event()
t2_read_done = threading.Event()


def t1():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT SUM(balance) FROM accounts")
    total = cur.fetchone()[0]
    print(f"[T1] total = {total}, planning to withdraw {WITHDRAW} from Alice")

    t1_read_done.set()
    t2_read_done.wait()

    if total - WITHDRAW >= MIN_TOTAL:
        cur.execute("UPDATE accounts SET balance = balance - %s WHERE name = 'Alice'", (WITHDRAW,))
        conn.commit()
        print(f"[T1] withdrew {WITHDRAW} from Alice, committed")
    else:
        conn.rollback()

    conn.close()


def t2():
    t1_read_done.wait()

    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")

    cur.execute("SELECT SUM(balance) FROM accounts")
    total = cur.fetchone()[0]
    print(f"[T2] total = {total}, planning to withdraw {WITHDRAW} from Bob")
    t2_read_done.set()

    if total - WITHDRAW >= MIN_TOTAL:
        cur.execute("UPDATE accounts SET balance = balance - %s WHERE name = 'Bob'", (WITHDRAW,))
        conn.commit()
        print(f"[T2] withdrew {WITHDRAW} from Bob, committed")
    else:
        conn.rollback()

    conn.close()


show("Initial state:")
print(f"\nBoth T1 and T2 want to withdraw {WITHDRAW}. Each sees total=400 >= {MIN_TOTAL+WITHDRAW}, so both proceed.\n")

th1 = threading.Thread(target=t1)
th2 = threading.Thread(target=t2)
th1.start()
th2.start()
th1.join()
th2.join()

print()
show("Final state:")
print("\nInvariant (total >= 200) violated.")
print("REPEATABLE READ can't detect it: T1 wrote Alice, T2 wrote Bob — no row conflict.")
