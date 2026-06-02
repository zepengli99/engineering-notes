import psycopg2

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")


def show(cur, label):
    cur.execute("SELECT name, balance FROM accounts ORDER BY name")
    print(label)
    for name, bal in cur.fetchall():
        print(f"  {name}: {bal}")


def main():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False  # changes go into a staging area until COMMIT or ROLLBACK
    cur = conn.cursor()

    show(cur, "Initial state:")

    print("\nTransferring $200 from Alice to Bob (inside a transaction) ...")

    try:
        cur.execute("UPDATE accounts SET balance = balance - 200 WHERE name = 'Alice'")
        print("  [step 1] debited Alice  (not committed yet)")

        print("  [step 2] something goes wrong ...")
        raise ValueError("payment service unavailable")

        cur.execute("UPDATE accounts SET balance = balance + 200 WHERE name = 'Bob'")
        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"  ERROR: {e}")
        print("  ROLLBACK -- step 1 is undone")

    show(cur, "\nFinal state:")
    print("\nBoth accounts unchanged. The transaction treated two steps as one unit.")


if __name__ == "__main__":
    main()
