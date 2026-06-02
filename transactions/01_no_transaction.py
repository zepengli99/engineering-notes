import psycopg2

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")


def show(cur, label):
    cur.execute("SELECT name, balance FROM accounts ORDER BY name")
    print(label)
    for name, bal in cur.fetchall():
        print(f"  {name}: {bal}")


def main():
    conn = psycopg2.connect(**DB)
    conn.autocommit = True   # each statement commits immediately, no transaction wrapper
    cur = conn.cursor()

    show(cur, "Initial state:")

    print("\nTransferring $200 from Alice to Bob (no transaction) ...")
    cur.execute("UPDATE accounts SET balance = balance - 200 WHERE name = 'Alice'")
    print("  [step 1] debited Alice -- committed immediately")

    print("  [step 2] server crash -- credit never runs")
    # In a real crash the process dies here.
    # We just skip the second UPDATE to simulate it.

    show(cur, "\nFinal state:")
    print("\n$200 is gone. Alice was debited, Bob was never credited.")


if __name__ == "__main__":
    main()
