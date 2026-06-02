import psycopg2

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")


def main():
    conn = psycopg2.connect(**DB)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS accounts")
    cur.execute("""
        CREATE TABLE accounts (
            name    TEXT    PRIMARY KEY,
            balance INTEGER NOT NULL CHECK (balance >= 0)
        )
    """)
    cur.execute("INSERT INTO accounts (name, balance) VALUES ('Alice', 1000), ('Bob', 500)")

    cur.execute("SELECT name, balance FROM accounts ORDER BY name")
    print("DB reset. Current state:")
    for name, bal in cur.fetchall():
        print(f"  {name}: {bal}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
