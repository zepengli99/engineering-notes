import psycopg2

# Setup for write skew demo: small balances to make the invariant easy to violate.

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")

conn = psycopg2.connect(**DB)
conn.autocommit = True
cur = conn.cursor()
cur.execute("UPDATE accounts SET balance = 200 WHERE name = 'Alice'")
cur.execute("UPDATE accounts SET balance = 200 WHERE name = 'Bob'")
cur.execute("DELETE FROM accounts WHERE name NOT IN ('Alice', 'Bob')")

cur.execute("SELECT name, balance FROM accounts ORDER BY name")
print("State for write skew demo:")
for name, bal in cur.fetchall():
    print(f"  {name}: {bal}")
print(f"  total: 400")
print(f"  business rule: total must stay >= 200")
conn.close()
