import psycopg2

# PostgreSQL stores xmin and xmax as hidden columns on every row.
#
#   xmin  = transaction ID that created this row version
#   xmax  = transaction ID that deleted/replaced this row version (0 = still alive)
#
# INSERT → new row,  xmin = current txid,  xmax = 0
# UPDATE → old row:  xmin unchanged,       xmax = current txid  (marked dead)
#          new row:  xmin = current txid,  xmax = 0
# DELETE → row:      xmin unchanged,       xmax = current txid  (marked dead)

DB = dict(host="localhost", port=5433, dbname="postgres", user="postgres", password="pass")


def show_mvcc(cur, label):
    cur.execute("SELECT name, balance, xmin::text, xmax::text, ctid::text FROM accounts ORDER BY name")
    rows = cur.fetchall()
    print(f"\n{label}")
    print(f"  {'name':<10} {'balance':<10} {'xmin':<10} {'xmax':<14} {'ctid':<10}")
    print(f"  {'-'*54}")
    for name, bal, xmin, xmax, ctid in rows:
        xmax_display = xmax if xmax != '0' else '0 (alive)'
        print(f"  {name:<10} {str(bal):<10} {xmin:<10} {xmax_display:<14} {ctid:<10}")


def main():
    conn = psycopg2.connect(**DB)
    conn.autocommit = False
    cur = conn.cursor()

    show_mvcc(cur, "Initial state:")

    # --- INSERT ---
    cur.execute("SELECT txid_current()")
    txid = cur.fetchone()[0]
    print(f"\n[txid={txid}] INSERT Charlie (balance=900)")
    cur.execute("INSERT INTO accounts (name, balance) VALUES ('Charlie', 900)")
    show_mvcc(cur, "After INSERT:")
    print(f"  → Charlie.xmin = {txid}  (born in this transaction)")
    conn.commit()

    # --- UPDATE ---
    cur.execute("SELECT txid_current()")
    txid = cur.fetchone()[0]
    print(f"\n[txid={txid}] UPDATE Charlie 900 → 850")
    cur.execute("UPDATE accounts SET balance = 850 WHERE name = 'Charlie'")
    show_mvcc(cur, "After UPDATE (inside transaction, not committed yet):")
    print(f"  → Charlie.xmin = {txid}  (new version born in this transaction)")
    print(f"  → old row (balance=900) still in heap with xmax={txid}, invisible to us")
    print(f"     autovacuum will delete it once no transaction needs it")
    conn.commit()

    # --- DELETE ---
    cur.execute("SELECT txid_current()")
    txid = cur.fetchone()[0]
    print(f"\n[txid={txid}] DELETE Charlie")
    cur.execute("DELETE FROM accounts WHERE name = 'Charlie'")
    show_mvcc(cur, "After DELETE (Charlie gone from normal queries):")
    print(f"  → Charlie still exists in heap with xmax={txid}")
    print(f"     gone from SELECT results because xmax is set")
    print(f"     autovacuum will physically remove it later")
    conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
