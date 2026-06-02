# Database Transactions & ACID

Personal notes from working through these concepts. The acronym is easy to say and hard to actually understand — especially Isolation, which has four levels, three anomalies, and an implementation (MVCC) that goes surprisingly deep.

Scripts are meant to be run in order. Each one builds on the last.

---

## Prerequisites

```bash
# Start PostgreSQL (Docker)
docker run --rm --name pg-acid -e POSTGRES_PASSWORD=pass -p 5433:5432 -d postgres

pip install psycopg2-binary

# Reset DB before each demo
python 00_setup.py
```

---

## The 30-second isolation level guide

```
Default for most apps (web APIs, OLTP)           →  READ COMMITTED (PostgreSQL default)
Long reads that must see a consistent snapshot   →  REPEATABLE READ
Prevent all anomalies including write skew       →  SERIALIZABLE (SSI in PostgreSQL)
Two transactions might update the same row       →  atomic UPDATE, or SELECT FOR UPDATE
```

---

## Scripts

| File | Concept |
|---|---|
| [00_setup.py](00_setup.py) | Reset DB: Alice=1000, Bob=500 |
| [01_no_transaction.py](01_no_transaction.py) | Partial failure without a transaction |
| [02_atomicity.py](02_atomicity.py) | Rollback undoes everything |
| [03_nonrepeatable_read.py](03_nonrepeatable_read.py) | Same row read twice, different value |
| [03b_event_demo.py](03b_event_demo.py) | threading.Event explained |
| [04_phantom_read.py](04_phantom_read.py) | Same WHERE clause, different row count |
| [04b_mvcc_versions.py](04b_mvcc_versions.py) | xmin/xmax/ctid made visible |
| [05_lost_update.py](05_lost_update.py) | Two writers, one update silently lost |
| [05b_lost_update_fixed.py](05b_lost_update_fixed.py) | Atomic UPDATE eliminates the gap |
| [06_write_skew.py](06_write_skew.py) | Reads shared state, writes disjoint rows, invariant broken |
| [06a_setup.py](06a_setup.py) | Reset DB for write skew demo |
| [06b_write_skew_fixed.py](06b_write_skew_fixed.py) | FOR UPDATE and SERIALIZABLE fixes |

---

## Concepts

### Why transactions exist

Transfer $200 from Alice to Bob:

```
Step 1: Alice -200
Step 2: Bob   +200
```

What if the server crashes after step 1? Without a transaction, Alice lost $200 and Bob never got it. The database is in a state that should never exist.

With a transaction: either both steps happen, or neither does.

---

### Atomicity

All operations in a transaction commit together, or roll back together.

```
BEGIN
  UPDATE Alice -200   ← not yet permanent
  UPDATE Bob   +200   ← not yet permanent
COMMIT               ← both become permanent atomically
```

If anything fails: `ROLLBACK` undoes everything back to before the `BEGIN`.

> **Q I had: isn't this the same as a thread lock?**
> Same idea, different layer. A lock makes read-modify-write atomic in a single process. A transaction makes a sequence of SQL operations atomic across a whole database. The pattern — "bundle multiple steps into one indivisible unit" — appears at every layer: CPU LOCK prefix, mutex, DB transaction, distributed saga. Bigger scope, higher cost.

> **Q I had: COMMIT is like git commit — can I undo it?**
> Similar concept, but after SQL COMMIT there's no built-in undo. Git has `git revert` because it stores complete history. SQL databases don't keep application-level undo history after commit. To reverse a committed transaction you write a compensating transaction manually — a new operation that undoes the effect. Not a rollback, just a new entry.
>
> | | Git | SQL |
> |---|---|---|
> | Before committing | `git checkout` | `ROLLBACK` |
> | After committing | `git revert` (history preserved) | manual compensating query |

**How atomicity works — WAL (Write-Ahead Log):**

Before writing to data files, PostgreSQL writes to a sequential log:

```
[txn-42] BEGIN
[txn-42] UPDATE Alice: 1000 → 800  (old value recorded too)
[txn-42] UPDATE Bob:    500 → 700
[txn-42] COMMIT → WAL flushed to disk (fsync)
```

On crash: PostgreSQL reads the WAL. Transactions with COMMIT: redo. Transactions without COMMIT: undo using recorded old values.

> **Q I had: if WAL isn't deleted, can we restore to any point in time?**
> Yes — this is PITR (Point-in-Time Recovery). Take a base backup, then replay WAL forward to any target time. But PITR replays *forward* from an old snapshot; you can't play WAL backward. You restore to a point *before* the mistake and replay up to just before it.
>
> Replaying the entire WAL from the beginning would be slow, which is why PostgreSQL uses **checkpoints** — periodic markers that say "all data pages are flushed to disk as of this point." Crash recovery only replays from the nearest checkpoint, not from the start.
>
> ```
> Day 1 base backup → WAL → Checkpoint → WAL → Checkpoint → WAL → crash
>                                                              ↑
>                                              recovery starts here, not Day 1
> ```

**WAL vs MVCC:**

These are two separate mechanisms that both "keep old data" but for different purposes:

```
WAL    → operation log on disk, for crash recovery and replication
MVCC   → multiple row versions in the heap file, for concurrent reads
```

WAL records what happened. MVCC stores what the data looked like at different points. They don't depend on each other.

---

### Consistency

The C that's different from the others.

A, I, D are **database guarantees** — the engine enforces them mechanically.

C is an **application guarantee** — you enforce it by writing correct transactions.

What the database enforces: `CHECK`, `UNIQUE`, `FOREIGN KEY`, `NOT NULL`.

What the database does NOT enforce: business invariants.

```sql
-- DB enforces this:
CHECK (balance >= 0)

-- You enforce this by writing correct transactions:
-- "Total money across all accounts never changes"
```

> **Q I had: is ACID consistency the same as CAP consistency?**
> No — completely different use of the word. CAP "consistency" means every read sees the most recent write across distributed nodes. ACID "consistency" means the database stays in a valid state per defined rules. The naming collision causes endless confusion.

---

### Isolation

The most complex property. Many transactions run concurrently — without isolation, they interfere.

#### Three anomalies

**Dirty Read** — reading data another transaction wrote but hasn't committed yet.

```
T1: UPDATE Alice 1000 → 800  (not committed)
T2: SELECT Alice → 800       (reads T1's uncommitted value)
T1: ROLLBACK                 (800 never existed)
T2: made a decision based on a value that never was
```

PostgreSQL prevents dirty reads at ALL isolation levels — even `READ UNCOMMITTED` behaves like `READ COMMITTED`. No demo script for this one; it can't be triggered in PostgreSQL.

---

**Non-repeatable Read** — reading the same row twice in one transaction, getting different values.

```
T1: SELECT Alice → 1000
T2: UPDATE Alice = 800, COMMIT
T1: SELECT Alice → 800   ← same query, different result
```

---

**Phantom Read** — running the same WHERE query twice in one transaction, getting a different set of rows.

```
T1: SELECT WHERE balance > 500 → [Alice]
T2: INSERT Charlie (balance=900), COMMIT
T1: SELECT WHERE balance > 500 → [Alice, Charlie]   ← phantom appeared
```

> **Q I had: what's the difference between non-repeatable read and phantom read?**
> Non-repeatable: same row, different value (T2 did an UPDATE).
> Phantom: same WHERE clause, different rows (T2 did an INSERT or DELETE).

> **Q I had: does DELETE also cause phantom reads?**
> Yes, in the other direction — rows disappear instead of appearing. "Phantom" means the result set of a range query changed, whether rows were added or removed.

---

#### Four isolation levels

Higher level = fewer anomalies = less concurrency.

```
Level              | Dirty Read | Non-repeatable | Phantom
-------------------+-----------+----------------+--------
Read Uncommitted   | possible  | possible       | possible
Read Committed     | prevented | possible       | possible
Repeatable Read    | prevented | prevented      | possible *
Serializable       | prevented | prevented      | prevented

* PostgreSQL REPEATABLE READ also prevents phantoms (beyond SQL standard).
  MySQL InnoDB REPEATABLE READ does not.
```

**PostgreSQL defaults to READ COMMITTED. MySQL InnoDB defaults to REPEATABLE READ.**

---

### MVCC — how isolation actually works

PostgreSQL doesn't use locking to prevent dirty reads. It uses **MVCC (Multi-Version Concurrency Control)**: multiple versions of each row coexist in the heap file. Readers see old versions. Writers create new versions. **Readers never block writers. Writers never block readers.**

Every row has two hidden system columns:

```
xmin  = transaction ID that created this row version
xmax  = transaction ID that deleted/replaced it (0 = still alive)
```

Operations:

```
INSERT (txid=99):
  new row: xmin=99, xmax=0, balance=1000

UPDATE (txid=150):
  old row: xmin=99, xmax=150, balance=1000  ← xmax stamped in-place (dead)
  new row: xmin=150, xmax=0,  balance=800   ← new version written

DELETE (txid=200):
  row:     xmin=150, xmax=200, balance=800  ← marked dead, physically still in heap
```

> **Q I had: after INSERT + UPDATE, how many physical rows are there?**
> Two, not three. The INSERT creates one row. The UPDATE stamps xmax on that same row in-place and writes one new row. Don't confuse "before UPDATE" and "after UPDATE" states of the same row — they're not two separate rows.
>
> ```
> xmin=99,  xmax=150, balance=1000  ← original row, xmax changed in-place
> xmin=150, xmax=0,   balance=800   ← new row from UPDATE
> ```
>
> Why xmin doesn't change: xmin is "who created this row" — that's history, it can't be rewritten. Only xmax changes: it's "who killed this row."

**Snapshot — what each transaction sees:**

When a transaction reads, it uses a snapshot: a struct in memory that captures the state of all active transactions at a point in time.

```
snapshot {
  xmin:  oldest active txid at snapshot time  (all below are committed/done)
  xmax:  next txid to be assigned             (all at/above haven't started)
  xip[]: list of txids in-progress between xmin and xmax
}
```

Visibility rule for a row version with `xmin=X, xmax=Y`:

```
Visible if:
  X < snapshot.xmin  AND  CLOG says X committed   → created before snapshot, committed
  X not in xip[]     AND  X < snapshot.xmax        → also committed before snapshot

  AND

  Y = 0              → not yet deleted
  Y > snapshot.xmax  → deleted after snapshot
  Y in xip[]         → deleter still in-progress at snapshot time
```

**CLOG** (pg_xact): a compact map of `txid → status` (COMMITTED / ABORTED / IN_PROGRESS). xmin/xmax tell you *who* modified a row; CLOG tells you *whether that transaction committed*. Once confirmed, PostgreSQL writes a **hint bit** directly on the row to avoid querying CLOG again.

**Where snapshots come from — procarray:**

PostgreSQL maintains a table in shared memory called **procarray** — one row per active connection, recording the current txid. Taking a snapshot = scanning procarray to get `xmin`, `xmax`, and `xip[]`.

```
procarray (shared memory):
  pid=101  txid=20  in_transaction
  pid=102  txid=30  in_transaction
  pid=103  txid=--  idle

Snapshot for txid=30:
  xmin = 20   (oldest active)
  xmax = 31   (next to be assigned)
  xip  = [20] (active between xmin and xmax)
```

**Snapshot lifetime determines what you see:**

```
READ COMMITTED:   new snapshot per SQL statement
                  → each statement sees latest committed data
                  → non-repeatable reads possible

REPEATABLE READ:  one snapshot at transaction start
                  → entire transaction sees fixed state
                  → non-repeatable reads and phantoms prevented
```

> **Q I had: why does non-repeatable read happen even with MVCC?**
> MVCC mechanism is the same at both levels. The difference is when the snapshot is taken.
> READ COMMITTED refreshes the snapshot per statement — you see whatever was committed before each query ran. REPEATABLE READ takes one snapshot at the start and sticks with it. Same MVCC, different snapshot lifetime.

> **Q I had: isolation level low → MVCC snapshots created and discarded faster?**
> Yes. Low isolation = short snapshot lifetime = old row versions become unreachable sooner = autovacuum can clean them faster. Long-running transactions hold their snapshot open, preventing autovacuum from cleaning any row version newer than their xmin — this is **table bloat**, a real PostgreSQL operational problem.

**Self-visibility:**

A transaction always sees its own uncommitted writes. The visibility rule has an exception: if `xmin = current txid`, the row is visible regardless of commit status. Without this, you couldn't INSERT then SELECT in the same transaction. This is not a dirty read — dirty read means seeing *someone else's* uncommitted data.

---

### Index + ctid: how queries find the right version

Every row version has a physical address called **ctid** — `(page_number, slot_number)` in the heap file. An index stores `key → [all ctids for that key]`, including dead versions.

```
B-tree index on (name):
  'Alice' → [(0,1), (0,3)]   ← all versions, live and dead

Query: SELECT WHERE name = 'Alice'
  Step 1: index lookup → [(0,1), (0,3)]   O(log n)
  Step 2: heap fetch each ctid             O(1) per version, direct address
  Step 3: xmin/xmax visibility check
          (0,1): xmax=150, visible? → no (dead)
          (0,3): xmax=0,   visible? → yes
  Step 4: return (0,3)
```

The index returns all versions; the heap filters to the one that's visible to the current snapshot. There's always at most one alive version (xmax=0) for a given key.

> **Q I had: if name → ctid is one-to-many, how do we know which one to return?**
> The index doesn't determine uniqueness — it returns all ctids. The visibility filter guarantees only one passes: xmax=0 (alive) and xmin committed. Multiple alive versions for the same key can't exist in a consistent DB.

**HOT optimization:** if an UPDATE changes only columns not in any index, PostgreSQL skips updating the index. Instead, the old row gets a pointer to the new ctid. Index entry still points to old ctid; follow the chain to reach the new version. Saves index write overhead on non-indexed column updates.

**Multiple versions accumulate:**

```
txid=10  INSERT Alice 1000  → ctid=(0,1) xmin=10, xmax=20
txid=20  UPDATE Alice 800   → ctid=(0,1) dead, new (0,3) xmin=20, xmax=40
txid=40  UPDATE Alice 600   → ctid=(0,3) dead, new (0,5) xmin=40, xmax=0
```

Three physical rows for one logical row. All needed as long as there are active snapshots that predate any of them. autovacuum removes a dead row only when `global_xmin > row.xmax` — meaning no active transaction could possibly need to see it.

---

### PostgreSQL internals — the full picture

```
shared memory:
  procarray    → active transaction registry (txid, pid, status)
  CLOG         → txid → COMMITTED/ABORTED/IN_PROGRESS (2 bits each)
  lock table   → lock object → [holders, waiters]

disk:
  heap files   → table rows (xmin, xmax, ctid, business columns)
  index files  → key → ctid mapping
  WAL          → operation log for durability

background:
  autovacuum   → clean dead tuples when global_xmin > row.xmax
  checkpointer → flush dirty pages to disk, write checkpoint marker to WAL
```

**SELECT path:**
```
txid assigned → registered in procarray → snapshot taken from procarray
→ index lookup → ctid list → heap fetch → xmin/xmax + CLOG visibility check → return
```

**UPDATE path:**
```
txid assigned → snapshot → index lookup → lock manager acquires row lock
→ check if row was modified by concurrent committed txn:
    READ COMMITTED:   wait, re-read current value, proceed
    REPEATABLE READ:  abort (SerializationFailure)
→ write new row version to heap → write WAL → COMMIT → update CLOG → clear procarray
```

**Lock table:**

Row locks aren't stored in the lock table — they're stored in the heap itself (xmax with a "lock only" flag, not "delete"). The lock table handles *waiting*: when a transaction can't acquire a row lock, it registers in the lock table and sleeps via OS semaphore until the holder commits.

Deadlock detection: a background thread periodically scans the wait graph. If it finds a cycle (T1 waits for T2, T2 waits for T1), it aborts one transaction.

> **Practical mental model:** don't need to memorize lock table internals. What matters: SELECT runs on snapshots, UPDATE acquires row locks and detects conflicts, COMMIT updates CLOG and frees locks.

---

### Lost update

Two transactions read the same value, compute a new value independently in application memory, write back. The second write overwrites the first.

```
T1: SELECT Alice → 1000 (into Python variable)
T2: SELECT Alice → 1000 (into Python variable)
T1: UPDATE Alice = 1100, COMMIT
T2: UPDATE Alice = 1200, COMMIT  ← computed from 1000, overwrites T1's 1100
Final: 1200. T1's +100 is gone.
```

This is an **application-layer problem**: read-compute-write split across three steps with a gap in between.

**Fix 1: atomic UPDATE** — keep the computation inside the database:

```sql
UPDATE accounts SET balance = balance + 100
```

The database reads the current committed value and applies the expression atomically. No gap, no lost update. Works at READ COMMITTED.

**Fix 2: SELECT FOR UPDATE** — acquire an exclusive row lock before reading:

```sql
SELECT balance FROM accounts WHERE name = 'Alice' FOR UPDATE
-- T2 blocks here until T1 commits
```

> **Q I had: does the abort depend on whether the SET clause is an absolute value or expression?**
> No. At REPEATABLE READ, PostgreSQL aborts if the target row was modified by another committed transaction after the snapshot — regardless of what the SET clause contains. At READ COMMITTED, PostgreSQL re-reads the current value and proceeds — again regardless of the SET clause. The isolation level is what determines behavior, not the expression type.

> **Q I had: why doesn't REPEATABLE READ abort the atomic UPDATE version?**
> It does, if run under REPEATABLE READ. The atomic UPDATE demo works because it uses READ COMMITTED (the default). Under READ COMMITTED, the UPDATE statement re-reads the current committed value when it executes. The expression `balance + 100` is evaluated against the current row, not T2's snapshot.

---

### Write skew

Two transactions each read shared state, each decides it's safe to proceed, each writes to a **different** row. No row conflict → MVCC/REPEATABLE READ can't detect it. The combination violates a business invariant.

```
Rule: total balance must stay >= 200. Alice=200, Bob=200, total=400.

T1: SELECT SUM → 400, safe to withdraw 150 from Alice
T2: SELECT SUM → 400, safe to withdraw 150 from Bob
T1: UPDATE Alice -150, COMMIT
T2: UPDATE Bob  -150, COMMIT
Final: Alice=50, Bob=50, total=100 — rule violated
```

T1 and T2 wrote different rows — no conflict detected, both committed.

> **Q I had: can we solve this by putting the check in the SQL WHERE clause?**
> No. `UPDATE ... WHERE (SELECT SUM(balance)) >= 200` — the subquery and the UPDATE use the same statement-level snapshot. If T1 and T2 start their statements concurrently, both subqueries see SUM=400, both pass the check. The snapshot prevents them from seeing each other's uncommitted writes. The check and the write are still not serialized.

**Fix 1: SELECT FOR UPDATE + READ COMMITTED** — lock all rows being read:

```sql
SELECT balance FROM accounts FOR UPDATE   -- locks all rows
-- T2 blocks until T1 commits
-- T2 then reads SUM=250, check fails, correctly rolls back
```

Note: `FOR UPDATE` can't be combined with aggregate functions in PostgreSQL. Lock first, compute aggregate in application code.

**Fix 2: SERIALIZABLE** — PostgreSQL SSI detects read-write dependency cycles and aborts one transaction. No locking, no blocking — but the application must catch `SerializationFailure` and retry.

```
SERIALIZABLE is not serial execution.
It means: the result is equivalent to *some* serial order.
Concurrent execution + conflict detection + abort on violation.
```

---

### Durability

A committed transaction survives crashes, power failures, OS panics.

```
Transaction runs → changes in memory (not yet on disk)
COMMIT issued
→ WAL record flushed to disk (fsync — waits for physical write)
→ COMMIT returns to client
→ Data files updated asynchronously later
```

If crash before fsync: no COMMIT in WAL → transaction never committed → safe.
If crash after fsync: WAL has the COMMIT record → recovery replays it → data preserved.

**Checkpoint:** PostgreSQL periodically flushes all dirty pages to disk and writes a checkpoint marker to WAL. Crash recovery only replays WAL from the last checkpoint, not from the beginning. Default: every 5 minutes or every 1GB of WAL.

**PITR (Point-in-Time Recovery):**

```
base backup (full snapshot) + archived WAL segments
→ replay WAL forward to any target time
→ restore to any point after the base backup

Without archiving: can only recover from last checkpoint to crash.
With archiving: can recover to any point in WAL history.
```

Data files are single current state — always overwritten in-place. Old versions only exist in MVCC heap rows and WAL; there's no "multiple checkpoint snapshots" on disk.

---

## Summary

| Property | Who guarantees it | Mechanism |
|---|---|---|
| Atomicity | Database | WAL + rollback |
| Consistency | Application (+ DB constraints) | Correct transactions + CHECK/FK |
| Isolation | Database | MVCC snapshots + locks |
| Durability | Database | WAL + fsync + checkpoint |

| Anomaly | Caused by | Prevented at |
|---|---|---|
| Dirty read | Reading uncommitted writes | READ COMMITTED (always in PG) |
| Non-repeatable read | Snapshot per statement | REPEATABLE READ |
| Phantom read | Snapshot per statement | REPEATABLE READ (in PG), SERIALIZABLE (standard) |
| Lost update | Read-compute-write gap in app | Atomic UPDATE, or SELECT FOR UPDATE |
| Write skew | Disjoint writes on shared read | SELECT FOR UPDATE, or SERIALIZABLE |
