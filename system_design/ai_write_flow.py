"""
AI-integrated multi-step write flow.

Demonstrates:
  - resource_id derived from key business fields (username + email)
    so only genuinely duplicate requests are blocked; others proceed in parallel
  - Pre-emptive reservation with partial unique index on active statuses
  - Idempotency key to prevent double-billing on AI API retry
  - Soft-delete Saga: compensation marks rows as 'failed' instead of deleting;
    a background job handles eventual cleanup

Run:
    python ai_write_flow.py
"""

import sqlite3
import hashlib
import time
import uuid
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            user_id     INTEGER,
            username    TEXT,
            email       TEXT,
            title       TEXT,
            content     TEXT,
            status      TEXT DEFAULT 'draft'
        );

        CREATE TABLE IF NOT EXISTS reservations (
            id          TEXT PRIMARY KEY,
            user_id     INTEGER,
            resource_id TEXT,
            status      TEXT,          -- pending | completed | failed | expired
            expires_at  REAL,
            updated_at  REAL
        );

        -- Only active reservations participate in uniqueness.
        -- failed/expired rows do not block new attempts for the same resource.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_active_reservation
            ON reservations(resource_id)
            WHERE status IN ('pending', 'completed');

        CREATE TABLE IF NOT EXISTS ai_results (
            id              TEXT PRIMARY KEY,
            document_id     TEXT,
            idempotency_key TEXT UNIQUE,
            result          TEXT,
            status          TEXT DEFAULT 'ok'   -- ok | failed
        );

        CREATE TABLE IF NOT EXISTS vector_entries (
            id          TEXT PRIMARY KEY,
            document_id TEXT,
            embedding   TEXT,
            status      TEXT DEFAULT 'ok'   -- ok | failed
        );
    """)
    conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Mock external services
# ---------------------------------------------------------------------------

class AIAPIError(Exception):
    pass


def mock_ai_api(prompt: str, idempotency_key: str, fail: bool = False) -> str:
    print(f"  [AI API] calling with key={idempotency_key[:16]}...")
    time.sleep(0.05)
    if fail:
        raise AIAPIError("AI API returned 500")
    return f"AI-generated summary for: {prompt[:30]}"


def mock_vector_ingest(document_id: str, text: str, fail: bool = False) -> str:
    print(f"  [Vector DB] ingesting document {document_id}...")
    time.sleep(0.02)
    if fail:
        raise RuntimeError("Vector DB unavailable")
    return f"embedding:{hash(text) % 10000}"


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def derive_resource_id(username: str, email: str) -> str:
    """
    Derive a stable resource key from the form's unique business fields.
    Only requests with identical username + email share the same key and
    will contend on the reservation; all other requests proceed independently.
    """
    return hashlib.sha256(f"{username}:{email}".encode()).hexdigest()


def step1_write_basic_info(conn, user_id: int, username: str, email: str, title: str) -> tuple:
    """
    Write basic document info and claim a reservation keyed on username+email.
    Returns (doc_id, resource_id).
    Raises if an active reservation for the same key already exists.
    Expiry of stale reservations is handled entirely by the background job.
    """
    resource_id = derive_resource_id(username, email)

    try:
        doc_id = str(uuid.uuid4())
        reservation_id = str(uuid.uuid4())
        expires_at = time.time() + 300  # 5 minutes

        conn.execute(
            "INSERT INTO documents (id, user_id, username, email, title, status) "
            "VALUES (?, ?, ?, ?, ?, 'draft')",
            (doc_id, user_id, username, email, title)
        )
        # partial unique index on active statuses guards this insert
        conn.execute(
            "INSERT INTO reservations (id, user_id, resource_id, status, expires_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?)",
            (reservation_id, user_id, resource_id, expires_at, time.time())
        )
        conn.commit()
        print(f"  [Step 1] document {doc_id[:8]}... created, resource reserved for ({username}, {email})")
        return doc_id, resource_id

    except sqlite3.IntegrityError:
        conn.rollback()
        raise RuntimeError(
            f"Active reservation already exists for ({username}, {email}) — "
            "duplicate request blocked before any AI call"
        )


def step2_call_ai(conn, doc_id: str, user_id: int, prompt: str,
                  fail: bool = False, max_retries: int = 2) -> str:
    """
    Call the AI API with an idempotency key derived from user_id + doc_id.
    Retries on transient failure without double-billing.
    """
    raw = f"{user_id}:{doc_id}:step2"
    idempotency_key = hashlib.sha256(raw.encode()).hexdigest()

    row = conn.execute(
        "SELECT result FROM ai_results WHERE idempotency_key = ? AND status = 'ok'",
        (idempotency_key,)
    ).fetchone()
    if row:
        print(f"  [Step 2] idempotency hit — reusing cached result")
        return row["result"]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = mock_ai_api(prompt, idempotency_key, fail=fail and attempt < max_retries)
            result_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO ai_results (id, document_id, idempotency_key, result, status) "
                "VALUES (?, ?, ?, ?, 'ok')",
                (result_id, doc_id, idempotency_key, result)
            )
            conn.commit()
            print(f"  [Step 2] AI result saved (attempt {attempt})")
            return result
        except AIAPIError as e:
            last_error = e
            print(f"  [Step 2] attempt {attempt} failed: {e}, retrying...")

    raise AIAPIError(f"AI API failed after {max_retries} attempts: {last_error}")


def step3_vector_ingest(conn, doc_id: str, resource_id: str, text: str, fail: bool = False) -> None:
    """Ingest document into vector DB and mark reservation completed."""
    embedding = mock_vector_ingest(doc_id, text, fail=fail)
    entry_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO vector_entries (id, document_id, embedding, status) VALUES (?, ?, ?, 'ok')",
        (entry_id, doc_id, embedding)
    )
    conn.execute(
        "UPDATE reservations SET status = 'completed', updated_at = ? WHERE resource_id = ?",
        (time.time(), resource_id)
    )
    conn.execute(
        "UPDATE documents SET status = 'published', content = ? WHERE id = ?",
        (text, doc_id)
    )
    conn.commit()
    print(f"  [Step 3] vector entry saved, reservation completed")


# ---------------------------------------------------------------------------
# Soft-delete Saga compensation
# ---------------------------------------------------------------------------

def compensate(conn, doc_id: str, resource_id: str, reached_step: int) -> None:
    """
    Mark completed steps as failed instead of deleting.
    A background job handles eventual hard-delete after the retention window.
    """
    print(f"  [Saga] marking steps failed from step {reached_step} down...")
    if reached_step >= 3:
        conn.execute(
            "UPDATE vector_entries SET status = 'failed' WHERE document_id = ?", (doc_id,)
        )
        print("  [Saga] vector entries marked failed")
    if reached_step >= 2:
        conn.execute(
            "UPDATE ai_results SET status = 'failed' WHERE document_id = ?", (doc_id,)
        )
        print("  [Saga] AI results marked failed")
    if reached_step >= 1:
        conn.execute(
            "UPDATE reservations SET status = 'failed', updated_at = ? WHERE resource_id = ?",
            (time.time(), resource_id)
        )
        conn.execute(
            "UPDATE documents SET status = 'failed' WHERE id = ?", (doc_id,)
        )
        print("  [Saga] reservation marked failed (background job will clean up later)")
    conn.commit()


def background_cleanup(conn) -> None:
    """
    Single owner of all expiry and cleanup logic — step1 does not touch this.

    1. pending reservations past their deadline -> expired
       (covers abandoned flows where no compensation ever ran)
    2. hard-delete terminal rows past the retention window
       (demo uses 0s retention; production would use 7 days)
    """
    now = time.time()
    conn.execute(
        "UPDATE reservations SET status = 'expired', updated_at = ? "
        "WHERE status = 'pending' AND expires_at < ?",
        (now, now)
    )
    # in production: WHERE updated_at < now - 7 days; here we delete immediately for demo
    conn.execute(
        "DELETE FROM reservations WHERE status IN ('failed', 'expired')"
    )
    conn.execute(
        "DELETE FROM documents WHERE status = 'failed'"
    )
    conn.execute(
        "DELETE FROM ai_results WHERE status = 'failed'"
    )
    conn.execute(
        "DELETE FROM vector_entries WHERE status = 'failed'"
    )
    conn.commit()
    print("  [Background job] terminal rows cleaned up")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_flow(conn, user_id: int, username: str, email: str,
             title: str, prompt: str, fail_step: int = 0) -> None:
    """
    Run the three-step flow with soft-delete Saga compensation on failure.
    fail_step: 0 = success, 2 = fail at step 2, 3 = fail at step 3
    """
    print(f"\n--- Flow: user={user_id} ({username}, {email}) ---")
    doc_id = None
    resource_id = None
    reached = 0

    try:
        doc_id, resource_id = step1_write_basic_info(conn, user_id, username, email, title)
        reached = 1

        ai_result = step2_call_ai(conn, doc_id, user_id, prompt, fail=(fail_step == 2))
        reached = 2

        step3_vector_ingest(conn, doc_id, resource_id, ai_result, fail=(fail_step == 3))
        reached = 3

        print(f"  [OK] flow completed successfully")

    except RuntimeError as e:
        print(f"  [FAIL] {e}")
        if doc_id and reached > 0:
            compensate(conn, doc_id, resource_id, reached)

    except AIAPIError as e:
        print(f"  [FAIL] {e}")
        if doc_id and reached > 0:
            compensate(conn, doc_id, resource_id, reached)


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

def print_state(conn):
    docs = conn.execute("SELECT id, username, email, status FROM documents").fetchall()
    res  = conn.execute("SELECT resource_id, status FROM reservations").fetchall()
    ai   = conn.execute("SELECT document_id, status FROM ai_results").fetchall()
    vec  = conn.execute("SELECT document_id, status FROM vector_entries").fetchall()

    print("\n=== DB state ===")
    print(f"documents:      {[dict(r) for r in docs]}")
    print(f"reservations:   {[{**dict(r), 'resource_id': r['resource_id'][:12]+'...'} for r in res]}")
    print(f"ai_results:     {[dict(r) for r in ai]}")
    print(f"vector_entries: {[dict(r) for r in vec]}")


if __name__ == "__main__":
    with get_conn() as conn:

        print("\n### Scenario 1: happy path ###")
        run_flow(conn, user_id=1, username="alice", email="alice@x.com",
                 title="Alice Doc", prompt="Summarise our product roadmap")
        print_state(conn)

        print("\n### Scenario 2: duplicate request — same username+email blocks before AI call ###")
        run_flow(conn, user_id=2, username="alice", email="alice@x.com",
                 title="Alice Doc (duplicate)", prompt="This should never reach the AI API")
        print_state(conn)

        print("\n### Scenario 3: different user, different fields — proceeds independently ###")
        run_flow(conn, user_id=3, username="bob", email="bob@x.com",
                 title="Bob Doc", prompt="Summarise Q3 results")
        print_state(conn)

        print("\n### Scenario 4: AI API hard failure -> soft compensation ###")
        run_flow(conn, user_id=4, username="carol", email="carol@x.com",
                 title="Carol Doc", prompt="Summarise the architecture", fail_step=2)
        print_state(conn)

        print("\n### Scenario 5: vector DB failure -> soft compensation ###")
        run_flow(conn, user_id=5, username="dave", email="dave@x.com",
                 title="Dave Doc", prompt="Summarise the roadmap", fail_step=3)
        print_state(conn)

        print("\n### Background cleanup job runs ###")
        background_cleanup(conn)
        print_state(conn)

        print("\n### Scenario 6: dave retries after compensation — resource_id now free ###")
        run_flow(conn, user_id=5, username="dave", email="dave@x.com",
                 title="Dave Doc (retry)", prompt="Summarise the roadmap")
        print_state(conn)
