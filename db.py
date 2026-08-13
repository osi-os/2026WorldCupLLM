"""
Database layer for monitoring the FIFA World Cup 2026 assistant.

Logs every question/answer to Postgres (with tokens, cost, latency,
search type) and records user thumbs up/down feedback. Grafana reads
these tables to render the monitoring dashboard.
"""

import os
from datetime import datetime, timezone

import psycopg


# --- Pricing for cost tracking (USD per token) ---
# Claude Haiku 4.5: $1 / M input tokens, $5 / M output tokens.
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
}


def compute_cost(model, input_tokens, output_tokens):
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens or 0) * p["input"] + (output_tokens or 0) * p["output"]


def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "worldcup"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )


def init_db():
    """Create the conversations and feedback tables if they don't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    search_type TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost DOUBLE PRECISION,
                    response_time DOUBLE PRECISION,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_conversation(question, answer, search_type, model,
                      input_tokens, output_tokens, cost, response_time):
    """Insert one conversation row and return its new id."""
    total = (input_tokens or 0) + (output_tokens or 0)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations
                    (question, answer, search_type, model, input_tokens,
                     output_tokens, total_tokens, cost, response_time, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (question, answer, search_type, model, input_tokens,
                  output_tokens, total, cost, response_time,
                  datetime.now(timezone.utc)))
            conv_id = cur.fetchone()[0]
        conn.commit()
        return conv_id
    finally:
        conn.close()


def save_feedback(conversation_id, feedback):
    """Record +1 (thumbs up) or -1 (thumbs down) for a conversation."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback (conversation_id, feedback, timestamp)
                VALUES (%s, %s, %s)
            """, (conversation_id, feedback, datetime.now(timezone.utc)))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized (conversations, feedback).")