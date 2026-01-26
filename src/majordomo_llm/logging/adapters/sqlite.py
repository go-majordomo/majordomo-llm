"""SQLite logging adapter."""

import aiosqlite

from majordomo_llm.logging.interfaces import DatabaseAdapter
from majordomo_llm.logging.models import LogEntry

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_requests (
    request_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    response_time REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    input_cost REAL,
    output_cost REAL,
    total_cost REAL,
    storage_request_key TEXT,
    storage_response_key TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    api_key_hash TEXT,
    api_key_alias TEXT
)
"""

INSERT_SQL = """
INSERT INTO llm_requests (
    request_id, provider, model, timestamp, response_time,
    input_tokens, output_tokens, cached_tokens,
    input_cost, output_cost, total_cost,
    storage_request_key, storage_response_key, status, error_message,
    api_key_hash, api_key_alias
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class SqliteAdapter(DatabaseAdapter):
    """SQLite adapter for logging LLM requests.

    Provides a lightweight, zero-setup option for local development and examples.
    The database file and table are created automatically.

    Example:
        >>> db = await SqliteAdapter.create("llm_logs.db")
        >>> logged_llm = LoggingLLM(llm, db)
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    @classmethod
    async def create(cls, database_path: str) -> "SqliteAdapter":
        """Create a new SqliteAdapter.

        Args:
            database_path: Path to the SQLite database file.
                Use ":memory:" for an in-memory database.

        Returns:
            A configured SqliteAdapter instance.
        """
        connection = await aiosqlite.connect(database_path)
        await connection.execute(CREATE_TABLE_SQL)
        await connection.commit()
        return cls(connection)

    async def insert(self, entry: LogEntry) -> None:
        """Insert a log entry into the database."""
        await self._connection.execute(
            INSERT_SQL,
            (
                str(entry.request_id),
                entry.provider,
                entry.model,
                entry.timestamp.isoformat(),
                entry.response_time,
                entry.input_tokens,
                entry.output_tokens,
                entry.cached_tokens,
                entry.input_cost,
                entry.output_cost,
                entry.total_cost,
                entry.s3_request_key,
                entry.s3_response_key,
                entry.status,
                entry.error_message,
                entry.api_key_hash,
                entry.api_key_alias,
            ),
        )
        await self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""
        await self._connection.close()
