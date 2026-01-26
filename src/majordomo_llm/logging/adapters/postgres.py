"""PostgreSQL logging adapter."""

import asyncpg

from majordomo_llm.logging.interfaces import DatabaseAdapter
from majordomo_llm.logging.models import LogEntry


class PostgresAdapter(DatabaseAdapter):
    """PostgreSQL adapter for logging LLM requests."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_size: int = 1,
        max_size: int = 10,
    ) -> "PostgresAdapter":
        """Create a new PostgresAdapter with a connection pool."""
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=min_size,
            max_size=max_size,
        )
        return cls(pool)

    async def insert(self, entry: LogEntry) -> None:
        """Insert a log entry into the database."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_requests (
                    request_id, provider, model, timestamp, response_time,
                    input_tokens, output_tokens, cached_tokens,
                    input_cost, output_cost, total_cost,
                    s3_request_key, s3_response_key, status, error_message,
                    api_key_hash, api_key_alias
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17
                )
                """,
                entry.request_id,
                entry.provider,
                entry.model,
                entry.timestamp,
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
            )

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
