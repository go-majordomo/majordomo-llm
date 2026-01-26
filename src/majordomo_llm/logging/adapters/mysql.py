"""MySQL logging adapter."""

import aiomysql

from majordomo_llm.logging.interfaces import DatabaseAdapter
from majordomo_llm.logging.models import LogEntry


class MySQLAdapter(DatabaseAdapter):
    """MySQL adapter for logging LLM requests."""

    def __init__(self, pool: aiomysql.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        minsize: int = 1,
        maxsize: int = 10,
    ) -> "MySQLAdapter":
        """Create a new MySQLAdapter with a connection pool."""
        pool = await aiomysql.create_pool(
            host=host,
            port=port,
            db=database,
            user=user,
            password=password,
            minsize=minsize,
            maxsize=maxsize,
        )
        return cls(pool)

    async def insert(self, entry: LogEntry) -> None:
        """Insert a log entry into the database."""
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO llm_requests (
                        request_id, provider, model, timestamp, response_time,
                        input_tokens, output_tokens, cached_tokens,
                        input_cost, output_cost, total_cost,
                        s3_request_key, s3_response_key, status, error_message,
                        api_key_hash, api_key_alias
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(entry.request_id),
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
                    ),
                )
            await conn.commit()

    async def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()
        await self._pool.wait_closed()
