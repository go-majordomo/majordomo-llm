"""Logging module for majordomo-llm.

Provides asynchronous request logging with support for:
- PostgreSQL and MySQL for metrics storage
- S3 for request/response body storage

Usage:
    >>> from majordomo_llm import get_llm_instance
    >>> from majordomo_llm.logging import LoggingLLM, PostgresAdapter, S3Adapter
    >>>
    >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
    >>> db = await PostgresAdapter.create(
    ...     host="localhost", port=5432, database="llm_logs",
    ...     user="postgres", password="password"
    ... )
    >>> storage = await S3Adapter.create(bucket="my-llm-logs")
    >>> logged_llm = LoggingLLM(llm, db, storage)
    >>>
    >>> # All requests are now logged asynchronously
    >>> response = await logged_llm.get_response("Hello!")

Note:
    Requires optional dependencies: pip install majordomo-llm[logging]
"""

from majordomo_llm.logging.adapters import MySQLAdapter, PostgresAdapter, S3Adapter
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter
from majordomo_llm.logging.models import LogEntry
from majordomo_llm.logging.wrapper import LoggingLLM

__all__ = [
    "LoggingLLM",
    "DatabaseAdapter",
    "StorageAdapter",
    "PostgresAdapter",
    "MySQLAdapter",
    "S3Adapter",
    "LogEntry",
]
