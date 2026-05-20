"""Logging module for majordomo-llm.

Provides asynchronous request logging with support for:
- PostgreSQL, MySQL, and SQLite for metrics storage
- Mixpanel for analytics
- S3 or local file system for request/response body storage

Usage:
    >>> from majordomo_llm import get_llm_instance
    >>> from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter
    >>>
    >>> llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
    >>> db = await SqliteAdapter.create("llm_logs.db")
    >>> # or: db = await MixpanelAdapter.create("your_mixpanel_token")
    >>> storage = await FileStorageAdapter.create("./llm_logs")
    >>> logged_llm = LoggingLLM(llm, db, storage)
    >>>
    >>> # All requests are now logged asynchronously
    >>> response = await logged_llm.get_response("Hello!")

Note:
    Requires optional dependencies: pip install 'majordomo-llm[logging]'
"""

from majordomo_llm.logging.adapters import (
    FileStorageAdapter,
    MixpanelAdapter,
    MySQLAdapter,
    PostgresAdapter,
    S3Adapter,
    SqliteAdapter,
)
from majordomo_llm.logging.interfaces import DatabaseAdapter, StorageAdapter
from majordomo_llm.logging.models import LogEntry
from majordomo_llm.logging.wrapper import LoggingLLM

__all__ = [
    "LoggingLLM",
    "DatabaseAdapter",
    "StorageAdapter",
    "FileStorageAdapter",
    "MixpanelAdapter",
    "MySQLAdapter",
    "PostgresAdapter",
    "S3Adapter",
    "SqliteAdapter",
    "LogEntry",
]
