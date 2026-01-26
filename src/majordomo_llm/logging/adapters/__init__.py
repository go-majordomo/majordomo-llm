"""Logging adapters for databases and storage."""

from majordomo_llm.logging.adapters.file import FileStorageAdapter
from majordomo_llm.logging.adapters.mysql import MySQLAdapter
from majordomo_llm.logging.adapters.postgres import PostgresAdapter
from majordomo_llm.logging.adapters.s3 import S3Adapter
from majordomo_llm.logging.adapters.sqlite import SqliteAdapter

__all__ = [
    "FileStorageAdapter",
    "MySQLAdapter",
    "PostgresAdapter",
    "S3Adapter",
    "SqliteAdapter",
]
