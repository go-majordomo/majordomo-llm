"""Logging adapters for databases and storage."""

from majordomo_llm.logging.adapters.mysql import MySQLAdapter
from majordomo_llm.logging.adapters.postgres import PostgresAdapter
from majordomo_llm.logging.adapters.s3 import S3Adapter

__all__ = [
    "MySQLAdapter",
    "PostgresAdapter",
    "S3Adapter",
]
