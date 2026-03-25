"""Database sub-package.

Exposes the SQLAlchemy engine / session factory and the SQL-driven schema
utilities.  ``schema.sql`` is the single source of truth for table
definitions; :func:`apply_schema` reads and executes it.
"""

from src.database.connection import get_engine, get_session
from src.database.schema import TABLE_COLUMNS, TABLE_PRIMARY_KEYS, apply_schema

__all__ = [
    "get_engine",
    "get_session",
    "apply_schema",
    "TABLE_COLUMNS",
    "TABLE_PRIMARY_KEYS",
]
