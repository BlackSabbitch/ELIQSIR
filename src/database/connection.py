"""SQLAlchemy engine and session factory.

All database interaction in ELIQSIR goes through this module so that the
connection string lives in exactly one place (:class:`~src.config.Settings`).

Usage::

    from src.database.connection import get_engine, get_session
    from src.database.schema import apply_schema

    engine = get_engine()

    # DDL – create all tables if they don't exist yet
    apply_schema(engine)

    # DML – use a managed session
    with get_session() as session:
        session.execute(text("SELECT 1"))
        session.commit()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised)
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None  # type: ignore[type-arg]


def get_engine(echo: bool = False) -> Engine:
    """Return (or create) the shared SQLAlchemy engine.

    Parameters
    ----------
    echo:
        When ``True``, SQLAlchemy will log every SQL statement it issues.
        Useful for debugging; should be ``False`` in production.

    Returns
    -------
    sqlalchemy.Engine
        A connected engine instance using the MySQL URL from ``settings``.
    """
    global _engine
    if _engine is None:
        url = settings.mysql_url()
        logger.info("Creating SQLAlchemy engine for database '%s'.", settings.mysql_db)
        _engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=True,   # detect stale connections before use
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_factory() -> sessionmaker:  # type: ignore[type-arg]
    """Return (or create) the session factory bound to the shared engine."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # keep attribute access after commit
        )
    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a transactional ``Session``.

    Commits on clean exit, rolls back on any exception, and always closes
    the session – so callers never need to manage commit/rollback manually.

    Example::

        with get_session() as session:
            session.add(dim_protein_row)
            # commit is automatic on __exit__
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Session rolled back due to an unhandled exception.")
        raise
    finally:
        session.close()
