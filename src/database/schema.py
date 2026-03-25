"""Schema metadata for the ELIQSIR data warehouse.

``schema.sql`` is the **single source of truth** for the physical schema.  This
module provides two things that the Python ETL layer needs:

1. :data:`TABLE_COLUMNS` – ordered column lists per table, used by
   :class:`~src.loading.warehouse_loader.WarehouseLoader` to build
   ``INSERT … ON DUPLICATE KEY UPDATE`` statements without hard-coding strings
   elsewhere.

2. :func:`apply_schema` – reads ``schema.sql`` and executes every DDL
   statement against the supplied SQLAlchemy engine, creating all tables if
   they do not already exist.

Star schema overview::

    dim_protein ──┐
    dim_drug    ──┤
    dim_article ──┼──► fact_bioactivity
    dim_structure  (FK to dim_protein.uniprot_id)

Usage::

    from src.database.schema import TABLE_COLUMNS, apply_schema
    from src.database.connection import get_engine

    engine = get_engine()
    apply_schema(engine)          # DDL – safe to call repeatedly
    cols = TABLE_COLUMNS["dim_protein"]
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import Engine, text

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Filesystem reference to the DDL source
# ---------------------------------------------------------------------------

#: Absolute path to the MySQL DDL file – the single source of truth.
SCHEMA_SQL_PATH: Path = Path(__file__).parent / "schema.sql"

# ---------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------

#: Ordered column lists per warehouse table.
#:
#: These lists mirror the ``CREATE TABLE`` column order in ``schema.sql`` and
#: are used by :class:`~src.loading.warehouse_loader.WarehouseLoader` when
#: building ``INSERT`` / ``ON DUPLICATE KEY UPDATE`` statements.  Keeping them
#: here (rather than duplicated in the loader) means a schema column rename
#: only needs to be updated in *one* place.
TABLE_COLUMNS: dict[str, list[str]] = {
    "dim_protein": [
        "protein_key",
        "uniprot_id",
        "gene_names",
        "protein_name",
    ],
    "dim_drug": [
        "drug_key",
        "drug_chembl_id",
        "drug_name",
    ],
    "dim_article": [
        "article_key",
        "pubmed_id",
        "article_title",
        "journal",
        "year",
        "abstract",
        "doi",
        "first_author",
    ],
    "dim_structure": [
        "structure_key",
        "pdb_id",
        "chain_id",
        "uniprot_id",
        "resolution",
        "coverage",
        "method",
        "unp_start",
        "unp_end",
    ],
    "fact_bioactivity": [
        "activity_id",
        "protein_key",
        "drug_key",
        "article_key",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "confidence_score",
        "assay_type",
        "assay_description",
    ],
}

#: Primary key column per table (used by the loader to exclude from UPDATE set).
TABLE_PRIMARY_KEYS: dict[str, str] = {
    "dim_protein":      "protein_key",
    "dim_drug":         "drug_key",
    "dim_article":      "article_key",
    "dim_structure":    "structure_key",
    "fact_bioactivity": "activity_id",
}

# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------


def _mysql_to_sqlite_ddl(sql: str) -> str:
    """Translate a MySQL DDL script into SQLite-compatible statements.

    This shim is used exclusively during testing (in-memory SQLite) so that
    the test suite does not require a running MySQL server.  It strips or
    rewrites MySQL-only clauses while preserving the logical schema:

    * Removes ``ENGINE``, ``DEFAULT CHARSET``, ``COLLATE``, ``ROW_FORMAT``
      table options.
    * Removes ``FULLTEXT KEY`` index definitions.
    * Removes ``SET FOREIGN_KEY_CHECKS`` statements.
    * Removes inline ``COMMENT`` column clauses.
    * Rewrites ``TINYINT UNSIGNED`` → ``INTEGER``.
    * Rewrites ``BIGINT`` / ``SMALLINT`` / ``DOUBLE`` → ``INTEGER`` / ``REAL``.
    * Rewrites ``VARCHAR(n)`` → ``TEXT``.
    * Removes ``KEY …`` and ``UNIQUE KEY …`` inline index declarations
      (SQLite handles uniqueness via ``UNIQUE`` on column definitions).
    * Removes trailing commas left after stripping index lines.
    """
    lines: list[str] = []
    for raw in sql.splitlines():
        line = raw.rstrip()

        # ── skip MySQL-only table options and noise ──────────────────────────
        if re.search(
            r"SET\s+FOREIGN_KEY_CHECKS|ENGINE\s*=|DEFAULT\s+CHARSET|"
            r"COLLATE\s*=|ROW_FORMAT\s*=|FULLTEXT\s+KEY",
            line, re.IGNORECASE,
        ):
            continue

        # ── skip standalone KEY / INDEX declarations ─────────────────────────
        # Matches lines like:  KEY `idx_name` (`col`),
        #                      UNIQUE KEY `uq` (`col`),
        # but NOT PRIMARY KEY or CONSTRAINT … FOREIGN KEY lines.
        if re.match(r"\s+(UNIQUE\s+)?KEY\s+`", line, re.IGNORECASE):
            continue

        # ── strip inline COMMENT clauses ─────────────────────────────────────
        line = re.sub(r"\s+COMMENT\s+'[^']*'", "", line, flags=re.IGNORECASE)

        # ── type substitutions ───────────────────────────────────────────────
        line = re.sub(r"\bTINYINT\s+UNSIGNED\b", "INTEGER", line, flags=re.IGNORECASE)
        line = re.sub(r"\bBIGINT\b",             "INTEGER", line, flags=re.IGNORECASE)
        line = re.sub(r"\bSMALLINT\b",           "INTEGER", line, flags=re.IGNORECASE)
        line = re.sub(r"\bDOUBLE\b",             "REAL",    line, flags=re.IGNORECASE)
        line = re.sub(r"\bVARCHAR\(\d+\)\b",     "TEXT",    line, flags=re.IGNORECASE)

        # ── strip ENGINE / CHARSET / COLLATE from closing line ───────────────
        line = re.sub(
            r"\)\s*ENGINE\s*=.*$", ");", line, flags=re.IGNORECASE
        )

        lines.append(line)

    # ── remove trailing commas before closing parenthesis ────────────────────
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        # If this line ends with a comma and the next non-empty line is ')',
        # strip the comma.
        stripped = line.rstrip()
        if stripped.endswith(","):
            rest = [l.strip() for l in lines[i + 1:] if l.strip()]
            if rest and rest[0].startswith(")"):
                line = stripped[:-1]
        cleaned.append(line)

    return "\n".join(cleaned)


def apply_schema(engine: Engine, *, sqlite_compat: bool = False) -> None:
    """Execute ``schema.sql`` against *engine* to create all tables.

    Parameters
    ----------
    engine:
        A SQLAlchemy :class:`~sqlalchemy.engine.Engine` instance.  Works with
        both MySQL (production) and SQLite (tests / notebooks).
    sqlite_compat:
        When ``True``, the MySQL DDL is automatically translated to
        SQLite-compatible syntax before execution.  Set this flag when
        *engine* connects to an SQLite database.

    Notes
    -----
    * Every ``CREATE TABLE`` statement uses ``IF NOT EXISTS``, so calling this
      function repeatedly is safe and idempotent.
    * Statements are split on ``;`` and executed one at a time so that engines
      that do not support multi-statement execution (e.g. SQLite) work
      correctly.
    """
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(
            f"Schema SQL file not found: {SCHEMA_SQL_PATH}. "
            "Ensure the repository is complete."
        )

    raw_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")

    if sqlite_compat:
        raw_sql = _mysql_to_sqlite_ddl(raw_sql)

    # Split on ';' and execute each non-empty statement individually.
    statements = [s.strip() for s in raw_sql.split(";") if s.strip()]

    logger.info(
        "Applying schema from '%s' (%d statements).",
        SCHEMA_SQL_PATH.name,
        len(statements),
    )

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    logger.info("Schema applied successfully.")
