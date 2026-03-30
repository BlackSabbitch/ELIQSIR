"""Schema metadata for the ELIQSIR data warehouse.

``schema.sql`` is the **single source of truth** for the physical schema.  This
module provides two things that the Python ETL layer needs:

1. :data:`TABLE_COLUMNS` - ordered column lists per table, used by
   :class:`~src.loading.warehouse_loader.WarehouseLoader` to build
   ``INSERT ... ON DUPLICATE KEY UPDATE`` statements without hard-coding strings
   elsewhere.

2. :func:`apply_schema` - reads ``schema.sql`` and executes every DDL
   statement against the supplied SQLAlchemy engine, creating all tables if
   they do not already exist.

Star schema overview::

    dim_protein ──┐
    dim_drug    ──┤
    dim_article ──┼──► fact_bioactivity
    dim_structure  (FK to dim_protein.protein_key)

Usage::

    from src.database.schema import TABLE_COLUMNS, apply_schema
    from src.database.connection import get_engine

    engine = get_engine()
    apply_schema(engine)          # DDL - safe to call repeatedly
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

#: Absolute path to the MySQL DDL file - the single source of truth.
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
        "sequence",          # Added for ML
        "sequence_length",   # Added for ML
        "protein_families",  # Added for ML
    ],
    "dim_drug": [
        "drug_key",
        "drug_chembl_id",
        "drug_name",
        "molecule_type",      # Added for ML filtering
        "molecular_weight",   # Added for ML filtering
        "canonical_smiles",   # Added for ML graph generation
        "standard_inchi_key", # Added for ML duplicate prevention
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
        "protein_key",        # Replaced uniprot_id for DW surrogate key standard
        "pdb_id",
        "chain_id",
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
        "assay_organism",     # Added for ML target species filtering
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
    * Rewrites ``TINYINT UNSIGNED`` -> ``INTEGER``.
    * Rewrites ``BIGINT`` / ``SMALLINT`` / ``DOUBLE`` -> ``INTEGER`` / ``REAL``.
    * Rewrites ``VARCHAR(n)`` -> ``TEXT``.
    * Removes ``KEY ...`` and ``UNIQUE KEY ...`` inline index declarations
      (SQLite handles uniqueness via ``UNIQUE`` on column definitions).
    * Removes trailing commas left after stripping index lines.
    """