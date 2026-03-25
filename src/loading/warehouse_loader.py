"""Data warehouse loader.

Loads cleaned, dimensionally modelled DataFrames into the MySQL star schema
defined in ``src/database/schema.sql``.

Strategy
--------
Each ``load_*`` method uses **upsert** semantics (``INSERT … ON DUPLICATE KEY
UPDATE …``) so that the ETL pipeline is *idempotent*: re-running it with the
same source data never creates duplicate rows and never errors on existing
primary keys.

Because pandas ``DataFrame.to_sql`` does not support native MySQL upserts, we
implement the upsert manually:

1. Write the batch to a temporary staging table (``to_sql`` with
   ``if_exists="replace"``).
2. Issue a single ``INSERT INTO dim_X … SELECT … FROM _stage
   ON DUPLICATE KEY UPDATE …`` statement.
3. Drop the staging table.

This is efficient for bulk loads and safe for incremental updates.

Usage::

    from src.loading import WarehouseLoader

    loader = WarehouseLoader()
    loader.initialise_schema()   # DDL: CREATE TABLE IF NOT EXISTS …
    loader.load_dim_protein(dim_protein_df)
    loader.load_dim_drug(dim_drug_df)
    loader.load_dim_article(dim_article_df)
    loader.load_dim_structure(dim_structure_df)
    loader.load_fact_bioactivity(fact_df)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine
from src.database.schema import TABLE_PRIMARY_KEYS, apply_schema
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Upsert template
# ---------------------------------------------------------------------------

_UPSERT_TMPL = """
INSERT INTO `{table}` ({columns})
SELECT {columns} FROM `{stage}`
ON DUPLICATE KEY UPDATE {updates};
"""


def _build_upsert(table: str, columns: list[str], pk: str) -> str:
    """Build the MySQL upsert SQL string.

    All columns except the primary key are updated on conflict.
    """
    update_cols = [c for c in columns if c != pk]
    updates = ", ".join(f"`{c}` = VALUES(`{c}`)" for c in update_cols)
    cols_str = ", ".join(f"`{c}`" for c in columns)
    return _UPSERT_TMPL.format(
        table=table,
        stage=f"_stage_{table}",
        columns=cols_str,
        updates=updates,
    )


# ---------------------------------------------------------------------------
# WarehouseLoader
# ---------------------------------------------------------------------------


class WarehouseLoader:
    """Load star-schema DataFrames into the MySQL data warehouse.

    Parameters
    ----------
    chunksize:
        Number of rows written per SQLAlchemy execute call during staging.
        Tune this for network / memory trade-offs.
    """

    def __init__(self, chunksize: int = 1000) -> None:
        self.chunksize = chunksize
        self._engine = get_engine()

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------

    def initialise_schema(self) -> None:
        """Create all tables (DDL) if they do not already exist.

        Reads ``schema.sql`` and executes every ``CREATE TABLE IF NOT EXISTS``
        statement.  Safe to call on every pipeline run.
        """
        logger.info("Initialising warehouse schema from schema.sql …")
        apply_schema(self._engine)
        logger.info("Schema ready.")

    # ------------------------------------------------------------------
    # Private upsert helper
    # ------------------------------------------------------------------

    def _upsert(
        self,
        df: pd.DataFrame,
        table_name: str,
        pk_column: str,
    ) -> int:
        """Stage *df* and upsert it into *table_name*.

        Returns the number of rows in *df*.
        """
        if df.empty:
            logger.info("Skipping load for '%s' – DataFrame is empty.", table_name)
            return 0

        stage_table = f"_stage_{table_name}"
        columns = list(df.columns)

        logger.debug(
            "Staging %d rows for '%s' into '%s'.", len(df), table_name, stage_table
        )

        with self._engine.begin() as conn:
            # 1. Write staging table.
            df.to_sql(
                stage_table,
                con=conn,
                if_exists="replace",
                index=False,
                chunksize=self.chunksize,
            )

            # 2. Upsert from staging into target.
            upsert_sql = _build_upsert(table_name, columns, pk_column)
            conn.execute(text(upsert_sql))

            # 3. Clean up staging table.
            conn.execute(text(f"DROP TABLE IF EXISTS `{stage_table}`"))

        logger.info("Upserted %d rows into '%s'.", len(df), table_name)
        return len(df)

    # ------------------------------------------------------------------
    # Dimension loaders
    # ------------------------------------------------------------------

    def load_dim_protein(self, df: pd.DataFrame) -> int:
        """Upsert protein dimension rows into ``dim_protein``.

        Expected columns: ``protein_key``, ``uniprot_id``, ``gene_names``,
        ``protein_name``.  See :data:`~src.database.schema.TABLE_COLUMNS`.
        """
        return self._upsert(df, "dim_protein", TABLE_PRIMARY_KEYS["dim_protein"])

    def load_dim_drug(self, df: pd.DataFrame) -> int:
        """Upsert drug dimension rows into ``dim_drug``."""
        return self._upsert(df, "dim_drug", TABLE_PRIMARY_KEYS["dim_drug"])

    def load_dim_article(self, df: pd.DataFrame) -> int:
        """Upsert article dimension rows into ``dim_article``."""
        return self._upsert(df, "dim_article", TABLE_PRIMARY_KEYS["dim_article"])

    def load_dim_structure(self, df: pd.DataFrame) -> int:
        """Upsert structure dimension rows into ``dim_structure``."""
        return self._upsert(df, "dim_structure", TABLE_PRIMARY_KEYS["dim_structure"])

    # ------------------------------------------------------------------
    # Fact loader
    # ------------------------------------------------------------------

    def load_fact_bioactivity(self, df: pd.DataFrame) -> int:
        """Upsert fact rows into ``fact_bioactivity``.

        The primary key (``activity_id``) comes from ChEMBL and is stable
        across pipeline runs, making idempotent upserts straightforward.
        """
        return self._upsert(df, "fact_bioactivity", TABLE_PRIMARY_KEYS["fact_bioactivity"])

    # ------------------------------------------------------------------
    # Bulk convenience
    # ------------------------------------------------------------------

    def load_all(
        self,
        dim_protein: pd.DataFrame,
        dim_drug: pd.DataFrame,
        dim_article: pd.DataFrame,
        dim_structure: pd.DataFrame,
        fact_bioactivity: pd.DataFrame,
    ) -> dict[str, int]:
        """Load all dimensions and the fact table in the correct FK order.

        Returns a dict mapping each table name to the number of rows loaded.
        """
        logger.info("Beginning bulk warehouse load.")
        results: dict[str, int] = {}

        # Dimensions first (no cross-dependencies).
        results["dim_protein"] = self.load_dim_protein(dim_protein)
        results["dim_drug"] = self.load_dim_drug(dim_drug)
        results["dim_article"] = self.load_dim_article(dim_article)
        results["dim_structure"] = self.load_dim_structure(dim_structure)

        # Fact table last (references all dimensions).
        results["fact_bioactivity"] = self.load_fact_bioactivity(fact_bioactivity)

        total = sum(results.values())
        logger.info("Bulk load complete – %d total rows written across %d tables.", total, len(results))
        return results
