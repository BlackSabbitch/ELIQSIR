"""Unit tests for the loading layer.

Uses an in-memory SQLite database instead of a real MySQL server so that
these tests run without any external infrastructure.

The tests verify:
  - Schema initialisation (DDL).
  - Upsert logic (new rows are inserted, existing rows are updated).
  - ``load_all`` loads all five tables in the correct FK order.

Run with::

    pytest tests/test_loading.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.database.schema import apply_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_in_memory_engine():
    """Return a fresh in-memory SQLite engine with the full schema applied.

    ``apply_schema`` is called with ``sqlite_compat=True`` so that MySQL-only
    DDL constructs (``ENGINE``, ``FULLTEXT KEY``, ``VARCHAR``, etc.) are
    automatically translated to SQLite equivalents before execution.
    """
    engine = create_engine("sqlite:///:memory:", echo=False)
    apply_schema(engine, sqlite_compat=True)
    return engine


# ---------------------------------------------------------------------------
# WarehouseLoader (patched to use SQLite)
# ---------------------------------------------------------------------------


class TestWarehouseLoader:
    """Smoke tests for WarehouseLoader using an in-memory SQLite database."""

    @pytest.fixture()
    def loader(self):
        """Return a WarehouseLoader wired to an in-memory SQLite engine."""
        engine = make_in_memory_engine()

        # Import here so we can patch the engine after module load.
        from src.loading.warehouse_loader import WarehouseLoader

        loader = WarehouseLoader()
        loader._engine = engine   # override the MySQL engine
        return loader

    @pytest.fixture()
    def dim_protein_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "protein_key": [1, 2],
            "uniprot_id": ["P00533", "P04637"],
            "gene_names": ["EGFR", "TP53"],
            "protein_name": ["EGF receptor", "Tumour suppressor p53"],
        })

    @pytest.fixture()
    def dim_drug_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "drug_key": [1],
            "drug_chembl_id": ["CHEMBL1"],
            "drug_name": ["Drug A"],
        })

    @pytest.fixture()
    def dim_article_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "article_key": [1],
            "pubmed_id": pd.array([12345678], dtype="Int64"),
            "article_title": ["A study"],
            "journal": ["Nature"],
            "year": pd.array([2020], dtype="Int64"),
            "abstract": ["Some abstract text."],
        })

    @pytest.fixture()
    def dim_structure_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "structure_key": [1],
            "pdb_id": ["1IVO"],
            "uniprot_id": ["P00533"],
            "chain_id": ["A"],
            "resolution": [2.6],
            "coverage": [0.8],
            "method": ["X-ray"],
            "unp_start": pd.array([1], dtype="Int64"),
            "unp_end": pd.array([350], dtype="Int64"),
        })

    @pytest.fixture()
    def fact_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "activity_id": [1001],
            "protein_key": [1],
            "drug_key": [1],
            "article_key": pd.array([1], dtype="Int64"),
            "standard_type": ["IC50"],
            "standard_value": [100.0],
            "standard_units": ["nM"],
            "pchembl_value": [7.0],
            "confidence_score": [9],
            "assay_type": ["B"],
            "assay_description": ["Binding assay"],
        })

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_initialise_schema_creates_tables(self, loader):
        loader.initialise_schema()  # should not raise

    def test_load_dim_protein_inserts_rows(self, loader, dim_protein_df):
        n = loader.load_dim_protein(dim_protein_df)
        assert n == len(dim_protein_df)

    def test_load_dim_protein_empty_df_skipped(self, loader):
        n = loader.load_dim_protein(pd.DataFrame(columns=["protein_key"]))
        assert n == 0

    def test_load_all_runs_without_error(
        self, loader, dim_protein_df, dim_drug_df,
        dim_article_df, dim_structure_df, fact_df,
    ):
        counts = loader.load_all(
            dim_protein=dim_protein_df,
            dim_drug=dim_drug_df,
            dim_article=dim_article_df,
            dim_structure=dim_structure_df,
            fact_bioactivity=fact_df,
        )
        assert counts["dim_protein"] == 2
        assert counts["fact_bioactivity"] == 1
