"""Unit tests for the transformation layer.

Run with::

    pytest tests/test_transformation.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.transformation.cleaner import DataCleaner
from src.transformation.dimensional_builder import DimensionalModelBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_uniprot() -> pd.DataFrame:
    return pd.DataFrame({
        "accession": ["P00533", "p04637", "P00533", None, "  "],
        "gene_names": ["EGFR", "TP53", "EGFR_dup", "BRCA1", ""],
        "protein_name": ["EGF receptor", "Tumour suppressor", "EGF dup", "BRCA1", ""],
    })


@pytest.fixture()
def raw_chembl() -> pd.DataFrame:
    return pd.DataFrame({
        "activity_id": [1, 2, 1],       # one duplicate
        "drug_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL1"],
        "drug_name": ["Drug A", "Drug B", "Drug A"],
        "target_chembl_id": ["CT1", "CT2", "CT1"],
        "target_name": ["Target A", "Target B", "Target A"],
        "organism": ["Homo sapiens"] * 3,
        "standard_type": ["IC50", "Ki", "IC50"],
        "standard_value": [100.0, None, 200.0],
        "standard_units": ["nM", "uM", "nM"],
        "pchembl_value": [7.0, None, 6.7],
        "assay_type": ["B", "F", "B"],
        "assay_description": ["Binding assay", "Functional", "Binding assay"],
        "assay_organism": ["Homo sapiens"] * 3,
        "confidence_score": [9, 8, 9],
        "article_title": ["Title A", "Title B", "Title A"],
        "journal": ["Nature", "Science", "Nature"],
        "year": [2020, 2021, 2020],
        "pubmed_id": [12345678, 87654321, 12345678],
        "uniprot_id": [" p00533 ", "P04637", "P00533"],
    })


@pytest.fixture()
def raw_pdbe() -> pd.DataFrame:
    return pd.DataFrame({
        "uniprot_id": ["P00533", "P04637", None],
        "pdb_id": ["1IVO", "2OCJ", None],
        "chain_id": ["A", "B", "A"],
        "resolution": ["2.6", "3.1", "1.9"],
        "coverage": ["0.8", "0.75", "0.9"],
        "method": ["X-ray", "X-ray", "Cryo-EM"],
        "unp_start": [1, 10, 5],
        "unp_end": [350, 390, 300],
    })


@pytest.fixture()
def raw_pubmed() -> pd.DataFrame:
    return pd.DataFrame({
        "pubmed_id": [12345678, 87654321, None, 12345678],   # one null, one dup
        "abstract": [
            "Abstract text A.",
            "  Abstract text B.  ",
            "Should be dropped.",
            "Duplicate row.",
        ],
    })


# ---------------------------------------------------------------------------
# DataCleaner – UniProt
# ---------------------------------------------------------------------------


class TestDataCleanerUniProt:
    def test_drops_null_accession(self, raw_uniprot):
        out = DataCleaner().clean_uniprot(raw_uniprot)
        assert out["accession"].notna().all()
        assert "" not in out["accession"].values

    def test_deduplicates_accession(self, raw_uniprot):
        out = DataCleaner().clean_uniprot(raw_uniprot)
        assert out["accession"].nunique() == len(out)

    def test_uppercases_accession(self, raw_uniprot):
        out = DataCleaner().clean_uniprot(raw_uniprot)
        assert all(v == v.upper() for v in out["accession"])

    def test_raises_on_missing_column(self):
        with pytest.raises(KeyError):
            DataCleaner().clean_uniprot(pd.DataFrame({"accession": ["X"]}))


# ---------------------------------------------------------------------------
# DataCleaner – ChEMBL
# ---------------------------------------------------------------------------


class TestDataCleanerChembl:
    def test_drops_null_standard_value(self, raw_chembl):
        out = DataCleaner().clean_chembl(raw_chembl)
        assert out["standard_value"].notna().all()

    def test_deduplicates_activity_id(self, raw_chembl):
        out = DataCleaner().clean_chembl(raw_chembl)
        assert out["activity_id"].nunique() == len(out)

    def test_normalises_uniprot_id(self, raw_chembl):
        out = DataCleaner().clean_chembl(raw_chembl)
        # " p00533 " should become "P00533"
        assert "P00533" in out["uniprot_id"].values

    def test_year_is_nullable_int(self, raw_chembl):
        out = DataCleaner().clean_chembl(raw_chembl)
        assert str(out["year"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# DataCleaner – PDBe
# ---------------------------------------------------------------------------


class TestDataCleanerPdbe:
    def test_drops_null_pdb_id(self, raw_pdbe):
        out = DataCleaner().clean_pdbe(raw_pdbe)
        assert out["pdb_id"].notna().all()

    def test_uppercases_pdb_id(self, raw_pdbe):
        out = DataCleaner().clean_pdbe(raw_pdbe)
        assert all(v == v.upper() for v in out["pdb_id"])

    def test_resolution_is_float(self, raw_pdbe):
        out = DataCleaner().clean_pdbe(raw_pdbe)
        assert out["resolution"].dtype == float


# ---------------------------------------------------------------------------
# DataCleaner – PubMed
# ---------------------------------------------------------------------------


class TestDataCleanerPubMed:
    def test_drops_null_pubmed_id(self, raw_pubmed):
        out = DataCleaner().clean_pubmed(raw_pubmed)
        assert out["pubmed_id"].notna().all()

    def test_deduplicates_pubmed_id(self, raw_pubmed):
        out = DataCleaner().clean_pubmed(raw_pubmed)
        assert out["pubmed_id"].nunique() == len(out)

    def test_strips_abstract_whitespace(self, raw_pubmed):
        out = DataCleaner().clean_pubmed(raw_pubmed)
        row = out[out["pubmed_id"] == 87654321]
        assert row["abstract"].iloc[0] == "Abstract text B."


# ---------------------------------------------------------------------------
# DimensionalModelBuilder
# ---------------------------------------------------------------------------


class TestDimensionalModelBuilder:
    """Smoke tests – verify shape, key types, and FK resolution."""

    @pytest.fixture()
    def clean_data(self, raw_uniprot, raw_chembl, raw_pdbe, raw_pubmed):
        c = DataCleaner()
        return {
            "uniprot": c.clean_uniprot(raw_uniprot),
            "chembl": c.clean_chembl(raw_chembl),
            "pdbe": c.clean_pdbe(raw_pdbe),
            "pubmed": c.clean_pubmed(raw_pubmed),
        }

    def test_dim_protein_has_surrogate_key(self, clean_data):
        builder = DimensionalModelBuilder()
        dim = builder.build_dim_protein(clean_data["uniprot"])
        assert "protein_key" in dim.columns
        assert dim["protein_key"].dtype == int

    def test_dim_drug_no_duplicates(self, clean_data):
        builder = DimensionalModelBuilder()
        dim = builder.build_dim_drug(clean_data["chembl"])
        assert dim["drug_chembl_id"].nunique() == len(dim)

    def test_fact_resolves_protein_key(self, clean_data):
        builder = DimensionalModelBuilder()
        builder.build_dim_protein(clean_data["uniprot"])
        builder.build_dim_drug(clean_data["chembl"])
        builder.build_dim_article(clean_data["chembl"], clean_data["pubmed"])
        fact = builder.build_fact_bioactivity(clean_data["chembl"])
        assert "protein_key" in fact.columns
        assert fact["protein_key"].notna().all()

    def test_incremental_load_no_duplicates(self, clean_data):
        """Re-running the builder with the same data must not create new rows."""
        builder = DimensionalModelBuilder()
        dim1 = builder.build_dim_protein(clean_data["uniprot"])
        dim2 = builder.build_dim_protein(clean_data["uniprot"])  # second run
        assert dim2.empty, "Second run should produce zero new rows."
