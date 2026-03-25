"""Unit tests for the extraction layer.

Tests are deliberately lightweight – they use mocked HTTP responses and a
mocked MySQL connection so that no real network calls or databases are needed.

Run with::

    pytest tests/test_extraction.py -v
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.extraction.chembl_extractor import ChemblExtractor
from src.extraction.pdbe_extractor import PdbeExtractor
from src.extraction.pubmed_extractor import PubMedExtractor
from src.extraction.uniprot_extractor import UniProtExtractor


# ---------------------------------------------------------------------------
# UniProtExtractor
# ---------------------------------------------------------------------------


class TestUniProtExtractor:
    """Tests for UniProtExtractor."""

    def _make_extractor(self) -> UniProtExtractor:
        return UniProtExtractor(organism_id=9606, reviewed=True)

    def test_build_params_reviewed(self):
        ext = self._make_extractor()
        params = ext._build_params()
        assert "reviewed:true" in params["query"]
        assert params["format"] == "tsv"

    def test_build_params_not_reviewed(self):
        ext = UniProtExtractor(reviewed=False)
        params = ext._build_params()
        assert "reviewed:false" in params["query"]

    def test_normalise_columns_renames_entry(self):
        raw = pd.DataFrame({
            "Entry": ["P00533"],
            "Gene Names": ["EGFR"],
            "Protein names": ["Epidermal growth factor receptor"],
        })
        ext = self._make_extractor()
        out = ext._normalise_columns(raw)
        assert "accession" in out.columns
        assert "gene_names" in out.columns
        assert "protein_name" in out.columns

    @patch("src.extraction.uniprot_extractor.requests.Session.get")
    def test_extract_returns_dataframe(self, mock_get):
        tsv_content = "Entry\tGene Names\tProtein names\nP00533\tEGFR\tEGF receptor\n"
        mock_response = MagicMock()
        mock_response.text = tsv_content
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        ext = self._make_extractor()
        df = ext.extract()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["accession"] == "P00533"


# ---------------------------------------------------------------------------
# ChemblExtractor
# ---------------------------------------------------------------------------

# Expected columns returned by _BIOACTIVITY_QUERY
_CHEMBL_COLUMNS = [
    "activity_id", "drug_chembl_id", "drug_name", "molecule_type",
    "molecular_weight", "canonical_smiles", "target_chembl_id", "target_name",
    "organism", "standard_type", "standard_value", "standard_units",
    "pchembl_value", "assay_type", "assay_description", "assay_organism",
    "confidence_score", "article_title", "journal", "year", "pubmed_id",
    "doi", "uniprot_id",
]


def _make_chembl_extractor(dump_dir=None) -> ChemblExtractor:
    return ChemblExtractor(
        host="localhost",
        user="test_user",
        password="test_pass",
        database="chembl_36",
        dump_dir=dump_dir,
    )


class TestChemblExtractor:
    """Tests for ChemblExtractor (MySQL-backed, fully mocked)."""

    @patch("src.extraction.chembl_extractor.pd.read_sql_query")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_extract_returns_expected_rows(self, mock_connect, mock_read_sql):
        row = dict.fromkeys(_CHEMBL_COLUMNS)
        row["drug_chembl_id"] = "CHEMBL1"
        row["uniprot_id"] = "P00533"
        row["activity_id"] = 1
        row["standard_value"] = 100.0
        expected_df = pd.DataFrame([row])

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = expected_df

        ext = _make_chembl_extractor()
        df = ext.extract(uniprot_ids=["P00533"])

        assert len(df) == 1
        assert df.iloc[0]["drug_chembl_id"] == "CHEMBL1"
        assert df.iloc[0]["uniprot_id"] == "P00533"

    @patch("src.extraction.chembl_extractor.pd.read_sql_query")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_extract_empty_when_no_match(self, mock_connect, mock_read_sql):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(columns=_CHEMBL_COLUMNS)

        ext = _make_chembl_extractor()
        df = ext.extract(uniprot_ids=["Q99999"])
        assert df.empty

    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_raises_on_empty_ids(self, mock_connect):
        ext = _make_chembl_extractor()
        with pytest.raises(ValueError):
            ext.extract(uniprot_ids=[])

    @patch("src.extraction.chembl_extractor.pd.read_sql_query")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_temp_table_created_and_populated(self, mock_connect, mock_read_sql):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(columns=_CHEMBL_COLUMNS)

        ext = _make_chembl_extractor()
        ext.extract(uniprot_ids=["P00533", "P04637"])

        drop_call = call("DROP TEMPORARY TABLE IF EXISTS _protein_filter")
        assert drop_call in mock_cursor.execute.call_args_list
        create_calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("CREATE TEMPORARY TABLE _protein_filter" in c for c in create_calls)
        mock_cursor.executemany.assert_called_once()

    @patch("src.extraction.chembl_extractor.pd.read_sql_query")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_deduplicates_uniprot_ids(self, mock_connect, mock_read_sql):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame(columns=_CHEMBL_COLUMNS)

        ext = _make_chembl_extractor()
        ext.extract(uniprot_ids=["P00533", "P00533", "P04637"])

        args = mock_cursor.executemany.call_args[0]
        inserted_ids = [row[0] for row in args[1]]
        assert len(inserted_ids) == 2
        assert "P00533" in inserted_ids
        assert "P04637" in inserted_ids

    # ------------------------------------------------------------------
    # setup_database tests
    # ------------------------------------------------------------------

    def test_setup_database_raises_when_no_dump_dir(self):
        """setup_database must fail fast when dump_dir is not set."""
        ext = _make_chembl_extractor(dump_dir=None)
        with pytest.raises(FileNotFoundError, match="dump_dir"):
            ext.setup_database()

    def test_setup_database_raises_when_dmp_missing(self, tmp_path):
        """setup_database must raise if the .dmp file is absent."""
        ext = _make_chembl_extractor(dump_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="chembl_36_mysql.dmp"):
            ext.setup_database()

    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_setup_database_skips_when_already_populated(self, mock_connect, tmp_path):
        """setup_database exits without touching mysql CLI when DB is populated."""
        # Make _database_is_populated() return True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create a dummy .dmp file so path checks pass
        dmp = tmp_path / "chembl_36_mysql.dmp"
        dmp.write_bytes(b"")

        ext = _make_chembl_extractor(dump_dir=tmp_path)
        # Only one connect call (for the population check); no subprocess
        with patch("src.extraction.chembl_extractor.subprocess.run") as mock_sub:
            ext.setup_database()
            mock_sub.assert_not_called()

    @patch("src.extraction.chembl_extractor.subprocess.run")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_setup_database_loads_dump_when_empty(self, mock_connect, mock_run, tmp_path):
        """setup_database runs CREATE DATABASE then mysql CLI when DB is empty."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")

        dmp = tmp_path / "chembl_36_mysql.dmp"
        dmp.write_bytes(b"-- mock dump")

        ext = _make_chembl_extractor(dump_dir=tmp_path)
        # Force _database_is_populated to return False so the import runs
        with patch.object(ext, "_database_is_populated", return_value=False):
            ext.setup_database()

        # mysql CLI must have been called exactly once
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert "mysql" in cmd_args[0]
        assert "chembl_36" in cmd_args   # database name passed to CLI

    @patch("src.extraction.chembl_extractor.subprocess.run")
    @patch("src.extraction.chembl_extractor.mysql.connector.connect")
    def test_setup_database_raises_on_nonzero_exit(self, mock_connect, mock_run, tmp_path):
        """setup_database raises RuntimeError when mysql CLI fails."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_run.return_value = MagicMock(returncode=1, stderr=b"Access denied")

        dmp = tmp_path / "chembl_36_mysql.dmp"
        dmp.write_bytes(b"-- mock dump")

        ext = _make_chembl_extractor(dump_dir=tmp_path)
        with patch.object(ext, "_database_is_populated", return_value=False):
            with pytest.raises(RuntimeError, match="mysql import failed"):
                ext.setup_database()




# ---------------------------------------------------------------------------
# PdbeExtractor
# ---------------------------------------------------------------------------


class TestPdbeExtractor:
    """Tests for PdbeExtractor."""

    @patch("src.extraction.pdbe_extractor.requests.Session.get")
    def test_extract_single_protein(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "P00533": [
                {
                    "pdb_id": "1IVO",
                    "chain_id": "A",
                    "resolution": 2.6,
                    "coverage": 0.8,
                    "experimental_method": "X-ray",
                    "unp_start": 1,
                    "unp_end": 350,
                }
            ]
        }
        mock_get.return_value = mock_response

        ext = PdbeExtractor(request_delay_s=0)
        df = ext.extract(["P00533"])

        assert len(df) == 1
        assert df.iloc[0]["pdb_id"] == "1IVO"
        assert df.iloc[0]["method"] == "X-ray"

    @patch("src.extraction.pdbe_extractor.requests.Session.get")
    def test_extract_handles_404(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        ext = PdbeExtractor(request_delay_s=0)
        df = ext.extract(["Q99999"])
        assert df.empty


# ---------------------------------------------------------------------------
# PubMedExtractor
# ---------------------------------------------------------------------------


class TestPubMedExtractor:
    """Tests for PubMedExtractor."""

    def test_raises_on_empty_email(self):
        with pytest.raises(ValueError):
            PubMedExtractor(email="")

    def test_extract_empty_ids(self):
        ext = PubMedExtractor(email="test@example.com")
        df = ext.extract([])
        assert df.empty
        assert list(df.columns) == ["pubmed_id", "abstract", "authors", "pub_date", "year", "month", "doi"]
