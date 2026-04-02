"""Data cleaning and normalisation for the ELIQSIR pipeline.

``DataCleaner`` receives raw ``pd.DataFrame`` objects produced by the
extraction layer and returns cleaned, type-correct DataFrames ready for
dimensional modelling.

Design principles
-----------------
* **Non-destructive** - the original DataFrame is never mutated; a copy is returned.
* **Transparent** - every cleaning step is logged at DEBUG level.
* **Fail-fast** - missing *required* columns raise ``KeyError`` immediately.
"""

from __future__ import annotations
import re
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _assert_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Raise ``KeyError`` if *df* is missing any of *required*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{context}] Required columns missing from DataFrame: {missing}"
        )

def _log_shape(df: pd.DataFrame, stage: str) -> None:
    logger.debug("[%s] shape=%s", stage, df.shape)


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """Stateless collection of cleaning methods for each raw dataset."""

    # ------------------------------------------------------------------
    # UniProt
    # ------------------------------------------------------------------

    def clean_uniprot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the raw UniProt protein table.
        
        Preserves ML features like sequence, sequence_length, and protein_families.
        """
        required = [
            "uniprot_id", 
            "gene_names", 
            "protein_name", 
            "sequence", 
            "sequence_length", 
            "protein_families"
        ]
        _assert_columns(df, required, "clean_uniprot")

        # Copy the entire DataFrame to avoid losing any extra extracted columns
        out = df.copy()
        _log_shape(out, "clean_uniprot - before")

        # 1. Drop null accessions.
        out = out.dropna(subset=["uniprot_id"])
        out = out[out["uniprot_id"].astype(str).str.strip() != ""]

        # 2. Deduplicate.
        before = len(out)
        out = out.drop_duplicates(subset=["uniprot_id"], keep="first")
        if (dropped := before - len(out)):
            logger.debug("clean_uniprot - dropped %d duplicate accessions.", dropped)

        # 3. Uppercase accession (UniProt canonical form).
        out["uniprot_id"] = out["uniprot_id"].str.upper().str.strip()

        # 4. Clean sequences (remove whitespace/newlines if any).
        if "sequence" in out.columns:
            out["sequence"] = out["sequence"].astype(str).str.replace(r"\s+", "", regex=True)

        # 5. Cast sequence_length to nullable integer.
        out["sequence_length"] = pd.to_numeric(out["sequence_length"], errors="coerce").astype("Int64")

        # 6. Strip remaining string columns and replace empty with NA.
        str_cols = out.select_dtypes(include="object").columns
        for col in str_cols:
            out[col] = out[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

        _log_shape(out, "clean_uniprot - after")
        logger.info("UniProt cleaning complete - %d proteins retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # ChEMBL bioactivity
    # ------------------------------------------------------------------

    def clean_chembl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw ChEMBL bioactivity table."""
        required = [
            "activity_id",
            "drug_chembl_id",
            "uniprot_id",
            "standard_value",
            "molecule_type",
            "molecular_weight",
            "canonical_smiles"
        ]
        _assert_columns(df, required, "clean_chembl")

        out = df.copy()
        _log_shape(out, "clean_chembl - before")

        # 1. Drop critical nulls (we can't train ML without a value or target).
        out = out.dropna(subset=["standard_value", "uniprot_id", "drug_chembl_id"])

        # 2. Float casts (including the new ML molecular_weight).
        for col in ("standard_value", "pchembl_value", "molecular_weight"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # 3. Integer casts with nullable dtype.
        for col in ("year", "pubmed_id", "confidence_score"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

        # 4. Normalise UniProt ID.
        out["uniprot_id"] = (
            out["uniprot_id"]
            .astype(str)
            .apply(lambda x: re.sub(r"[^a-zA-Z0-9]", "", x).upper())
        )

        # 5. Deduplicate.
        before = len(out)
        out = out.drop_duplicates(subset=["activity_id"], keep="first")
        if (dropped := before - len(out)):
            logger.debug("clean_chembl - dropped %d duplicate activity_ids.", dropped)

        # 6. Clean SMILES (remove any accidental whitespace).
        if "canonical_smiles" in out.columns:
            out["canonical_smiles"] = out["canonical_smiles"].astype(str).str.strip()

        # 7. Strip string columns.
        str_cols = out.select_dtypes(include="object").columns
        for col in str_cols:
            out[col] = out[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

        _log_shape(out, "clean_chembl - after")
        logger.info("ChEMBL cleaning complete - %d bioactivity records retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PDBe structures
    # ------------------------------------------------------------------

    def clean_pdbe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw PDBe structure table."""
        required = ["uniprot_id", "pdb_id"]
        _assert_columns(df, required, "clean_pdbe")

        out = df.copy()
        _log_shape(out, "clean_pdbe - before")

        # 1. Drop critical nulls.
        out = out.dropna(subset=["pdb_id", "uniprot_id"])

        # 2. Uppercase PDB ID and UniProt ID.
        out["pdb_id"] = out["pdb_id"].astype(str).str.upper().str.strip()
        out["uniprot_id"] = out["uniprot_id"].astype(str).str.upper().str.strip()

        # 3. Float casts.
        for col in ("resolution", "coverage"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # 4. Integer casts.
        for col in ("unp_start", "unp_end"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

        # 5. Deduplicate.
        dedup_keys = [c for c in ("uniprot_id", "pdb_id", "chain_id") if c in out.columns]
        before = len(out)
        out = out.drop_duplicates(subset=dedup_keys, keep="first")
        if (dropped := before - len(out)):
            logger.debug("clean_pdbe - dropped %d duplicate structure entries.", dropped)

        _log_shape(out, "clean_pdbe - after")
        logger.info("PDBe cleaning complete - %d structure entries retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PubMed abstracts
    # ------------------------------------------------------------------

    def clean_pubmed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw PubMed abstract table.
           Preserves authors, doi, journal, year, title, and pub_date for temporal modeling.
        """
        required = ["pubmed_id", "abstract", "authors"]
        _assert_columns(df, required, "clean_pubmed")

        out = df.copy()
        _log_shape(out, "clean_pubmed - before")

        # 1 & 2. Null-drop and integer cast for primary key.
        out = out.dropna(subset=["pubmed_id"])
        out["pubmed_id"] = pd.to_numeric(out["pubmed_id"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["pubmed_id"])

        # 3. Cast year to integer.
        if "year" in out.columns:
            out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

        # 4. Clean pub_date (prepare for datetime parsing in builder)
        # We strip whitespace to prevent pd.to_datetime from failing silently on edge cases.
        if "pub_date" in out.columns:
            out["pub_date"] = out["pub_date"].astype(str).str.strip()

        # 5. Normalise abstract and authors text (collapse multiple spaces).
        for col in ("abstract", "authors"):
            if col in out.columns:
                out[col] = (
                    out[col]
                    .astype(str)
                    .str.strip()
                    .apply(lambda t: re.sub(r"\s+", " ", t))
                )

        # 6. Empty/Invalid strings -> NA for all string columns.
        str_cols = out.select_dtypes(include="object").columns
        for col in str_cols:
            out[col] = out[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

        # 7. Deduplicate.
        before = len(out)
        out = out.drop_duplicates(subset=["pubmed_id"], keep="first")
        if (dropped := before - len(out)):
            logger.debug("clean_pubmed - dropped %d duplicate pubmed_ids.", dropped)

        _log_shape(out, "clean_pubmed - after")
        logger.info("PubMed cleaning complete - %d abstracts retained.", len(out))
        return out.reset_index(drop=True)