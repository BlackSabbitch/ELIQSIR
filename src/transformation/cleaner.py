"""Data cleaning and normalisation for the ELIQSIR pipeline.

``DataCleaner`` receives raw ``pd.DataFrame`` objects produced by the
extraction layer and returns cleaned, type-correct DataFrames ready for
dimensional modelling.

Each ``clean_*`` method documents:
  - Which columns are expected on input.
  - Which columns are guaranteed on output.
  - Every transformation rule applied.

Design principles
-----------------
* **Non-destructive** – the original DataFrame is never mutated; a copy is
  always returned.
* **Transparent** – every cleaning step is logged at DEBUG level so that
  pipeline runs can be audited.
* **Fail-fast** – missing *required* columns raise ``KeyError`` immediately
  rather than producing silent NaNs downstream.
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
    """Stateless collection of cleaning methods for each raw dataset.

    All methods follow the signature::

        clean_*(raw_df: pd.DataFrame) -> pd.DataFrame

    and return a *new* DataFrame.
    """

    # ------------------------------------------------------------------
    # UniProt
    # ------------------------------------------------------------------

    def clean_uniprot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise the raw UniProt protein table.

        Input columns (from :class:`~src.extraction.UniProtExtractor`):
            ``accession``, ``gene_names``, ``protein_name``

        Transformations:
            1. Drop rows where ``accession`` is null or empty.
            2. Strip leading / trailing whitespace from all string columns.
            3. Deduplicate on ``accession`` (keep first occurrence).
            4. Normalise ``accession`` to uppercase (UniProt canonical form).
            5. Replace empty strings with ``pd.NA``.

        Output columns:
            ``accession``, ``gene_names``, ``protein_name``
        """
        required = ["accession", "gene_names", "protein_name"]
        _assert_columns(df, required, "clean_uniprot")

        out = df[required].copy()
        _log_shape(out, "clean_uniprot – before")

        # 1. Drop null accessions.
        out = out.dropna(subset=["accession"])
        out = out[out["accession"].str.strip() != ""]

        # 2. Strip whitespace from all string columns.
        for col in required:
            out[col] = out[col].astype(str).str.strip()

        # 3. Deduplicate.
        before = len(out)
        out = out.drop_duplicates(subset=["accession"], keep="first")
        dropped = before - len(out)
        if dropped:
            logger.debug("clean_uniprot – dropped %d duplicate accessions.", dropped)

        # 4. Uppercase accession.
        out["accession"] = out["accession"].str.upper()

        # 5. Empty strings → NA.
        out = out.replace({"": pd.NA})

        _log_shape(out, "clean_uniprot – after")
        logger.info("UniProt cleaning complete – %d proteins retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # ChEMBL bioactivity
    # ------------------------------------------------------------------

    def clean_chembl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw ChEMBL bioactivity table.

        Input columns (from :class:`~src.extraction.ChemblExtractor`):
            ``activity_id``, ``drug_chembl_id``, ``drug_name``,
            ``target_chembl_id``, ``target_name``, ``organism``,
            ``standard_type``, ``standard_value``, ``standard_units``,
            ``pchembl_value``, ``assay_type``, ``assay_description``,
            ``assay_organism``, ``confidence_score``, ``article_title``,
            ``journal``, ``year``, ``pubmed_id``, ``uniprot_id``

        Transformations:
            1. Drop rows with null ``standard_value`` or ``uniprot_id``.
            2. Cast ``standard_value`` and ``pchembl_value`` to float.
            3. Cast ``year``, ``pubmed_id``, ``confidence_score`` to
               nullable integer (``pd.Int64Dtype``).
            4. Normalise ``uniprot_id`` – strip non-alphanumeric chars
               and uppercase (matches UniProt canonical form).
            5. Deduplicate on ``activity_id``.
            6. Strip whitespace from all remaining string columns.
        """
        required = [
            "activity_id",
            "drug_chembl_id",
            "uniprot_id",
            "standard_value",
            "pubmed_id",
        ]
        _assert_columns(df, required, "clean_chembl")

        out = df.copy()
        _log_shape(out, "clean_chembl – before")

        # 1. Drop critical nulls.
        out = out.dropna(subset=["standard_value", "uniprot_id"])

        # 2. Numeric casts.
        for col in ("standard_value", "pchembl_value"):
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
            logger.debug("clean_chembl – dropped %d duplicate activity_ids.", dropped)

        # 6. Strip string columns.
        str_cols = out.select_dtypes(include="object").columns
        for col in str_cols:
            out[col] = out[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

        _log_shape(out, "clean_chembl – after")
        logger.info("ChEMBL cleaning complete – %d bioactivity records retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PDBe structures
    # ------------------------------------------------------------------

    def clean_pdbe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw PDBe structure table.

        Input columns (from :class:`~src.extraction.PdbeExtractor`):
            ``uniprot_id``, ``pdb_id``, ``chain_id``, ``resolution``,
            ``coverage``, ``method``, ``unp_start``, ``unp_end``

        Transformations:
            1. Drop rows with null ``pdb_id`` or ``uniprot_id``.
            2. Uppercase ``pdb_id`` (PDB canonical form is uppercase).
            3. Cast ``resolution`` and ``coverage`` to float.
            4. Cast ``unp_start`` and ``unp_end`` to nullable integer.
            5. Deduplicate on ``(uniprot_id, pdb_id, chain_id)``.
        """
        required = ["uniprot_id", "pdb_id"]
        _assert_columns(df, required, "clean_pdbe")

        out = df.copy()
        _log_shape(out, "clean_pdbe – before")

        # 1. Drop critical nulls.
        out = out.dropna(subset=["pdb_id", "uniprot_id"])

        # 2. Uppercase PDB ID.
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
            logger.debug("clean_pdbe – dropped %d duplicate structure entries.", dropped)

        _log_shape(out, "clean_pdbe – after")
        logger.info("PDBe cleaning complete – %d structure entries retained.", len(out))
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # PubMed abstracts
    # ------------------------------------------------------------------

    def clean_pubmed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw PubMed abstract table.

        Input columns (from :class:`~src.extraction.PubMedExtractor`):
            ``pubmed_id``, ``abstract``

        Transformations:
            1. Drop rows with null ``pubmed_id``.
            2. Cast ``pubmed_id`` to nullable integer.
            3. Strip whitespace and collapse multiple internal spaces in
               ``abstract``.
            4. Replace empty abstracts with ``pd.NA``.
            5. Deduplicate on ``pubmed_id``.
        """
        required = ["pubmed_id", "abstract"]
        _assert_columns(df, required, "clean_pubmed")

        out = df[required].copy()
        _log_shape(out, "clean_pubmed – before")

        # 1 & 2. Null-drop and integer cast.
        out = out.dropna(subset=["pubmed_id"])
        out["pubmed_id"] = pd.to_numeric(out["pubmed_id"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["pubmed_id"])  # drop any that failed conversion

        # 3. Normalise abstract text.
        out["abstract"] = (
            out["abstract"]
            .astype(str)
            .str.strip()
            .apply(lambda t: re.sub(r"\s+", " ", t))
        )

        # 4. Empty → NA.
        out["abstract"] = out["abstract"].replace({"nan": pd.NA, "": pd.NA})

        # 5. Deduplicate.
        before = len(out)
        out = out.drop_duplicates(subset=["pubmed_id"], keep="first")
        if (dropped := before - len(out)):
            logger.debug("clean_pubmed – dropped %d duplicate pubmed_ids.", dropped)

        _log_shape(out, "clean_pubmed – after")
        logger.info("PubMed cleaning complete – %d abstracts retained.", len(out))
        return out.reset_index(drop=True)
