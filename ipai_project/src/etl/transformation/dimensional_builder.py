"""Dimensional model builder.

Transforms cleaned DataFrames into the star-schema dimension and fact tables
used by the ELIQSIR data warehouse.

Each ``build_dim_*`` and ``build_fact_*`` method:
  - Accepts one or more clean DataFrames from :class:`~src.transformation.DataCleaner`.
  - Returns a new DataFrame whose columns match the corresponding ORM
    model in :mod:`src.database.schema`.
  - Is *idempotent* – calling it twice with the same input produces the
    same output.

The builder itself is stateless; it holds no mutable state between calls.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lookup helpers (natural key → surrogate key)
# ---------------------------------------------------------------------------


def _assign_surrogate_keys(
    natural_keys: pd.Series,
    lookup: dict[str, int],
    next_sk: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, int], int]:
    """Map a Series of natural keys to integer surrogate keys.

    Parameters
    ----------
    natural_keys:
        Series of natural key values (must be string-compatible).
    lookup:
        Existing ``{natural_key: surrogate_key}`` mapping, possibly empty.
    next_sk:
        The next surrogate key integer to assign.

    Returns
    -------
    key_series:
        Integer surrogate key for every row in *natural_keys*.
    new_rows_df:
        DataFrame ``(natural_key, surrogate_key)`` for *newly* discovered keys.
    updated_lookup:
        The lookup dict extended with any new mappings.
    updated_next_sk:
        The next available surrogate key after this call.
    """
    # Work with string representations to ensure JSON-serialisable keys.
    str_keys = natural_keys.astype(str)
    new_keys = [k for k in pd.unique(str_keys) if k not in lookup]

    if new_keys:
        new_mapping = {
            k: sk for sk, k in enumerate(new_keys, start=next_sk)
        }
        lookup.update(new_mapping)
        next_sk += len(new_keys)
        new_rows_df = pd.DataFrame(
            {
                "natural_key": new_keys,
                "surrogate_key": [lookup[k] for k in new_keys],
            }
        )
    else:
        new_rows_df = pd.DataFrame(columns=["natural_key", "surrogate_key"])

    key_series = str_keys.map(lookup).astype(int)
    return key_series, new_rows_df, lookup, next_sk


# ---------------------------------------------------------------------------
# DimensionalModelBuilder
# ---------------------------------------------------------------------------


class DimensionalModelBuilder:
    """Build star-schema dimension and fact DataFrames.

    Parameters
    ----------
    protein_lookup:
        Optional seed ``{uniprot_id: surrogate_key}`` dict for incremental
        loads.  Defaults to an empty dict (full load).
    drug_lookup:
        Optional seed for drug dimension.
    article_lookup:
        Optional seed for article dimension.
    structure_lookup:
        Optional seed for structure dimension.
    """

    def __init__(
        self,
        protein_lookup: dict[str, int] | None = None,
        drug_lookup: dict[str, int] | None = None,
        article_lookup: dict[str, int] | None = None,
        structure_lookup: dict[str, int] | None = None,
    ) -> None:
        self._protein_lookup: dict[str, int] = protein_lookup or {}
        self._drug_lookup: dict[str, int] = drug_lookup or {}
        self._article_lookup: dict[str, int] = article_lookup or {}
        self._structure_lookup: dict[str, int] = structure_lookup or {}

        self._protein_next_sk: int = (max(self._protein_lookup.values()) + 1) if self._protein_lookup else 1
        self._drug_next_sk: int = (max(self._drug_lookup.values()) + 1) if self._drug_lookup else 1
        self._article_next_sk: int = (max(self._article_lookup.values()) + 1) if self._article_lookup else 1
        self._structure_next_sk: int = (max(self._structure_lookup.values()) + 1) if self._structure_lookup else 1

    # ------------------------------------------------------------------
    # DimProtein
    # ------------------------------------------------------------------

    def build_dim_protein(self, df_uniprot: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimProtein`` dimension table.

        Input columns (clean UniProt DataFrame):
            ``accession``, ``gene_names``, ``protein_name``

        Output columns:
            ``protein_key`` (int, surrogate PK),
            ``uniprot_id`` (str, natural key),
            ``gene_names`` (str | NA),
            ``protein_name`` (str | NA)
        """
        logger.info("Building DimProtein – %d input rows.", len(df_uniprot))

        keys, new_rows, self._protein_lookup, self._protein_next_sk = (
            _assign_surrogate_keys(
                df_uniprot["accession"],
                self._protein_lookup,
                self._protein_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimProtein – no new proteins to add.")
            return pd.DataFrame(
                columns=["protein_key", "uniprot_id", "gene_names", "protein_name"]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "protein_key", "natural_key": "uniprot_id"}
        )
        attrs = df_uniprot[["accession", "gene_names", "protein_name"]].rename(
            columns={"accession": "uniprot_id"}
        )
        dim = dim.merge(attrs.drop_duplicates("uniprot_id"), on="uniprot_id", how="left")
        dim = dim[["protein_key", "uniprot_id", "gene_names", "protein_name"]]

        logger.info("DimProtein – %d new proteins added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimDrug
    # ------------------------------------------------------------------

    def build_dim_drug(self, df_chembl: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimDrug`` dimension table.

        Input columns (clean ChEMBL DataFrame):
            ``drug_chembl_id``, ``drug_name``

        Output columns:
            ``drug_key`` (int),
            ``drug_chembl_id`` (str),
            ``drug_name`` (str | NA)
        """
        logger.info("Building DimDrug.")

        drug_df = (
            df_chembl[["drug_chembl_id", "drug_name"]]
            .drop_duplicates(subset=["drug_chembl_id"])
        )

        keys, new_rows, self._drug_lookup, self._drug_next_sk = (
            _assign_surrogate_keys(
                drug_df["drug_chembl_id"],
                self._drug_lookup,
                self._drug_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimDrug – no new drugs to add.")
            return pd.DataFrame(columns=["drug_key", "drug_chembl_id", "drug_name"])

        dim = new_rows.rename(
            columns={"surrogate_key": "drug_key", "natural_key": "drug_chembl_id"}
        )
        dim = dim.merge(
            drug_df.rename(columns={}),
            on="drug_chembl_id",
            how="left",
        )
        dim = dim[["drug_key", "drug_chembl_id", "drug_name"]]

        logger.info("DimDrug – %d new drugs added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimArticle
    # ------------------------------------------------------------------

    def build_dim_article(
        self,
        df_chembl: pd.DataFrame,
        df_abstracts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the ``DimArticle`` dimension table.

        The article metadata (title, journal, year) comes from the ChEMBL
        bioactivity table; the full abstract text is joined from the PubMed
        extract.

        Input columns:
            *df_chembl*: ``pubmed_id``, ``article_title``, ``journal``, ``year``
            *df_abstracts*: ``pubmed_id``, ``abstract``

        Output columns:
            ``article_key`` (int),
            ``pubmed_id`` (Int64),
            ``article_title`` (str | NA),
            ``journal`` (str | NA),
            ``year`` (Int64 | NA),
            ``abstract`` (str | NA)
        """
        logger.info("Building DimArticle.")

        # Deduplicate article metadata from ChEMBL.
        article_meta = (
            df_chembl[["pubmed_id", "article_title", "journal", "year"]]
            .drop_duplicates(subset=["pubmed_id"])
            .dropna(subset=["pubmed_id"])
        )
        article_meta["pubmed_id"] = article_meta["pubmed_id"].astype("Int64")

        # Join abstract text.
        df_abs = df_abstracts[["pubmed_id", "abstract"]].copy()
        df_abs["pubmed_id"] = df_abs["pubmed_id"].astype("Int64")

        article_full = article_meta.merge(df_abs, on="pubmed_id", how="left")

        keys, new_rows, self._article_lookup, self._article_next_sk = (
            _assign_surrogate_keys(
                article_full["pubmed_id"].astype(str),
                self._article_lookup,
                self._article_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimArticle – no new articles to add.")
            return pd.DataFrame(
                columns=["article_key", "pubmed_id", "article_title", "journal", "year", "abstract"]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "article_key", "natural_key": "_pmid_str"}
        )
        article_full["_pmid_str"] = article_full["pubmed_id"].astype(str)
        dim = dim.merge(article_full, on="_pmid_str", how="left")
        dim = dim[["article_key", "pubmed_id", "article_title", "journal", "year", "abstract"]]

        logger.info("DimArticle – %d new articles added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimStructure
    # ------------------------------------------------------------------

    def build_dim_structure(self, df_pdbe: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimStructure`` dimension table.

        Input columns (clean PDBe DataFrame):
            ``uniprot_id``, ``pdb_id``, ``chain_id``, ``resolution``,
            ``coverage``, ``method``, ``unp_start``, ``unp_end``

        Output columns:
            ``structure_key`` (int),
            ``pdb_id`` (str),
            ``uniprot_id`` (str),
            ``chain_id`` (str | NA),
            ``resolution`` (float | NA),
            ``coverage`` (float | NA),
            ``method`` (str | NA),
            ``unp_start`` (Int64 | NA),
            ``unp_end`` (Int64 | NA)
        """
        logger.info("Building DimStructure – %d input rows.", len(df_pdbe))

        # Natural key = pdb_id + chain_id
        df_pdbe = df_pdbe.copy()
        df_pdbe["_nk"] = df_pdbe["pdb_id"] + "_" + df_pdbe["chain_id"].fillna("?")

        keys, new_rows, self._structure_lookup, self._structure_next_sk = (
            _assign_surrogate_keys(
                df_pdbe["_nk"],
                self._structure_lookup,
                self._structure_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimStructure – no new structures to add.")
            return pd.DataFrame(
                columns=[
                    "structure_key", "pdb_id", "uniprot_id", "chain_id",
                    "resolution", "coverage", "method", "unp_start", "unp_end",
                ]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "structure_key", "natural_key": "_nk"}
        )
        dim = dim.merge(df_pdbe, on="_nk", how="left")
        dim = dim[[
            "structure_key", "pdb_id", "uniprot_id", "chain_id",
            "resolution", "coverage", "method", "unp_start", "unp_end",
        ]]

        logger.info("DimStructure – %d new structures added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # FactBioactivity
    # ------------------------------------------------------------------

    def build_fact_bioactivity(self, df_chembl: pd.DataFrame) -> pd.DataFrame:
        """Build the ``FactBioactivity`` fact table.

        Resolves all foreign keys from the current state of the builder's
        internal lookup tables.  Must be called *after* the four
        ``build_dim_*`` methods so that all lookups are populated.

        Input columns (clean ChEMBL DataFrame):
            ``activity_id``, ``uniprot_id``, ``drug_chembl_id``, ``pubmed_id``,
            ``standard_type``, ``standard_value``, ``standard_units``,
            ``pchembl_value``, ``confidence_score``, ``assay_type``,
            ``assay_description``

        Output columns:
            ``activity_id``, ``protein_key``, ``drug_key``, ``article_key``,
            ``standard_type``, ``standard_value``, ``standard_units``,
            ``pchembl_value``, ``confidence_score``, ``assay_type``,
            ``assay_description``
        """
        logger.info("Building FactBioactivity – %d input rows.", len(df_chembl))

        fact = df_chembl.copy()

        # Resolve foreign keys; rows without a match become NaN → drop them.
        fact["protein_key"] = (
            fact["uniprot_id"].astype(str).map(self._protein_lookup)
        )
        fact["drug_key"] = (
            fact["drug_chembl_id"].astype(str).map(self._drug_lookup)
        )
        fact["article_key"] = (
            fact["pubmed_id"].astype(str).map(self._article_lookup)
        )

        before = len(fact)
        fact = fact.dropna(subset=["protein_key", "drug_key"])
        dropped = before - len(fact)
        if dropped:
            logger.warning(
                "FactBioactivity – dropped %d rows with unresolved foreign keys.", dropped
            )

        fact["protein_key"] = fact["protein_key"].astype(int)
        fact["drug_key"] = fact["drug_key"].astype(int)
        # article_key may be null (no abstract available) – keep as nullable int.
        fact["article_key"] = (
            pd.to_numeric(fact["article_key"], errors="coerce").astype("Int64")
        )

        output_cols = [
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
        ]
        available = [c for c in output_cols if c in fact.columns]
        fact = fact[available]

        logger.info(
            "FactBioactivity complete – %d rows, %d columns.", len(fact), len(available)
        )
        return fact.reset_index(drop=True)
