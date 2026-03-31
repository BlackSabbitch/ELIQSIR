"""Dimensional model builder.

Transforms cleaned DataFrames into the star-schema dimension and fact tables
used by the ELIQSIR data warehouse.

The builder itself is stateless; it holds no mutable state between calls,
other than the running lookup dictionaries for surrogate keys.
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
    """Map a Series of natural keys to integer surrogate keys."""
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
    """Build star-schema dimension and fact DataFrames."""

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
        MUST BE CALLED BEFORE build_dim_structure and build_fact_bioactivity.
        """
        logger.info("Building DimProtein - %d input rows.", len(df_uniprot))

        keys, new_rows, self._protein_lookup, self._protein_next_sk = (
            _assign_surrogate_keys(
                df_uniprot["uniprot_id"],
                self._protein_lookup,
                self._protein_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimProtein - no new proteins to add.")
            return pd.DataFrame(
                columns=[
                    "protein_key", "uniprot_id", "gene_names", 
                    "protein_name", "sequence", "sequence_length", "protein_families"
                ]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "protein_key", "natural_key": "uniprot_id"}
        )
        
        # Include all ML features
        attrs_cols = [
            "uniprot_id", "gene_names", "protein_name", 
            "sequence", "sequence_length", "protein_families"
        ]
        
        # Safely extract columns that exist in the clean dataframe
        available_attrs = [c for c in attrs_cols if c in df_uniprot.columns]
        attrs = df_uniprot[available_attrs]
        
        dim = dim.merge(attrs.drop_duplicates("uniprot_id"), on="uniprot_id", how="left")
        
        # Ensure final column order
        final_cols = [c for c in attrs_cols if c in dim.columns]
        dim = dim[["protein_key"] + final_cols]

        logger.info("DimProtein - %d new proteins added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimDrug
    # ------------------------------------------------------------------

    def build_dim_drug(self, df_chembl: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimDrug`` dimension table."""
        logger.info("Building DimDrug.")

        drug_cols = [
            "drug_chembl_id", "drug_name", "molecule_type", 
            "molecular_weight", "canonical_smiles", "standard_inchi_key"
        ]
        available_cols = [c for c in drug_cols if c in df_chembl.columns]

        drug_df = (
            df_chembl[available_cols]
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
            logger.info("DimDrug - no new drugs to add.")
            return pd.DataFrame(columns=["drug_key"] + drug_cols)

        dim = new_rows.rename(
            columns={"surrogate_key": "drug_key", "natural_key": "drug_chembl_id"}
        )
        dim = dim.merge(drug_df, on="drug_chembl_id", how="left")

        logger.info("DimDrug - %d new drugs added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimArticle
    # ------------------------------------------------------------------

    def build_dim_article(
        self,
        df_chembl: pd.DataFrame,
        df_abstracts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the ``DimArticle`` dimension table."""
        logger.info("Building DimArticle.")

        # Deduplicate article metadata from ChEMBL
        chembl_cols = ["pubmed_id", "article_title", "journal", "year"]
        avail_chembl = [c for c in chembl_cols if c in df_chembl.columns]
        
        article_meta = (
            df_chembl[avail_chembl]
            .drop_duplicates(subset=["pubmed_id"])
            .dropna(subset=["pubmed_id"])
        )
        article_meta["pubmed_id"] = article_meta["pubmed_id"].astype("Int64")

        # Join abstract text, authors, and doi from PubMed
        abs_cols = ["pubmed_id", "abstract", "authors", "doi"]
        avail_abs = [c for c in abs_cols if c in df_abstracts.columns]
        
        df_abs = df_abstracts[avail_abs].copy()
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
            logger.info("DimArticle - no new articles to add.")
            return pd.DataFrame(
                columns=["article_key", "pubmed_id", "article_title", "journal", "year", "abstract", "doi", "authors"]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "article_key", "natural_key": "_pmid_str"}
        )
        article_full["_pmid_str"] = article_full["pubmed_id"].astype(str)
        dim = dim.merge(article_full, on="_pmid_str", how="left")
        
        final_cols = [c for c in ["pubmed_id", "article_title", "journal", "year", "abstract", "doi", "authors"] if c in dim.columns]
        dim = dim[["article_key"] + final_cols]

        logger.info("DimArticle - %d new articles added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimStructure
    # ------------------------------------------------------------------

    def build_dim_structure(self, df_pdbe: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimStructure`` dimension table."""
        logger.info("Building DimStructure - %d input rows.", len(df_pdbe))

        df_pdbe = df_pdbe.copy()
        
        # Map uniprot_id to protein_key using the existing lookup
        df_pdbe["protein_key"] = df_pdbe["uniprot_id"].astype(str).map(self._protein_lookup)
        
        # Drop structures whose proteins didn't exist in our protein lookup
        before = len(df_pdbe)
        df_pdbe = df_pdbe.dropna(subset=["protein_key"])
        df_pdbe["protein_key"] = df_pdbe["protein_key"].astype(int)
        
        if (dropped := before - len(df_pdbe)):
            logger.warning("DimStructure - dropped %d structures with unknown uniprot_ids.", dropped)

        # Natural key = pdb_id + chain_id
        df_pdbe["_nk"] = df_pdbe["pdb_id"] + "_" + df_pdbe["chain_id"].fillna("?")

        keys, new_rows, self._structure_lookup, self._structure_next_sk = (
            _assign_surrogate_keys(
                df_pdbe["_nk"],
                self._structure_lookup,
                self._structure_next_sk,
            )
        )

        if new_rows.empty:
            logger.info("DimStructure - no new structures to add.")
            return pd.DataFrame(
                columns=[
                    "structure_key", "protein_key", "pdb_id", "chain_id",
                    "resolution", "coverage", "method", "unp_start", "unp_end",
                ]
            )

        dim = new_rows.rename(
            columns={"surrogate_key": "structure_key", "natural_key": "_nk"}
        )
        dim = dim.merge(df_pdbe, on="_nk", how="left")
        
        # Note: uniprot_id is explicitly removed from output, replaced by protein_key
        dim = dim[[
            "structure_key", "protein_key", "pdb_id", "chain_id",
            "resolution", "coverage", "method", "unp_start", "unp_end",
        ]]

        logger.info("DimStructure - %d new structures added.", len(dim))
        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # FactBioactivity
    # ------------------------------------------------------------------

    def build_fact_bioactivity(self, df_chembl: pd.DataFrame) -> pd.DataFrame:
        """Build the ``FactBioactivity`` fact table."""
        logger.info("Building FactBioactivity - %d input rows.", len(df_chembl))

        fact = df_chembl.copy()

        # Resolve foreign keys
        fact["protein_key"] = fact["uniprot_id"].astype(str).map(self._protein_lookup)
        fact["drug_key"] = fact["drug_chembl_id"].astype(str).map(self._drug_lookup)
        fact["article_key"] = fact["pubmed_id"].astype(str).map(self._article_lookup)

        before = len(fact)
        fact = fact.dropna(subset=["protein_key", "drug_key"])
        if (dropped := before - len(fact)):
            logger.warning(
                "FactBioactivity - dropped %d rows with unresolved foreign keys.", dropped
            )

        fact["protein_key"] = fact["protein_key"].astype(int)
        fact["drug_key"] = fact["drug_key"].astype(int)
        fact["article_key"] = pd.to_numeric(fact["article_key"], errors="coerce").astype("Int64")

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
            "assay_organism",  
        ]
        
        available = [c for c in output_cols if c in fact.columns]
        fact = fact[available]

        logger.info(
            "FactBioactivity complete - %d rows, %d columns.", len(fact), len(available)
        )
        return fact.reset_index(drop=True)