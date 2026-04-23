"""Dimensional model builder.

Transforms cleaned DataFrames into the star-schema dimension and fact tables
used by the ELIQSIR data warehouse.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lookup helpers (natural key -> surrogate key)
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
    # DimDate 
    # ------------------------------------------------------------------

    def build_dim_date(self, fact_df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically generates the Data-Driven Date Dimension.
        Collects keys from ALL date columns in the fact table and 
        generates human-readable attributes.
        """
        # 1. Собираем даты из ВСЕХ колонок, где есть 'date_key'
        date_cols = [c for c in fact_df.columns if 'date_key' in c]
        unique_dates = pd.Series(dtype='Int64')
        
        for col in date_cols:
            unique_dates = pd.concat([unique_dates, fact_df[col].dropna()])
            
        unique_dates = pd.concat([unique_dates, pd.Series([19000101])])
        unique_dates = unique_dates.drop_duplicates().astype(int)
        
        dim_date = pd.DataFrame({'date_key': unique_dates})
        
        dim_date['_temp_date'] = pd.to_datetime(dim_date['date_key'], format='%Y%m%d', errors='coerce')
        dim_date['_temp_date'] = dim_date['_temp_date'].fillna(pd.Timestamp('1900-01-01'))

        dim_date['full_date'] = dim_date['_temp_date'].dt.strftime('%Y-%m-%d')
        dim_date['full_date_desc'] = dim_date['_temp_date'].dt.strftime('%B %d, %Y, %A')
        
        dim_date['year'] = dim_date['date_key'] // 10000
        dim_date['month_name'] = dim_date['_temp_date'].dt.strftime('%B')
        dim_date['day'] = dim_date['date_key'] % 100
        dim_date['quarter'] = dim_date['_temp_date'].dt.quarter
        dim_date['day_name'] = dim_date['_temp_date'].dt.strftime('%A')
        
        dim_date['is_weekend'] = dim_date['_temp_date'].dt.dayofweek.apply(
            lambda x: 'weekend' if x >= 5 else 'non-weekend'
        )

        dim_date['fractional_year'] = dim_date.apply(
            lambda row: 1900.0 if row['date_key'] == 19000101 else row['year'] + ((row['date_key'] % 10000) // 100 - 1) / 12.0, 
            axis=1
        ).round(3)
        
        dim_date['epoch_time'] = dim_date['_temp_date'].astype('int64') // 10**9

        dim_date = dim_date.drop(columns=['_temp_date'])

        columns_order = [
            'date_key', 'full_date', 'full_date_desc', 'year', 'month_name', 
            'day', 'quarter', 'day_name', 'is_weekend', 'fractional_year', 'epoch_time'
        ]

        return dim_date[columns_order].sort_values('date_key').reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimProtein
    # ------------------------------------------------------------------

    def build_dim_protein(self, df_uniprot: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimProtein`` dimension table."""
        logger.info("Building DimProtein - %d input rows.", len(df_uniprot))

        keys, new_rows, self._protein_lookup, self._protein_next_sk = (
            _assign_surrogate_keys(
                df_uniprot["uniprot_id"], self._protein_lookup, self._protein_next_sk
            )
        )

        if new_rows.empty:
            return pd.DataFrame(columns=[
                "protein_key", "uniprot_id", "gene_names", 
                "protein_name", "sequence", "sequence_length", "protein_families"
            ])

        dim = new_rows.rename(columns={"surrogate_key": "protein_key", "natural_key": "uniprot_id"})
        
        attrs_cols = [
            "uniprot_id", "gene_names", "protein_name", 
            "sequence", "sequence_length", "protein_families"
        ]
        available_attrs = [c for c in attrs_cols if c in df_uniprot.columns]
        dim = dim.merge(df_uniprot[available_attrs].drop_duplicates("uniprot_id"), on="uniprot_id", how="left")
        
        final_cols = [c for c in attrs_cols if c in dim.columns]
        return dim[["protein_key"] + final_cols].reset_index(drop=True)

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

        drug_df = df_chembl[available_cols].drop_duplicates(subset=["drug_chembl_id"])

        keys, new_rows, self._drug_lookup, self._drug_next_sk = (
            _assign_surrogate_keys(
                drug_df["drug_chembl_id"], self._drug_lookup, self._drug_next_sk
            )
        )

        if new_rows.empty:
            return pd.DataFrame(columns=["drug_key"] + drug_cols)

        dim = new_rows.rename(columns={"surrogate_key": "drug_key", "natural_key": "drug_chembl_id"})
        dim = dim.merge(drug_df, on="drug_chembl_id", how="left")

        return dim.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimArticle
    # ------------------------------------------------------------------

    def build_dim_article(self, df_chembl: pd.DataFrame, df_abstracts: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimArticle`` dimension table."""
        logger.info("Building DimArticle.")

        chembl_cols = ["pubmed_id", "article_title", "journal", "year"]
        avail_chembl = [c for c in chembl_cols if c in df_chembl.columns]
        
        article_meta = (
            df_chembl[avail_chembl]
            .drop_duplicates(subset=["pubmed_id"])
            .dropna(subset=["pubmed_id"])
        )
        article_meta["pubmed_id"] = article_meta["pubmed_id"].astype("Int64")

        abs_cols = ["pubmed_id", "abstract", "authors", "doi", "pub_date"]
        avail_abs = [c for c in abs_cols if c in df_abstracts.columns]
        
        df_abs = df_abstracts[avail_abs].copy()
        df_abs["pubmed_id"] = df_abs["pubmed_id"].astype("Int64")

        article_full = article_meta.merge(df_abs, on="pubmed_id", how="left")

        keys, new_rows, self._article_lookup, self._article_next_sk = (
            _assign_surrogate_keys(
                article_full["pubmed_id"].astype(str), self._article_lookup, self._article_next_sk
            )
        )

        if new_rows.empty:
            return pd.DataFrame(columns=["article_key", "pubmed_id", "article_title", "journal", "year", "abstract", "doi", "authors", "pub_date"])

        dim = new_rows.rename(columns={"surrogate_key": "article_key", "natural_key": "_pmid_str"})
        article_full["_pmid_str"] = article_full["pubmed_id"].astype(str)
        dim = dim.merge(article_full, on="_pmid_str", how="left")
        
        final_cols = [c for c in ["pubmed_id", "article_title", "journal", "year", "abstract", "doi", "authors", "pub_date"] if c in dim.columns]
        return dim[["article_key"] + final_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # DimStructure
    # ------------------------------------------------------------------

    def build_dim_structure(self, df_pdbe: pd.DataFrame) -> pd.DataFrame:
        """Build the ``DimStructure`` dimension table."""
        logger.info("Building DimStructure - %d input rows.", len(df_pdbe))

        df_pdbe = df_pdbe.copy()
        df_pdbe["protein_key"] = df_pdbe["uniprot_id"].astype(str).map(self._protein_lookup)
        df_pdbe = df_pdbe.dropna(subset=["protein_key"])
        df_pdbe["protein_key"] = df_pdbe["protein_key"].astype(int)
        
        df_pdbe["_nk"] = df_pdbe["pdb_id"] + "_" + df_pdbe["chain_id"].fillna("?")

        keys, new_rows, self._structure_lookup, self._structure_next_sk = (
            _assign_surrogate_keys(df_pdbe["_nk"], self._structure_lookup, self._structure_next_sk)
        )

        if new_rows.empty:
            return pd.DataFrame(columns=[
                "structure_key", "protein_key", "pdb_id", "chain_id",
                "resolution", "coverage", "method", "unp_start", "unp_end"
            ])

        dim = new_rows.rename(columns={"surrogate_key": "structure_key", "natural_key": "_nk"})
        dim = dim.merge(df_pdbe, on="_nk", how="left")
        
        return dim[[
            "structure_key", "protein_key", "pdb_id", "chain_id",
            "resolution", "coverage", "method", "unp_start", "unp_end"
        ]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # FactBioactivity
    # ------------------------------------------------------------------

    def build_fact_bioactivity(self, df_chembl: pd.DataFrame, df_abstracts: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build the ``FactBioactivity`` fact table.
        Integrates temporal imputation to map dates to DimDate surrogate keys.
        Prepares the Role-Playing dimensions for the enrichment phase.
        """
        logger.info("Building FactBioactivity - %d input rows.", len(df_chembl))

        fact = df_chembl.copy()

        # 1. Resolve standard foreign keys
        fact["protein_key"] = fact["uniprot_id"].astype(str).map(self._protein_lookup)
        fact["drug_key"] = fact["drug_chembl_id"].astype(str).map(self._drug_lookup)
        fact["article_key"] = fact["pubmed_id"].astype(str).map(self._article_lookup)

        fact = fact.dropna(subset=["protein_key", "drug_key"])
        fact["protein_key"] = fact["protein_key"].astype(int)
        fact["drug_key"] = fact["drug_key"].astype(int)
        fact["article_key"] = pd.to_numeric(fact["article_key"], errors="coerce").astype("Int64")

        # 2. Resolve Temporal Logic (Date Dimension Mapping)
        fact["date_key"] = 19000101  # Default unknown
        fact["date_precision"] = "Unknown"

        # Base fallback: Year from ChEMBL
        if "year" in fact.columns:
            valid_year = fact["year"].notna()
            # Impute to YYYY0101
            fact.loc[valid_year, "date_key"] = (fact.loc[valid_year, "year"].astype(int) * 10000) + 101
            fact.loc[valid_year, "date_precision"] = "Year"

        # High-precision override: Exact date from PubMed abstracts
        if df_abstracts is not None and "pub_date" in df_abstracts.columns:
            abs_dates = df_abstracts.dropna(subset=["pubmed_id", "pub_date"])[["pubmed_id", "pub_date"]].copy()
            abs_dates["pubmed_id"] = abs_dates["pubmed_id"].astype("Int64")
            
            # Merge dates into fact table
            fact = fact.merge(abs_dates, on="pubmed_id", how="left")
            
            # Parse PubMed dates. Assuming format is YYYY-MM-DD or parseable
            valid_pub_date = fact["pub_date"].notna()
            
            if valid_pub_date.any():
                parsed_dates = pd.to_datetime(fact.loc[valid_pub_date, "pub_date"], errors="coerce")
                valid_parsed = parsed_dates.notna()
                
                # Get the exact indices where dates were successfully parsed
                valid_indices = parsed_dates[valid_parsed].index
                
                # Apply high precision key
                fact.loc[valid_indices, "date_key"] = (
                    parsed_dates[valid_parsed].dt.strftime("%Y%m%d").astype(int)
                )
                fact.loc[valid_indices, "date_precision"] = "Exact"

        # 3. Finalize output columns (Initial filtering)
        output_cols = [
            "activity_id", "protein_key", "drug_key", "article_key", "date_key", "date_precision",
            "standard_type", "standard_value", "standard_units", "pchembl_value",
            "confidence_score", "assay_type", "assay_description", "assay_organism"
        ]
        
        available = [c for c in output_cols if c in fact.columns]
        fact = fact[available]

        # 4. Role-Playing Dimension Alignment
        # Rename the primary temporal key to match the new architecture
        if 'date_key' in fact.columns:
            fact = fact.rename(columns={'date_key': 'publication_date_key'})
            
        # Initialize the 5 temporal placeholders for the Enrichment phase
        temporal_roles = [
            'received_date_key', 
            'revised_date_key', 
            'accepted_date_key', 
            'epub_date_key', 
            'ppub_date_key'
        ]
        for role in temporal_roles:
            fact[role] = 19000101

        logger.info("FactBioactivity complete - %d rows.", len(fact))
        
        return fact.reset_index(drop=True)