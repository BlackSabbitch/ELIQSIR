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
    # DimDate (Static Generation)
    # ------------------------------------------------------------------

    def build_dim_date(self, start_year: int = 1950, end_year: int = 2030) -> pd.DataFrame:
        """
        Build the ``DimDate`` dimension table.
        Generates a continuous temporal calendar with ML-specific features
        (e.g., fractional_year) to prevent data leakage during graph training.
        """
        logger.info("Building DimDate from %d to %d.", start_year, end_year)

        date_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
        df = pd.DataFrame({"full_date": date_range})

        # Generate YYYYMMDD integer surrogate key for optimal B-Tree indexing
        df["date_key"] = df["full_date"].dt.strftime("%Y%m%d").astype(int)

        df["year"] = df["full_date"].dt.year.astype("Int16")
        df["month"] = df["full_date"].dt.month.astype("Int8")
        df["day"] = df["full_date"].dt.day.astype("Int8")
        df["quarter"] = df["full_date"].dt.quarter.astype("Int8")
        df["day_of_week"] = df["full_date"].dt.dayofweek.astype("Int8") 
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # ML temporal feature
        days_in_year = df["full_date"].dt.is_leap_year.map({True: 366.0, False: 365.0})
        df["fractional_year"] = df["year"] + (df["full_date"].dt.dayofyear - 1) / days_in_year
        df["epoch_time"] = df["full_date"].astype("int64") // 10**9

        columns = [
            "date_key", "full_date", "year", "month", "day", 
            "quarter", "day_of_week", "is_weekend", 
            "fractional_year", "epoch_time"
        ]
        
        # Add default unknown date row to satisfy Foreign Key constraints for missing data
        unknown_row = pd.DataFrame([{
            "date_key": 19000101, "full_date": pd.Timestamp("1900-01-01"),
            "year": 1900, "month": 1, "day": 1, "quarter": 1, 
            "day_of_week": 0, "is_weekend": 0, "fractional_year": 1900.0, "epoch_time": 0
        }])
        
        df = pd.concat([unknown_row, df], ignore_index=True)
        return df[columns]

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

        # 3. Finalize output columns
        output_cols = [
            "activity_id", "protein_key", "drug_key", "article_key", "date_key", "date_precision",
            "standard_type", "standard_value", "standard_units", "pchembl_value",
            "confidence_score", "assay_type", "assay_description", "assay_organism"
        ]
        
        available = [c for c in output_cols if c in fact.columns]
        fact = fact[available]

        logger.info("FactBioactivity complete - %d rows.", len(fact))
        return fact.reset_index(drop=True)