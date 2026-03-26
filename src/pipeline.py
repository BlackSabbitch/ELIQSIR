"""ETL pipeline orchestrator.

Wires together the extraction, transformation, and loading layers into a
single callable function ``run_pipeline``.

The pipeline runs in the following order:

    1. **Extract**  – UniProt, ChEMBL, PDBe, PubMed
    2. **Clean**    – :class:`~src.transformation.DataCleaner`
    3. **Model**    – :class:`~src.transformation.DimensionalModelBuilder`
    4. **Load**     – :class:`~src.loading.WarehouseLoader`

Each stage is independently logged; the function returns a summary dict so
that notebook cells can inspect intermediate DataFrames.

Checkpointing
-------------
By default every intermediate DataFrame is saved to ``data/checkpoints/``
after it is computed.  On the next run, stages whose output already exists on
disk are **skipped automatically** – their saved file is loaded instead.

This means:

* You can safely interrupt the pipeline mid-run and restart it; only the
  stages that did not finish will re-run.
* Changing code in a part of the pipeline that does not affect earlier stages
  (e.g. only editing the loader) still gives you the fast path for extraction
  and cleaning.
* Pass ``force_rerun=True`` to ignore all checkpoints and start fresh, or call
  ``ckpt.invalidate("stage_name")`` / ``ckpt.invalidate_stage("extraction")``
  to selectively invalidate individual stages.

Usage::

    from src.pipeline import run_pipeline

    # Normal run – skips stages already on disk
    summary = run_pipeline()

    # Force full re-run (overwrite every checkpoint)
    summary = run_pipeline(force_rerun=True)

    # Inspect what is cached
    from src.utils import CheckpointManager
    ckpt = CheckpointManager("data/checkpoints")
    print(ckpt.summary())

    print(summary["row_counts"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import settings
from src.extraction import (
    ChemblExtractor,
    PdbeExtractor,
    PubMedExtractor,
    UniProtExtractor,
)
from src.loading import WarehouseLoader
from src.transformation import DataCleaner, DimensionalModelBuilder
from src.utils.checkpoint import CheckpointManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    load_to_db: bool = True,
    save_csv: bool = True,
    csv_dir: Path | str | None = None,
    save_format: str = "parquet",
    checkpoint_dir: Path | str | None = None,
    use_checkpoints: bool = True,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Execute the full ELIQSIR ETL pipeline.

    Parameters
    ----------
    load_to_db:
        When ``True``, load the dimension and fact DataFrames into MySQL.
        Set to ``False`` for dry-run / notebook exploration mode.
    save_csv:
        When ``True``, persist each dimension and the fact table to the
        *csv_dir* directory in the format specified by *save_format*.
    csv_dir:
        Directory to write final output files.  Defaults to
        ``settings.data_dir / "staging"``.
    save_format:
        File format for final staging files.  One of ``"parquet"``
        *(default)*, ``"pickle"``, or ``"csv"``.
    checkpoint_dir:
        Directory for intermediate checkpoint files.  Defaults to
        ``settings.data_dir / "checkpoints"``.
    use_checkpoints:
        When ``True`` *(default)*, each stage is skipped if its checkpoint
        file already exists on disk.  Set to ``False`` to disable the
        checkpoint system entirely (equivalent to always passing
        ``force_rerun=True``).
    force_rerun:
        When ``True``, ignore existing checkpoints and re-execute every
        stage, overwriting saved files.  Equivalent to calling
        ``ckpt.clear()`` before the run.

    Returns
    -------
    dict
        Keys:

        * ``"dim_protein"``       – :class:`pd.DataFrame`
        * ``"dim_drug"``          – :class:`pd.DataFrame`
        * ``"dim_article"``       – :class:`pd.DataFrame`
        * ``"dim_structure"``     – :class:`pd.DataFrame`
        * ``"fact_bioactivity"``  – :class:`pd.DataFrame`
        * ``"row_counts"``        – ``dict[str, int]``
        * ``"checkpoint_manager"``– :class:`~src.utils.CheckpointManager`
    """
    settings.validate()

    # ------------------------------------------------------------------ #
    # Resolve output directories                                           #
    # ------------------------------------------------------------------ #
    out_dir = Path(csv_dir) if csv_dir else settings.data_dir / "staging"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else settings.data_dir / "checkpoints"

    ckpt = CheckpointManager(
        directory=ckpt_dir,
        fmt=save_format,
        force_rerun=(force_rerun or not use_checkpoints),
    )

    logger.info(
        "Pipeline starting.  Checkpoints: %s | directory: '%s' | force_rerun: %s",
        "enabled" if use_checkpoints else "disabled",
        ckpt_dir,
        force_rerun,
    )
    logger.info("Checkpoint status:\n%s", ckpt.summary())

    # ------------------------------------------------------------------ #
    # 1.  EXTRACTION                                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== STAGE 1: EXTRACTION ===")

    # 1a. UniProt
    def _extract_uniprot() -> pd.DataFrame:
        ext = UniProtExtractor(
            organism_id=settings.uniprot_organism_id,
            reviewed=settings.uniprot_reviewed_only,
        )
        return ext.extract()

    raw_uniprot = ckpt.load_or_run("raw_uniprot", _extract_uniprot, stage="extraction",
                                   description="Raw UniProt Swiss-Prot dump")

    # 1b. ChEMBL – ensure DB is set up, then filter on proteins from UniProt
    def _extract_chembl() -> pd.DataFrame:
        ext = ChemblExtractor(
            host=settings.mysql_host,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.chembl_mysql_db,
            port=settings.mysql_port,
            dump_dir=settings.chembl_dump_dir,
        )
        ext.setup_database()   # no-op if already populated
        accessions = raw_uniprot["accession"].dropna().unique().tolist()
        return ext.extract(uniprot_ids=accessions)

    raw_chembl = ckpt.load_or_run("raw_chembl", _extract_chembl, stage="extraction",
                                  description="Raw ChEMBL bioactivity records")

    # 1c. PDBe – fetch structures for proteins that have ChEMBL activity
    def _extract_pdbe() -> pd.DataFrame:
        active_accessions = raw_chembl["uniprot_id"].dropna().unique().tolist()
        ext = PdbeExtractor(
            request_delay_s=settings.pdbe_request_delay_s,
            timeout_s=settings.pdbe_timeout_s,
        )
        return ext.extract(uniprot_ids=active_accessions)

    raw_pdbe = ckpt.load_or_run("raw_pdbe", _extract_pdbe, stage="extraction",
                                description="Raw PDBe structure records")

    # 1d. PubMed – fetch abstracts for articles cited in ChEMBL
    def _extract_pubmed() -> pd.DataFrame:
        pmids = raw_chembl["pubmed_id"].dropna().unique().tolist()
        ext = PubMedExtractor(
            email=settings.ncbi_email,
            batch_size=settings.ncbi_batch_size,
            request_delay_s=settings.ncbi_request_delay_s,
        )
        return ext.extract(pubmed_ids=pmids)

    raw_pubmed = ckpt.load_or_run("raw_pubmed", _extract_pubmed, stage="extraction",
                                  description="Raw PubMed article abstracts")

    logger.info("Extraction stage complete.")

    # ------------------------------------------------------------------ #
    # 2.  CLEANING                                                         #
    # ------------------------------------------------------------------ #
    logger.info("=== STAGE 2: CLEANING ===")

    cleaner = DataCleaner()

    clean_uniprot = ckpt.load_or_run(
        "clean_uniprot",
        lambda: cleaner.clean_uniprot(raw_uniprot),
        stage="cleaning",
        description="Cleaned UniProt records",
    )
    clean_chembl = ckpt.load_or_run(
        "clean_chembl",
        lambda: cleaner.clean_chembl(raw_chembl),
        stage="cleaning",
        description="Cleaned ChEMBL bioactivity records",
    )
    clean_pdbe = ckpt.load_or_run(
        "clean_pdbe",
        lambda: cleaner.clean_pdbe(raw_pdbe),
        stage="cleaning",
        description="Cleaned PDBe structure records",
    )
    clean_pubmed = ckpt.load_or_run(
        "clean_pubmed",
        lambda: cleaner.clean_pubmed(raw_pubmed),
        stage="cleaning",
        description="Cleaned PubMed article metadata",
    )

    logger.info("Cleaning stage complete.")

    # ------------------------------------------------------------------ #
    # 3.  DIMENSIONAL MODELLING                                            #
    # ------------------------------------------------------------------ #
    logger.info("=== STAGE 3: DIMENSIONAL MODELLING ===")

    builder = DimensionalModelBuilder()

    dim_protein = ckpt.load_or_run(
        "dim_protein",
        lambda: builder.build_dim_protein(clean_uniprot),
        stage="modelling",
        description="Protein dimension table",
    )
    dim_drug = ckpt.load_or_run(
        "dim_drug",
        lambda: builder.build_dim_drug(clean_chembl),
        stage="modelling",
        description="Drug dimension table",
    )
    dim_article = ckpt.load_or_run(
        "dim_article",
        lambda: builder.build_dim_article(clean_chembl, clean_pubmed),
        stage="modelling",
        description="Article dimension table",
    )
    dim_structure = ckpt.load_or_run(
        "dim_structure",
        lambda: builder.build_dim_structure(clean_pdbe),
        stage="modelling",
        description="Structure dimension table",
    )
    fact_bioactivity = ckpt.load_or_run(
        "fact_bioactivity",
        lambda: builder.build_fact_bioactivity(clean_chembl),
        stage="modelling",
        description="Bioactivity fact table",
    )

    logger.info("Dimensional modelling stage complete.")

    # ------------------------------------------------------------------ #
    # 4a.  PERSIST STAGING FILES                                           #
    # ------------------------------------------------------------------ #
    _SAVERS: dict[str, tuple[str, Any]] = {
        "parquet": (".parquet", lambda df, p: df.to_parquet(p, index=False)),
        "pickle":  (".pkl",     lambda df, p: df.to_pickle(p)),
        "csv":     (".csv",     lambda df, p: df.to_csv(p, index=False)),
    }
    if save_format not in _SAVERS:
        raise ValueError(
            f"save_format must be one of {list(_SAVERS)}; got {save_format!r}"
        )
    ext, _saver = _SAVERS[save_format]

    if save_csv:
        logger.info(
            "=== STAGE 4a: SAVING '%s' files to '%s' ===", save_format.upper(), out_dir
        )
        _save_df(dim_protein,      out_dir / f"dim_protein{ext}",      _saver)
        _save_df(dim_drug,         out_dir / f"dim_drug{ext}",         _saver)
        _save_df(dim_article,      out_dir / f"dim_article{ext}",      _saver)
        _save_df(dim_structure,    out_dir / f"dim_structure{ext}",    _saver)
        _save_df(fact_bioactivity, out_dir / f"fact_bioactivity{ext}", _saver)

    # ------------------------------------------------------------------ #
    # 4b.  LOAD TO DATABASE                                                #
    # ------------------------------------------------------------------ #
    row_counts: dict[str, int] = {
        "dim_protein":      len(dim_protein),
        "dim_drug":         len(dim_drug),
        "dim_article":      len(dim_article),
        "dim_structure":    len(dim_structure),
        "fact_bioactivity": len(fact_bioactivity),
    }

    if load_to_db:
        logger.info("=== STAGE 4b: LOADING TO DATABASE ===")
        loader = WarehouseLoader()
        loader.initialise_schema()
        db_counts = loader.load_all(
            dim_protein=dim_protein,
            dim_drug=dim_drug,
            dim_article=dim_article,
            dim_structure=dim_structure,
            fact_bioactivity=fact_bioactivity,
        )
        row_counts.update(db_counts)
    else:
        logger.info("Database load skipped (load_to_db=False).")

    logger.info("=== PIPELINE COMPLETE ===  Row counts: %s", row_counts)
    logger.info("Final checkpoint state:\n%s", ckpt.summary())

    return {
        "dim_protein":        dim_protein,
        "dim_drug":           dim_drug,
        "dim_article":        dim_article,
        "dim_structure":      dim_structure,
        "fact_bioactivity":   fact_bioactivity,
        "row_counts":         row_counts,
        "checkpoint_manager": ckpt,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_df(df: pd.DataFrame, path: Path, saver: Any) -> None:
    """Persist *df* to *path* using the supplied *saver* callable."""
    saver(df, path)
    logger.info("Saved %d rows to '%s'.", len(df), settings.display_path(path))

