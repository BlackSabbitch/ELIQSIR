"""ChEMBL SQLite extractor.

Queries a local ChEMBL **SQLite** database and returns results as a
``pandas.DataFrame``.  The extractor automatically detects the ChEMBL
version from the folder structure and caches query results as Parquet
files for fast subsequent loads.

Expected folder structure after downloading and extracting from
https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/::

    data/ChEMBL/
    └── chembl_36/                      # ← version detected from folder name
        └── chembl_36_sqlite/
            ├── chembl_36.db            # ← SQLite database file
            └── INSTALL_sqlite

Typical usage::

    from src.extraction import ChemblExtractor

    extractor = ChemblExtractor(
        chembl_dir="data/ChEMBL",
    )

    # Run the production bioactivity query
    df = extractor.extract(uniprot_ids=["P00533", "P04637"])

    # …or an ad-hoc query
    df_raw = extractor.query("SELECT chembl_id FROM molecule_dictionary LIMIT 5")
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Shorthand for safe display of paths (project-relative, never absolute)
_dp = settings.display_path


# ---------------------------------------------------------------------------
# SQL query template (SQLite compatible)
# ---------------------------------------------------------------------------

_BIOACTIVITY_QUERY = """
SELECT
    act.activity_id,
    mol.chembl_id                  AS drug_chembl_id,
    mol.pref_name                  AS drug_name,
    mol.molecule_type,
    cp.mw_freebase                 AS molecular_weight,
    cs.canonical_smiles,
    tar.chembl_id                  AS target_chembl_id,
    tar.pref_name                  AS target_name,
    tar.organism,
    act.standard_type,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    ass.assay_type,
    ass.description                AS assay_description,
    ass.assay_organism,
    ass.confidence_score,
    doc.title                      AS article_title,
    doc.journal,
    doc.year,
    doc.pubmed_id,
    doc.doi,
    seq.accession                  AS uniprot_id
FROM activities              act
JOIN molecule_dictionary     mol ON act.molregno        = mol.molregno
LEFT JOIN compound_properties cp  ON mol.molregno        = cp.molregno
LEFT JOIN compound_structures cs  ON mol.molregno        = cs.molregno
JOIN assays                  ass ON act.assay_id        = ass.assay_id
JOIN target_dictionary       tar ON ass.tid             = tar.tid
JOIN docs                    doc ON ass.doc_id          = doc.doc_id
JOIN target_components       tc  ON tar.tid             = tc.tid
JOIN component_sequences     seq ON tc.component_id     = seq.component_id
WHERE tar.target_type    = 'SINGLE PROTEIN'
  AND act.standard_value IS NOT NULL
  AND seq.accession IN ({placeholders})
"""


# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------


def _print_progress(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print a simple progress bar to stdout."""
    percent = current / total if total > 0 else 1.0
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} |{bar}| {current:,}/{total:,} ({percent:.1%})", end="", flush=True)
    if current >= total:
        print()  # newline when complete


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------


def discover_chembl_database(chembl_dir: Path) -> tuple[str, Path]:
    """Auto-discover the ChEMBL version and database path.

    Scans ``chembl_dir`` for folders matching the pattern ``chembl_XX``
    (where XX is the version number), then locates the SQLite database
    file inside the ``chembl_XX_sqlite/`` subdirectory.

    Parameters
    ----------
    chembl_dir:
        Root directory containing extracted ChEMBL downloads (e.g.
        ``data/ChEMBL/``).

    Returns
    -------
    tuple[str, Path]
        A tuple of (version_string, db_path), e.g.
        ``("36", Path("data/ChEMBL/chembl_36/chembl_36_sqlite/chembl_36.db"))``.

    Raises
    ------
    FileNotFoundError
        If no ChEMBL version folder or database file is found.
    """
    chembl_dir = Path(chembl_dir)
    if not chembl_dir.exists():
        raise FileNotFoundError(f"ChEMBL directory not found: {_dp(chembl_dir)}")

    # Pattern: chembl_XX where XX is one or more digits
    version_pattern = re.compile(r"^chembl_(\d+)$")

    # Find all matching version directories and sort by version number (descending)
    versions: list[tuple[int, Path]] = []
    for item in chembl_dir.iterdir():
        if item.is_dir():
            match = version_pattern.match(item.name)
            if match:
                version_num = int(match.group(1))
                versions.append((version_num, item))

    if not versions:
        raise FileNotFoundError(
            f"No ChEMBL version folder found in '{_dp(chembl_dir)}'. "
            "Expected folder structure: chembl_XX/ (e.g. chembl_36/). "
            "Download from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ "
            "and extract into this directory."
        )

    # Use the latest version (highest number)
    versions.sort(key=lambda x: x[0], reverse=True)
    version_num, version_dir = versions[0]
    version_str = str(version_num)

    # Look for the SQLite subdirectory
    sqlite_subdir = version_dir / f"chembl_{version_num}_sqlite"
    if not sqlite_subdir.exists():
        raise FileNotFoundError(
            f"SQLite subdirectory not found at '{_dp(sqlite_subdir)}'. "
            f"Expected: {_dp(version_dir)}/chembl_{version_num}_sqlite/"
        )

    # Look for the .db file
    db_file = sqlite_subdir / f"chembl_{version_num}.db"
    if not db_file.exists():
        # Try to find any .db file in the directory
        db_files = list(sqlite_subdir.glob("*.db"))
        if db_files:
            db_file = db_files[0]
        else:
            raise FileNotFoundError(
                f"SQLite database file not found at '{_dp(db_file)}'. "
                "Download the SQLite version from "
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ "
                f"and extract it into '{_dp(sqlite_subdir)}'."
            )

    return version_str, db_file


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class ChemblExtractor:
    """Extract bioactivity data from a local ChEMBL SQLite database.

    The class automatically detects the ChEMBL version from the folder
    structure and caches extraction results as Parquet files for fast
    subsequent loads.

    Parameters
    ----------
    chembl_dir:
        Root directory containing ChEMBL data (e.g. ``data/ChEMBL/``).
        The extractor will auto-discover the latest version folder.
    db_path:
        Explicit path to the SQLite database file. If provided, this
        overrides auto-discovery.
    cache_dir:
        Directory for caching Parquet files. Defaults to
        ``chembl_dir / "cache"``.
    use_cache:
        Whether to use cached Parquet files if available. Defaults to
        ``True``.

    Attributes
    ----------
    version : str
        Detected ChEMBL version (e.g. ``"36"``).
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(
        self,
        chembl_dir: str | Path | None = None,
        db_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        if db_path is not None:
            # Explicit path provided
            self.db_path = Path(db_path)
            if not self.db_path.exists():
                raise FileNotFoundError(f"ChEMBL database not found at '{_dp(self.db_path)}'")
            # Extract version from filename (e.g. chembl_36.db → 36)
            match = re.search(r"chembl_(\d+)", self.db_path.name)
            self.version = match.group(1) if match else "unknown"
            self._chembl_dir = self.db_path.parent.parent.parent
        elif chembl_dir is not None:
            # Auto-discover from directory
            self._chembl_dir = Path(chembl_dir)
            self.version, self.db_path = discover_chembl_database(self._chembl_dir)
        else:
            raise ValueError("Either chembl_dir or db_path must be provided.")

        # Setup cache directory
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = self._chembl_dir / "cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._use_cache = use_cache

        # Log discovery
        logger.info("=" * 60)
        logger.info("ChEMBL SQLite Extractor initialized")
        logger.info("  Version  : %s", self.version)
        logger.info("  Database : %s", _dp(self.db_path))
        logger.info("  Cache dir: %s", _dp(self._cache_dir))
        logger.info("  Use cache: %s", self._use_cache)
        logger.info("=" * 60)

        # Print to stdout for notebook visibility
        print(f"✓ ChEMBL version {self.version} detected")
        print(f"  Database: {_dp(self.db_path)}")
        print(f"  Size: {self.db_path.stat().st_size / (1024**3):.2f} GB")

    def _connect(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Return the path for a cache file."""
        safe_key = re.sub(r"[^\w\-]", "_", cache_key)
        return self._cache_dir / f"chembl_{self.version}_{safe_key}.parquet"

    def _load_from_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Load DataFrame from cache if it exists."""
        if not self._use_cache:
            return None
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            logger.info("Loading from cache: %s", _dp(cache_path))
            print(f"📂 Loading from cache: {_dp(cache_path)}")
            return pd.read_parquet(cache_path)
        return None

    def _save_to_cache(self, df: pd.DataFrame, cache_key: str) -> Path:
        """Save DataFrame to cache and return the path."""
        cache_path = self._get_cache_path(cache_key)
        df.to_parquet(cache_path, index=False, compression="snappy")
        size_mb = cache_path.stat().st_size / (1024**2)
        logger.info("Saved to cache: %s (%.1f MB)", _dp(cache_path), size_mb)
        print(f"💾 Saved to cache: {_dp(cache_path)} ({size_mb:.1f} MB)")
        return cache_path

    def query(self, sql: str) -> pd.DataFrame:
        """Execute *sql* and return the result as a ``DataFrame``.

        Parameters
        ----------
        sql:
            Any valid SQLite SELECT statement.

        Returns
        -------
        pd.DataFrame
            Query result; column names match the SELECT aliases.
        """
        logger.info("Executing ad-hoc ChEMBL query:\n%s", sql.strip()[:200])
        conn = self._connect()
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        logger.info("Query returned %d rows.", len(df))
        return df

    def extract(
        self,
        uniprot_ids: list[str],
        batch_size: int = 500,
        use_cache: bool | None = None,
    ) -> pd.DataFrame:
        """Extract bioactivity records for the supplied UniProt accessions.

        The method batches queries to handle large ID lists efficiently and
        displays a progress bar during extraction.  Results are cached as a
        Parquet file for fast subsequent loads.

        Parameters
        ----------
        uniprot_ids:
            List of UniProt accession strings (e.g. ``["P00533", "P04637"]``).
        batch_size:
            Number of UniProt IDs to query per batch. Defaults to 500.
        use_cache:
            Override instance-level cache setting. If ``None``, uses the
            instance setting.

        Returns
        -------
        pd.DataFrame
            Bioactivity records with 23 columns including ``activity_id``,
            ``drug_chembl_id``, ``uniprot_id``, etc.

        Raises
        ------
        ValueError
            If *uniprot_ids* is empty.
        """
        if not uniprot_ids:
            raise ValueError("uniprot_ids must not be empty.")

        # Deduplicate and sort for consistent cache keys
        unique_ids = sorted(set(uniprot_ids))
        num_proteins = len(unique_ids)

        logger.info(
            "ChEMBL extraction requested for %d unique proteins.", num_proteins
        )
        print(f"\n🔬 Extracting bioactivity data for {num_proteins:,} proteins...")

        # Check cache
        should_use_cache = use_cache if use_cache is not None else self._use_cache
        if should_use_cache:
            # Create a cache key based on protein count and hash
            # (Full ID list would make filename too long)
            id_hash = hash(tuple(unique_ids)) % 10**8
            cache_key = f"bioactivity_{num_proteins}p_{id_hash}"
            cached_df = self._load_from_cache(cache_key)
            if cached_df is not None:
                print(f"✓ Loaded {len(cached_df):,} records from cache")
                return cached_df
        else:
            cache_key = None

        # Query in batches with progress bar
        conn = self._connect()
        try:
            all_dfs: list[pd.DataFrame] = []
            total_batches = (num_proteins + batch_size - 1) // batch_size

            print(f"  Querying ChEMBL v{self.version} in {total_batches} batches...")

            for i in range(0, num_proteins, batch_size):
                batch = unique_ids[i : i + batch_size]
                batch_num = i // batch_size + 1

                # Build query with placeholders
                placeholders = ",".join("?" * len(batch))
                query = _BIOACTIVITY_QUERY.format(placeholders=placeholders)

                # Execute query
                df_batch = pd.read_sql_query(query, conn, params=batch)
                all_dfs.append(df_batch)

                # Update progress bar
                _print_progress(
                    min(i + batch_size, num_proteins),
                    num_proteins,
                    prefix="  Progress",
                )

                logger.debug(
                    "Batch %d/%d: %d proteins → %d records",
                    batch_num,
                    total_batches,
                    len(batch),
                    len(df_batch),
                )

            # Combine all batches
            df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        finally:
            conn.close()

        logger.info(
            "ChEMBL extraction complete – %d bioactivity records retrieved.", len(df)
        )
        print(f"\n✓ Extracted {len(df):,} bioactivity records")

        # Summary statistics
        if len(df) > 0:
            print(f"  Unique drugs   : {df['drug_chembl_id'].nunique():,}")
            print(f"  Unique targets : {df['target_chembl_id'].nunique():,}")
            print(f"  Unique proteins: {df['uniprot_id'].nunique():,}")
            print(f"  With PubMed ID : {df['pubmed_id'].notna().sum():,}")

        # Save to cache
        if cache_key is not None and len(df) > 0:
            self._save_to_cache(df, cache_key)

        return df

    def get_table_info(self) -> pd.DataFrame:
        """Return information about all tables in the database."""
        sql = """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        ORDER BY name
        """
        return self.query(sql)

    def get_row_counts(self, tables: list[str] | None = None) -> dict[str, int]:
        """Return row counts for specified tables (or key tables if None)."""
        if tables is None:
            tables = [
                "activities",
                "molecule_dictionary",
                "assays",
                "target_dictionary",
                "docs",
                "compound_structures",
            ]

        counts = {}
        conn = self._connect()
        try:
            cursor = conn.cursor()
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                    counts[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    counts[table] = -1  # Table doesn't exist
        finally:
            conn.close()

        return counts

    def print_database_info(self) -> None:
        """Print summary information about the ChEMBL database."""
        print(f"\n{'='*60}")
        print(f"ChEMBL v{self.version} Database Summary")
        print(f"{'='*60}")
        print(f"Database file: {_dp(self.db_path)}")
        print(f"Size: {self.db_path.stat().st_size / (1024**3):.2f} GB")
        print("\nKey table row counts:")
        
        counts = self.get_row_counts()
        for table, count in counts.items():
            if count >= 0:
                print(f"  {table:25s}: {count:>12,}")
            else:
                print(f"  {table:25s}: (not found)")
        print(f"{'='*60}\n")
