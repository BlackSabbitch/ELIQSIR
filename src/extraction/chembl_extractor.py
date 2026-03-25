"""ChEMBL MySQL extractor.

Queries a *local* ChEMBL MySQL database and returns results as a
``pandas.DataFrame``.  The extractor can also **bootstrap** the ChEMBL
database from the official MySQL dump file that ships with the project
(``data/ChEMBLE/chembl_36/chembl_36_mysql/chembl_36_mysql.dmp``).

Typical usage::

    from src.extraction import ChemblExtractor

    extractor = ChemblExtractor(
        host="localhost",
        user="root",
        password="secret",
        database="chembl_36",
        dump_dir="data/ChEMBLE/chembl_36/chembl_36_mysql",
    )

    # First run – creates the DB and loads the dump (~10–30 min):
    extractor.setup_database()

    # Subsequent runs re-use the already-loaded database:
    extractor.setup_database()          # no-op if DB already populated

    # Run the production bioactivity query
    df = extractor.extract(uniprot_ids=["P00533", "P04637"])

    # …or an ad-hoc query
    df_raw = extractor.query("SELECT chembl_id FROM molecule_dictionary LIMIT 5")
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mysql.connector
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Name of the sentinel table we check to decide whether the dump has already
# been loaded.  ``molecule_dictionary`` is always present in ChEMBL.
_SENTINEL_TABLE = "molecule_dictionary"

# Name of the MySQL dump file (relative to dump_dir).
_DUMP_FILENAME = "chembl_36_mysql.dmp"


# ---------------------------------------------------------------------------
# SQL query templates
# ---------------------------------------------------------------------------

# Full bioactivity extraction joining 7 tables; the caller must supply the
# list of UniProt accession IDs as a temp-table approach (see ``extract``).
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
JOIN _protein_filter         pf  ON seq.accession       = pf.uniprot_id
WHERE tar.target_type    = 'SINGLE PROTEIN'
  AND act.standard_value IS NOT NULL
"""


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class ChemblExtractor:
    """Extract bioactivity data from a local ChEMBL MySQL database.

    The class also handles **first-time database setup**: call
    :py:meth:`setup_database` once to create the ``chembl_36`` schema and
    import the official MySQL dump.  Subsequent calls are safe no-ops.

    Parameters
    ----------
    host:
        MySQL server hostname or IP (e.g. ``"localhost"``).
    user:
        MySQL username (same credentials used for the warehouse).
    password:
        MySQL password.
    database:
        Name of the ChEMBL database to create/use (e.g. ``"chembl_36"``).
    port:
        MySQL port; defaults to ``3306``.
    dump_dir:
        Directory that contains ``chembl_36_mysql.dmp``.  Defaults to
        ``data/ChEMBLE/chembl_36/chembl_36_mysql/`` inside the project.
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        dump_dir: str | Path | None = None,
    ) -> None:
        self._config: dict[str, Any] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        # Config without a database selected – used for DB creation.
        self._admin_config: dict[str, Any] = {
            k: v for k, v in self._config.items() if k != "database"
        }
        self.dump_dir = Path(dump_dir) if dump_dir else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self, admin: bool = False) -> mysql.connector.MySQLConnection:
        """Open and return a new MySQL connection.

        Parameters
        ----------
        admin:
            When ``True`` connects without selecting a database (needed
            to run ``CREATE DATABASE``).
        """
        cfg = self._admin_config if admin else self._config
        return mysql.connector.connect(**cfg)

    def _database_is_populated(self) -> bool:
        """Return ``True`` when the sentinel table exists and has rows."""
        try:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT 1 FROM {_SENTINEL_TABLE} LIMIT 1"  # noqa: S608
                )
                return cursor.fetchone() is not None
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 – DB absent, wrong credentials, etc.
            return False

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def setup_database(self, force: bool = False) -> None:
        """Create the ChEMBL database and load the MySQL dump.

        This is a **safe, idempotent** operation:

        * If the database already exists **and** contains data, the method
          exits immediately (unless ``force=True``).
        * If the database is missing or empty, it creates it and pipes the
          ``.dmp`` file through the ``mysql`` CLI — exactly the two-step
          process described in the ``INSTALL_mysql`` file shipped with the
          ChEMBL download.

        Parameters
        ----------
        force:
            When ``True`` the dump is always re-loaded, even if the
            database already appears populated.  Useful after a corrupt
            import.

        Raises
        ------
        FileNotFoundError
            If ``dump_dir`` is not set or the ``.dmp`` file cannot be
            found at the expected path.
        RuntimeError
            If the ``mysql`` CLI command exits with a non-zero status.
        """
        if self.dump_dir is None:
            raise FileNotFoundError(
                "dump_dir was not supplied.  Pass it to ChemblExtractor() "
                "or set CHEMBL_DUMP_DIR in your .env file."
            )

        dump_file = self.dump_dir / _DUMP_FILENAME
        if not dump_file.exists():
            raise FileNotFoundError(
                f"ChEMBL dump file not found at '{dump_file}'. "
                "Download chembl_36_mysql.tar.gz from "
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ "
                f"and extract it into '{self.dump_dir}'."
            )

        if not force and self._database_is_populated():
            logger.info(
                "ChEMBL database '%s' is already populated – skipping setup.",
                self._config["database"],
            )
            return

        db_name = self._config["database"]
        host = self._config["host"]
        port = self._config["port"]
        user = self._config["user"]
        password = self._config["password"]

        # ── Step 1: CREATE DATABASE IF NOT EXISTS ──────────────────────
        logger.info("Creating ChEMBL database '%s' if it does not exist…", db_name)
        admin_conn = self._connect(admin=True)
        try:
            cursor = admin_conn.cursor()
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "DEFAULT CHARACTER SET utf8 "
                "DEFAULT COLLATE utf8_general_ci"
            )
            admin_conn.commit()
        finally:
            admin_conn.close()
        logger.info("Database '%s' ready.", db_name)

        # ── Step 2: Load dump via mysql CLI ────────────────────────────
        logger.info(
            "Loading ChEMBL dump from '%s' into '%s'… "
            "This may take 10–30 minutes.",
            dump_file,
            db_name,
        )
        cmd = [
            "mysql",
            f"-u{user}",
            f"-p{password}",
            f"-h{host}",
            f"-P{port}",
            db_name,
        ]
        with dump_file.open("rb") as dmp:
            result = subprocess.run(
                cmd,
                stdin=dmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"mysql import failed (exit {result.returncode}):\n"
                + result.stderr.decode(errors="replace")
            )

        logger.info("ChEMBL dump loaded successfully into '%s'.", db_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Execute *sql* and return the result as a ``DataFrame``.

        Parameters
        ----------
        sql:
            Any valid MySQL SELECT statement.

        Returns
        -------
        pd.DataFrame
            Query result; column names match the SELECT aliases.

        Raises
        ------
        mysql.connector.Error
            On any database-level error.
        """
        logger.info("Executing ad-hoc ChEMBL query:\n%s", sql.strip())
        conn = self._connect()
        try:
            df = pd.read_sql_query(sql, conn)
        finally:
            conn.close()
        logger.info("Query returned %d rows.", len(df))
        return df

    def extract(self, uniprot_ids: list[str]) -> pd.DataFrame:
        """Extract bioactivity records for the supplied UniProt accessions.

        The method uses a *temporary table* strategy to avoid generating
        huge ``IN (…)`` clauses: it creates a session-scoped temporary
        table ``_protein_filter``, bulk-inserts the accession list, and
        then performs an inner JOIN against it.  The table is automatically
        dropped when the connection is closed.

        Parameters
        ----------
        uniprot_ids:
            List of UniProt accession strings (e.g. ``["P00533", "P04637"]``).

        Returns
        -------
        pd.DataFrame
            Bioactivity records with columns: ``activity_id``,
            ``drug_chembl_id``, ``drug_name``, ``molecule_type``,
            ``molecular_weight``, ``canonical_smiles``,
            ``target_chembl_id``, ``target_name``, ``organism``,
            ``standard_type``, ``standard_value``, ``standard_units``,
            ``pchembl_value``, ``assay_type``, ``assay_description``,
            ``assay_organism``, ``confidence_score``, ``article_title``,
            ``journal``, ``year``, ``pubmed_id``, ``doi``,
            ``uniprot_id``.

        Raises
        ------
        ValueError
            If *uniprot_ids* is empty.
        mysql.connector.Error
            On any database-level error.
        """
        if not uniprot_ids:
            raise ValueError("uniprot_ids must not be empty.")

        unique_ids = list(dict.fromkeys(uniprot_ids))  # preserve order, deduplicate
        logger.info(
            "Connecting to ChEMBL MySQL at '%s/%s' – filtering on %d unique proteins.",
            self._config["host"],
            self._config["database"],
            len(unique_ids),
        )

        conn = self._connect()
        try:
            cursor = conn.cursor()

            # Temporary table is session-scoped in MySQL: dropped automatically
            # when the connection closes.
            cursor.execute("DROP TEMPORARY TABLE IF EXISTS _protein_filter")
            cursor.execute(
                "CREATE TEMPORARY TABLE _protein_filter "
                "(uniprot_id VARCHAR(20) NOT NULL PRIMARY KEY)"
            )
            cursor.executemany(
                "INSERT IGNORE INTO _protein_filter (uniprot_id) VALUES (%s)",
                [(uid,) for uid in unique_ids],
            )
            conn.commit()
            logger.debug(
                "Temporary filter table loaded with %d accessions.", len(unique_ids)
            )

            df = pd.read_sql_query(_BIOACTIVITY_QUERY, conn)
        finally:
            conn.close()

        logger.info(
            "ChEMBL extraction complete – %d bioactivity records retrieved.", len(df)
        )
        return df
