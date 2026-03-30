"""Central configuration module.

All runtime settings are read from environment variables (populated by the
``.env`` file at the project root via ``python-dotenv``).  The module exposes
a single ``Settings`` dataclass instance (``settings``) that the rest of the
codebase imports directly:

    from src.config import settings
    print(settings.mysql_host)

No hard-coded credentials or paths are stored here – only defaults that are
safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load .env
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable application-wide settings.

    Attributes are populated once at import time from the environment.
    Freezing the dataclass prevents accidental mutation after startup.
    """

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data")))
    )

    # ------------------------------------------------------------------
    # ChEMBL SQLite source database
    # The extractor auto-discovers the ChEMBL version from folder structure.
    # Expected structure: data/ChEMBL/chembl_XX/chembl_XX_sqlite/chembl_XX.db
    # ------------------------------------------------------------------
    chembl_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "CHEMBL_DIR",
                str(_PROJECT_ROOT / "data" / "ChEMBL"),
            )
        )
    )

    # ------------------------------------------------------------------
    # UniProt API
    # ------------------------------------------------------------------
    uniprot_base_url: str = "https://rest.uniprot.org/uniprotkb/stream"
    uniprot_organism_id: int = field(
        default_factory=lambda: int(os.getenv("UNIPROT_ORGANISM_ID", "9606"))
    )
    uniprot_reviewed_only: bool = field(
        default_factory=lambda: os.getenv("UNIPROT_REVIEWED_ONLY", "true").lower() == "true"
    )

    # ------------------------------------------------------------------
    # PDBe API
    # ------------------------------------------------------------------
    pdbe_base_url: str = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures"
    pdbe_request_delay_s: float = field(
        default_factory=lambda: float(os.getenv("PDBE_REQUEST_DELAY_S", "0.1"))
    )
    pdbe_timeout_s: int = field(
        default_factory=lambda: int(os.getenv("PDBE_TIMEOUT_S", "15"))
    )

    # ------------------------------------------------------------------
    # NCBI / PubMed
    # ------------------------------------------------------------------
    ncbi_email: str = field(
        default_factory=lambda: os.getenv("NCBI_EMAIL", "")
    )
    ncbi_batch_size: int = field(
        default_factory=lambda: int(os.getenv("NCBI_BATCH_SIZE", "200"))
    )
    ncbi_request_delay_s: float = field(
        default_factory=lambda: float(os.getenv("NCBI_REQUEST_DELAY_S", "0.5"))
    )

    # ------------------------------------------------------------------
    # MySQL / Data Warehouse
    # ------------------------------------------------------------------
    mysql_host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "localhost"))
    mysql_port: int = field(
        default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306"))
    )
    mysql_user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", ""))
    mysql_password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    mysql_db: str = field(default_factory=lambda: os.getenv("MYSQL_DB", "eliqsir_dw"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    # ------------------------------------------------------------------
    # Derived helpers (properties are not supported on frozen dataclasses,
    # so we use regular methods)
    # ------------------------------------------------------------------
    def mysql_url(self) -> str:
        """Return a SQLAlchemy-compatible MySQL connection URL."""
        return (
            f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
        )

    def display_path(self, path: str | Path) -> str:
        """Return a project-relative path string safe for display/logging.

        Converts an absolute path to a path relative to the project root
        (e.g. ``data/ChEMBL/chembl_36/...``), so that the user's home
        directory is never leaked into notebook output or log files.

        Falls back to ``str(path)`` when the path is not under the project
        root (e.g. ``/tmp/something``).
        """
        try:
            return str(Path(path).resolve().relative_to(self.project_root))
        except ValueError:
            # Path is outside the project tree — return as-is
            return str(path)

    def validate(self) -> None:
        """Raise ``ValueError`` for missing required settings."""
        missing: list[str] = []
        if not self.ncbi_email:
            missing.append("NCBI_EMAIL")
        if not self.mysql_user:
            missing.append("MYSQL_USER")
        if not self.mysql_password:
            missing.append("MYSQL_PASSWORD")
        if missing:
            raise ValueError(
                f"The following required environment variables are not set: "
                f"{', '.join(missing)}. "
                f"Please add them to the .env file at the project root."
            )


# ---------------------------------------------------------------------------
# Module-level singleton – import this everywhere
# ---------------------------------------------------------------------------

settings = Settings()
