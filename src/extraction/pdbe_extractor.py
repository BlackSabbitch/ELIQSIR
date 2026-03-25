"""PDBe Graph API extractor.

Fetches the *best 3-D structures* for a list of UniProt accessions from the
PDBe Graph API (``/mappings/best_structures/{uniprot_id}``).

The endpoint returns a curated, high-resolution subset of the Protein Data
Bank, so no additional quality filtering is required at extraction time.

Typical usage::

    from src.extraction import PdbeExtractor

    extractor = PdbeExtractor()
    df = extractor.extract(uniprot_ids=["P00533", "P04637"])
    df.to_csv("data/PDB/human_protein_pdb_structures.csv", index=False)
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URL = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/best_structures"
_DEFAULT_TIMEOUT = 15       # seconds per request
_DEFAULT_DELAY = 0.1        # seconds between requests (EBI rate limit)
_LOG_EVERY_N = 100          # progress log frequency

# Keys returned by the PDBe JSON response that we want to keep.
_STRUCTURE_KEYS = (
    "pdb_id",
    "chain_id",
    "resolution",
    "coverage",
    "experimental_method",
    "unp_start",
    "unp_end",
)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class PdbeExtractor:
    """Fetch best 3-D protein structures from the PDBe Graph API.

    Parameters
    ----------
    base_url:
        Override the API root URL (useful for testing).
    request_delay_s:
        Seconds to sleep between consecutive requests to comply with EBI
        server rate limits.
    timeout_s:
        Per-request HTTP timeout in seconds.
    max_retries:
        Number of automatic retries on transient network failures.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        request_delay_s: float = _DEFAULT_DELAY,
        timeout_s: int = _DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_delay_s = request_delay_s
        self.timeout_s = timeout_s

        self._session = self._build_session(max_retries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _fetch_one(self, uniprot_id: str) -> list[dict]:
        """Fetch structure records for a single UniProt accession.

        Returns an empty list when the protein has no deposited structures
        or when the API returns a non-200 status (e.g. 404).
        """
        url = f"{self.base_url}/{uniprot_id}"
        try:
            response = self._session.get(url, timeout=self.timeout_s)
            if response.status_code == 404:
                logger.debug("No structures found for %s (404).", uniprot_id)
                return []
            response.raise_for_status()
            data = response.json()
            structures = data.get(uniprot_id, [])
            records = []
            for s in structures:
                row = {"uniprot_id": uniprot_id}
                row.update({k: s.get(k) for k in _STRUCTURE_KEYS})
                # Rename key to match the schema convention.
                row["method"] = row.pop("experimental_method", None)
                records.append(row)
            return records
        except requests.RequestException as exc:
            logger.warning("Skipping %s – request failed: %s", uniprot_id, exc)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, uniprot_ids: Iterable[str]) -> pd.DataFrame:
        """Fetch PDB structure records for all supplied UniProt accessions.

        Parameters
        ----------
        uniprot_ids:
            Any iterable of UniProt accession strings.  Duplicates are
            silently deduplicated.

        Returns
        -------
        pd.DataFrame
            Columns: ``uniprot_id``, ``pdb_id``, ``chain_id``,
            ``resolution``, ``coverage``, ``method``, ``unp_start``,
            ``unp_end``.  Empty DataFrame when no structures are found.
        """
        unique_ids = list(dict.fromkeys(uniprot_ids))
        total = len(unique_ids)
        logger.info("Starting PDB structure fetch for %d unique proteins.", total)

        all_records: list[dict] = []
        for i, uid in enumerate(unique_ids, start=1):
            records = self._fetch_one(uid)
            all_records.extend(records)

            if i % _LOG_EVERY_N == 0:
                logger.info(
                    "Progress: %d / %d proteins processed – %d structures collected.",
                    i,
                    total,
                    len(all_records),
                )

            time.sleep(self.request_delay_s)

        df = pd.DataFrame(all_records)
        logger.info(
            "PDBe extraction complete – %d structures across %d proteins.",
            len(df),
            total,
        )
        return df
