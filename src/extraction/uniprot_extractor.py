"""UniProt REST API extractor.

Retrieves reviewed (Swiss-Prot) human proteome entries from the UniProt
``/stream`` endpoint and returns a tidy ``pandas.DataFrame``.

Typical usage::

    from src.extraction import UniProtExtractor

    extractor = UniProtExtractor()
    df = extractor.extract()
    df.to_csv("data/UniProt/human_swissprot_uniprot.csv", index=False)
"""

from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FIELDS = (
    "accession,"
    "gene_names,"
    "protein_name,"
    "organism_name,"
    "sequence,"
    "protein_families,"
    "ec,"
    "cc_catalytic_activity"
)
_DEFAULT_BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"

# Columns as returned by the UniProt TSV stream (order may vary by field list).
# Keys are the raw TSV header names; values are the snake_case aliases used
# throughout the pipeline.
_EXPECTED_COLUMNS = {
    "Entry": "accession",
    "Gene Names": "gene_names",
    "Protein names": "protein_name",
    "Organism": "organism_name",
    "Sequence": "protein_sequence",
    "Protein families": "protein_class",
    "EC number": "ec_number",
    "Catalytic activity": "catalyzed_reaction",
}


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class UniProtExtractor:
    """Fetch reviewed human proteins from the UniProt REST API.

    Parameters
    ----------
    organism_id:
        NCBI taxonomy identifier.  Defaults to 9606 (Homo sapiens).
    reviewed:
        When ``True`` only Swiss-Prot (manually curated) entries are fetched.
    fields:
        Comma-separated list of UniProt return columns.
    base_url:
        Override the API endpoint (useful for testing with a mock server).
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of automatic retries on transient network errors.
    """

    def __init__(
        self,
        organism_id: int = 9606,
        reviewed: bool = True,
        fields: str = _DEFAULT_FIELDS,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.organism_id = organism_id
        self.reviewed = reviewed
        self.fields = fields
        self.base_url = base_url
        self.timeout = timeout

        self._session = self._build_session(max_retries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        """Return a ``requests.Session`` with automatic retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _build_params(self) -> dict[str, str]:
        """Construct query parameters for the API request."""
        reviewed_str = "true" if self.reviewed else "false"
        return {
            "query": f"organism_id:{self.organism_id} AND reviewed:{reviewed_str}",
            "fields": self.fields,
            "format": "tsv",
        }

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename UniProt column headers to snake_case identifiers."""
        rename_map = {k: v for k, v in _EXPECTED_COLUMNS.items() if k in df.columns}
        return df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """Fetch proteins and return a normalised ``DataFrame``.

        Returns
        -------
        pd.DataFrame
            Columns: ``accession``, ``gene_names``, ``protein_name``,
            ``organism_name``, ``protein_sequence``, ``protein_class``,
            ``ec_number``, ``catalyzed_reaction``.

        Raises
        ------
        requests.HTTPError
            If the server returns a non-2xx status code after all retries.
        """
        params = self._build_params()
        logger.info(
            "Fetching UniProt data – organism_id=%s, reviewed=%s",
            self.organism_id,
            self.reviewed,
        )

        response = self._session.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), sep="\t")
        df = self._normalise_columns(df)

        logger.info("UniProt extraction complete – %d proteins retrieved.", len(df))
        return df
