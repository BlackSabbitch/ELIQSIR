"""NCBI PubMed abstract extractor.

Given a list of PubMed IDs, fetches full article abstracts via the NCBI
E-utilities API (``efetch``) in configurable batches.

Typical usage::

    from src.extraction import PubMedExtractor

    extractor = PubMedExtractor(email="your@email.com")
    df = extractor.extract(pubmed_ids=["12345678", "87654321"])
    df.to_csv("data/PubMedAbstracts/article_abstracts.csv", index=False)
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
from Bio import Entrez

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE = 200
_DEFAULT_DELAY_S = 0.5          # sleep between batches (NCBI policy: max 3 req/s)
_RETMODE = "xml"
_DB = "pubmed"


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class PubMedExtractor:
    """Fetch article abstracts from NCBI PubMed via the Entrez E-utils API.

    Parameters
    ----------
    email:
        **Required by NCBI** – identifies your application to the server.
        Supply a valid e-mail address to avoid being blocked.
    batch_size:
        Number of PubMed IDs sent per HTTP request.  NCBI recommends
        keeping this below 500; 200 is a safe default.
    request_delay_s:
        Seconds to sleep between batch requests to comply with NCBI's
        rate-limit policy (max 3 requests per second without an API key).
    """

    def __init__(
        self,
        email: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        request_delay_s: float = _DEFAULT_DELAY_S,
    ) -> None:
        if not email:
            raise ValueError(
                "An e-mail address is required by NCBI.  "
                "Set NCBI_EMAIL in your .env file."
            )
        self.email = email
        self.batch_size = batch_size
        self.request_delay_s = request_delay_s

        Entrez.email = self.email

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_abstract(article: dict) -> str:
        """Extract the abstract text from a parsed Entrez article record.

        Returns an empty string when the article has no abstract (e.g.
        editorials, errata, letters without an ``Abstract`` section).
        """
        try:
            parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            # ``AbstractText`` can be a list of structured sections or a
            # plain string – handle both.
            if isinstance(parts, list):
                return " ".join(str(p) for p in parts)
            return str(parts)
        except KeyError:
            return ""

    @staticmethod
    def _parse_pmid(article: dict) -> str:
        return str(article["MedlineCitation"]["PMID"])

    @staticmethod
    def _parse_authors(article: dict) -> str:
        """Return semicolon-separated 'LastName, ForeName' author strings.

        Collective authors (e.g. consortia) that have no ``LastName`` key
        are included via their ``CollectiveName`` if available.
        Returns an empty string when no ``AuthorList`` is present.
        """
        try:
            author_list = article["MedlineCitation"]["Article"]["AuthorList"]
            parts: list[str] = []
            for author in author_list:
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                name = f"{last}, {fore}".strip(", ")
                if not name:
                    name = str(author.get("CollectiveName", "")).strip()
                if name:
                    parts.append(name)
            return "; ".join(parts)
        except KeyError:
            return ""

    @staticmethod
    def _parse_pub_date(article: dict) -> tuple[str, int | None, int | None]:
        """Return ``(pub_date_str, year_int, month_int)`` from the PubDate node.

        *pub_date_str* is a best-effort ISO-like string (``"YYYY-MM-DD"``,
        ``"YYYY-MM"``, ``"YYYY"``, or ``""``).  *year_int* and *month_int*
        are ``None`` when the corresponding value is missing or unparseable.
        """
        _MONTH_MAP: dict[str, int] = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        try:
            pub_date = (
                article["MedlineCitation"]["Article"]["Journal"]
                       ["JournalIssue"]["PubDate"]
            )
            year_raw  = str(pub_date.get("Year",  "")).strip()
            month_raw = str(pub_date.get("Month", "")).strip()
            day_raw   = str(pub_date.get("Day",   "")).strip()

            year = int(year_raw) if year_raw.isdigit() else None
            if month_raw.isdigit():
                month_int: int | None = int(month_raw)
            else:
                month_int = _MONTH_MAP.get(month_raw)

            month_str = str(month_int) if month_int is not None else month_raw
            date_str = "-".join(part for part in [year_raw, month_str, day_raw] if part)
            return date_str, year, month_int
        except (KeyError, TypeError, ValueError):
            return "", None, None

    @staticmethod
    def _parse_doi(article: dict) -> str:
        """Extract the DOI string from ``ArticleIdList``, or return ``""``."""
        try:
            for aid in article["PubmedData"]["ArticleIdList"]:
                if aid.attributes.get("IdType") == "doi":
                    return str(aid)
            return ""
        except (KeyError, AttributeError):
            return ""

    def _fetch_batch(self, batch: list[str]) -> list[dict]:
        """Fetch and parse a single batch of PubMed IDs.

        Returns a list of dicts with keys:
        ``pubmed_id``, ``abstract``, ``authors``, ``pub_date``,
        ``year``, ``month``, ``doi``.
        On failure the entire batch is logged and skipped (no partial
        results are lost silently).
        """
        try:
            handle = Entrez.efetch(
                db=_DB,
                id=",".join(batch),
                rettype="abstract",
                retmode=_RETMODE,
            )
            records = Entrez.read(handle)
            rows: list[dict] = []
            for article in records["PubmedArticle"]:
                pub_date_str, year, month = self._parse_pub_date(article)
                rows.append(
                    {
                        "pubmed_id": self._parse_pmid(article),
                        "abstract":  self._parse_abstract(article),
                        "authors":   self._parse_authors(article),
                        "pub_date":  pub_date_str,
                        "year":      year,
                        "month":     month,
                        "doi":       self._parse_doi(article),
                    }
                )
            return rows
        except Exception as exc:  # noqa: BLE001 – intentional broad catch
            logger.error(
                "Failed to fetch batch starting with PMID %s: %s",
                batch[0],
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pubmed_ids: Iterable[str | int]) -> pd.DataFrame:
        """Fetch abstracts for all supplied PubMed IDs.

        Parameters
        ----------
        pubmed_ids:
            Any iterable of PubMed ID values (strings or integers).
            ``None`` / ``NaN`` values are automatically dropped.

        Returns
        -------
        pd.DataFrame
            Columns: ``pubmed_id`` (str), ``abstract`` (str),
            ``authors`` (str), ``pub_date`` (str), ``year`` (Int64),
            ``month`` (Int64), ``doi`` (str).
            Rows whose metadata could not be retrieved contain empty
            strings / ``NaN`` rather than raising an error.
        """
        # Normalise to a deduplicated list of strings, dropping nulls.
        id_series = pd.Series(pubmed_ids).dropna().astype(int).astype(str)
        unique_ids = id_series.unique().tolist()
        total = len(unique_ids)

        if total == 0:
            logger.warning("No valid PubMed IDs supplied – returning empty DataFrame.")
            return pd.DataFrame(
                columns=["pubmed_id", "abstract", "authors", "pub_date", "year", "month", "doi"]
            )

        logger.info(
            "Fetching abstracts for %d PubMed articles in batches of %d.",
            total,
            self.batch_size,
        )

        all_rows: list[dict] = []
        for start in range(0, total, self.batch_size):
            batch = unique_ids[start : start + self.batch_size]
            rows = self._fetch_batch(batch)
            all_rows.extend(rows)
            fetched = min(start + self.batch_size, total)
            logger.info("Progress: %d / %d abstracts retrieved.", fetched, total)
            time.sleep(self.request_delay_s)

        df = pd.DataFrame(
            all_rows,
            columns=["pubmed_id", "abstract", "authors", "pub_date", "year", "month", "doi"],
        )
        logger.info("PubMed extraction complete – %d abstracts retrieved.", len(df))
        return df
