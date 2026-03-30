"""Extraction sub-package.

Exposes the four extractor classes at the package level so that callers can
write ``from src.extraction import UniProtExtractor`` instead of the full
dotted path.
"""

from src.extraction.uniprot_extractor import UniProtExtractor
from src.extraction.chembl_extractor import ChemblExtractor
from src.extraction.pdbe_extractor import PdbeExtractor
from src.extraction.pubmed_extractor import PubMedExtractor

__all__ = [
    "UniProtExtractor",
    "ChemblExtractor",
    "PdbeExtractor",
    "PubMedExtractor",
]
